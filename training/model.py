"""
Deletion pathogenicity predictor with robust imbalance handling.

Features:
- Uses class weights and stratified sampling instead of upsampling
- Handles imbalanced datasets (e.g., 5000 pathogenic vs 2000 benign)
- Stratifies train/test by deletion size bins to avoid length confounding
- Uses weighted cross-validation
- Ensemble: RandomForest + GradientBoosting (+ XGBoost if installed)
"""
from typing import List, Dict, Any, Optional
import logging
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, precision_recall_curve, auc
)
from sklearn.utils.class_weight import compute_sample_weight, compute_class_weight
from sklearn.base import clone
import math

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class DeletionPathogenicityPredictor:
    def __init__(self, threshold: float = 0.5, n_jobs: int = -1, random_state: int = 42):
        self.threshold = threshold
        self.random_state = random_state
        self.n_jobs = n_jobs

        self.scaler = RobustScaler()
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.feature_columns: Optional[List[str]] = None
        self.categorical_columns = ['gene']  # Only gene remains

        # Placeholder for model
        self.model: Optional[VotingClassifier] = None
        self.has_xgboost = False
        self.class_weights = None

    def _calculate_sequence_features(self, seq: str) -> dict:
        if not seq or not isinstance(seq, str):
            return {
                'gc_content': 0.5,
                'repeat_content': 0.0,
                'homopolymer_run': 0.0,
                'cpg_islands': 0.0,
                'at_content': 0.5,
                'complexity_score': 0.5
            }
        seq = seq.upper()
        seq_len = len(seq)
        gc_count = sum(1 for b in seq if b in 'GC')
        gc_content = gc_count / seq_len if seq_len > 0 else 0.5
        at_content = 1.0 - gc_content
        cpg_count = seq.count('CG')
        cpg_islands = cpg_count / (seq_len - 1) if seq_len > 1 else 0.0

        max_hpoly = 0
        cur = 1
        for i in range(1, seq_len):
            if seq[i] == seq[i-1]:
                cur += 1
                if cur > max_hpoly:
                    max_hpoly = cur
            else:
                cur = 1
        homopolymer_run = max_hpoly / seq_len if seq_len > 0 else 0.0

        repeat_score = 0.0
        for repeat_len in (2, 3, 4):
            if seq_len >= repeat_len * 2:
                for i in range(seq_len - 2 * repeat_len):
                    pattern = seq[i:i+repeat_len]
                    if seq[i+repeat_len:i+2*repeat_len] == pattern:
                        repeat_score += 1
        repeat_content = min(repeat_score / seq_len if seq_len > 0 else 0.0, 1.0)

        counts = {b: seq.count(b) for b in 'ACGT'}
        complexity = 0.0
        for c in counts.values():
            if c > 0:
                p = c / seq_len
                complexity -= p * math.log2(p)
        complexity_score = complexity / 2.0 if seq_len > 0 else 0.5

        return {
            'gc_content': gc_content,
            'repeat_content': repeat_content,
            'homopolymer_run': homopolymer_run,
            'cpg_islands': cpg_islands,
            'at_content': at_content,
            'complexity_score': complexity_score
        }

    def _encode_gene_features(self, gene: str) -> dict:
        known_disease_genes = {
            'BRCA1', 'BRCA2', 'TP53', 'PTEN', 'RB1', 'APC', 'MLH1', 'MSH2',
            'VHL', 'NF1', 'NF2', 'TSC1', 'TSC2', 'ATM', 'CHEK2', 'PALB2',
            'CDKN2A', 'STK11', 'CDH1', 'SMAD4', 'BMPR1A', 'MUTYH', 'MSH6',
            'PMS2', 'EPCAM', 'POLD1', 'POLE', 'RAD51C', 'RAD51D', 'BRIP1'
        }
        has_gene = bool(gene) and gene != 'N/A'
        gene_symbol = gene
        if gene and gene.startswith('ENSG'):
            gene_symbol = gene.split('.')[0] if '.' in gene else gene
        return {
            'has_gene': 1.0 if has_gene else 0.0,
            'is_known_disease_gene': 1.0 if gene_symbol in known_disease_genes else 0.0,
            'gene_length': len(gene) if gene else 0,
            'is_ensembl_id': 1.0 if (gene and gene.startswith('ENSG')) else 0.0
        }

    def _encode_features(self, variants: List[Dict[str, Any]], fit: bool = False) -> pd.DataFrame:
        df = pd.DataFrame(variants)
        for col in ('chr', 'start', 'end'):
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")

        df['chr'] = df['chr'].astype(str).str.replace('chr', '', regex=False)
        df['chr'] = df['chr'].replace({'X': '23', 'Y': '24', 'M': '25'})
        df['chr'] = pd.to_numeric(df['chr'], errors='coerce').fillna(0).astype(int)

        df['start'] = pd.to_numeric(df['start'], errors='coerce').fillna(0).astype(int)
        df['end'] = pd.to_numeric(df['end'], errors='coerce').fillna(0).astype(int)
        df['deletion_length'] = (df['end'] - df['start']).clip(lower=0)
        df['log_deletion_length'] = np.log1p(df['deletion_length'])
        df['normalized_chr_position'] = df['start'] / 250_000_000.0

        try:
            df['size_bin'] = pd.qcut(df['deletion_length'] + 1, q=10, labels=False, duplicates='drop')
        except Exception:
            df['size_bin'] = (df['deletion_length'] > 10000).astype(int)

        df['is_small_del'] = (df['deletion_length'] < 1000).astype(float)
        df['is_medium_del'] = ((df['deletion_length'] >= 1000) & (df['deletion_length'] < 10000)).astype(float)
        df['is_large_del'] = (df['deletion_length'] >= 10000).astype(float)

        # Sequence features
        seq_feats = df.apply(lambda r: pd.Series(self._calculate_sequence_features(r.get('sequence', '') or r.get('ref_seq', ''))), axis=1)
        df = pd.concat([df, seq_feats], axis=1)

        # Gene features
        gene_feats = df.apply(lambda r: pd.Series(self._encode_gene_features(r.get('gene', ''))), axis=1)
        df = pd.concat([df, gene_feats], axis=1)

        # Encode gene (only categorical feature remaining)
        for col in self.categorical_columns:
            if col not in df.columns:
                df[col] = 'N/A'
            df[col] = df[col].fillna('N/A').astype(str)
            if fit:
                le = LabelEncoder()
                df[col] = df[col].astype(str)
                le.fit(df[col])
                self.label_encoders[col] = le
                df[f'{col}_encoded'] = le.transform(df[col])
            else:
                if col not in self.label_encoders:
                    df[f'{col}_encoded'] = -1
                else:
                    le = self.label_encoders[col]
                    def enc(x):
                        try:
                            return int(le.transform([x])[0])
                        except Exception:
                            return -1
                    df[f'{col}_encoded'] = df[col].apply(enc)

        # CLEANED FEATURE SET: Only biological features
        numeric_features = [
            # Genomic location
            'chr', 
            'deletion_length', 
            'log_deletion_length', 
            'normalized_chr_position',
            
            # Deletion size categories
            'is_small_del', 
            'is_medium_del', 
            'is_large_del',
            
            # Sequence composition (biological)
            'gc_content', 
            'repeat_content', 
            'homopolymer_run', 
            'cpg_islands', 
            'at_content', 
            'complexity_score',
            
            # Gene context (biological)
            'has_gene', 
            'is_known_disease_gene', 
            'gene_length', 
            'is_ensembl_id',
            
            # Gene encoding
            'gene_encoded'
        ]

        if fit:
            self.feature_columns = numeric_features

        X = df[numeric_features].copy().fillna(0)
        return X

    def train(self, variants: List[Dict[str, Any]], test_size: float = 0.2, cv_folds: int = 5):
        """
        Train predictor on imbalanced data using only biological features.
        
        Args:
            variants: List of variant dictionaries with 'clinical_significance'
            test_size: Fraction for test set
            cv_folds: Number of CV folds
        """
        logger.info("=== Training Deletion Pathogenicity Predictor ===")
        
        # Build labels
        rows = []
        labels = []
        for v in variants:
            sig = str(v.get('clinical_significance', '')).lower()
            if 'pathogenic' in sig and 'benign' not in sig:
                rows.append(v)
                labels.append(1)
            elif 'benign' in sig and 'pathogenic' not in sig:
                rows.append(v)
                labels.append(0)

        if len(rows) == 0:
            raise ValueError("No labeled variants found after filtering uncertain labels")

        X_all = self._encode_features(rows, fit=True)
        y_all = np.array(labels, dtype=int)

        n_pathogenic = y_all.sum()
        n_benign = len(y_all) - n_pathogenic
        
        logger.info(f"Dataset: {n_pathogenic} pathogenic, {n_benign} benign")
        logger.info(f"Imbalance ratio: {n_pathogenic/n_benign:.2f}:1 (pathogenic:benign)")

        # Compute class weights
        self.class_weights = compute_class_weight(
            class_weight='balanced',
            classes=np.array([0, 1]),
            y=y_all
        )
        
        logger.info(f"Computed class weights: benign={self.class_weights[0]:.3f}, pathogenic={self.class_weights[1]:.3f}")

        # Stratified train/test split
        df_tmp = pd.DataFrame(rows)
        df_tmp['deletion_length'] = (pd.to_numeric(df_tmp.get('end', 0)) - pd.to_numeric(df_tmp.get('start', 0))).clip(lower=0)
        
        try:
            n_bins = min(5, len(df_tmp) // 100)
            if n_bins >= 2:
                df_tmp['size_bin'] = pd.qcut(
                    df_tmp['deletion_length'] + 1, 
                    q=n_bins,
                    labels=False, 
                    duplicates='drop'
                )
                stratify_col = df_tmp['size_bin'].astype(str) + "_" + pd.Series(y_all).astype(str)
                
                unique_classes, class_counts = np.unique(stratify_col, return_counts=True)
                min_class_count = class_counts.min()
                
                if min_class_count < 2:
                    logger.warning(f"Size-based stratification has classes with only {min_class_count} sample(s)")
                    stratify_col = y_all
            else:
                logger.warning(f"Too few samples ({len(df_tmp)}) for size-based stratification")
                stratify_col = y_all
                
        except Exception as e:
            logger.warning(f"Could not create size bins: {e}. Using label-only stratification")
            stratify_col = y_all

        X_train, X_test, y_train, y_test = train_test_split(
            X_all, y_all, test_size=test_size,
            random_state=self.random_state, stratify=stratify_col
        )

        logger.info(f"Train set: {y_train.sum()} pathogenic, {len(y_train)-y_train.sum()} benign")
        logger.info(f"Test set: {y_test.sum()} pathogenic, {len(y_test)-y_test.sum()} benign")

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Build ensemble
        logger.info("Building weighted ensemble (RF + GB + XGB)")
        
        rf = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=10,
            min_samples_leaf=4,
            max_features='sqrt',
            class_weight='balanced',
            random_state=self.random_state,
            n_jobs=self.n_jobs
        )

        gb = GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=8,
            min_samples_split=10,
            min_samples_leaf=4,
            subsample=0.8,
            random_state=self.random_state
        )

        estimators = [('rf', rf), ('gb', gb)]

        try:
            from xgboost import XGBClassifier
            self.has_xgboost = True
            
            neg, pos = len(y_train) - y_train.sum(), y_train.sum()
            scale_pos = neg / pos if pos > 0 else 1.0
            
            logger.info(f"XGBoost scale_pos_weight: {scale_pos:.2f}")
            
            xgb = XGBClassifier(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=6,
                min_child_weight=4,
                subsample=0.8,
                colsample_bytree=0.8,
                scale_pos_weight=scale_pos,
                random_state=self.random_state,
                n_jobs=self.n_jobs,
                eval_metric='logloss'
            )
            estimators.append(('xgb', xgb))
            logger.info("XGBoost included in ensemble")
        except ImportError:
            self.has_xgboost = False
            logger.warning("XGBoost not available, using RF + GB only")

        self.model = VotingClassifier(
            estimators=estimators,
            voting='soft',
            n_jobs=self.n_jobs
        )

        sample_weights_train = compute_sample_weight(class_weight='balanced', y=y_train)

        logger.info("Fitting ensemble with sample weights...")
        
        fitted_estimators = []
        for name, est in self.model.estimators:
            if name == 'gb':
                est.fit(X_train_scaled, y_train, sample_weight=sample_weights_train)
            elif name == 'xgb':
                est.fit(X_train_scaled, y_train, sample_weight=sample_weights_train)
            else:
                est.fit(X_train_scaled, y_train, sample_weight=sample_weights_train)
            fitted_estimators.append((name, est))

        self.model = VotingClassifier(
            estimators=fitted_estimators,
            voting='soft',
            n_jobs=self.n_jobs
        )
        self.model.fit(X_train_scaled, y_train)

        # Cross-validation
        logger.info(f"Running {cv_folds}-fold weighted cross-validation...")
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
        oof_probs = np.zeros(len(X_train_scaled))
        oof_preds = np.zeros(len(X_train_scaled), dtype=int)

        for fold, (train_idx, val_idx) in enumerate(cv.split(X_train_scaled, y_train), 1):
            X_tr, X_val = X_train_scaled[train_idx], X_train_scaled[val_idx]
            y_tr, y_val = y_train[train_idx], y_train[val_idx]
            sw_tr = compute_sample_weight(class_weight='balanced', y=y_tr)

            fold_estimators = []
            for name, est in self.model.estimators:
                est_clone = clone(est)
                if name == 'xgb' and self.has_xgboost:
                    neg_f, pos_f = (len(y_tr) - y_tr.sum()), y_tr.sum()
                    est_clone.set_params(scale_pos_weight=neg_f/pos_f if pos_f > 0 else 1.0)
                
                if name in ['gb', 'xgb']:
                    est_clone.fit(X_tr, y_tr, sample_weight=sw_tr)
                else:
                    est_clone.fit(X_tr, y_tr, sample_weight=sw_tr)
                
                fold_estimators.append((name, est_clone))

            fold_model = VotingClassifier(estimators=fold_estimators, voting='soft', n_jobs=self.n_jobs)
            fold_model.fit(X_tr, y_tr)

            probs_val = fold_model.predict_proba(X_val)[:, 1]
            preds_val = (probs_val >= self.threshold).astype(int)
            oof_probs[val_idx] = probs_val
            oof_preds[val_idx] = preds_val

        # CV metrics
        cv_precision = precision_score(y_train, oof_preds, zero_division=0)
        cv_recall = recall_score(y_train, oof_preds, zero_division=0)
        cv_f1 = f1_score(y_train, oof_preds, zero_division=0)
        try:
            cv_auc = roc_auc_score(y_train, oof_probs)
        except Exception:
            cv_auc = 0.0
        
        tn_cv, fp_cv, fn_cv, tp_cv = confusion_matrix(y_train, oof_preds).ravel()
        cv_specificity = tn_cv / (tn_cv + fp_cv) if (tn_cv + fp_cv) > 0 else 0.0

        # Test set evaluation
        logger.info("Evaluating on held-out test set...")
        y_test_proba = self.model.predict_proba(X_test_scaled)[:, 1]
        y_test_pred = (y_test_proba >= self.threshold).astype(int)
        
        test_precision = precision_score(y_test, y_test_pred, zero_division=0)
        test_recall = recall_score(y_test, y_test_pred, zero_division=0)
        test_f1 = f1_score(y_test, y_test_pred, zero_division=0)
        try:
            test_auc = roc_auc_score(y_test, y_test_proba)
        except Exception:
            test_auc = 0.0

        tn_test, fp_test, fn_test, tp_test = confusion_matrix(y_test, y_test_pred).ravel()
        test_specificity = tn_test / (tn_test + fp_test) if (tn_test + fp_test) > 0 else 0.0

        # Feature importance
        rf_imp = None
        try:
            rf_est = None
            for name, est in self.model.estimators:
                if name == 'rf':
                    rf_est = est
                    break
            if rf_est and hasattr(rf_est, 'feature_importances_'):
                rf_imp = sorted(
                    zip(self.feature_columns, rf_est.feature_importances_),
                    key=lambda x: x[1],
                    reverse=True
                )[:15]
                logger.info("Top 15 feature importances (Random Forest):")
                for feat, imp in rf_imp:
                    logger.info(f"  {feat}: {imp:.4f}")
        except Exception as e:
            logger.warning(f"Could not extract feature importances: {e}")

        results = {
            'cv_precision': cv_precision,
            'cv_recall': cv_recall,
            'cv_f1': cv_f1,
            'cv_auc': cv_auc,
            'cv_specificity': cv_specificity,
            'cv_tp': int(tp_cv),
            'cv_tn': int(tn_cv),
            'cv_fp': int(fp_cv),
            'cv_fn': int(fn_cv),
            
            'test_precision': test_precision,
            'test_recall': test_recall,
            'test_f1': test_f1,
            'test_auc': test_auc,
            'test_specificity': test_specificity,
            'test_tp': int(tp_test),
            'test_tn': int(tn_test),
            'test_fp': int(fp_test),
            'test_fn': int(fn_test),
            
            'n_features': X_all.shape[1],
            'n_train': len(X_train),
            'n_test': len(X_test),
            'n_pathogenic_train': int(y_train.sum()),
            'n_benign_train': int(len(y_train) - y_train.sum()),
            'n_pathogenic_test': int(y_test.sum()),
            'n_benign_test': int(len(y_test) - y_test.sum()),
            'imbalance_ratio': float(n_pathogenic / n_benign),
            'class_weight_benign': float(self.class_weights[0]),
            'class_weight_pathogenic': float(self.class_weights[1]),
            
            'feature_importances_rf_top15': rf_imp,
            'has_xgboost': self.has_xgboost,
            
            'y_test': y_test,
            'y_test_pred': y_test_pred,
            'y_test_proba': y_test_proba
        }

        logger.info("=== Training Complete ===")
        logger.info(f"CV: Precision={cv_precision:.3f}, Recall={cv_recall:.3f}, F1={cv_f1:.3f}, AUC={cv_auc:.3f}")
        logger.info(f"Test: Precision={test_precision:.3f}, Recall={test_recall:.3f}, F1={test_f1:.3f}, AUC={test_auc:.3f}")
        logger.info(f"Models used: {'RF + GB + XGB' if self.has_xgboost else 'RF + GB'}")

        return results

    def predict_proba(self, variants: List[Dict[str, Any]]) -> np.ndarray:
        """Predict pathogenicity probabilities."""
        if self.model is None:
            raise ValueError("Model has not been trained")
        X = self._encode_features(variants, fit=False)
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)[:, 1]

    def predict(self, variants: List[Dict[str, Any]]) -> np.ndarray:
        """Predict binary labels (pathogenic=1, benign=0)."""
        probs = self.predict_proba(variants)
        return (probs >= self.threshold).astype(int)