"""
Train and evaluate deletion pathogenicity prediction models.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler
from sklearn.metrics import (
    mean_squared_error, precision_score, recall_score, 
    f1_score, roc_auc_score, classification_report, confusion_matrix
)
from sklearn.utils import resample
import logging
import re

logger = logging.getLogger(__name__)


class DeletionPathogenicityPredictor:
    """Predict pathogenicity of deletion variants using ensemble models."""
    
    def __init__(self, threshold: float = 0.5):
        """Initialize predictor.
        
        Args:
            threshold: Classification threshold for pathogenic/benign
        """
        self.threshold = threshold
        self.model = None
        self.scaler = RobustScaler()  # More robust to outliers than StandardScaler
        self.label_encoders = {}
        self.feature_columns = None
        # REMOVED: 'condition' and 'consequence' from categorical columns
        # These are clinical annotations not available from BAM files
        self.categorical_columns = ['gene', 'variant_type', 'review_status']
        
    def _calculate_sequence_features(self, seq: str) -> dict:
        """Calculate sequence-based features.
        
        Args:
            seq: DNA sequence string
            
        Returns:
            Dictionary of sequence features
        """
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
        
        # GC content
        gc_count = sum(1 for b in seq if b in 'GC')
        gc_content = gc_count / seq_len if seq_len > 0 else 0.5
        
        # AT content
        at_content = 1.0 - gc_content
        
        # CpG islands (CG dinucleotides)
        cpg_count = seq.count('CG')
        cpg_islands = cpg_count / (seq_len - 1) if seq_len > 1 else 0.0
        
        # Homopolymer runs (consecutive same bases)
        max_homopolymer = 0
        current_run = 1
        for i in range(1, seq_len):
            if seq[i] == seq[i-1]:
                current_run += 1
                max_homopolymer = max(max_homopolymer, current_run)
            else:
                current_run = 1
        homopolymer_run = max_homopolymer / seq_len if seq_len > 0 else 0.0
        
        # Simple repeat content (di/tri-nucleotide repeats)
        repeat_score = 0.0
        for repeat_len in [2, 3, 4]:
            if seq_len >= repeat_len * 2:
                for i in range(seq_len - repeat_len * 2):
                    pattern = seq[i:i+repeat_len]
                    if seq[i+repeat_len:i+2*repeat_len] == pattern:
                        repeat_score += 1
        repeat_content = min(repeat_score / seq_len if seq_len > 0 else 0.0, 1.0)
        
        # Sequence complexity (Shannon entropy)
        base_counts = {b: seq.count(b) for b in 'ACGT'}
        complexity = 0.0
        for count in base_counts.values():
            if count > 0:
                p = count / seq_len
                complexity -= p * np.log2(p)
        complexity_score = complexity / 2.0  # Normalize to [0, 1]
        
        return {
            'gc_content': gc_content,
            'repeat_content': repeat_content,
            'homopolymer_run': homopolymer_run,
            'cpg_islands': cpg_islands,
            'at_content': at_content,
            'complexity_score': complexity_score
        }
    
    def _encode_gene_features(self, gene: str) -> dict:
        """Encode gene-related features.
        
        Args:
            gene: Gene symbol or identifier
            
        Returns:
            Dictionary of gene features
        """
        # Expanded list of known disease-associated genes
        # In production, load from ClinGen, OMIM, or gene panels
        known_disease_genes = {
            'BRCA1', 'BRCA2', 'TP53', 'PTEN', 'RB1', 'APC', 'MLH1', 'MSH2',
            'VHL', 'NF1', 'NF2', 'TSC1', 'TSC2', 'ATM', 'CHEK2', 'PALB2',
            'CDKN2A', 'STK11', 'CDH1', 'SMAD4', 'BMPR1A', 'MUTYH', 'MSH6',
            'PMS2', 'EPCAM', 'POLD1', 'POLE', 'RAD51C', 'RAD51D', 'BRIP1'
        }
        
        has_gene = gene and gene != 'N/A' and gene != ''
        
        # Extract gene symbol if it's an Ensembl ID
        gene_symbol = gene
        if gene and gene.startswith('ENSG'):
            gene_symbol = gene.split('.')[0] if '.' in gene else gene
        
        return {
            'has_gene': 1.0 if has_gene else 0.0,
            'is_known_disease_gene': 1.0 if gene_symbol in known_disease_genes else 0.0,
            'gene_length': len(gene) if gene else 0,
            'is_ensembl_id': 1.0 if (gene and gene.startswith('ENSG')) else 0.0
        }
    
    def _encode_review_status_confidence(self, review_status: str) -> float:
        """Encode review status as confidence score.
        
        Args:
            review_status: ClinVar review status
            
        Returns:
            Confidence score [0-1]
        """
        if not review_status:
            return 0.0
        
        review_status = review_status.lower()
        
        confidence_map = {
            'practice guideline': 1.0,
            'reviewed by expert panel': 0.9,
            'criteria provided, multiple submitters': 0.7,
            'criteria provided, single submitter': 0.5,
            'no assertion criteria provided': 0.2,
            'no assertion provided': 0.1,
            'conflicting': 0.3,
            'from_bam': 0.0  # BAM-extracted variants have no review
        }
        
        for status, confidence in confidence_map.items():
            if status in review_status:
                return confidence
        
        return 0.0
    
    def _encode_features(self, variants: list, fit: bool = False) -> pd.DataFrame:
        """Encode variant features for model training/prediction.
        
        IMPORTANT: This method does NOT use 'consequence' or 'condition' fields
        since these are clinical annotations unavailable from BAM files.
        
        Args:
            variants: List of variant dictionaries
            fit: If True, fit encoders; if False, use existing encoders
            
        Returns:
            DataFrame with encoded features
        """
        df = pd.DataFrame(variants)
        
        # Ensure required columns exist
        required_cols = ['chr', 'start', 'end']
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")
        
        # Convert chromosome to numeric
        df['chr'] = df['chr'].astype(str).str.replace('chr', '').replace('X', '23').replace('Y', '24').replace('M', '25')
        df['chr'] = pd.to_numeric(df['chr'], errors='coerce').fillna(0).astype(int)
        
        # Convert start/end to numeric
        df['start'] = pd.to_numeric(df['start'], errors='coerce').fillna(0).astype(int)
        df['end'] = pd.to_numeric(df['end'], errors='coerce').fillna(0).astype(int)
        
        # Basic deletion metrics
        df['deletion_length'] = df['end'] - df['start']
        df['log_deletion_length'] = np.log1p(df['deletion_length'])
        
        # Chromosome position features
        df['chr_position'] = df['start']
        df['normalized_chr_position'] = df['start'] / 250000000.0  # Normalize by ~max chr length
        
        # Size categories
        df['is_small_del'] = (df['deletion_length'] < 1000).astype(float)
        df['is_medium_del'] = ((df['deletion_length'] >= 1000) & (df['deletion_length'] < 10000)).astype(float)
        df['is_large_del'] = (df['deletion_length'] >= 10000).astype(float)
        
        # Add sequence features
        logger.info("Calculating sequence features...")
        sequence_features = df.apply(
            lambda row: pd.Series(self._calculate_sequence_features(row.get('ref_seq', ''))),
            axis=1
        )
        df = pd.concat([df, sequence_features], axis=1)
        
        # Add gene features
        logger.info("Encoding gene features...")
        gene_features = df.apply(
            lambda row: pd.Series(self._encode_gene_features(row.get('gene', ''))),
            axis=1
        )
        df = pd.concat([df, gene_features], axis=1)
        
        # Add review status confidence
        df['review_confidence'] = df.apply(
            lambda row: self._encode_review_status_confidence(row.get('review_status', '')),
            axis=1
        )
        
        # Population frequency (if available)
        if 'population_af' not in df.columns:
            df['population_af'] = 0.0
        df['population_af'] = pd.to_numeric(df['population_af'], errors='coerce').fillna(0.0)
        df['is_rare'] = (df['population_af'] < 0.01).astype(float)
        
        # Quality metrics (if available from BAM)
        if 'mapping_quality' not in df.columns:
            df['mapping_quality'] = 30
        df['mapping_quality'] = pd.to_numeric(df['mapping_quality'], errors='coerce').fillna(30) / 60.0
        
        if 'read_depth' not in df.columns:
            df['read_depth'] = 30
        df['read_depth'] = np.log1p(pd.to_numeric(df['read_depth'], errors='coerce').fillna(30))
        
        # Encode categorical features with handling for unseen categories
        for col in self.categorical_columns:
            if col not in df.columns:
                df[col] = 'N/A'
            
            df[col] = df[col].fillna('N/A').astype(str)
            
            if fit:
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                
                self.label_encoders[col].fit(df[col])
                df[f'{col}_encoded'] = self.label_encoders[col].transform(df[col])
            else:
                if col not in self.label_encoders:
                    logger.warning(f"No encoder for {col}, using default encoding")
                    df[f'{col}_encoded'] = 0
                else:
                    encoder = self.label_encoders[col]
                    
                    def encode_with_unknown(value):
                        try:
                            return encoder.transform([value])[0]
                        except ValueError:
                            return 0
                    
                    df[f'{col}_encoded'] = df[col].apply(encode_with_unknown)
        
        # Select numeric features for model
        # REMOVED: consequence_severity, is_lof, is_coding, is_regulatory
        # These depend on 'consequence' field which BAM files don't have
        numeric_features = [
            # Basic position/size features
            'chr',
            'deletion_length',
            'log_deletion_length',
            'normalized_chr_position',
            
            # Size categories
            'is_small_del',
            'is_medium_del',
            'is_large_del',
            
            # Sequence features (from reference genome)
            'gc_content',
            'repeat_content',
            'homopolymer_run',
            'cpg_islands',
            'at_content',
            'complexity_score',
            
            # Gene features (from GTF annotation)
            'has_gene',
            'is_known_disease_gene',
            'gene_length',
            'is_ensembl_id',
            
            # Quality features
            'review_confidence',
            'population_af',
            'is_rare',
            'mapping_quality',
            'read_depth'
        ]
        
        # Add encoded categorical features
        for col in self.categorical_columns:
            numeric_features.append(f'{col}_encoded')
        
        # Store feature columns during training
        if fit:
            self.feature_columns = numeric_features
            logger.info(f"Training features: {', '.join(numeric_features)}")
        
        # Ensure we have the same features as training
        X = df[numeric_features].copy()
        
        # Fill any remaining NaN values
        X = X.fillna(0)
        
        return X

    def train(self, variants: list, test_size: float = 0.2, cv_folds: int = 5, balance_classes: bool = True):
        """Train the pathogenicity predictor with improved model architecture.
        
        Args:
            variants: List of variant dictionaries with 'clinical_significance'
            test_size: Fraction of data to use for testing
            cv_folds: Number of cross-validation folds
            balance_classes: Whether to balance pathogenic/benign classes
            
        Returns:
            Dictionary with training metrics
        """
        logger.info(f"Training pathogenicity predictor on {len(variants)} variants")
        logger.info("Note: Model does NOT use 'consequence' or 'condition' fields")
        logger.info("      This allows prediction on BAM-extracted deletions without clinical annotations")
        
        # Create binary labels
        y = []
        for variant in variants:
            sig = variant.get('clinical_significance', '').lower()
            if 'pathogenic' in sig and 'benign' not in sig:
                y.append(1)  # Pathogenic
            else:
                y.append(0)  # Benign or uncertain
        
        y = np.array(y)
        
        initial_pathogenic = sum(y)
        initial_benign = len(y) - sum(y)
        logger.info(f"Initial distribution: {initial_pathogenic} pathogenic, {initial_benign} benign/uncertain")
        
        # Balance classes if requested
        if balance_classes and initial_pathogenic > 0 and initial_benign > 0:
            logger.info("Balancing classes using resampling...")
            
            pathogenic_variants = [v for v, label in zip(variants, y) if label == 1]
            benign_variants = [v for v, label in zip(variants, y) if label == 0]
            
            # Upsample minority class
            if len(pathogenic_variants) < len(benign_variants):
                pathogenic_variants = resample(
                    pathogenic_variants,
                    n_samples=len(benign_variants),
                    random_state=42
                )
            else:
                benign_variants = resample(
                    benign_variants,
                    n_samples=len(pathogenic_variants),
                    random_state=42
                )
            
            # Combine balanced sets
            variants = pathogenic_variants + benign_variants
            y = np.array([1] * len(pathogenic_variants) + [0] * len(benign_variants))
            
            logger.info(f"Balanced distribution: {sum(y)} pathogenic, {len(y) - sum(y)} benign")
        
        # Encode features (fit encoders during training)
        X = self._encode_features(variants, fit=True)

        
        logger.info(f"Training with {X.shape[1]} features")
        
        # Split into train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Build ensemble model with XGBoost
        logger.info("Training ensemble classifier (Random Forest + Gradient Boosting + XGBoost)...")
        
        # Try to import XGBoost
        try:
            from xgboost import XGBClassifier
            has_xgboost = True
            logger.info("XGBoost available, using 3-model ensemble")
        except ImportError:
            has_xgboost = False
            logger.warning("XGBoost not available, using 2-model ensemble. Install with: pip install xgboost")
        
        rf = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=10,
            min_samples_leaf=4,
            max_features='sqrt',
            random_state=42,
            class_weight='balanced',
            n_jobs=-1
        )
        
        gb = GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=8,
            min_samples_split=10,
            min_samples_leaf=4,
            subsample=0.8,
            random_state=42
        )
        
        # Build estimators list
        estimators = [
            ('rf', rf),
            ('gb', gb)
        ]
        
        # Add XGBoost if available
        if has_xgboost:
            scale_pos_weight = (len(y_train) - sum(y_train)) / sum(y_train) if sum(y_train) > 0 else 1.0
            
            xgb = XGBClassifier(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=8,
                min_child_weight=4,
                subsample=0.8,
                colsample_bytree=0.8,
                scale_pos_weight=scale_pos_weight,
                random_state=42,
                n_jobs=-1,
                eval_metric='logloss'
            )
            estimators.append(('xgb', xgb))
        
        self.model = VotingClassifier(
            estimators=estimators,
            voting='soft',
            n_jobs=-1
        )
        
        self.model.fit(X_train_scaled, y_train)
        
        # Cross-validation on training set
        logger.info(f"Running {cv_folds}-fold stratified cross-validation...")
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        
        from sklearn.model_selection import cross_val_predict
        y_train_pred_cv = cross_val_predict(self.model, X_train_scaled, y_train, cv=cv)
        y_train_proba_cv = cross_val_predict(self.model, X_train_scaled, y_train, cv=cv, method='predict_proba')[:, 1]
        
        # Calculate CV metrics
        cv_precision = precision_score(y_train, y_train_pred_cv, zero_division=0)
        cv_recall = recall_score(y_train, y_train_pred_cv, zero_division=0)
        cv_f1 = f1_score(y_train, y_train_pred_cv, zero_division=0)
        
        try:
            cv_auc = roc_auc_score(y_train, y_train_proba_cv)
        except:
            cv_auc = 0.0
        
        # Confusion matrix for CV
        tn, fp, fn, tp = confusion_matrix(y_train, y_train_pred_cv).ravel()
        cv_specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        
        # Evaluate on test set
        y_test_proba = self.model.predict_proba(X_test_scaled)[:, 1]
        y_test_pred = (y_test_proba >= self.threshold).astype(int)
        
        test_precision = precision_score(y_test, y_test_pred, zero_division=0)
        test_recall = recall_score(y_test, y_test_pred, zero_division=0)
        test_f1 = f1_score(y_test, y_test_pred, zero_division=0)
        
        try:
            test_auc = roc_auc_score(y_test, y_test_proba)
        except:
            test_auc = 0.0
        
        # Test confusion matrix
        tn_test, fp_test, fn_test, tp_test = confusion_matrix(y_test, y_test_pred).ravel()
        test_specificity = tn_test / (tn_test + fp_test) if (tn_test + fp_test) > 0 else 0.0
        
        # Feature importance (from Random Forest)
        if hasattr(self.model.estimators_[0], 'feature_importances_'):
            feature_importance = self.model.estimators_[0].feature_importances_
            top_features = sorted(
                zip(self.feature_columns, feature_importance),
                key=lambda x: x[1],
                reverse=True
            )[:15]
            logger.info("Top 15 most important features:")
            for feat, imp in top_features:
                logger.info(f"  {feat}: {imp:.4f}")
        
        results = {
            # Cross-validation metrics
            'cv_precision': cv_precision,
            'cv_recall': cv_recall,
            'cv_f1': cv_f1,
            'cv_specificity': cv_specificity,
            'cv_auc': cv_auc,
            
            # Test set metrics
            'test_precision': test_precision,
            'test_recall': test_recall,
            'test_f1': test_f1,
            'test_specificity': test_specificity,
            'test_auc': test_auc,
            
            # Confusion matrix
            'test_tp': int(tp_test),
            'test_tn': int(tn_test),
            'test_fp': int(fp_test),
            'test_fn': int(fn_test),
            
            # Predictions for analysis
            'y_train_pred': y_train_pred_cv,
            'y_test': y_test,
            'y_test_pred': y_test_pred,
            'y_test_proba': y_test_proba,
            
            # Data info
            'n_features': X.shape[1],
            'n_train': len(X_train),
            'n_test': len(X_test),
            'has_xgboost': has_xgboost
        }
        
        logger.info(f"\nTraining Complete:")
        logger.info(f"  Models: {'RF + GB + XGB' if has_xgboost else 'RF + GB'}")
        logger.info(f"  Features: {X.shape[1]} (genomic + sequence + gene)")
        logger.info(f"  CV Precision: {cv_precision:.3f}, Recall: {cv_recall:.3f}, F1: {cv_f1:.3f}, AUC: {cv_auc:.3f}")
        logger.info(f"  Test Precision: {test_precision:.3f}, Recall: {test_recall:.3f}, F1: {test_f1:.3f}, AUC: {test_auc:.3f}")
        
        return results
    
    def predict_proba(self, variants: list) -> np.ndarray:
        """Predict pathogenicity probabilities for variants.
        
        Args:
            variants: List of variant dictionaries
            
        Returns:
            Array of pathogenicity probabilities [0-1]
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet")
        
        # Encode features (use existing encoders, don't fit)
        X = self._encode_features(variants, fit=False)
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Predict probabilities
        probs = self.model.predict_proba(X_scaled)[:, 1]
        
        return probs
    
    def predict(self, variants: list) -> np.ndarray:
        """Predict pathogenic (1) or benign (0) for variants.
        
        Args:
            variants: List of variant dictionaries
            
        Returns:
            Array of predictions (0 or 1)
        """
        probs = self.predict_proba(variants)
        return (probs >= self.threshold).astype(int)