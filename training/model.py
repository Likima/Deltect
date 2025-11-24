"""
Train and evaluate deletion pathogenicity prediction models.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, precision_score, recall_score
import logging

logger = logging.getLogger(__name__)


class DeletionPathogenicityPredictor:
    """Predict pathogenicity of deletion variants using Random Forest."""
    
    def __init__(self, threshold: float = 0.5):
        """Initialize predictor.
        
        Args:
            threshold: Classification threshold for pathogenic/benign
        """
        self.threshold = threshold
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_columns = None
        self.categorical_columns = ['gene', 'variant_type', 'consequence', 'condition', 'review_status']
        
    def _encode_features(self, variants: list, fit: bool = False) -> pd.DataFrame:
        """Encode variant features for model training/prediction.
        
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
        
        # Calculate deletion length
        df['deletion_length'] = df['end'] - df['start']
        
        # Calculate chromosome position (for positional encoding)
        df['chr_position'] = df['start']
        
        # Encode categorical features with handling for unseen categories
        for col in self.categorical_columns:
            if col not in df.columns:
                df[col] = 'N/A'
            
            # Fill NaN with 'N/A'
            df[col] = df[col].fillna('N/A').astype(str)
            
            if fit:
                # During training: fit encoder
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                
                # Fit encoder on training data
                self.label_encoders[col].fit(df[col])
                df[f'{col}_encoded'] = self.label_encoders[col].transform(df[col])
            else:
                # During inference: handle unseen categories
                if col not in self.label_encoders:
                    logger.warning(f"No encoder for {col}, using default encoding")
                    df[f'{col}_encoded'] = 0
                else:
                    # Map unseen categories to a default value
                    encoder = self.label_encoders[col]
                    
                    def encode_with_unknown(value):
                        try:
                            return encoder.transform([value])[0]
                        except ValueError:
                            # Unseen category - return the most common class (index 0)
                            return 0
                    
                    df[f'{col}_encoded'] = df[col].apply(encode_with_unknown)
        
        # Select numeric features for model
        numeric_features = [
            'chr',
            'start', 
            'end',
            'deletion_length',
            'chr_position'
        ]
        
        # Add encoded categorical features
        for col in self.categorical_columns:
            numeric_features.append(f'{col}_encoded')
        
        # Store feature columns during training
        if fit:
            self.feature_columns = numeric_features
        
        # Ensure we have the same features as training
        X = df[numeric_features].copy()
        
        return X
    
    def train(self, variants: list, test_size: float = 0.2, cv_folds: int = 5):
        """Train the pathogenicity predictor.
        
        Args:
            variants: List of variant dictionaries with 'clinical_significance'
            test_size: Fraction of data to use for testing
            cv_folds: Number of cross-validation folds
            
        Returns:
            Dictionary with training metrics
        """
        logger.info(f"Training pathogenicity predictor on {len(variants)} variants")
        
        # Encode features (fit encoders during training)
        X = self._encode_features(variants, fit=True)
        
        # Create binary labels (pathogenic = 1, benign/uncertain = 0)
        y = []
        for variant in variants:
            sig = variant.get('clinical_significance', '').lower()
            if 'pathogenic' in sig and 'benign' not in sig:
                y.append(1)  # Pathogenic
            else:
                y.append(0)  # Benign or uncertain
        
        y = np.array(y)
        
        logger.info(f"Training set: {sum(y)} pathogenic, {len(y) - sum(y)} benign/uncertain")
        
        # Split into train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train Random Forest
        logger.info("Training Random Forest classifier...")
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            class_weight='balanced'
        )
        
        self.model.fit(X_train_scaled, y_train)
        
        # Cross-validation on training set
        logger.info(f"Running {cv_folds}-fold cross-validation...")
        cv_scores_mse = cross_val_score(
            self.model, X_train_scaled, y_train,
            cv=cv_folds, scoring='neg_mean_squared_error'
        )
        
        # Calculate precision, recall, specificity for CV
        from sklearn.model_selection import cross_val_predict
        y_train_pred = cross_val_predict(self.model, X_train_scaled, y_train, cv=cv_folds)
        
        # Calculate metrics
        cv_precision = precision_score(y_train, y_train_pred, zero_division=0)
        cv_recall = recall_score(y_train, y_train_pred, zero_division=0)
        
        # Specificity = TN / (TN + FP)
        tn = np.sum((y_train == 0) & (y_train_pred == 0))
        fp = np.sum((y_train == 0) & (y_train_pred == 1))
        cv_specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        # Evaluate on test set
        y_test_pred = self.model.predict(X_test_scaled)
        y_test_proba = self.model.predict_proba(X_test_scaled)[:, 1]
        
        # Apply threshold
        y_test_pred_threshold = (y_test_proba >= self.threshold).astype(int)
        
        test_mse = mean_squared_error(y_test, y_test_pred_threshold)
        test_precision = precision_score(y_test, y_test_pred_threshold, zero_division=0)
        test_recall = recall_score(y_test, y_test_pred_threshold, zero_division=0)
        
        # Test specificity
        tn_test = np.sum((y_test == 0) & (y_test_pred_threshold == 0))
        fp_test = np.sum((y_test == 0) & (y_test_pred_threshold == 1))
        test_specificity = tn_test / (tn_test + fp_test) if (tn_test + fp_test) > 0 else 0
        
        results = {
            # Cross-validation metrics
            'cv_mse_mean': -cv_scores_mse.mean(),
            'cv_mse_std': cv_scores_mse.std(),
            'cv_precision_mean': cv_precision,
            'cv_precision_std': 0.0,  # Would need to calculate per fold
            'cv_recall_mean': cv_recall,
            'cv_recall_std': 0.0,
            'cv_specificity_mean': cv_specificity,
            'cv_specificity_std': 0.0,
            
            # Test set metrics
            'mse': test_mse,
            'precision': test_precision,
            'recall': test_recall,
            'specificity': test_specificity,
            
            # Predictions for analysis
            'y_train_pred': y_train_pred,
            'y_test': y_test,
            'y_test_pred': y_test_pred_threshold,
            'y_test_proba': y_test_proba
        }
        
        logger.info(f"Training complete - Test MSE: {test_mse:.4f}")
        
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