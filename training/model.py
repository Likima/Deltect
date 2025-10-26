import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, precision_score, recall_score
import re

class DeletionPathogenicityPredictor:
    """Random Forest regressor that predicts a pathogenicity probability for deletions."""

    def __init__(self, threshold=0.5):
        """Initialize the predictor
        
        Args:
            threshold: Probability threshold for converting predictions to binary classes
            e.g. if something is 80% predicted of being pathogenic its above the threshold and is classified as YES to pathogenic
        """
        self.model = None  # Random Forest model (not trained yet)
        self.label_encoders = {}  # Dictionary to store encoders for our features
        self.scaler = StandardScaler()  # Scaler for numerical features
        self.feature_names = []  # List of feature column names
        self.threshold = threshold  # Threshold for binary classification

    def _calculate_metrics(self, y_true, y_pred_proba, threshold=None):
        """Calculate precision, recall, and specificity from probabilities."""
        if threshold is None:
            threshold = self.threshold
        
        # Convert probabilities to binary classes
        y_true_binary = (y_true >= threshold).astype(int)
        y_pred_binary = (y_pred_proba >= threshold).astype(int)
        
        # Calculate metrics
        precision = precision_score(y_true_binary, y_pred_binary, zero_division=0)
        recall = recall_score(y_true_binary, y_pred_binary, zero_division=0)
        
        # Calculate specificity (true negative rate)
        tn = np.sum((y_true_binary == 0) & (y_pred_binary == 0))
        fp = np.sum((y_true_binary == 0) & (y_pred_binary == 1))
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        
        return {
            'precision': precision,
            'recall': recall,
            'specificity': specificity
        }

    def prepare_features(self, variants):
        """Convert variant data into features and a probabilistic target for ML.
        
        Args:
            variants: List of variants obtained via the ClinVar API
            
        Returns:
            X: Feature matrix (numpy array)
            y: Target probabilities (numpy array)
            df: Processed dataframe
        """
        df = pd.DataFrame(variants) # dataframe easier to use

        #TODO: is it possible to query ClinVar to give us the length rather than computing mathematically? It gets quite lengthy...

        # Create numerical features from deletion coordinates
        # Calculate deletion size as the difference between end and start positions
        df['deletion_size'] = pd.to_numeric(df.get('end', None), errors='coerce') - pd.to_numeric(df.get('start', None), errors='coerce')
        df['deletion_size'] = df['deletion_size'].fillna(df['deletion_size'].median())

        # Store chromosomal position (start coordinate)
        df['chr_position'] = pd.to_numeric(df.get('start', None), errors='coerce')
        df['chr_position'] = df['chr_position'].fillna(df['chr_position'].median())

        # Build a probabilistic label from clinical_significance:
        # - "pathogenic" -> 1.0 (definitely pathogenic)
        # - "likely pathogenic" -> 0.9 (probably pathogenic)
        # - "benign" -> 0.0 (definitely benign)
        # - "likely benign" -> 0.1 (probably benign)
        # Anything ambiguous (both terms / none) is dropped.

        #TODO: reach out to bioinformatics people to see what "likely" means numerically

        def map_prob(text):
            """Map clinical significance text to a probability score."""
            t = (text or '').lower()
            
            # Check for 'likely' phrases first (lower confidence)
            if re.search(r'likely pathogenic', t):
                return 0.9
            if re.search(r'likely benign', t):
                return 0.1
            
            # Check for explicit terms (higher confidence)
            if re.search(r'pathogenic', t) and not re.search(r'benign', t):
                return 1.0
            if re.search(r'benign', t) and not re.search(r'pathogenic', t):
                return 0.0
            
            # Return NaN for ambiguous cases (will be dropped)
            return np.nan

        # Apply probability mapping to clinical significance column
        df['pathogenic_prob'] = df['clinical_significance'].apply(map_prob)

        print("After Filter")
        print(df)

        # Drop rows without a clear label (NaN values)
        # -> if our regex doesnt work we drop it
        # TODO: implement more realistic probabilities such that we dont have to drop unclassified values
        df = df[~df['pathogenic_prob'].isna()].copy()

        # Ensure we have data to train on
        # NOTE: if this fails we probably regexed wrong
        if len(df) == 0:
            raise ValueError("No variants with assignable probabilistic labels found for training")

        # Encode features that are strings (gene, consequence, condition) into integers
        # label encoding
        for feature in ['gene', 'consequence', 'condition']:
            # Replace missing values with 'Unknown'
            df[feature] = df[feature].fillna('Unknown')

            if feature not in self.label_encoders:
                # First time seeing this feature - create and fit encoder
                self.label_encoders[feature] = LabelEncoder()
                df[f'{feature}_encoded'] = self.label_encoders[feature].fit_transform(df[feature])
            else:
                # Encoder already exists - handle unseen categories
                le = self.label_encoders[feature]
                df[f'{feature}_encoded'] = df[feature].apply(
                    # below function reads: "check if the category is seen during training. If it has, transform it to an integer, else to -1"
                    lambda x: le.transform([x])[0] if x in le.classes_ else -1 
                )
        
        # what columns are features
        feature_cols = [
            'deletion_size',
            'chr_position',
            'gene_encoded',
            'consequence_encoded',
            'condition_encoded'
        ]

        # Extract feature matrix and target vector
        X = df[feature_cols].values.astype(float)
        y = df['pathogenic_prob'].astype(float).values

        # Store feature names for later reference
        self.feature_names = feature_cols

        return X, y, df

    def train(self, variants, test_size=0.2, cv_folds=10, random_state=42):
        """Train Random Forest regressor to predict pathogenicity probability with 10-fold CV."""
        
        print("\nTRAINING RANDOM FOREST REGRESSOR (probabilistic labels)")

        # Prepare features and target from variant data
        X, y, df = self.prepare_features(variants)

        print(f"Dataset: {len(X)} samples, {X.shape[1]} features")
        print(f"Target: mean={y.mean():.3f}, std={y.std():.3f}")

        # Split data into training and test sets via built in function
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        print(f"\nTrain samples: {len(X_train)}, Test samples: {len(X_test)}")

        # Scale features to zero mean and unit variance
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Initialize Random Forest model with specific hyperparameters
        # TODO: we can play with these hyperparameters and see how it helps us
        self.model = RandomForestRegressor(
            n_estimators=100,  # Number of trees in the forest
            max_depth=10,  # Maximum depth of each tree 
            min_samples_split=5,  # Minimum samples required to split a node
            min_samples_leaf=2,  # Minimum samples required at a leaf node
            random_state=random_state,
            n_jobs=-1  # idk lol
        )

        print(f"\nPerforming {cv_folds}-Fold Cross-Validation")
        
        cv_results = cross_validate(
            self.model, 
            X_train_scaled, 
            y_train,
            cv=cv_folds,  # Number of folds
            scoring=['neg_mean_squared_error', 'neg_mean_absolute_error'],
            return_train_score=True,  # Also compute training scores
            n_jobs=-1
        )

        # Calculate precision, recall, specificity for each CV fold manually
        from sklearn.model_selection import KFold
        
        kf = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
        cv_precision_test, cv_recall_test, cv_specificity_test = [], [], []
        cv_precision_train, cv_recall_train, cv_specificity_train = [], [], []
        
        for train_idx, val_idx in kf.split(X_train_scaled):
            X_fold_train, X_fold_val = X_train_scaled[train_idx], X_train_scaled[val_idx]
            y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]
            
            # Train model on this fold
            fold_model = RandomForestRegressor(
                n_estimators=100, max_depth=10, min_samples_split=5,
                min_samples_leaf=2, random_state=random_state, n_jobs=-1
            )
            fold_model.fit(X_fold_train, y_fold_train)
            
            # Validation metrics
            y_val_pred = np.clip(fold_model.predict(X_fold_val), 0.0, 1.0)
            val_metrics = self._calculate_metrics(y_fold_val, y_val_pred)
            cv_precision_test.append(val_metrics['precision'])
            cv_recall_test.append(val_metrics['recall'])
            cv_specificity_test.append(val_metrics['specificity'])
            
            # Training metrics
            y_train_pred = np.clip(fold_model.predict(X_fold_train), 0.0, 1.0)
            train_metrics = self._calculate_metrics(y_fold_train, y_train_pred)
            cv_precision_train.append(train_metrics['precision'])
            cv_recall_train.append(train_metrics['recall'])
            cv_specificity_train.append(train_metrics['specificity'])

        print(f"\n{cv_folds}-Fold Cross-Validation Results (on training set):")
        
        # Extract MSE/MAE metrics
        cv_mse_test = -cv_results['test_neg_mean_squared_error']
        cv_mse_train = -cv_results['train_neg_mean_squared_error']

        # Average CV metrics for validation folds
        print(f"  Test MSE:         {cv_mse_test.mean():.4f}")
        print(f"  Test Precision:   {np.mean(cv_precision_test):.4f}")
        print(f"  Test Recall:      {np.mean(cv_recall_test):.4f}")
        print(f"  Test Specificity: {np.mean(cv_specificity_test):.4f}")
        
        # Average CV metrics for training folds
        print(f"\n  Train MSE:         {cv_mse_train.mean():.4f}")
        print(f"  Train Precision:   {np.mean(cv_precision_train):.4f}")
        print(f"  Train Recall:      {np.mean(cv_recall_train):.4f}")
        print(f"  Train Specificity: {np.mean(cv_specificity_train):.4f}")

        # Individual fold results 
        print(f"\n  Individual Fold Results:")
        for i in range(cv_folds):
            print(f"    Fold {i+1}: MSE={cv_mse_test[i]:.4f}, "
                  f"Precision={cv_precision_test[i]:.4f}, Recall={cv_recall_test[i]:.4f}, "
                  f"Specificity={cv_specificity_test[i]:.4f}")

        # Train final model on entire training set
        print(f"\nTraining final model on entire training set")
        self.model.fit(X_train_scaled, y_train)

        # Evaluate on hold-out test set (validation set)
        print(f"\nHold-out Test Set Evaluation:")
        
        # Make predictions on test set
        y_pred = self.model.predict(X_test_scaled)

        # make sure the predictions are from 0 to 1
        y_pred = np.clip(y_pred, 0.0, 1.0)

        # make predictions on training set
        train_pred = self.model.predict(X_train_scaled)
        train_pred = np.clip(train_pred, 0.0, 1.0)

        # Calculate test set metrics
        mse = mean_squared_error(y_test, y_pred)
        test_metrics = self._calculate_metrics(y_test, y_pred)

        print(f"  Test MSE:         {mse:.4f}")
        print(f"  Test Precision:   {test_metrics['precision']:.4f}")
        print(f"  Test Recall:      {test_metrics['recall']:.4f}")
        print(f"  Test Specificity: {test_metrics['specificity']:.4f}")

        # Calculate training set metrics to check for overfitting
        train_mse = mean_squared_error(y_train, train_pred)
        train_metrics = self._calculate_metrics(y_train, train_pred)
        
        print(f"\n  Train MSE:         {train_mse:.4f}")
        print(f"  Train Precision:   {train_metrics['precision']:.4f}")
        print(f"  Train Recall:      {train_metrics['recall']:.4f}")
        print(f"  Train Specificity: {train_metrics['specificity']:.4f}")
    
        # Print feature importance scores
        print("\nFeature Importance:")
        for feature, importance in zip(self.feature_names, self.model.feature_importances_):
            print(f"  {feature}: {importance:.4f}")

        # Return comprehensive results dictionary
        return {
            'mse': mse,
            'precision': test_metrics['precision'],
            'recall': test_metrics['recall'],
            'specificity': test_metrics['specificity'],
            'cv_results': cv_results,
            'cv_mse_mean': cv_mse_test.mean(),
            'cv_mse_std': cv_mse_test.std(),
            'cv_precision_mean': np.mean(cv_precision_test),
            'cv_precision_std': np.std(cv_precision_test),
            'cv_recall_mean': np.mean(cv_recall_test),
            'cv_recall_std': np.std(cv_recall_test),
            'cv_specificity_mean': np.mean(cv_specificity_test),
            'cv_specificity_std': np.std(cv_specificity_test),
            'y_test': y_test,
            'y_pred': y_pred,
            'y_train_pred': train_pred,
            'cv_folds': cv_folds
        }

    def predict_proba(self, variants):
        """Return predicted pathogenic probabilities for input variants."""
        # Ensure model has been trained
        if self.model is None:
            raise ValueError("Model not trained yet")

        # Prepare features from input variants
        X, _, _ = self.prepare_features(variants)
        
        # Scale features using the fitted scaler
        X_scaled = self.scaler.transform(X)
        
        # Make predictions
        preds = self.model.predict(X_scaled)
        
        # Clip predictions to valid probability range [0, 1]
        return np.clip(preds, 0.0, 1.0)