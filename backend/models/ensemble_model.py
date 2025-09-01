# 📄 backend/models/ensemble_model.py - Advanced Ensemble Model
# ================================================================================

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, BaggingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import xgboost as xgb
import logging
import joblib

logger = logging.getLogger(__name__)

class AdvancedEnsembleModel:
    def __init__(self):
        """Initialize advanced ensemble model for banking APK detection"""
        self.base_models = {}
        self.meta_model = None
        self.ensemble_model = None
        self.is_trained = False
        self.performance_metrics = {}
        
        # Initialize base models with optimal hyperparameters
        self._initialize_base_models()
    
    def _initialize_base_models(self):
        """Initialize base models with banking-optimized hyperparameters"""
        
        # Random Forest - excellent for permission features
        self.base_models['random_forest'] = RandomForestClassifier(
            n_estimators=300,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
        
        # XGBoost - great for structured features with interactions
        self.base_models['xgboost'] = xgb.XGBClassifier(
            n_estimators=250,
            max_depth=15,
            learning_rate=0.1,
            subsample=0.9,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=0.1,
            scale_pos_weight=1,
            random_state=42,
            eval_metric='logloss',
            use_label_encoder=False
        )
        
        # SVM with RBF kernel - good for complex decision boundaries
        self.base_models['svm'] = SVC(
            kernel='rbf',
            C=1.0,
            gamma='scale',
            probability=True,
            class_weight='balanced',
            random_state=42
        )
        
        # Logistic Regression - interpretable baseline
        self.base_models['logistic'] = LogisticRegression(
            C=1.0,
            solver='liblinear',
            class_weight='balanced',
            random_state=42,
            max_iter=1000
        )
        
        # Neural Network - for complex pattern recognition
        self.base_models['neural_network'] = MLPClassifier(
            hidden_layer_sizes=(128, 64, 32),
            activation='relu',
            solver='adam',
            alpha=0.001,
            learning_rate='adaptive',
            max_iter=500,
            random_state=42
        )
        
        logger.info("Base models initialized successfully")
    
    def hyperparameter_tuning(self, X_train, y_train, model_name='random_forest'):
        """Perform hyperparameter tuning for specific model"""
        try:
            logger.info(f"Tuning hyperparameters for {model_name}...")
            
            param_grids = {
                'random_forest': {
                    'n_estimators': [200, 300, 400],
                    'max_depth': [15, 20, 25],
                    'min_samples_split': [5, 10],
                    'min_samples_leaf': [2, 4]
                },
                'xgboost': {
                    'n_estimators': [200, 300],
                    'max_depth': [10, 15, 20],
                    'learning_rate': [0.05, 0.1, 0.2],
                    'subsample': [0.8, 0.9]
                },
                'svm': {
                    'C': [0.5, 1.0, 2.0],
                    'gamma': ['scale', 'auto'],
                    'kernel': ['rbf', 'poly']
                }
            }
            
            if model_name in param_grids:
                model = self.base_models[model_name]
                param_grid = param_grids[model_name]
                
                # Perform grid search with cross-validation
                grid_search = GridSearchCV(
                    model, param_grid,
                    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
                    scoring='f1',
                    n_jobs=-1,
                    verbose=1
                )
                
                grid_search.fit(X_train, y_train)
                
                # Update model with best parameters
                self.base_models[model_name] = grid_search.best_estimator_
                
                logger.info(f"Best parameters for {model_name}: {grid_search.best_params_}")
                logger.info(f"Best CV score: {grid_search.best_score_:.4f}")
                
                return grid_search.best_estimator_, grid_search.best_params_
            
        except Exception as e:
            logger.error(f"Hyperparameter tuning failed for {model_name}: {str(e)}")
            return self.base_models[model_name], {}
    
    def train_ensemble(self, X_train, y_train, use_stacking=True):
        """Train advanced ensemble model"""
        try:
            logger.info("Training advanced ensemble model...")
            
            # Step 1: Train individual base models
            trained_models = []
            model_performances = {}
            
            for model_name, model in self.base_models.items():
                logger.info(f"Training {model_name}...")
                
                # Train model
                model.fit(X_train, y_train)
                
                # Evaluate with cross-validation
                cv_scores = cross_val_score(
                    model, X_train, y_train,
                    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
                    scoring='f1'
                )
                
                model_performances[model_name] = {
                    'cv_mean': cv_scores.mean(),
                    'cv_std': cv_scores.std(),
                    'cv_scores': cv_scores
                }
                
                trained_models.append((model_name, model))
                logger.info(f"{model_name} CV F1: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
            
            # Step 2: Create weighted voting ensemble
            # Calculate weights based on cross-validation performance
            total_performance = sum(perf['cv_mean'] for perf in model_performances.values())
            weights = [
                model_performances[name]['cv_mean'] / total_performance 
                for name, _ in trained_models
            ]
            
            self.ensemble_model = VotingClassifier(
                estimators=trained_models,
                voting='soft',
                weights=weights
            )
            
            # Train ensemble
            self.ensemble_model.fit(X_train, y_train)
            
            # Final ensemble performance
            ensemble_cv_scores = cross_val_score(
                self.ensemble_model, X_train, y_train,
                cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
                scoring='f1'
            )
            
            self.performance_metrics = {
                'individual_models': model_performances,
                'ensemble_cv_mean': ensemble_cv_scores.mean(),
                'ensemble_cv_std': ensemble_cv_scores.std(),
                'model_weights': dict(zip([name for name, _ in trained_models], weights))
            }
            
            logger.info(f"Ensemble CV F1: {ensemble_cv_scores.mean():.4f} (+/- {ensemble_cv_scores.std() * 2:.4f})")
            logger.info(f"Model weights: {self.performance_metrics['model_weights']}")
            
            self.is_trained = True
            return self.ensemble_model
            
        except Exception as e:
            logger.error(f"Ensemble training failed: {str(e)}")
            raise
    
    def evaluate_model(self, X_test, y_test):
        """Comprehensive model evaluation"""
        try:
            if not self.is_trained:
                raise ValueError("Model not trained yet")
            
            # Make predictions
            y_pred = self.ensemble_model.predict(X_test)
            y_pred_proba = self.ensemble_model.predict_proba(X_test)
            
            # Calculate comprehensive metrics
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, average='weighted'),
                'recall': recall_score(y_test, y_pred, average='weighted'),
                'f1_score': f1_score(y_test, y_pred, average='weighted'),
                'roc_auc': roc_auc_score(y_test, y_pred_proba[:, 1])
            }
            
            # Banking-specific metrics
            banking_metrics = self._calculate_banking_metrics(y_test, y_pred, y_pred_proba)
            metrics.update(banking_metrics)
            
            logger.info("Model Evaluation Results:")
            for metric, value in metrics.items():
                logger.info(f"  {metric}: {value:.4f}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Model evaluation failed: {str(e)}")
            return {}
    
    def _calculate_banking_metrics(self, y_true, y_pred, y_pred_proba):
        """Calculate banking-specific performance metrics"""
        try:
            # False positive rate (legitimate apps flagged as malicious)
            fp_rate = np.sum((y_true == 0) & (y_pred == 1)) / np.sum(y_true == 0)
            
            # False negative rate (malicious apps missed)
            fn_rate = np.sum((y_true == 1) & (y_pred == 0)) / np.sum(y_true == 1)
            
            # High-confidence predictions (confidence > 90%)
            high_conf_mask = np.max(y_pred_proba, axis=1) > 0.9
            high_conf_accuracy = accuracy_score(
                y_true[high_conf_mask], 
                y_pred[high_conf_mask]
            ) if np.any(high_conf_mask) else 0
            
            return {
                'false_positive_rate': fp_rate,
                'false_negative_rate': fn_rate,
                'high_confidence_accuracy': high_conf_accuracy,
                'high_confidence_samples': np.sum(high_conf_mask)
            }
            
        except Exception as e:
            logger.error(f"Banking metrics calculation failed: {str(e)}")
            return {}
    
    def predict_with_uncertainty(self, X):
        """Make predictions with uncertainty quantification"""
        try:
            if not self.is_trained:
                raise ValueError("Model not trained yet")
            
            # Get predictions from ensemble
            predictions = self.ensemble_model.predict(X)
            probabilities = self.ensemble_model.predict_proba(X)
            
            # Calculate prediction uncertainty
            uncertainty = 1 - np.max(probabilities, axis=1)
            
            # Get individual model predictions for diversity assessment
            individual_preds = []
            for name, model in self.ensemble_model.named_estimators_.items():
                individual_preds.append(model.predict(X))
            
            # Calculate prediction diversity
            individual_preds = np.array(individual_preds).T
            diversity = np.std(individual_preds, axis=1)
            
            return {
                'predictions': predictions,
                'probabilities': probabilities,
                'uncertainty': uncertainty,
                'diversity': diversity,
                'confidence': np.max(probabilities, axis=1)
            }
            
        except Exception as e:
            logger.error(f"Prediction with uncertainty failed: {str(e)}")
            return {}
