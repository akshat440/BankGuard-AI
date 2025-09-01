# 📄 backend/models/feature_extractor.py - Feature Engineering for DroidRL Dataset
# ================================================================================

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import SelectKBest, chi2, mutual_info_classif
import logging

logger = logging.getLogger(__name__)

class FeatureExtractor:
    def __init__(self):
        """Initialize feature extractor for DroidRL dataset"""
        self.scaler = StandardScaler()
        self.feature_selector = None
        self.selected_features = None
        self.feature_importance = None
        
        # Banking-specific feature categories
        self.banking_critical_permissions = [
            'permission_SEND_SMS', 'permission_READ_SMS', 'permission_CALL_PHONE',
            'permission_READ_PHONE_STATE', 'permission_SYSTEM_ALERT_WINDOW',
            'permission_BIND_ACCESSIBILITY_SERVICE', 'permission_CAMERA',
            'permission_RECORD_AUDIO', 'permission_ACCESS_FINE_LOCATION',
            'permission_WRITE_EXTERNAL_STORAGE'
        ]
        
        self.banking_normal_permissions = [
            'permission_INTERNET', 'permission_ACCESS_NETWORK_STATE',
            'permission_ACCESS_WIFI_STATE', 'permission_VIBRATE',
            'permission_WAKE_LOCK', 'permission_READ_EXTERNAL_STORAGE'
        ]
        
        # Intent patterns relevant to banking malware
        self.suspicious_intents = [
            'intent_BOOT_COMPLETED', 'intent_SMS_RECEIVED', 'intent_PHONE_STATE',
            'intent_NEW_OUTGOING_CALL', 'intent_USER_PRESENT'
        ]
    
    def extract_banking_features(self, df):
        """Extract banking-specific features from DroidRL dataset"""
        try:
            logger.info("Extracting banking-specific features...")
            
            # Create copy of dataframe
            banking_df = df.copy()
            
            # Add banking-specific aggregated features
            banking_features = {}
            
            # 1. Critical permission count
            critical_perm_cols = [col for col in df.columns if col in self.banking_critical_permissions]
            banking_features['critical_permission_count'] = df[critical_perm_cols].sum(axis=1)
            
            # 2. Normal permission count
            normal_perm_cols = [col for col in df.columns if col in self.banking_normal_permissions]
            banking_features['normal_permission_count'] = df[normal_perm_cols].sum(axis=1)
            
            # 3. Total permission count
            perm_cols = [col for col in df.columns if col.startswith('permission_')]
            banking_features['total_permission_count'] = df[perm_cols].sum(axis=1)
            
            # 4. Permission risk ratio
            banking_features['permission_risk_ratio'] = (
                banking_features['critical_permission_count'] / 
                (banking_features['total_permission_count'] + 1)  # Add 1 to avoid division by zero
            )
            
            # 5. SMS-related permissions
            sms_perm_cols = [col for col in df.columns if 'SMS' in col]
            banking_features['sms_permission_count'] = df[sms_perm_cols].sum(axis=1)
            
            # 6. Phone-related permissions
            phone_perm_cols = [col for col in df.columns if 'PHONE' in col or 'CALL' in col]
            banking_features['phone_permission_count'] = df[phone_perm_cols].sum(axis=1)
            
            # 7. Location permissions
            location_perm_cols = [col for col in df.columns if 'LOCATION' in col]
            banking_features['location_permission_count'] = df[location_perm_cols].sum(axis=1)
            
            # 8. Storage permissions
            storage_perm_cols = [col for col in df.columns if 'STORAGE' in col]
            banking_features['storage_permission_count'] = df[storage_perm_cols].sum(axis=1)
            
            # 9. Suspicious intent count
            suspicious_intent_cols = [col for col in df.columns if col in self.suspicious_intents]
            banking_features['suspicious_intent_count'] = df[suspicious_intent_cols].sum(axis=1)
            
            # 10. Intent diversity (number of different intents)
            intent_cols = [col for col in df.columns if col.startswith('intent_')]
            banking_features['intent_diversity'] = df[intent_cols].sum(axis=1)
            
            # Add banking features to dataframe
            for feature_name, feature_values in banking_features.items():
                banking_df[feature_name] = feature_values
            
            logger.info(f"Added {len(banking_features)} banking-specific features")
            return banking_df
            
        except Exception as e:
            logger.error(f"Banking feature extraction failed: {str(e)}")
            return df
    
    def select_best_features(self, X, y, k=200):
        """Select top k features using multiple selection methods"""
        try:
            logger.info(f"Selecting top {k} features...")
            
            # Method 1: Chi-square test for categorical features
            chi2_selector = SelectKBest(chi2, k=k//2)
            chi2_features = chi2_selector.fit_transform(X, y)
            chi2_feature_indices = chi2_selector.get_support(indices=True)
            
            # Method 2: Mutual information for mixed features
            mi_selector = SelectKBest(mutual_info_classif, k=k//2)
            mi_features = mi_selector.fit_transform(X, y)
            mi_feature_indices = mi_selector.get_support(indices=True)
            
            # Combine selected features
            combined_indices = np.unique(np.concatenate([chi2_feature_indices, mi_feature_indices]))
            
            # If still too many features, take top k
            if len(combined_indices) > k:
                # Use chi2 scores to rank
                chi2_scores = chi2_selector.scores_
                top_indices = chi2_feature_indices[np.argsort(chi2_scores[chi2_feature_indices])[-k:]]
                combined_indices = top_indices
            
            self.selected_features = combined_indices
            logger.info(f"Selected {len(combined_indices)} best features")
            
            return X[:, combined_indices], combined_indices
            
        except Exception as e:
            logger.error(f"Feature selection failed: {str(e)}")
            return X, np.arange(X.shape[1])
    
    def engineer_interaction_features(self, df):
        """Create interaction features between important permissions"""
        try:
            interaction_features = {}
            
            # SMS + Phone combination (suspicious for banking)
            if 'permission_SEND_SMS' in df.columns and 'permission_CALL_PHONE' in df.columns:
                interaction_features['sms_phone_combo'] = (
                    df['permission_SEND_SMS'] * df['permission_CALL_PHONE']
                )
            
            # Overlay + Accessibility combination (very suspicious)
            if 'permission_SYSTEM_ALERT_WINDOW' in df.columns and 'permission_BIND_ACCESSIBILITY_SERVICE' in df.columns:
                interaction_features['overlay_accessibility_combo'] = (
                    df['permission_SYSTEM_ALERT_WINDOW'] * df['permission_BIND_ACCESSIBILITY_SERVICE']
                )
            
            # Camera + Audio combination (privacy invasion)
            if 'permission_CAMERA' in df.columns and 'permission_RECORD_AUDIO' in df.columns:
                interaction_features['camera_audio_combo'] = (
                    df['permission_CAMERA'] * df['permission_RECORD_AUDIO']
                )
            
            # Location + Storage combination (data collection)
            if 'permission_ACCESS_FINE_LOCATION' in df.columns and 'permission_WRITE_EXTERNAL_STORAGE' in df.columns:
                interaction_features['location_storage_combo'] = (
                    df['permission_ACCESS_FINE_LOCATION'] * df['permission_WRITE_EXTERNAL_STORAGE']
                )
            
            # Add interaction features to dataframe
            for feature_name, feature_values in interaction_features.items():
                df[feature_name] = feature_values
            
            logger.info(f"Created {len(interaction_features)} interaction features")
            return df
            
        except Exception as e:
            logger.error(f"Interaction feature creation failed: {str(e)}")
            return df
    
    def calculate_feature_importance(self, model, feature_names):
        """Calculate and store feature importance"""
        try:
            if hasattr(model, 'feature_importances_'):
                # For tree-based models
                importances = model.feature_importances_
            elif hasattr(model, 'coef_'):
                # For linear models
                importances = np.abs(model.coef_[0])
            else:
                # For ensemble models, get average importance
                importances = []
                for estimator_name, estimator in model.named_estimators_.items():
                    if hasattr(estimator, 'feature_importances_'):
                        importances.append(estimator.feature_importances_)
                    elif hasattr(estimator, 'coef_'):
                        importances.append(np.abs(estimator.coef_[0]))
                
                if importances:
                    importances = np.mean(importances, axis=0)
                else:
                    importances = np.ones(len(feature_names))
            
            # Create importance dictionary
            self.feature_importance = dict(zip(feature_names, importances))
            
            # Sort by importance
            sorted_importance = sorted(
                self.feature_importance.items(), 
                key=lambda x: x[1], 
                reverse=True
            )
            
            logger.info("Top 10 most important features:")
            for feature, importance in sorted_importance[:10]:
                logger.info(f"  {feature}: {importance:.4f}")
            
            return self.feature_importance
            
        except Exception as e:
            logger.error(f"Feature importance calculation failed: {str(e)}")
            return {}
    
    def preprocess_droidrl_dataset(self, df, target_column='class'):
        """Complete preprocessing pipeline for DroidRL dataset"""
        try:
            logger.info("Starting comprehensive feature preprocessing...")
            
            # 1. Add banking-specific features
            df_enhanced = self.extract_banking_features(df)
            
            # 2. Create interaction features
            df_enhanced = self.engineer_interaction_features(df_enhanced)
            
            # 3. Prepare features and labels
            if target_column in df_enhanced.columns:
                X = df_enhanced.drop([target_column], axis=1)
                y = df_enhanced[target_column]
            else:
                # Assume last column is label
                X = df_enhanced.iloc[:, :-1]
                y = df_enhanced.iloc[:, -1]
            
            # 4. Convert to numpy arrays
            X_array = X.values.astype(np.float32)
            y_array = y.values.astype(np.int32)
            feature_names = X.columns.tolist()
            
            # 5. Handle any missing values
            if np.any(np.isnan(X_array)):
                logger.warning("Missing values detected, filling with zeros")
                X_array = np.nan_to_num(X_array, nan=0.0)
            
            # 6. Feature selection (optional)
            if X_array.shape[1] > 300:  # If too many features
                X_selected, selected_indices = self.select_best_features(X_array, y_array, k=300)
                selected_feature_names = [feature_names[i] for i in selected_indices]
                logger.info(f"Feature selection: {X_array.shape[1]} -> {len(selected_indices)} features")
                
                return X_selected, y_array, selected_feature_names
            else:
                return X_array, y_array, feature_names
            
        except Exception as e:
            logger.error(f"Dataset preprocessing failed: {str(e)}")
            raise
