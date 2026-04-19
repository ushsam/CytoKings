"""
Enhanced XGBoost + Multiple Models Comparison
Features: Improved hyperparameter tuning, class balancing, ensemble methods
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score, StratifiedKFold, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, confusion_matrix, 
                             classification_report, roc_curve)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import warnings
import joblib
from pathlib import Path
warnings.filterwarnings('ignore')

print("="*80)
print("ENHANCED ML PIPELINE: XGBoost + Alternative Models")
print("="*80)

# ============================================================================
# PATHS
# ============================================================================
BASE_DIR    = Path(__file__).resolve().parent.parent.parent.parent
DATA_DIR    = BASE_DIR / "Data"
OUTPUT_DIR  = BASE_DIR / "XGBoost" / "sex" /"sex_enhanced" /"Outputs"
FIGURES_DIR = OUTPUT_DIR / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# 1. LOAD AND PREPARE DATA
# ============================================================================
print("\n1. Loading data...")

X = np.load(BASE_DIR / "PCA+KNN-Emma" / "Outputs" / "cytokines_only" / "X_processed.npy")
y = np.load(BASE_DIR / "PCA+KNN-Emma" / "Outputs" / "cytokines_only" / "y_labels.npy")

cytokines = [
    'IFN-gamma', 'IL-12p70', 'IL-13', 'IL-1beta', 'IL-2', 'IL-4', 'IL-5', 
    'IL-6', 'TNF-alpha', 'GM-CSF', 'IL-18', 'IL-10', 'IL-17A', 'IL-21', 
    'IL-22', 'IL-23', 'IL-27', 'IL-9'
]

print(f"Data shape: X={X.shape}, y={y.shape}")
print(f"Class distribution: {np.bincount(y)}")

# ============================================================================
# 2. CLASS BALANCING WITH SMOTE
# ============================================================================
print("\n2. Applying SMOTE for class balancing...")
smote = SMOTE(random_state=42, k_neighbors=3)
X_balanced, y_balanced = smote.fit_resample(X, y)
print(f"Balanced class distribution: {np.bincount(y_balanced)}")

# ============================================================================
# 3. DEFINE MODELS WITH OPTIMIZED PARAMETERS
# ============================================================================
print("\n3. Training multiple models...")

models = {
    'XGBoost': XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss'
    ),
    'Random Forest': RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1
    ),
    'Gradient Boosting': GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=5,
        random_state=42
    ),
    'SVM (RBF)': SVC(
        kernel='rbf',
        C=1.0,
        gamma='scale',
        probability=True,
        random_state=42
    ),
    'Logistic Regression': LogisticRegression(
        max_iter=1000,
        random_state=42,
        class_weight='balanced'
    )
}

# ============================================================================
# 4. CROSS-VALIDATION EVALUATION
# ============================================================================
print("\nCross-validation Results (5-Fold Stratified):")
print("-" * 80)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results = {}

for model_name, model in models.items():
    print(f"\n{model_name}:")
    
    # Train on balanced data
    cv_scores = cross_val_score(model, X_balanced, y_balanced, cv=cv, scoring='accuracy')
    print(f"  Accuracy:  {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    
    # Also test on original data for comparison
    cv_scores_orig = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
    print(f"  Accuracy (original data): {cv_scores_orig.mean():.4f} (+/- {cv_scores_orig.std():.4f})")
    
    results[model_name] = {
        'balanced': cv_scores.mean(),
        'original': cv_scores_orig.mean(),
        'model': model
    }

# ============================================================================
# 5. ENSEMBLE MODEL (VOTING CLASSIFIER)
# ============================================================================
print("\n" + "="*80)
print("5. Training Ensemble Voting Classifier...")
print("-"*80)

ensemble = VotingClassifier(
    estimators=[
        ('rf',  models['Random Forest']),
        ('gb',  models['Gradient Boosting']),
        ('lr',  models['Logistic Regression'])
    ],
    voting='soft'
)

cv_scores_ensemble = cross_val_score(ensemble, X_balanced, y_balanced, cv=cv, scoring='accuracy')
print(f"Ensemble Voting Accuracy: {cv_scores_ensemble.mean():.4f} (+/- {cv_scores_ensemble.std():.4f})")

results['Ensemble (Voting)'] = {
    'balanced': cv_scores_ensemble.mean(),
    'original': None,
    'model': ensemble
}

# ============================================================================
# 6. TRAIN FINAL BEST MODEL ON FULL DATA
# ============================================================================
print("\n" + "="*80)
print("6. Training Best Model on Full Dataset...")
print("-"*80)

best_model = models['XGBoost']
best_model.fit(X_balanced, y_balanced)

# Predictions
y_pred = best_model.predict(X)
y_pred_proba = best_model.predict_proba(X)[:, 1]

# Metrics
accuracy = accuracy_score(y, y_pred)
precision = precision_score(y, y_pred)
recall = recall_score(y, y_pred)
f1 = f1_score(y, y_pred)
roc_auc = roc_auc_score(y, y_pred_proba)

print(f"\nTraining Metrics (XGBoost on original data):")
print(f"  Accuracy:  {accuracy:.4f}")
print(f"  Precision: {precision:.4f}")
print(f"  Recall:    {recall:.4f}")
print(f"  F1-Score:  {f1:.4f}")
print(f"  ROC-AUC:   {roc_auc:.4f}")

print("\nClassification Report:")
print(classification_report(y, y_pred, target_names=['Female', 'Male']))

# ============================================================================
# 7. FEATURE IMPORTANCE
# ============================================================================
print("\n" + "="*80)
print("7. Feature Importance Analysis")
print("-"*80)

feature_importance = pd.DataFrame({
    'Cytokine': cytokines,
    'Importance': best_model.feature_importances_
}).sort_values('Importance', ascending=False)

print("\nTop 10 Cytokines:")
print(feature_importance.head(10).to_string(index=False))

# ============================================================================
# 8. SAVE RESULTS
# ============================================================================
print("\n" + "="*80)
print("8. Saving Results...")
print("-"*80)

# Model comparison
comparison_df = pd.DataFrame([
    {
        'Model': name,
        'CV Accuracy (Balanced)': f"{results[name]['balanced']:.4f}",
        'CV Accuracy (Original)': f"{results[name]['original']:.4f}" if results[name]['original'] else "N/A"
    }
    for name in results.keys()
])

comparison_df.to_csv(OUTPUT_DIR / "model_comparison.csv", index=False)
print("✓ Model comparison saved")

feature_importance.to_csv(OUTPUT_DIR / "feature_importance_enhanced.csv", index=False)
print("✓ Feature importance saved")

joblib.dump(best_model, OUTPUT_DIR / "xgboost_smote.pkl")
print("✓ Best model saved")

# ============================================================================
# 9. VISUALIZATION
# ============================================================================
print("\nGenerating visualizations...")

fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# 1. Model Comparison
models_list = list(results.keys())
balanced_scores = [results[m]['balanced'] for m in models_list]
original_scores = [results[m]['original'] if results[m]['original'] else results[m]['balanced'] for m in models_list]

x = np.arange(len(models_list))
width = 0.35
axes[0, 0].bar(x - width/2, balanced_scores, width, label='Balanced Data', alpha=0.8)
axes[0, 0].bar(x + width/2, original_scores, width, label='Original Data', alpha=0.8)
axes[0, 0].set_ylabel('Accuracy')
axes[0, 0].set_title('Model Comparison (5-Fold CV)')
axes[0, 0].set_xticks(x)
axes[0, 0].set_xticklabels(models_list, rotation=45, ha='right')
axes[0, 0].legend()
axes[0, 0].set_ylim([0.4, 0.75])

# 2. Top 10 Features
top_features = feature_importance.head(10)
axes[0, 1].barh(top_features['Cytokine'], top_features['Importance'], color='steelblue')
axes[0, 1].set_xlabel('Importance')
axes[0, 1].set_title('Top 10 Most Important Cytokines')
axes[0, 1].invert_yaxis()

# 3. Confusion Matrix
cm = confusion_matrix(y, y_pred)
im = axes[0, 2].imshow(cm, interpolation='nearest', cmap='Blues')
axes[0, 2].set_title('Confusion Matrix (XGBoost)')
axes[0, 2].set_ylabel('True Label')
axes[0, 2].set_xlabel('Predicted Label')
axes[0, 2].set_xticks([0, 1])
axes[0, 2].set_yticks([0, 1])
axes[0, 2].set_xticklabels(['Female', 'Male'])
axes[0, 2].set_yticklabels(['Female', 'Male'])
for i in range(2):
    for j in range(2):
        axes[0, 2].text(j, i, cm[i, j], ha='center', va='center', 
                       color='white' if cm[i, j] > cm.max()/2 else 'black', fontsize=12)

# 4. ROC Curve
fpr, tpr, _ = roc_curve(y, y_pred_proba)
axes[1, 0].plot(fpr, tpr, label=f'XGBoost (AUC = {roc_auc:.3f})', linewidth=2)
axes[1, 0].plot([0, 1], [0, 1], 'k--', label='Random')
axes[1, 0].set_xlabel('False Positive Rate')
axes[1, 0].set_ylabel('True Positive Rate')
axes[1, 0].set_title('ROC Curve')
axes[1, 0].legend()

# 5. Class Distribution
classes = ['Female', 'Male']
axes[1, 1].bar(classes, np.bincount(y), alpha=0.7, label='Original')
axes[1, 1].bar(classes, np.bincount(y_balanced), alpha=0.7, label='After SMOTE')
axes[1, 1].set_ylabel('Sample Count')
axes[1, 1].set_title('Class Balance: Before vs After SMOTE')
axes[1, 1].legend()

# 6. All Feature Importances
axes[1, 2].barh(feature_importance['Cytokine'][::-1], feature_importance['Importance'][::-1], color='coral')
axes[1, 2].set_xlabel('Importance')
axes[1, 2].set_title('All Cytokines - Feature Importance')

plt.tight_layout()
plt.savefig(FIGURES_DIR / "xgboost_results_visualization.png",
            dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("✓ Visualization saved")

print("\n" + "="*80)
print("✅ Enhanced pipeline completed!")
print("="*80)
