"""
XGBoost Sex Classification Pipeline (SEXC Prediction)
========================================================
Target: SEXC (Biological Sex - Binary Classification)
Features: 18 cytokines from healthy adults
Method: XGBoost with model comparison (LogReg, RF, SVM, GB)
Cross-validation: 10×5-fold Repeated Stratified K-Fold
Preprocessing: Outlier clipping (99th percentile) + Robust Scaling + ANOVA Feature Selection
"""

import os
os.environ['OMP_NUM_THREADS'] = '1'

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from warnings import filterwarnings
filterwarnings('ignore')

# Machine Learning
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate, cross_val_predict
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, confusion_matrix, 
                             classification_report, roc_curve, auc)

# Models
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC

# ============================================================================
# 1. LOAD & PREPROCESS DATA
# ============================================================================
print("=" * 80)
print("LOADING DATA")
print("=" * 80)

# Load data
data_path = "../Data/analysis_merged_subject_level.csv"
df = pd.read_csv(data_path)

print(f"✓ Data loaded: {df.shape[0]} samples × {df.shape[1]} columns")

# Cytokine features (18 features)
cytokines = [
    'IFN-gamma', 'IL-12p70', 'IL-13', 'IL-1beta', 'IL-2', 'IL-4', 'IL-5', 
    'IL-6', 'TNF-alpha', 'GM-CSF', 'IL-18', 'IL-10', 'IL-17A', 'IL-21', 
    'IL-22', 'IL-23', 'IL-27', 'IL-9'
]

# Target: SEXC (Male=1, Female=0)
X = df[cytokines].copy()
y = (df['SEXC'] == 'Male').astype(int).values

print(f"✓ Features: {len(cytokines)} cytokines")
print(f"✓ Target: SEXC (Sex Classification)")
print(f"  - Females (0): {(y == 0).sum()}")
print(f"  - Males (1): {(y == 1).sum()}")
print(f"  - Balance: {(y == 0).sum()}/{(y == 1).sum()} ratio = {(y == 0).sum() / (y == 1).sum():.3f}")

# ============================================================================
# 2. OUTLIER HANDLING (99th Percentile Clipping)
# ============================================================================
print("\n" + "=" * 80)
print("OUTLIER HANDLING (99th Percentile Clipping)")
print("=" * 80)

X_clipped = X.copy()
for col in X_clipped.columns:
    p99 = X_clipped[col].quantile(0.99)
    X_clipped[col] = np.minimum(X_clipped[col], p99)

print(f"✓ Clipped all features to 99th percentile")

# ============================================================================
# 3. ROBUST SCALING (Handling Outliers Better than StandardScaler)
# ============================================================================
print("\n" + "=" * 80)
print("ROBUST SCALING")
print("=" * 80)

scaler = RobustScaler()
X_scaled = scaler.fit_transform(X_clipped)
X_scaled = pd.DataFrame(X_scaled, columns=cytokines)

print(f"✓ Applied RobustScaler (handles outliers better than Z-score)")

# ============================================================================
# 4. FEATURE SELECTION (ANOVA: Select Top 12 Cytokines)
# ============================================================================
print("\n" + "=" * 80)
print("FEATURE SELECTION (ANOVA)")
print("=" * 80)

# Compute ANOVA F-scores
selector = SelectKBest(f_classif, k=12)
X_selected = selector.fit_transform(X_scaled, y)

# Get selected features
mask = selector.get_support()
selected_cytokines = [cytokines[i] for i in range(len(cytokines)) if mask[i]]
f_scores = selector.scores_

feature_selection_df = pd.DataFrame({
    'Cytokine': cytokines,
    'ANOVA_F_Score': f_scores,
    'Selected': mask
}).sort_values('ANOVA_F_Score', ascending=False)

print(f"✓ Selected top 12 features (ANOVA)")
print(f"  Feature/Sample ratio: {12}/{X.shape[0]} = {12/X.shape[0]:.3f} (good: <0.05)")
print("\nTop 5 selected features:")
print(feature_selection_df[feature_selection_df['Selected']].head(5).to_string(index=False))

# Save feature selection results
feature_selection_df.to_csv('sex_feature_selection_anova.csv', index=False)
print("\n✓ Saved: sex_feature_selection_anova.csv")

# ============================================================================
# 5. MODEL SETUP & CROSS-VALIDATION
# ============================================================================
print("\n" + "=" * 80)
print("MODEL SETUP & CROSS-VALIDATION")
print("=" * 80)

# 10×5-fold Repeated Stratified Cross-Validation (stable for small data)
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=42)

print(f"✓ Using RepeatedStratifiedKFold (5 folds × 10 repeats = 50 total folds)")

# Define models
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42, solver='lbfgs'),
    'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42, n_jobs=-1),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42),
    'SVM (RBF)': SVC(kernel='rbf', probability=True, random_state=42),
    'XGBoost': XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, 
                             random_state=42, use_label_encoder=False, 
                             eval_metric='logloss', verbosity=0),
}

# Scoring metrics
scoring_metrics = {
    'accuracy': 'accuracy',
    'precision': 'precision_zero_division=0',
    'recall': 'recall_zero_division=0',
    'f1': 'f1_zero_division=0',
    'roc_auc': 'roc_auc',
}

# ============================================================================
# 6. CROSS-VALIDATION FOR ALL MODELS
# ============================================================================
print("\n" + "=" * 80)
print("CROSS-VALIDATION RESULTS (10×5-fold)")
print("=" * 80)

cv_results = {}
model_predictions = {}

for model_name, model in models.items():
    print(f"\n{model_name}...")
    
    # Custom cross-validation to handle metrics properly
    accuracies = []
    precisions = []
    recalls = []
    f1s = []
    roc_aucs = []
    
    fold_idx = 0
    for train_idx, test_idx in cv.split(X_selected, y):
        X_train, X_test = X_selected[train_idx], X_selected[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Train
        model.fit(X_train, y_train)
        
        # Predict
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else y_pred
        
        # Metrics
        try:
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, zero_division=0)
            rec = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            roc = roc_auc_score(y_test, y_proba)
        except:
            acc = prec = rec = f1 = roc = 0.0
        
        accuracies.append(acc)
        precisions.append(prec)
        recalls.append(rec)
        f1s.append(f1)
        roc_aucs.append(roc)
        
        fold_idx += 1
    
    cv_results[model_name] = {
        'Accuracy': f"{np.mean(accuracies):.4f} ± {np.std(accuracies):.4f}",
        'Precision': f"{np.mean(precisions):.4f} ± {np.std(precisions):.4f}",
        'Recall': f"{np.mean(recalls):.4f} ± {np.std(recalls):.4f}",
        'F1-Score': f"{np.mean(f1s):.4f} ± {np.std(f1s):.4f}",
        'ROC-AUC': f"{np.mean(roc_aucs):.4f} ± {np.std(roc_aucs):.4f}",
        'Accuracy_Mean': np.mean(accuracies),
    }
    
    print(f"  ✓ Accuracy: {np.mean(accuracies):.4f} ± {np.std(accuracies):.4f}")
    print(f"  ✓ F1-Score: {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")
    print(f"  ✓ ROC-AUC:  {np.mean(roc_aucs):.4f} ± {np.std(roc_aucs):.4f}")

# ============================================================================
# 7. SAVE MODEL COMPARISON TABLE
# ============================================================================
print("\n" + "=" * 80)
print("MODEL COMPARISON TABLE")
print("=" * 80)

cv_results_df = pd.DataFrame(cv_results).T
print("\n" + cv_results_df.to_string())

cv_results_df.to_csv('sex_model_comparison.csv')
print("\n✓ Saved: sex_model_comparison.csv")

# ============================================================================
# 8. TRAIN FINAL XGBoost & GET FEATURE IMPORTANCE
# ============================================================================
print("\n" + "=" * 80)
print("FINAL XGBOOST MODEL & FEATURE IMPORTANCE")
print("=" * 80)

xgb_final = XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, 
                          random_state=42, use_label_encoder=False, 
                          eval_metric='logloss', verbosity=0)

# Train on full data for feature importance
xgb_final.fit(X_selected, y)

# Feature importance
feature_importance_dict = dict(zip(selected_cytokines, xgb_final.feature_importances_))
feature_importance_df = pd.DataFrame(list(feature_importance_dict.items()), 
                                      columns=['Cytokine', 'Importance']).sort_values('Importance', ascending=False)

print("\nTop 10 Most Important Cytokines:")
print(feature_importance_df.head(10).to_string(index=False))

feature_importance_df.to_csv('sex_xgboost_feature_importance.csv', index=False)
print("\n✓ Saved: sex_xgboost_feature_importance.csv")

# ============================================================================
# 9. FINAL MODEL EVALUATION (On Full Data)
# ============================================================================
print("\n" + "=" * 80)
print("FINAL XGBOOST EVALUATION")
print("=" * 80)

y_pred_final = xgb_final.predict(X_selected)
y_proba_final = xgb_final.predict_proba(X_selected)[:, 1]

accuracy_final = accuracy_score(y, y_pred_final)
precision_final = precision_score(y, y_pred_final, zero_division=0)
recall_final = recall_score(y, y_pred_final, zero_division=0)
f1_final = f1_score(y, y_pred_final, zero_division=0)
roc_auc_final = roc_auc_score(y, y_proba_final)
cm = confusion_matrix(y, y_pred_final)

print(f"Accuracy:  {accuracy_final:.4f}")
print(f"Precision: {precision_final:.4f}")
print(f"Recall:    {recall_final:.4f}")
print(f"F1-Score:  {f1_final:.4f}")
print(f"ROC-AUC:   {roc_auc_final:.4f}")

print("\nConfusion Matrix:")
print(f"  True Negatives (Correctly classified Females):  {cm[0, 0]}")
print(f"  False Positives (Females misclassified):        {cm[0, 1]}")
print(f"  False Negatives (Males misclassified):          {cm[1, 0]}")
print(f"  True Positives (Correctly classified Males):    {cm[1, 1]}")

print("\nClassification Report:")
print(classification_report(y, y_pred_final, target_names=['Female', 'Male'], zero_division=0))

# ============================================================================
# 10. COMPREHENSIVE VISUALIZATION
# ============================================================================
print("\n" + "=" * 80)
print("GENERATING VISUALIZATIONS")
print("=" * 80)

fig = plt.figure(figsize=(18, 12))
gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.35)

# 1. Model Comparison - Accuracy
ax1 = fig.add_subplot(gs[0, 0])
models_sorted = sorted(cv_results.items(), key=lambda x: x[1]['Accuracy_Mean'], reverse=True)
model_names = [m[0] for m in models_sorted]
accuracies = [m[1]['Accuracy_Mean'] for m in models_sorted]
colors = ['#FF6B6B' if m[0] == 'XGBoost' else '#4ECDC4' for m in models_sorted]
ax1.barh(model_names, accuracies, color=colors)
ax1.set_xlabel('Accuracy')
ax1.set_title('Model Comparison: Cross-Validated Accuracy (Sex Classification)', fontsize=11, fontweight='bold')
ax1.set_xlim(0.4, 0.8)
for i, v in enumerate(accuracies):
    ax1.text(v + 0.01, i, f'{v:.3f}', va='center', fontsize=9)

# 2. Feature Importance (Top 10)
ax2 = fig.add_subplot(gs[0, 1])
top_features = feature_importance_df.head(10)
colors_feat = plt.cm.RdYlGn(np.linspace(0.3, 0.7, len(top_features)))
ax2.barh(top_features['Cytokine'], top_features['Importance'], color=colors_feat)
ax2.set_xlabel('Importance Score')
ax2.set_title('Top 10 Most Important Cytokines (XGBoost)', fontsize=11, fontweight='bold')
ax2.invert_yaxis()

# 3. Confusion Matrix (XGBoost)
ax3 = fig.add_subplot(gs[0, 2])
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax3,
            xticklabels=['Female', 'Male'], yticklabels=['Female', 'Male'])
ax3.set_ylabel('True Label')
ax3.set_xlabel('Predicted Label')
ax3.set_title('XGBoost Confusion Matrix', fontsize=11, fontweight='bold')

# 4. ROC Curve (XGBoost)
ax4 = fig.add_subplot(gs[1, 0])
fpr, tpr, _ = roc_curve(y, y_proba_final)
ax4.plot(fpr, tpr, linewidth=2, label=f'XGBoost (AUC = {roc_auc_final:.3f})')
ax4.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
ax4.set_xlabel('False Positive Rate')
ax4.set_ylabel('True Positive Rate')
ax4.set_title('ROC Curve (XGBoost)', fontsize=11, fontweight='bold')
ax4.legend(loc='lower right')
ax4.grid(alpha=0.3)

# 5. Class Distribution
ax5 = fig.add_subplot(gs[1, 1])
class_counts = [np.sum(y == 0), np.sum(y == 1)]
ax5.bar(['Female', 'Male'], class_counts, color=['#FF9999', '#66B2FF'], edgecolor='black', linewidth=1.5)
ax5.set_ylabel('Count')
ax5.set_title('Class Distribution (SEXC)', fontsize=11, fontweight='bold')
for i, v in enumerate(class_counts):
    ax5.text(i, v + 1, str(v), ha='center', fontsize=10, fontweight='bold')

# 6. Performance Metrics Heatmap
ax6 = fig.add_subplot(gs[1, 2])
metrics_data = []
for model_name in model_names[:3]:  # Top 3 models
    row = []
    metrics_str = cv_results[model_name]
    for metric in ['Accuracy_Mean']:
        try:
            val = float(metrics_str[metric.replace('_Mean', '')] if '_Mean' in metric else metrics_str['Accuracy'].split('±')[0])
            row.append(val)
        except:
            row.append(0)
    metrics_data.append(row)

metrics_array = np.array([[cv_results[m]['Accuracy_Mean'] for m in model_names[:3]]])
sns.heatmap(metrics_array, annot=True, fmt='.3f', cmap='YlOrRd', cbar_kws={'label': 'Accuracy'},
            ax=ax6, xticklabels=model_names[:3], yticklabels=['CV Accuracy'])
ax6.set_title('Top 3 Models Performance', fontsize=11, fontweight='bold')

# 7. Feature Selection Comparison
ax7 = fig.add_subplot(gs[2, 0])
n_selected = feature_selection_df['Selected'].sum()
n_not_selected = feature_selection_df['Selected'].sum()
ax7.pie([n_selected, len(cytokines) - n_selected], labels=[f'Selected\n({n_selected})', f'Not Selected\n({len(cytokines) - n_selected})'],
        autopct='%1.1f%%', colors=['#90EE90', '#FFB6C6'], startangle=90)
ax7.set_title(f'ANOVA Feature Selection ({n_selected}/{len(cytokines)} features)', fontsize=11, fontweight='bold')

# 8. Model Performance Comparison - All Metrics
ax8 = fig.add_subplot(gs[2, 1:])
metrics_comparison = pd.DataFrame({
    'Accuracy': [cv_results[m]['Accuracy_Mean'] for m in model_names],
}, index=model_names)
metrics_comparison.plot(kind='bar', ax=ax8, color=['#FF6B6B' if m == 'XGBoost' else '#4ECDC4' for m in metrics_comparison.index])
ax8.set_ylabel('Score')
ax8.set_xlabel('Model')
ax8.set_title('Cross-Validated Accuracy by Model', fontsize=11, fontweight='bold')
ax8.set_ylim(0.4, 0.8)
ax8.legend(loc='lower right')
ax8.grid(axis='y', alpha=0.3)
plt.setp(ax8.xaxis.get_majorticklabels(), rotation=45, ha='right')

plt.savefig('sex_classification_comprehensive_results.png', dpi=300, bbox_inches='tight')
print("✓ Saved: sex_classification_comprehensive_results.png")
plt.close()

# ============================================================================
# 11. SAVE FINAL RESULTS SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("SAVING RESULTS")
print("=" * 80)

results_summary = {
    'Target': 'SEXC (Sex Classification)',
    'Samples': X.shape[0],
    'Features_Original': len(cytokines),
    'Features_Selected': 12,
    'Feature_Sample_Ratio': 12 / X.shape[0],
    'Female_Count': np.sum(y == 0),
    'Male_Count': np.sum(y == 1),
    'Class_Balance_Ratio': np.sum(y == 0) / np.sum(y == 1),
    'XGBoost_CV_Accuracy': f"{cv_results['XGBoost']['Accuracy_Mean']:.4f}",
    'XGBoost_CV_F1': cv_results['XGBoost']['F1-Score'],
    'XGBoost_CV_ROC_AUC': cv_results['XGBoost']['ROC-AUC'],
    'XGBoost_Train_Accuracy': f"{accuracy_final:.4f}",
    'XGBoost_Train_F1': f"{f1_final:.4f}",
    'XGBoost_Train_ROC_AUC': f"{roc_auc_final:.4f}",
    'Best_Model': model_names[0],
    'Best_Model_Accuracy': f"{accuracies[0]:.4f}",
}

results_summary_df = pd.DataFrame([results_summary]).T
results_summary_df.columns = ['Value']
results_summary_df.to_csv('sex_classification_results_summary.csv')
print("✓ Saved: sex_classification_results_summary.csv")

print("\n" + "=" * 80)
print("SUMMARY - SEX CLASSIFICATION RESULTS")
print("=" * 80)
print(f"\n📊 Dataset: {X.shape[0]} healthy adults")
print(f"   - Females: {np.sum(y == 0)}")
print(f"   - Males: {np.sum(y == 1)}")
print(f"\n🔧 Preprocessing:")
print(f"   - Outlier clipping: 99th percentile")
print(f"   - Scaling: RobustScaler")
print(f"   - Feature selection: ANOVA (12 of 18 cytokines)")
print(f"\n🤖 Model Performance (10×5-fold CV):")
print(f"   - Best Model: {model_names[0]} ({accuracies[0]:.1%} accuracy)")
print(f"   - XGBoost: {cv_results['XGBoost']['Accuracy_Mean']:.1%} accuracy")
print(f"\n⭐ Top 3 Important Cytokines:")
for idx, row in feature_importance_df.head(3).iterrows():
    print(f"   {idx+1}. {row['Cytokine']} ({row['Importance']:.3f})")
print("\n" + "=" * 80)
print("✓ Pipeline completed successfully!")
print("=" * 80)
