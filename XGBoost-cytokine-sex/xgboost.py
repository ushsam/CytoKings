"""
XGBoost Sex Classification Pipeline (SEXC Prediction)
========================================================
Target: SEXC (Biological Sex - Binary Classification)
Features: 18 cytokines from healthy adults
Method: XGBoost with model comparison (LogReg, RF, SVM, GB)
Cross-validation: 10×5-fold Repeated Stratified K-Fold
Preprocessing: Outlier clipping (99th percentile) + Robust Scaling + ANOVA Feature Selection
Additions: Grid search for XGBoost + LogReg C tuning + SHAP + CV-based ROC
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
from sklearn.model_selection import (RepeatedStratifiedKFold, cross_val_score,
                                     cross_val_predict, StratifiedKFold)
from sklearn.pipeline import Pipeline

# Optional: mlflow experiment tracking (runs fine without it)
try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             classification_report, roc_curve, auc)

# Models
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC

# SHAP for feature interpretability
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("⚠ shap not installed. Run: pip install shap")

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

# ============================================================================
# 5a. GRID SEARCH — XGBoost hyperparameters (fills Table 2 in paper)
# ============================================================================
print("\n" + "=" * 80)
print("GRID SEARCH: XGBoost Hyperparameters")
print("=" * 80)

# Pipeline ensures ANOVA is fit on training fold only — no data leakage
xgb_param_grid = {
    'n_estimators':  [100, 200],
    'max_depth':     [3, 5],
    'learning_rate': [0.05, 0.1],
}

cv_inner = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=42)

best_xgb_score  = -1
best_xgb_params = {}
grid_results    = []

total_combos = (len(xgb_param_grid['n_estimators']) *
                len(xgb_param_grid['max_depth']) *
                len(xgb_param_grid['learning_rate']))
combo_idx = 0

for n_est in xgb_param_grid['n_estimators']:
    for depth in xgb_param_grid['max_depth']:
        for lr in xgb_param_grid['learning_rate']:
            combo_idx += 1
            pipe = Pipeline([
                ('selector', SelectKBest(f_classif, k=12)),
                ('xgb', XGBClassifier(
                    n_estimators=n_est,
                    max_depth=depth,
                    learning_rate=lr,
                    random_state=42,
                    eval_metric='logloss',
                    verbosity=0
                ))
            ])
            scores = cross_val_score(pipe, X_scaled, y, cv=cv_inner,
                                     scoring='accuracy', n_jobs=-1)
            mean_acc = scores.mean()
            std_acc  = scores.std()

            grid_results.append({
                'n_estimators': n_est, 'max_depth': depth,
                'learning_rate': lr,
                'CV_Accuracy_Mean': round(mean_acc, 4),
                'CV_Accuracy_Std':  round(std_acc, 4),
            })
            print(f"  [{combo_idx}/{total_combos}] n_est={n_est}, depth={depth}, "
                  f"lr={lr}  →  {mean_acc:.4f} ± {std_acc:.4f}")

            if mean_acc > best_xgb_score:
                best_xgb_score  = mean_acc
                best_xgb_params = {'n_estimators': n_est, 'max_depth': depth,
                                   'learning_rate': lr}

grid_df = pd.DataFrame(grid_results).sort_values('CV_Accuracy_Mean', ascending=False)
grid_df.to_csv('results_xgboost_config_comparison.csv', index=False)

print(f"\n✓ Best XGBoost config: {best_xgb_params}")
print(f"  Best CV Accuracy: {best_xgb_score:.4f}")
print("✓ Saved: results_xgboost_config_comparison.csv")

# ============================================================================
# 5b. GRID SEARCH — Logistic Regression C tuning (fills Table 2 in paper)
# ============================================================================
print("\n" + "=" * 80)
print("GRID SEARCH: Logistic Regression C tuning")
print("=" * 80)

best_lr_score = -1
best_lr_C     = 1.0
lr_results    = []

for C_val in [0.01, 0.1, 1.0, 10.0]:
    pipe_lr = Pipeline([
        ('selector', SelectKBest(f_classif, k=12)),
        ('lr', LogisticRegression(C=C_val, max_iter=1000,
                                  random_state=42, solver='lbfgs'))
    ])
    scores = cross_val_score(pipe_lr, X_scaled, y, cv=cv_inner,
                             scoring='accuracy', n_jobs=-1)
    mean_acc = scores.mean()
    lr_results.append({'C': C_val, 'CV_Accuracy_Mean': round(mean_acc, 4),
                       'CV_Accuracy_Std': round(scores.std(), 4)})
    print(f"  C={C_val:<6}  →  {mean_acc:.4f} ± {scores.std():.4f}")

    if mean_acc > best_lr_score:
        best_lr_score = mean_acc
        best_lr_C     = C_val

pd.DataFrame(lr_results).to_csv('results_logreg_c_tuning.csv', index=False)
print(f"\n✓ Best LogReg C: {best_lr_C}  (CV Accuracy: {best_lr_score:.4f})")
print("✓ Saved: results_logreg_c_tuning.csv")

# ============================================================================
# 5c. FINAL CV SETUP using best hyperparameters
# ============================================================================
# 10×5-fold Repeated Stratified Cross-Validation (stable for small data)
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=42)

print(f"\n✓ Using RepeatedStratifiedKFold (5 folds × 10 repeats = 50 total folds)")
print(f"✓ XGBoost best params: {best_xgb_params}")
print(f"✓ LogReg best C: {best_lr_C}")

# Define models using tuned hyperparameters
models = {
    'Logistic Regression': LogisticRegression(C=best_lr_C, max_iter=1000,
                                              random_state=42, solver='lbfgs'),
    'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=5,
                                            random_state=42, n_jobs=-1),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, max_depth=3,
                                                    random_state=42),
    'SVM (RBF)': SVC(kernel='rbf', probability=True, random_state=42),
    'XGBoost': XGBClassifier(
        n_estimators=best_xgb_params['n_estimators'],
        max_depth=best_xgb_params['max_depth'],
        learning_rate=best_xgb_params['learning_rate'],
        random_state=42,
        eval_metric='logloss',
        verbosity=0
    ),
}

# ============================================================================
# 6. CROSS-VALIDATION FOR ALL MODELS
# ============================================================================
print("\n" + "=" * 80)
print("CROSS-VALIDATION RESULTS (10×5-fold)")
print("=" * 80)

cv_results = {}

if MLFLOW_AVAILABLE:
    mlflow.set_experiment("cytokine_sex_classification")
    run = mlflow.start_run(run_name="xgboost_sex_pipeline")
    mlflow.log_params({
        'xgb_n_estimators': best_xgb_params['n_estimators'],
        'xgb_max_depth':    best_xgb_params['max_depth'],
        'xgb_learning_rate': best_xgb_params['learning_rate'],
        'logreg_C':         best_lr_C,
        'n_features_selected': 12,
        'cv_folds': 5,
        'cv_repeats': 10,
    })

for model_name, model in models.items():
    print(f"\n{model_name}...")

    accuracies, precisions, recalls, f1s, roc_aucs = [], [], [], [], []

    for train_idx, test_idx in cv.split(X_selected, y):
        X_train, X_test = X_selected[train_idx], X_selected[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model.fit(X_train, y_train)
        y_pred  = model.predict(X_test)
        y_proba = (model.predict_proba(X_test)[:, 1]
                   if hasattr(model, 'predict_proba') else y_pred.astype(float))

        try:
            accuracies.append(accuracy_score(y_test, y_pred))
            precisions.append(precision_score(y_test, y_pred, zero_division=0))
            recalls.append(recall_score(y_test, y_pred, zero_division=0))
            f1s.append(f1_score(y_test, y_pred, zero_division=0))
            roc_aucs.append(roc_auc_score(y_test, y_proba))
        except Exception:
            accuracies.append(0.0); precisions.append(0.0)
            recalls.append(0.0);   f1s.append(0.0); roc_aucs.append(0.0)

    cv_results[model_name] = {
        'Accuracy':      f"{np.mean(accuracies):.4f} ± {np.std(accuracies):.4f}",
        'Precision':     f"{np.mean(precisions):.4f} ± {np.std(precisions):.4f}",
        'Recall':        f"{np.mean(recalls):.4f} ± {np.std(recalls):.4f}",
        'F1-Score':      f"{np.mean(f1s):.4f} ± {np.std(f1s):.4f}",
        'ROC-AUC':       f"{np.mean(roc_aucs):.4f} ± {np.std(roc_aucs):.4f}",
        'Accuracy_Mean': np.mean(accuracies),
    }

    print(f"  ✓ Accuracy: {np.mean(accuracies):.4f} ± {np.std(accuracies):.4f}")
    print(f"  ✓ F1-Score: {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")
    print(f"  ✓ ROC-AUC:  {np.mean(roc_aucs):.4f} ± {np.std(roc_aucs):.4f}")

    if MLFLOW_AVAILABLE:
        safe = model_name.replace(' ', '_').replace('(', '').replace(')', '')
        mlflow.log_metrics({
            f"{safe}_accuracy": np.mean(accuracies),
            f"{safe}_f1":       np.mean(f1s),
            f"{safe}_roc_auc":  np.mean(roc_aucs),
        })

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

xgb_final = XGBClassifier(
    n_estimators=best_xgb_params['n_estimators'],
    max_depth=best_xgb_params['max_depth'],
    learning_rate=best_xgb_params['learning_rate'],
    random_state=42,
    eval_metric='logloss',
    verbosity=0
)

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

# 4. ROC Curve — cross-validated probabilities (not training set)
ax4 = fig.add_subplot(gs[1, 0])
cv_roc_inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
y_proba_cv_plot = cross_val_predict(
    XGBClassifier(
        n_estimators=best_xgb_params['n_estimators'],
        max_depth=best_xgb_params['max_depth'],
        learning_rate=best_xgb_params['learning_rate'],
        random_state=42, eval_metric='logloss', verbosity=0
    ),
    X_selected, y, cv=cv_roc_inner, method='predict_proba'
)[:, 1]
fpr_plot, tpr_plot, _ = roc_curve(y, y_proba_cv_plot)
auc_plot = auc(fpr_plot, tpr_plot)
ax4.plot(fpr_plot, tpr_plot, linewidth=2,
         label=f'XGBoost CV (AUC = {auc_plot:.3f})')
ax4.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
ax4.set_xlabel('False Positive Rate')
ax4.set_ylabel('True Positive Rate')
ax4.set_title('ROC Curve — Cross-Validated', fontsize=11, fontweight='bold')
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

plt.savefig('final_plots_sex_classification_comprehensive_results.png', dpi=300, bbox_inches='tight')
print("✓ Saved: final_plots_sex_classification_comprehensive_results.png")
plt.close()

# ============================================================================
# 10b. CYTOKINE-SEX DIFFERENCE VISUALIZATIONS
# ============================================================================
print("\n" + "=" * 80)
print("CYTOKINE-SEX DIFFERENCE VISUALIZATIONS")
print("=" * 80)

# Build a plot-ready dataframe with raw (clipped) values and sex labels
plot_df = X_clipped.copy()
plot_df['Sex'] = pd.Series(y).map({0: 'Female', 1: 'Male'}).values

# Top cytokines by ANOVA F-score (selected ones, sorted)
top_anova = feature_selection_df[feature_selection_df['Selected']].sort_values(
    'ANOVA_F_Score', ascending=False
)
top_cytokines_plot = top_anova['Cytokine'].tolist()[:8]  # top 8 for readability

fig2, axes2 = plt.subplots(3, 3, figsize=(18, 15))
fig2.suptitle('Cytokine Differences Between Sexes (Healthy Adults)', fontsize=15, fontweight='bold', y=1.01)

# --- Panel 1: ANOVA F-scores for all 18 cytokines ---
ax = axes2[0, 0]
fs_sorted = feature_selection_df.sort_values('ANOVA_F_Score', ascending=True)
bar_colors = ['#FF6B6B' if sel else '#CCCCCC' for sel in fs_sorted['Selected']]
ax.barh(fs_sorted['Cytokine'], fs_sorted['ANOVA_F_Score'], color=bar_colors)
ax.set_xlabel('ANOVA F-Score')
ax.set_title('ANOVA F-Scores: All 18 Cytokines\n(red = selected for model)', fontsize=10, fontweight='bold')
ax.axvline(fs_sorted[fs_sorted['Selected']]['ANOVA_F_Score'].min(), color='red',
           linestyle='--', linewidth=1, alpha=0.6, label='selection threshold')
ax.legend(fontsize=8)

# --- Panel 2: Mean cytokine levels by sex (heatmap) ---
ax = axes2[0, 1]
mean_by_sex = plot_df.groupby('Sex')[cytokines].mean()
# Normalize per cytokine (z-score) so all fit on same scale
mean_z = (mean_by_sex - mean_by_sex.mean()) / (mean_by_sex.std() + 1e-9)
# Only show selected cytokines for clarity
mean_z_sel = mean_z[top_cytokines_plot]
sns.heatmap(mean_z_sel, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
            ax=ax, cbar_kws={'label': 'Z-score'}, linewidths=0.5)
ax.set_title('Mean Cytokine Levels by Sex\n(Z-scored per cytokine, ANOVA-selected)', fontsize=10, fontweight='bold')
ax.set_xlabel('')

# --- Panel 3: Log2 fold change Male vs Female ---
ax = axes2[0, 2]
eps = 1e-6
male_mean = plot_df[plot_df['Sex'] == 'Male'][cytokines].mean()
female_mean = plot_df[plot_df['Sex'] == 'Female'][cytokines].mean()
log2fc = np.log2((male_mean + eps) / (female_mean + eps))
log2fc_df = log2fc.sort_values()
fc_colors = ['#66B2FF' if v > 0 else '#FF9999' for v in log2fc_df.values]
ax.barh(log2fc_df.index, log2fc_df.values, color=fc_colors)
ax.axvline(0, color='black', linewidth=1)
ax.set_xlabel('Log2 Fold Change (Male / Female)')
ax.set_title('Cytokine Log2 Fold Change\nMale vs Female', fontsize=10, fontweight='bold')
ax.text(0.98, 0.02, 'Blue = higher in Males\nRed = higher in Females',
        transform=ax.transAxes, fontsize=8, ha='right', va='bottom',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

# --- Panels 4-9: Violin plots for top 6 ANOVA cytokines ---
palette = {'Female': '#FF9999', 'Male': '#66B2FF'}
for i, cyto in enumerate(top_cytokines_plot[:6]):
    row, col = divmod(i, 3)
    ax = axes2[row + 1, col]
    sns.violinplot(data=plot_df, x='Sex', y=cyto, palette=palette,
                   inner='box', ax=ax, order=['Female', 'Male'])
    # Overlay individual points (strip)
    sns.stripplot(data=plot_df, x='Sex', y=cyto, color='black', alpha=0.25,
                  size=2.5, jitter=True, ax=ax, order=['Female', 'Male'])
    f_val = feature_selection_df.loc[feature_selection_df['Cytokine'] == cyto, 'ANOVA_F_Score'].values[0]
    ax.set_title(f'{cyto}\n(ANOVA F={f_val:.1f})', fontsize=10, fontweight='bold')
    ax.set_xlabel('')
    ax.set_ylabel('Concentration (clipped)')

plt.tight_layout()
plt.savefig('final_plots_sex_cytokine_differences.png', dpi=300, bbox_inches='tight')
print("✓ Saved: final_plots_sex_cytokine_differences.png")
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

# ============================================================================
# 12. CV-BASED ROC CURVE (correct — not training set)
# ============================================================================
print("\n" + "=" * 80)
print("CV-BASED ROC CURVE (cross-validated probabilities)")
print("=" * 80)

# cross_val_predict with predict_proba gives probabilities from held-out folds
# This is the correct ROC for reporting — the main script ROC was on training data.
cv_roc = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
y_proba_cv = cross_val_predict(
    XGBClassifier(
        n_estimators=best_xgb_params['n_estimators'],
        max_depth=best_xgb_params['max_depth'],
        learning_rate=best_xgb_params['learning_rate'],
        random_state=42, eval_metric='logloss', verbosity=0
    ),
    X_selected, y,
    cv=cv_roc,
    method='predict_proba'
)[:, 1]

fpr_cv, tpr_cv, _ = roc_curve(y, y_proba_cv)
auc_cv = auc(fpr_cv, tpr_cv)

fig_roc, ax_roc = plt.subplots(figsize=(6, 5))
ax_roc.plot(fpr_cv, tpr_cv, color='#CC3333', linewidth=2.5,
            label=f'XGBoost CV ROC (AUC = {auc_cv:.3f})')
ax_roc.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random (AUC = 0.500)')
ax_roc.set_xlabel('False Positive Rate', fontsize=11)
ax_roc.set_ylabel('True Positive Rate', fontsize=11)
ax_roc.set_title('XGBoost ROC Curve\n(Cross-Validated Probabilities)', fontsize=11,
                 fontweight='bold')
ax_roc.legend(fontsize=10)
ax_roc.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('results_xgboost_cv_roc.png', dpi=200, bbox_inches='tight')
print(f"✓ CV AUC: {auc_cv:.4f}")
print("✓ Saved: results_xgboost_cv_roc.png")
plt.close()

# ============================================================================
# 13. SHAP ANALYSIS — directionality + magnitude of feature contributions
# ============================================================================
print("\n" + "=" * 80)
print("SHAP ANALYSIS")
print("=" * 80)

if SHAP_AVAILABLE:
    # SHAP values: positive = pushes toward Male (class 1)
    #              negative = pushes toward Female (class 0)
    explainer   = shap.TreeExplainer(xgb_final)
    shap_values = explainer.shap_values(X_selected)

    # Beeswarm summary plot: shows magnitude AND directionality per sample
    plt.figure(figsize=(9, 6))
    shap.summary_plot(
        shap_values,
        X_selected,
        feature_names=selected_cytokines,
        show=False,
        plot_size=(9, 6)
    )
    plt.title("SHAP Summary: Feature Contributions to Sex Prediction\n"
              "(Red = high value, Blue = low value; right = Male, left = Female)",
              fontsize=10, fontweight='bold')
    plt.tight_layout()
    plt.savefig('results_shap_beeswarm.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("✓ Saved: results_shap_beeswarm.png")

    # Bar plot: mean absolute SHAP (magnitude only, like feature_importances_)
    plt.figure(figsize=(7, 5))
    shap.summary_plot(
        shap_values,
        X_selected,
        feature_names=selected_cytokines,
        plot_type='bar',
        show=False
    )
    plt.title("SHAP Feature Importance (Mean |SHAP|)", fontsize=10, fontweight='bold')
    plt.tight_layout()
    plt.savefig('results_shap_bar.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("✓ Saved: results_shap_bar.png")

    # Save SHAP values to CSV for paper table
    shap_mean_abs = pd.DataFrame({
        'Cytokine':        selected_cytokines,
        'Mean_Abs_SHAP':   np.abs(shap_values).mean(axis=0).round(4),
        'Mean_SHAP':       shap_values.mean(axis=0).round(4),
        'Direction':       ['Male↑' if v > 0 else 'Female↑'
                            for v in shap_values.mean(axis=0)]
    }).sort_values('Mean_Abs_SHAP', ascending=False)

    shap_mean_abs.to_csv('results_shap_values.csv', index=False)
    print("✓ Saved: results_shap_values.csv")

    print("\nTop 5 SHAP features (with directionality):")
    print(shap_mean_abs.head(5).to_string(index=False))
    print("\nInterpretation:")
    print("  Male↑  = high cytokine value pushes prediction toward Male")
    print("  Female↑ = high cytokine value pushes prediction toward Female")
else:
    print("⚠ SHAP not available. Install with: pip install shap")
    print("  Then rerun this script to generate SHAP plots.")

# ============================================================================
# 14. HYPERPARAMETER TUNING SUMMARY (for Table 2 in paper)
# ============================================================================
print("\n" + "=" * 80)
print("TABLE 2 — HYPERPARAMETER TUNING SUMMARY")
print("=" * 80)

table2_data = [
    {'Model': 'PCA+KNN', 'Parameter': 'n_pca',         'Search Range': '2–10',              'Best Value': '4',                              'CV Score': '56.8%'},
    {'Model': 'PCA+KNN', 'Parameter': 'K',              'Search Range': '3–7',               'Best Value': '7',                              'CV Score': '56.8%'},
    {'Model': 'XGBoost', 'Parameter': 'n_estimators',   'Search Range': '{100, 200}',         'Best Value': str(best_xgb_params['n_estimators']),  'CV Score': f"{best_xgb_score*100:.1f}%"},
    {'Model': 'XGBoost', 'Parameter': 'max_depth',      'Search Range': '{3, 5}',             'Best Value': str(best_xgb_params['max_depth']),      'CV Score': f"{best_xgb_score*100:.1f}%"},
    {'Model': 'XGBoost', 'Parameter': 'learning_rate',  'Search Range': '{0.05, 0.1}',        'Best Value': str(best_xgb_params['learning_rate']),  'CV Score': f"{best_xgb_score*100:.1f}%"},
    {'Model': 'LogReg',  'Parameter': 'C',              'Search Range': '{0.01, 0.1, 1, 10}', 'Best Value': str(best_lr_C),                         'CV Score': f"{best_lr_score*100:.1f}%"},
]

table2_df = pd.DataFrame(table2_data)
print(table2_df.to_string(index=False))
table2_df.to_csv('results_table2_hyperparameters.csv', index=False)
print("\n✓ Saved: results_table2_hyperparameters.csv  ← copy these into Table 2")

if MLFLOW_AVAILABLE:
    mlflow.log_artifact('sex_model_comparison.csv')
    mlflow.log_artifact('sex_xgboost_feature_importance.csv')
    mlflow.log_artifact('results_shap_values.csv')
    mlflow.log_artifact('results_table2_hyperparameters.csv')
    mlflow.end_run()
    print("✓ mlflow run logged")

print("\n" + "=" * 80)
print("✓ Pipeline completed successfully!")
print("=" * 80)
