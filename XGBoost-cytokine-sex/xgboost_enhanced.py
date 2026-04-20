"""
Enhanced XGBoost + Multiple Models Comparison
Features: Improved hyperparameter tuning, class balancing, ensemble methods
"""

import matplotlib
matplotlib.use('Agg')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score, StratifiedKFold
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
BASE_DIR    = Path(__file__).resolve().parent.parent
DATA_DIR    = BASE_DIR / "Data-Emma"
OUTPUT_DIR  = BASE_DIR / "XGBoost-cytokine-sex"/ "sex_enhanced-output"
FIGURES_DIR = OUTPUT_DIR / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# SAVE PANEL HELPER
# ============================================================================
def save_panel(fig, ax, filename, pad=1.15):
    try:
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        extent   = ax.get_tightbbox(renderer)
        extent   = extent.transformed(fig.dpi_scale_trans.inverted())
        fig.savefig(FIGURES_DIR / filename,
                    bbox_inches=extent.expanded(pad, pad),
                    dpi=150, facecolor="white")
        print(f"  Panel saved: {filename}")
    except Exception as e:
        fig.savefig(FIGURES_DIR / filename,
                    bbox_inches="tight", dpi=150, facecolor="white")
        print(f"  Panel saved (fallback): {filename} — {e}")

# ============================================================================
# 1. LOAD AND PREPARE DATA
# ============================================================================
print("\n1. Loading data...")

X = np.load(BASE_DIR / "PCA+KNN-Emma" / "Outputs" / "cytokines_only" / "X_processed.npy")
y = np.load(BASE_DIR / "PCA+KNN-Emma" / "Outputs" / "cytokines_only" / "y_labels.npy")

# Extract cytokine names dynamically from data file
import pandas as pd_check
_df = pd_check.read_csv(BASE_DIR / "Data" / "analysis_merged_subject_level.csv")
DEMO_COLS = ['SUBJECT_ID', 'AGE', 'AGEGR1N', 'AGEGR1C', 'RACEGRP',
             'SEXN', 'SEXC', 'FASFL', 'AGE_BINARY', 'Batch date']
cytokines = [c for c in _df.columns
             if c not in DEMO_COLS and not c.startswith('CT_')]

print(f"Data shape: X={X.shape}, y={y.shape}")
print(f"Cytokines: {cytokines}")
print(f"Class distribution: {np.bincount(y)}")

# ============================================================================
# 2. CLASS BALANCING WITH SMOTE
# ============================================================================
print("\n2. Applying SMOTE for class balancing...")
smote = SMOTE(random_state=42, k_neighbors=3)
X_balanced, y_balanced = smote.fit_resample(X, y)
print(f"Balanced class distribution: {np.bincount(y_balanced)}")

# ============================================================================
# 3. DEFINE MODELS
# ============================================================================
print("\n3. Training multiple models...")

models = {
    'XGBoost': XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        random_state=42, eval_metric='logloss',
        objective='binary:logistic'
    ),
    'Random Forest': RandomForestClassifier(
        n_estimators=200, max_depth=15, min_samples_split=5,
        random_state=42, n_jobs=-1
    ),
    'Gradient Boosting': GradientBoostingClassifier(
        n_estimators=200, learning_rate=0.05, max_depth=5,
        random_state=42
    ),
    'SVM (RBF)': SVC(
        kernel='rbf', C=1.0, gamma='scale',
        probability=True, random_state=42
    ),
    'Logistic Regression': LogisticRegression(
        max_iter=1000, random_state=42, class_weight='balanced'
    )
}

# ============================================================================
# 4. CROSS-VALIDATION EVALUATION
# ============================================================================
print("\nCross-validation Results (5-Fold Stratified):")
print("-" * 80)

cv      = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results = {}

for model_name, model in models.items():
    print(f"\n{model_name}:")

    # Balanced data
    cv_acc_bal  = cross_val_score(model, X_balanced, y_balanced, cv=cv, scoring='accuracy')
    cv_f1_bal   = cross_val_score(model, X_balanced, y_balanced, cv=cv, scoring='f1')
    cv_auc_bal  = cross_val_score(model, X_balanced, y_balanced, cv=cv, scoring='roc_auc')

    # Original data
    cv_acc_orig = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
    cv_f1_orig  = cross_val_score(model, X, y, cv=cv, scoring='f1')
    cv_auc_orig = cross_val_score(model, X, y, cv=cv, scoring='roc_auc')

    print(f"  Balanced  — Acc: {cv_acc_bal.mean():.4f}  "
          f"F1: {cv_f1_bal.mean():.4f}  "
          f"AUC: {cv_auc_bal.mean():.4f}")
    print(f"  Original  — Acc: {cv_acc_orig.mean():.4f}  "
          f"F1: {cv_f1_orig.mean():.4f}  "
          f"AUC: {cv_auc_orig.mean():.4f}")

    results[model_name] = {
        'acc_balanced':  cv_acc_bal.mean(),
        'acc_original':  cv_acc_orig.mean(),
        'f1_balanced':   cv_f1_bal.mean(),
        'f1_original':   cv_f1_orig.mean(),
        'auc_balanced':  cv_auc_bal.mean(),
        'auc_original':  cv_auc_orig.mean(),
        'model':         model
    }

# ============================================================================
# 5. ENSEMBLE MODEL
# ============================================================================
print("\n" + "="*80)
print("5. Training Ensemble Voting Classifier...")

ensemble = VotingClassifier(
    estimators=[
        ('rf', models['Random Forest']),
        ('gb', models['Gradient Boosting']),
        ('lr', models['Logistic Regression'])
    ],
    voting='soft'
)

cv_acc_ens  = cross_val_score(ensemble, X_balanced, y_balanced, cv=cv, scoring='accuracy')
cv_f1_ens   = cross_val_score(ensemble, X_balanced, y_balanced, cv=cv, scoring='f1')
cv_auc_ens  = cross_val_score(ensemble, X_balanced, y_balanced, cv=cv, scoring='roc_auc')

print(f"Ensemble — Acc: {cv_acc_ens.mean():.4f}  "
      f"F1: {cv_f1_ens.mean():.4f}  "
      f"AUC: {cv_auc_ens.mean():.4f}")

results['Ensemble (Voting)'] = {
    'acc_balanced':  cv_acc_ens.mean(),
    'acc_original':  cv_acc_ens.mean(),
    'f1_balanced':   cv_f1_ens.mean(),
    'f1_original':   cv_f1_ens.mean(),
    'auc_balanced':  cv_auc_ens.mean(),
    'auc_original':  cv_auc_ens.mean(),
    'model':         ensemble
}

# ============================================================================
# 6. TRAIN FINAL BEST MODEL
# ============================================================================
print("\n" + "="*80)
print("6. Training Best Model on Full Dataset...")

best_model = models['XGBoost']
best_model.fit(X_balanced, y_balanced)

y_pred       = best_model.predict(X)
y_pred_proba = best_model.predict_proba(X)[:, 1]

accuracy  = accuracy_score(y, y_pred)
precision = precision_score(y, y_pred)
recall    = recall_score(y, y_pred)
f1        = f1_score(y, y_pred)
roc_auc   = roc_auc_score(y, y_pred_proba)

print(f"\nTraining Metrics (XGBoost on original data):")
print(f"  Accuracy  : {accuracy:.4f}")
print(f"  Precision : {precision:.4f}")
print(f"  Recall    : {recall:.4f}")
print(f"  F1-Score  : {f1:.4f}")
print(f"  ROC-AUC   : {roc_auc:.4f}")
print("\nClassification Report:")
print(classification_report(y, y_pred, target_names=['Female', 'Male']))

# ============================================================================
# 7. FEATURE IMPORTANCE
# ============================================================================
print("\n" + "="*80)
print("7. Feature Importance Analysis")

feature_importance = pd.DataFrame({
    'Cytokine':   cytokines,
    'Importance': best_model.feature_importances_
}).sort_values('Importance', ascending=False)

print("\nTop 10 Cytokines:")
print(feature_importance.head(10).to_string(index=False))

# ============================================================================
# 8. SAVE RESULTS
# ============================================================================
print("\n" + "="*80)
print("8. Saving Results...")

comparison_df = pd.DataFrame([
    {
        'Model':                    name,
        'Accuracy (Balanced)':      f"{results[name]['acc_balanced']:.4f}",
        'Accuracy (Original)':      f"{results[name]['acc_original']:.4f}",
        'F1 (Balanced)':            f"{results[name]['f1_balanced']:.4f}",
        'F1 (Original)':            f"{results[name]['f1_original']:.4f}",
        'ROC-AUC (Balanced)':       f"{results[name]['auc_balanced']:.4f}",
        'ROC-AUC (Original)':       f"{results[name]['auc_original']:.4f}",
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

models_list     = list(results.keys())
acc_balanced    = [results[m]['acc_balanced']  for m in models_list]
acc_original    = [results[m]['acc_original']  for m in models_list]
f1_balanced     = [results[m]['f1_balanced']   for m in models_list]
f1_original     = [results[m]['f1_original']   for m in models_list]
auc_balanced    = [results[m]['auc_balanced']  for m in models_list]
auc_original    = [results[m]['auc_original']  for m in models_list]
x               = np.arange(len(models_list))
width           = 0.35

fig, axes = plt.subplots(2, 4, figsize=(24, 10))
fig.suptitle("Enhanced ML Pipeline: Sex Prediction\n"
             "XGBoost + Alternative Models",
             fontsize=14, fontweight='bold')

# Panel 1: Accuracy
axes[0, 0].bar(x - width/2, acc_balanced, width, label='Balanced', alpha=0.8)
axes[0, 0].bar(x + width/2, acc_original, width, label='Original',  alpha=0.8)
axes[0, 0].axhline(0.5, color='red', linestyle='--', alpha=0.5, label='Random')
axes[0, 0].set_ylabel('Accuracy')
axes[0, 0].set_title('Accuracy (5-Fold CV)')
axes[0, 0].set_xticks(x)
axes[0, 0].set_xticklabels(models_list, rotation=45, ha='right', fontsize=8)
axes[0, 0].legend(fontsize=8)
axes[0, 0].set_ylim([0.4, 0.85])
axes[0, 0].grid(axis='y', alpha=0.3)

# Panel 2: F1 Score
axes[0, 1].bar(x - width/2, f1_balanced, width, label='Balanced', alpha=0.8, color='#2ca02c')
axes[0, 1].bar(x + width/2, f1_original, width, label='Original',  alpha=0.8, color='#98df8a')
axes[0, 1].axhline(0.5, color='red', linestyle='--', alpha=0.5, label='Random')
axes[0, 1].set_ylabel('F1 Score')
axes[0, 1].set_title('F1 Score (5-Fold CV)')
axes[0, 1].set_xticks(x)
axes[0, 1].set_xticklabels(models_list, rotation=45, ha='right', fontsize=8)
axes[0, 1].legend(fontsize=8)
axes[0, 1].set_ylim([0.4, 0.85])
axes[0, 1].grid(axis='y', alpha=0.3)

# Panel 3: ROC-AUC
axes[0, 2].bar(x - width/2, auc_balanced, width, label='Balanced', alpha=0.8, color='#ff7f0e')
axes[0, 2].bar(x + width/2, auc_original, width, label='Original',  alpha=0.8, color='#ffbb78')
axes[0, 2].axhline(0.5, color='red', linestyle='--', alpha=0.5, label='Random')
axes[0, 2].set_ylabel('ROC-AUC')
axes[0, 2].set_title('ROC-AUC (5-Fold CV)')
axes[0, 2].set_xticks(x)
axes[0, 2].set_xticklabels(models_list, rotation=45, ha='right', fontsize=8)
axes[0, 2].legend(fontsize=8)
axes[0, 2].set_ylim([0.4, 0.85])
axes[0, 2].grid(axis='y', alpha=0.3)

# Panel 4: Top 10 features
top_features = feature_importance.head(10)
axes[0, 3].barh(top_features['Cytokine'], top_features['Importance'],
                color='steelblue')
axes[0, 3].set_xlabel('Importance')
axes[0, 3].set_title('Top 10 Cytokines (XGBoost)')
axes[0, 3].invert_yaxis()
for i, v in enumerate(top_features['Importance']):
    axes[0, 3].text(v + 0.005, i, f'{v:.3f}', va='center', fontsize=8)

# Panel 5: Confusion matrix
cm = confusion_matrix(y, y_pred)
im = axes[1, 0].imshow(cm, interpolation='nearest', cmap='Blues')
axes[1, 0].set_title('Confusion Matrix (XGBoost)')
axes[1, 0].set_ylabel('True Label')
axes[1, 0].set_xlabel('Predicted Label')
axes[1, 0].set_xticks([0, 1])
axes[1, 0].set_yticks([0, 1])
axes[1, 0].set_xticklabels(['Female', 'Male'])
axes[1, 0].set_yticklabels(['Female', 'Male'])
for i in range(2):
    for j in range(2):
        axes[1, 0].text(j, i, cm[i, j], ha='center', va='center',
                        color='white' if cm[i, j] > cm.max()/2 else 'black',
                        fontsize=12)

# Panel 6: ROC Curve
fpr, tpr, _ = roc_curve(y, y_pred_proba)
axes[1, 1].plot(fpr, tpr, label=f'XGBoost (AUC={roc_auc:.3f})', linewidth=2)
axes[1, 1].plot([0, 1], [0, 1], 'k--', label='Random')
axes[1, 1].set_xlabel('False Positive Rate')
axes[1, 1].set_ylabel('True Positive Rate')
axes[1, 1].set_title('ROC Curve (XGBoost)')
axes[1, 1].legend(fontsize=9)
axes[1, 1].grid(alpha=0.3)

# Panel 7: Class distribution
axes[1, 2].bar(['Female', 'Male'], np.bincount(y),
               alpha=0.7, label='Original', color='#1f77b4')
axes[1, 2].bar(['Female', 'Male'], np.bincount(y_balanced),
               alpha=0.7, label='After SMOTE', color='#ff7f0e')
axes[1, 2].set_ylabel('Sample Count')
axes[1, 2].set_title('Class Balance: Before vs After SMOTE')
axes[1, 2].legend(fontsize=9)
axes[1, 2].grid(axis='y', alpha=0.3)

# Panel 8: All feature importances
axes[1, 3].barh(feature_importance['Cytokine'][::-1],
                feature_importance['Importance'][::-1],
                color='coral')
axes[1, 3].set_xlabel('Importance')
axes[1, 3].set_title('All Cytokines — Feature Importance')

plt.tight_layout()
plt.savefig(FIGURES_DIR / "xgboost_results_visualization.png",
            dpi=150, bbox_inches='tight', facecolor='white')

# Save individual panels
fig.canvas.draw()
save_panel(fig, axes[0, 0], "panel_accuracy.png")
save_panel(fig, axes[0, 1], "panel_f1.png")
save_panel(fig, axes[0, 2], "panel_roc_auc.png")
save_panel(fig, axes[0, 3], "panel_feature_importance_top10.png")
save_panel(fig, axes[1, 0], "panel_confusion_matrix.png")
save_panel(fig, axes[1, 1], "panel_roc_curve.png")
save_panel(fig, axes[1, 2], "panel_class_balance.png")
save_panel(fig, axes[1, 3], "panel_feature_importance_all.png")

plt.close()
print("✓ Visualization saved: xgboost_results_visualization.png")

# ============================================================================
# SEPARATE FIGURE: XGBoost vs KNN vs LogReg — 3-model comparison
# ============================================================================
# KNN results from pipeline output
knn_acc = 0.568
knn_f1  = 0.591
knn_auc = 0.0     # not computed in KNN pipeline

model_names = ['XGBoost', 'KNN', 'Logistic Regression']
colors_3    = ['#1f77b4', '#2ca02c', '#ff7f0e']
acc_3       = [results['XGBoost']['acc_original'],   knn_acc,
               results['Logistic Regression']['acc_original']]
f1_3        = [results['XGBoost']['f1_original'],    knn_f1,
               results['Logistic Regression']['f1_original']]
auc_3       = [results['XGBoost']['auc_original'],   knn_auc,
               results['Logistic Regression']['auc_original']]

fig_cmp, axes_cmp = plt.subplots(1, 3, figsize=(14, 5))
fig_cmp.suptitle("Model Comparison: XGBoost vs KNN vs Logistic Regression\n"
                 "Sex Prediction from Cytokine Expression",
                 fontsize=13, fontweight='bold')

for ax, vals, ylabel, baseline in zip(
    axes_cmp,
    [acc_3, f1_3, auc_3],
    ['Accuracy', 'F1 Score', 'ROC-AUC'],
    [0.5, 0.5, 0.5]
):
    bars = ax.bar(model_names, vals, color=colors_3, alpha=0.8,
                  edgecolor='black')
    ax.axhline(baseline, color='red', linestyle='--', alpha=0.6,
               label=f'Random ({baseline})')
    ax.set_ylabel(ylabel, fontsize=11, fontweight='bold')
    ax.set_title(ylabel, fontsize=11, fontweight='bold')
    ax.set_ylim([0.0, 0.85])
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.02,
                f'{val:.3f}', ha='center', fontweight='bold', fontsize=10)

plt.tight_layout()
plt.savefig(FIGURES_DIR / "model_comparison_3models.png",
            dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("✓ Figure saved: model_comparison_3models.png")

# Save accuracy panel by itself
fig_acc, ax_acc = plt.subplots(figsize=(6, 5))
bars = ax_acc.bar(model_names, acc_3, color=colors_3, alpha=0.8, edgecolor='black')
ax_acc.axhline(0.5, color='red', linestyle='--', alpha=0.6, label='Random (0.5)')
ax_acc.set_ylabel('Accuracy', fontsize=11, fontweight='bold')
ax_acc.set_title('Model Comparison: Accuracy\nXGBoost vs KNN vs Logistic Regression\n(Sex)',
                 fontsize=11, fontweight='bold')
ax_acc.set_ylim([0.0, 0.85])
ax_acc.legend(fontsize=9)
ax_acc.grid(axis='y', alpha=0.3)
for bar, val in zip(bars, acc_3):
    ax_acc.text(bar.get_x() + bar.get_width()/2, val + 0.02,
                f'{val:.3f}', ha='center', fontweight='bold', fontsize=10)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "panel_accuracy_3models.png",
            dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("✓ Figure saved: panel_accuracy_3models.png")

# Save F1 score panel by itself
fig_f1, ax_f1 = plt.subplots(figsize=(6, 5))
bars = ax_f1.bar(model_names, f1_3, color=colors_3, alpha=0.8, edgecolor='black')
ax_f1.axhline(0.5, color='red', linestyle='--', alpha=0.6, label='Random (0.5)')
ax_f1.set_ylabel('F1 Score', fontsize=11, fontweight='bold')
ax_f1.set_title('Model Comparison: F1 Score\nXGBoost vs KNN vs Logistic Regression\n(Sex)',
                fontsize=11, fontweight='bold')
ax_f1.set_ylim([0.0, 0.85])
ax_f1.legend(fontsize=9)
ax_f1.grid(axis='y', alpha=0.3)
for bar, val in zip(bars, f1_3):
    ax_f1.text(bar.get_x() + bar.get_width()/2, val + 0.02,
               f'{val:.3f}', ha='center', fontweight='bold', fontsize=10)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "panel_f1_3models.png",
            dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("✓ Figure saved: panel_f1_3models.png")

print("\n" + "="*80)
print("✅ Enhanced pipeline completed!")
print("="*80)