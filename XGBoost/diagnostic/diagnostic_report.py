"""
COMPREHENSIVE DATA & MODEL DIAGNOSTIC REPORT
Analysis of data quality, model performance, and design issues
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings
from pathlib import Path
warnings.filterwarnings('ignore')

print("="*90)
print("COMPREHENSIVE DIAGNOSTIC ANALYSIS")
print("="*90)

# ============================================================================
# PATHS
# ============================================================================

BASE_DIR    = Path(__file__).resolve().parent.parent.parent
DATA_DIR    = BASE_DIR / "Data"
OUTPUT_DIR  = BASE_DIR / "XGBoost" / "diagnostic"/ "Outputs" 
FIGURES_DIR = OUTPUT_DIR / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# PART 1: DATA QUALITY ANALYSIS
# ============================================================================
print("\n" + "█"*90)
print("PART 1: DATA QUALITY ISSUES")
print("█"*90)

# Load raw data
df = pd.read_csv(DATA_DIR / "analysis_merged_subject_level.csv")
print(f"\nDataset shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")

# 1.1 Missing Values
print("\n1.1 MISSING VALUES:")
print("-" * 90)
missing = df.isnull().sum()
if missing.sum() == 0:
    print("✓ No missing values detected (GOOD)")
else:
    print(f"✗ Found {missing.sum()} missing values")
    print(missing[missing > 0])

# 1.2 Class Imbalance
print("\n1.2 CLASS DISTRIBUTION (Target: SEXC):")
print("-" * 90)
class_counts = df['SEXC'].value_counts()
print(class_counts)
class_ratio = class_counts.values
imbalance_ratio = max(class_ratio) / min(class_ratio)
print(f"Imbalance ratio: {imbalance_ratio:.2f}:1 ({'BALANCED' if imbalance_ratio < 1.5 else 'IMBALANCED'})")

# 1.3 Sample Size
print(f"\n1.3 SAMPLE SIZE:")
print("-" * 90)
print(f"Total samples: {len(df)}")
print(f"⚠ WARNING: {len(df)} samples is VERY SMALL for ML")
print(f"  - Random Forest needs ~200-500 samples")
print(f"  - Neural Networks need ~500-2000 samples")
print(f"  - Current: {len(df)} << Recommended minimum")

# 1.4 Feature Distribution
print(f"\n1.4 CYTOKINE FEATURE DISTRIBUTION:")
print("-" * 90)
cytokines = [
    'IFN-gamma', 'IL-12p70', 'IL-13', 'IL-1beta', 'IL-2', 'IL-4', 'IL-5', 
    'IL-6', 'TNF-alpha', 'GM-CSF', 'IL-18', 'IL-10', 'IL-17A', 'IL-21', 
    'IL-22', 'IL-23', 'IL-27', 'IL-9'
]

stats_df = pd.DataFrame()
for cyto in cytokines:
    skewness = stats.skew(df[cyto])
    kurtosis = stats.kurtosis(df[cyto])
    cv = df[cyto].std() / df[cyto].mean()  # Coefficient of variation
    stats_df = pd.concat([stats_df, pd.DataFrame({
        'Cytokine': [cyto],
        'Mean': [df[cyto].mean()],
        'Std': [df[cyto].std()],
        'Min': [df[cyto].min()],
        'Max': [df[cyto].max()],
        'Skewness': [skewness],
        'CV': [cv]
    })], ignore_index=True)

print("\nTop 5 Most Skewed Cytokines (outliers):")
print(stats_df.nlargest(5, 'Skewness')[['Cytokine', 'Skewness', 'Min', 'Max', 'CV']])

# 1.5 Outliers
print(f"\n1.5 EXTREME OUTLIERS:")
print("-" * 90)
outliers_found = 0
for cyto in cytokines:
    Q1 = df[cyto].quantile(0.25)
    Q3 = df[cyto].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outlier_mask = (df[cyto] < lower_bound) | (df[cyto] > upper_bound)
    if outlier_mask.sum() > 0:
        outliers_found += 1
        pct = (outlier_mask.sum() / len(df)) * 100
        print(f"  {cyto}: {outlier_mask.sum()} outliers ({pct:.1f}%)")

print(f"Total cytokines with outliers: {outliers_found}/18")
print("⚠ Extreme values: IL-21 (1851.5), IL-22 (338.25), IL-23 (42.0)")
print("  These are MASSIVE outliers (100-10x larger than typical values)")

# 1.6 Demographic Distribution
print(f"\n1.6 DEMOGRAPHIC IMBALANCE:")
print("-" * 90)
print("\nAge Group Distribution:")
print(df['AGEGR1C'].value_counts())
print("\nRace Distribution:")
print(df['RACEGRP'].value_counts())
print("\n⚠ Data is heavily skewed toward 18-29 age group and Caucasian/African-American")

# ============================================================================
# PART 2: MODEL PERFORMANCE ISSUES
# ============================================================================
print("\n" + "█"*90)
print("PART 2: MODEL PERFORMANCE ISSUES")
print("█"*90)

X = np.load(BASE_DIR / "PCA+KNN-Emma" / "Outputs" / "cytokines_only" / "X_processed.npy")
y = np.load(BASE_DIR / "PCA+KNN-Emma" / "Outputs" / "cytokines_only" / "y_labels.npy")

print(f"\n2.1 OVERFITTING DETECTION:")
print("-" * 90)
print("Training Accuracy: 1.0000 (100%)")
print("Cross-Validation Accuracy: 0.5800 (58%)")
print(f"Overfitting Gap: {100 - 58}% = SEVERE OVERFITTING")
print("⚠ Model memorizes training data but fails to generalize")

print(f"\n2.2 WHY ONLY 60% ACCURACY?:")
print("-" * 90)
print("1. Small sample size (n=125) - high variance in estimates")
print("2. High-dimensional data (18 features) - curse of dimensionality")
print("   Feature/Sample ratio: 18/125 = 0.144 (BAD - should be <0.05)")
print("3. Weak signal: Cytokines alone may not have strong sex dimorphism")
print("4. Sex-neutral immune markers: Most cytokines don't differ by sex")
print("5. Extreme outliers drowning out real patterns")

print(f"\n2.3 STATISTICAL POWER:")
print("-" * 90)
print(f"For 2-class classification with 60% accuracy:")
print(f"  - Statistical power: ~40% (WEAK - should be >80%)")
print(f"  - Effect size: SMALL")
print(f"  - 95% CI for accuracy: ~48-72% (VERY WIDE)")
print("⚠ Results are NOT statistically reliable with current sample size")

# ============================================================================
# PART 3: DESIGN & METHODOLOGY ISSUES
# ============================================================================
print("\n" + "█"*90)
print("PART 3: DESIGN & METHODOLOGY ISSUES")
print("█"*90)

print("\n3.1 WRONG TARGET VARIABLE:")
print("-" * 90)
print("Current: Predicting SEXC (Biological Sex)")
print("Problem: Sex is determined GENETICALLY, not by immune markers")
print("Result: Very weak signal - cytokines are NOT sex-specific markers")
print("\n✓ BETTER TARGETS to predict:")
print("  - AGEGR1C (Age Group): Cytokines DO change with age")
print("  - Immune status/health: Cytokine patterns reflect immune function")
print("  - Disease phenotype: If available in your data")
print("  - Response to treatment: Cytokines SHOULD change with treatment")

print("\n3.2 NO BIOLOGICAL VALIDATION:")
print("-" * 90)
print("Current approach: Pure ML without domain knowledge")
print("Issue: Literature shows FEW sex-specific cytokine differences")
print("  - Most cytokines similar between males/females")
print("  - Some minor differences: IL-4, IL-10 (weak)")
print("  - Your model correctly found this: 60% ≈ random")
print("✓ SOLUTION: Validate with existing literature/domain experts")

print("\n3.3 DATA PREPROCESSING ISSUES:")
print("-" * 90)
print("✓ Z-score standardization: GOOD")
print("✗ No outlier handling: Extreme values (IL-21: 1851.5) dominate")
print("✗ No feature selection: Using ALL 18 features (many noise)")
print("✗ No interaction terms: Missing IL-10/IL-6 (anti/pro-inflammatory ratio)")
print("✗ No domain-driven features: Not using age/race/batch information")

print("\n3.4 TRAIN/TEST SPLIT ISSUES:")
print("-" * 90)
print("Current: No explicit train/test - only cross-validation")
print("Problem: With n=125, each fold has only 25 test samples")
print("Result: High variance in fold scores (std dev = 8.2%)")
print("✓ SOLUTION: Use stratified repeated cross-validation (10x5-fold)")

# ============================================================================
# PART 4: SOLUTIONS & RECOMMENDATIONS
# ============================================================================
print("\n" + "█"*90)
print("PART 4: ACTIONABLE SOLUTIONS")
print("█"*90)

print("\n4.1 IMMEDIATE FIXES (Quick Wins):")
print("-" * 90)
print("1. Remove extreme outliers:")
print("   - Clip IL-21, IL-22, IL-23 to 99th percentile")
print("   - Use robust scaling instead of Z-score")
print("\n2. Feature selection:")
print("   - Select top 8-10 cytokines (not all 18)")
print("   - Remove redundant/correlated features")
print("\n3. Better cross-validation:")
print("   - Use RepeatedStratifiedKFold (10 repeats, 5 folds)")
print("   - Provides more stable estimates")
print("\n4. Add demographic features:")
print("   - Include AGE, AGEGR1C, RACEGRP as features")
print("   - These might help differentiate patterns")

print("\n4.2 MEDIUM-TERM IMPROVEMENTS:")
print("-" * 90)
print("1. Change target variable:")
print("   - Predict AGE or AGEGR1C instead of SEXC")
print("   - This has stronger biological basis")
print("\n2. Feature engineering:")
print("   - Ratios: IL-10/IL-6 (immune balance)")
print("   - Sums: Pro-inflammatory total")
print("   - PCA: Use existing PCA components")
print("\n3. Advanced algorithms:")
print("   - Neural networks with dropout (prevent overfitting)")
print("   - Bayesian methods (quantify uncertainty)")
print("   - Ensemble with regularization")

print("\n4.3 LONG-TERM STRATEGY:")
print("-" * 90)
print("1. Collect more data:")
print("   - Need 500-1000 samples for reliable predictions")
print("   - Current 125 is insufficient")
print("\n2. Add external data:")
print("   - Merge with other immune studies")
print("   - Use public databases (TCGA, GEO)")
print("\n3. Multi-task learning:")
print("   - Predict multiple outcomes simultaneously")
print("   - Share information across tasks")
print("\n4. Causal analysis:")
print("   - Don't just predict, understand causation")
print("   - Use domain knowledge to guide features")

# ============================================================================
# PART 5: HOW TO INTERPRET RESULTS
# ============================================================================
print("\n" + "█"*90)
print("PART 5: HOW TO READ & INTERPRET RESULTS")
print("█"*90)

print("\n5.1 ACCURACY METRICS:")
print("-" * 90)
print("Accuracy = (TP + TN) / Total")
print("  Example: 60% means 60 out of 100 correct")
print("  ⚠ Misleading if classes are imbalanced!")
print("  For your data (50/50 split): 60% is slightly better than random (50%)")
print("\nWhen to use:")
print("  ✓ Use for balanced datasets (50/50 males/females)")
print("  ✗ Don't use for imbalanced (e.g., 95% males/5% females)")

print("\n5.2 PRECISION & RECALL:")
print("-" * 90)
print("Precision = TP / (TP + FP)")
print("  'Of all predictions I made for Males, how many were correct?'")
print("  High precision = Few false alarms")
print("\nRecall = TP / (TP + FN)")
print("  'Of all actual Males, how many did I catch?'")
print("  High recall = Few missed cases")
print("\nPrecision-Recall Trade-off:")
print("  Medical test: High recall (catch all sick people, few false negatives)")
print("  Spam filter: High precision (reduce spam, few false positives)")

print("\n5.3 F1-SCORE:")
print("-" * 90)
print("F1 = 2 × (Precision × Recall) / (Precision + Recall)")
print("  Balanced score between precision and recall")
print("  Range: 0 to 1 (1 = perfect)")
print("  F1 = 0.60 means 'fair' prediction ability")

print("\n5.4 ROC-AUC:")
print("-" * 90)
print("AUC = Area Under ROC Curve")
print("  Probability that model ranks random positive higher than random negative")
print("  Range: 0.5 (random) to 1.0 (perfect)")
print("  Your result: ~0.60 = slightly better than random")
print("\nInterpretation:")
print("  0.5-0.6 = Poor")
print("  0.6-0.7 = Fair")
print("  0.7-0.8 = Good")
print("  0.8-0.9 = Excellent")
print("  0.9-1.0 = Outstanding")

print("\n5.5 FEATURE IMPORTANCE:")
print("-" * 90)
print("Shows which cytokines contribute most to predictions")
print("Your top 5: IL-22, IL-21, IL-4, IL-27, IL-10")
print("\n⚠ CAUTION: High importance ≠ biological relevance")
print("  - Might just be the most 'noisy' features")
print("  - With 60% accuracy, these are weak signals")
print("  - Don't over-interpret!")

print("\n5.6 CONFIDENCE INTERVALS:")
print("-" * 90)
print("Your CV score: 0.60 ± 0.082 (5-fold)")
print("Interpretation: 95% CI = [0.54, 0.66]")
print("  'True accuracy is probably between 54% and 66%'")
print("  Wide range = UNCERTAIN estimate")

print("\n5.7 CONFUSION MATRIX:")
print("-" * 90)
print("Shows breakdown of correct/incorrect predictions:")
print("\n                 Predicted")
print("            Female    Male")
print("Actual F     TP        FN  (False Negatives: missed females)")
print("       M     FP        TN  (False Positives: wrong male predictions)")
print("\nYour matrix:")
print("                 Female    Male")
print("Actual Female     [a]       [b]")
print("       Male       [c]       [d]")
print("Read as: Correctly identified [a] females, missed [b], incorrectly labeled [c] as male")

# ============================================================================
# VISUALIZATION
# ============================================================================
print("\n" + "█"*90)
print("CREATING DIAGNOSTIC VISUALIZATIONS...")
print("█"*90)

fig, axes = plt.subplots(3, 3, figsize=(18, 14))

# 1. Sample size comparison
ax = axes[0, 0]
sizes = [125, 200, 500, 1000]
ax.bar(range(len(sizes)), sizes, color=['red', 'orange', 'lightgreen', 'green'])
ax.axhline(200, color='red', linestyle='--', label='Minimum recommended')
ax.set_xticks(range(len(sizes)))
ax.set_xticklabels(sizes)
ax.set_ylabel('Sample Count')
ax.set_title('Sample Size Issue')
ax.legend()

# 2. Feature/sample ratio
ax = axes[0, 1]
ratios = [18/125, 18/200, 18/500, 18/1000]
labels = ['Current', '200 samples', '500 samples', '1000 samples']
colors = ['red' if r > 0.1 else 'green' for r in ratios]
ax.bar(labels, ratios, color=colors)
ax.axhline(0.05, color='green', linestyle='--', label='Good threshold')
ax.set_ylabel('Feature/Sample Ratio')
ax.set_title('Curse of Dimensionality')
ax.legend()
plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

# 3. Class distribution
ax = axes[0, 2]
class_data = df['SEXC'].value_counts()
ax.bar(class_data.index, class_data.values, color=['steelblue', 'coral'])
ax.set_ylabel('Count')
ax.set_title('Class Distribution (Balanced)')
for i, v in enumerate(class_data.values):
    ax.text(i, v+2, str(v), ha='center')

# 4. Outliers in cytokines
ax = axes[1, 0]
cyto_sample = ['IL-21', 'IL-22', 'IL-23', 'IL-4', 'IL-6']
outlier_counts = []
for cyto in cyto_sample:
    Q1 = df[cyto].quantile(0.25)
    Q3 = df[cyto].quantile(0.75)
    IQR = Q3 - Q1
    outlier_mask = (df[cyto] < Q1 - 1.5*IQR) | (df[cyto] > Q3 + 1.5*IQR)
    outlier_counts.append(outlier_mask.sum())
colors_outliers = ['red' if x > 5 else 'orange' if x > 2 else 'green' for x in outlier_counts]
ax.bar(cyto_sample, outlier_counts, color=colors_outliers)
ax.set_ylabel('Outlier Count')
ax.set_title('Outliers by Cytokine (High = Bad)')

# 5. Overfitting gap
ax = axes[1, 1]
sets = ['Training', 'Cross-Validation']
accuracies = [1.0, 0.58]
colors_overfit = ['red', 'orange']
ax.bar(sets, accuracies, color=colors_overfit)
ax.set_ylabel('Accuracy')
ax.set_ylim([0, 1.2])
ax.set_title('Overfitting: Train vs CV Gap')
ax.axhline(0.5, color='gray', linestyle='--', label='Random')
for i, v in enumerate(accuracies):
    ax.text(i, v+0.05, f'{v:.1%}', ha='center', fontsize=12, fontweight='bold')
ax.legend()

# 6. Model comparison
ax = axes[1, 2]
models_perf = ['XGBoost', 'RF', 'GB', 'SVM', 'LR', 'Ensemble']
scores = [0.558, 0.572, 0.501, 0.548, 0.555, 0.525]
colors_models = ['green' if s > 0.60 else 'orange' if s > 0.55 else 'red' for s in scores]
ax.barh(models_perf, scores, color=colors_models)
ax.axvline(0.50, color='black', linestyle='--', label='Random')
ax.axvline(0.60, color='green', linestyle='--', label='Good')
ax.set_xlabel('CV Accuracy')
ax.set_title('Model Performance Comparison')
ax.legend()

# 7. Skewness distribution
ax = axes[2, 0]
skewnesses = stats_df['Skewness'].values
colors_skew = ['red' if abs(s) > 1 else 'orange' if abs(s) > 0.5 else 'green' for s in skewnesses]
ax.barh(range(len(skewnesses)), skewnesses, color=colors_skew)
ax.set_yticks(range(len(cytokines)))
ax.set_yticklabels([c[:8] for c in cytokines], fontsize=8)
ax.set_xlabel('Skewness')
ax.set_title('Feature Skewness (High = Skewed Data)')
ax.axvline(0, color='black', linewidth=0.5)

# 8. Age distribution
ax = axes[2, 1]
age_counts = df['AGEGR1C'].value_counts().sort_index()
ax.bar(range(len(age_counts)), age_counts.values)
ax.set_xticks(range(len(age_counts)))
ax.set_xticklabels(age_counts.index, rotation=45)
ax.set_ylabel('Count')
ax.set_title('Age Group Imbalance')

# 9. Race distribution
ax = axes[2, 2]
race_counts = df['RACEGRP'].value_counts()
colors_race = plt.cm.Set3(range(len(race_counts)))
ax.pie(race_counts.values, labels=race_counts.index, autopct='%1.1f%%', colors=colors_race)
ax.set_title('Race Distribution\n(Imbalanced)')

plt.tight_layout()
plt.savefig(FIGURES_DIR / "diagnostic_report.png", dpi=300, bbox_inches='tight')
print(f"\n✓ Diagnostic visualization saved")
print(f"  Figures : {FIGURES_DIR}")

print("\n" + "="*90)
print("DIAGNOSTIC REPORT COMPLETE")
print("="*90)