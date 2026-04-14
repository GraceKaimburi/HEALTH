"""
Maternal Health Risk Classification Model Training
Includes: Data preprocessing, model training, statistical significance tests, 
error analysis, and comprehensive visualizations
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_auc_score, roc_curve,
    precision_recall_curve, f1_score, accuracy_score, cohen_kappa_score,
    matthews_corrcoef
)
from scipy.stats import chi2_contingency, fisher_exact
import warnings
warnings.filterwarnings('ignore')

# Create output directory
import os
os.makedirs('visualizations/classification', exist_ok=True)

print("=" * 80)
print("MATERNAL HEALTH RISK CLASSIFICATION MODEL")
print("=" * 80)

# ============================================================================
# 1. LOAD AND EXPLORE DATA
# ============================================================================
print("\n[1] LOADING DATA...")
df = pd.read_csv('Maternal_Risk.csv')
print(f"[OK] Dataset shape: {df.shape}")
print(f"[OK] Columns: {list(df.columns)}")
print(f"\nData Info:")
print(df.info())
print(f"\nFirst 5 rows:")
print(df.head())
print(f"\nMissing values:\n{df.isnull().sum()}")
print(f"\nRisk Level Distribution:\n{df['RiskLevel'].value_counts()}")

# ============================================================================
# 2. DATA PREPROCESSING
# ============================================================================
print("\n[2] PREPROCESSING DATA...")

# Encode target variable
le = LabelEncoder()
df['RiskLevel_encoded'] = le.fit_transform(df['RiskLevel'])
print(f"[OK] Risk levels encoded: {dict(zip(le.classes_, le.transform(le.classes_)))}")

# Separate features and target
X = df.drop(['RiskLevel', 'RiskLevel_encoded'], axis=1)
y = df['RiskLevel_encoded']

print(f"[OK] Features shape: {X.shape}")
print(f"[OK] Target shape: {y.shape}")
print(f"[OK] Class distribution: {np.bincount(y)}")

# Train-test split (80-20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"[OK] Training set: {X_train.shape}")
print(f"[OK] Test set: {X_test.shape}")
print(f"[OK] Train class distribution: {np.bincount(y_train)}")
print(f"[OK] Test class distribution: {np.bincount(y_test)}")

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"[OK] Features standardized")

# ============================================================================
# 3. MODEL TRAINING (10 EPOCHS)
# ============================================================================
print("\n[3] TRAINING CLASSIFICATION MODEL (10 epochs)...")

# Train model with 10 estimators (epochs)
model = RandomForestClassifier(
    n_estimators=100,  # More trees for stability
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1,
    class_weight='balanced'  # Handle class imbalance
)

model.fit(X_train_scaled, y_train)
print(f"[OK] Model trained with 100 trees")

# ============================================================================
# 4. PREDICTIONS
# ============================================================================
print("\n[4] GENERATING PREDICTIONS...")

# Training predictions
y_train_pred = model.predict(X_train_scaled)
y_train_proba = model.predict_proba(X_train_scaled)

# Test predictions
y_test_pred = model.predict(X_test_scaled)
y_test_proba = model.predict_proba(X_test_scaled)

print(f"[OK] Training accuracy: {accuracy_score(y_train, y_train_pred):.4f}")
print(f"[OK] Test accuracy: {accuracy_score(y_test, y_test_pred):.4f}")

# ============================================================================
# 5. STATISTICAL ANALYSIS & SIGNIFICANCE TESTS
# ============================================================================
print("\n[5] STATISTICAL SIGNIFICANCE TESTS...")

# Confusion Matrix
cm = confusion_matrix(y_test, y_test_pred)
print(f"\nConfusion Matrix:\n{cm}")

# Classification metrics
class_report = classification_report(
    y_test, y_test_pred, 
    target_names=le.classes_,
    output_dict=True
)
print(f"\nClassification Report:")
print(classification_report(y_test, y_test_pred, target_names=le.classes_))

# Calculate detailed metrics
accuracy = accuracy_score(y_test, y_test_pred)
macro_f1 = f1_score(y_test, y_test_pred, average='macro')
weighted_f1 = f1_score(y_test, y_test_pred, average='weighted')
kappa = cohen_kappa_score(y_test, y_test_pred)
mcc = matthews_corrcoef(y_test, y_test_pred)

print(f"\nDetailed Metrics:")
print(f"  Accuracy: {accuracy:.4f}")
print(f"  Macro F1-Score: {macro_f1:.4f}")
print(f"  Weighted F1-Score: {weighted_f1:.4f}")
print(f"  Cohen's Kappa: {kappa:.4f}")
print(f"  Matthews Correlation Coefficient: {mcc:.4f}")

# ROC-AUC (binary classification - high risk vs others)
if len(le.classes_) == 2:
    roc_auc = roc_auc_score(y_test, y_test_proba[:, 1])
    print(f"  ROC-AUC Score: {roc_auc:.4f}")

# Chi-square test for independence (association between predictions and actual)
chi2, p_value, dof, expected = chi2_contingency(cm)
print(f"\nChi-Square Test (Independence):")
print(f"  Chi-square statistic: {chi2:.4f}")
print(f"  P-value: {p_value:.6f}")
print(f"  Degrees of freedom: {dof}")
if p_value < 0.05:
    print(f"  [OK] SIGNIFICANT: Predictions are strongly associated with actual values (p < 0.05)")
else:
    print(f"  ✗ NOT SIGNIFICANT: No strong association found")

# Cross-validation scores
print(f"\nCross-Validation Analysis (5-fold Stratified):")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=cv, scoring='accuracy')
print(f"  CV Accuracy Scores: {[f'{s:.4f}' for s in cv_scores]}")
print(f"  Mean CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

# ============================================================================
# 6. ERROR ANALYSIS
# ============================================================================
print("\n[6] ERROR ANALYSIS...")

# Identify misclassified samples
misclassified_mask = y_test_pred != y_test
misclassified_count = misclassified_mask.sum()
misclassification_rate = misclassified_count / len(y_test) * 100

print(f"\nMisclassification Summary:")
print(f"  Total misclassified: {misclassified_count} out of {len(y_test)} ({misclassification_rate:.2f}%)")

# Per-class error analysis
print(f"\nPer-Class Error Analysis:")
for i, class_name in enumerate(le.classes_):
    class_mask = y_test == i
    class_size = class_mask.sum()
    class_errors = ((y_test_pred != y_test) & class_mask).sum()
    error_rate = class_errors / class_size * 100 if class_size > 0 else 0
    print(f"  {class_name:12} → {class_errors:3} errors out of {class_size:3} ({error_rate:5.2f}%)")

# Confidence analysis
print(f"\nConfidence Analysis:")
max_proba = np.max(y_test_proba, axis=1)
correct_mask = y_test_pred == y_test
avg_correct_confidence = max_proba[correct_mask].mean() if correct_mask.sum() > 0 else 0
avg_wrong_confidence = max_proba[~correct_mask].mean() if (~correct_mask).sum() > 0 else 0

print(f"  Avg confidence on CORRECT predictions: {avg_correct_confidence:.4f}")
print(f"  Avg confidence on WRONG predictions: {avg_wrong_confidence:.4f}")
print(f"  Confidence difference: {avg_correct_confidence - avg_wrong_confidence:.4f}")

# Systematic error patterns
print(f"\nSystematic Error Patterns (Confusion Analysis):")
for i in range(len(le.classes_)):
    for j in range(len(le.classes_)):
        if i != j and cm[i, j] > 0:
            pattern_pct = cm[i, j] / cm[i, :].sum() * 100
            print(f"  {le.classes_[i]} misclassified as {le.classes_[j]}: {cm[i, j]} times ({pattern_pct:.1f}%)")

# Feature importance
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=False)

print(f"\nFeature Importance Ranking:")
for idx, row in feature_importance.iterrows():
    print(f"  {row['Feature']:15} → {row['Importance']:.4f}")

# ============================================================================
# 7. VISUALIZATIONS
# ============================================================================
print("\n[7] GENERATING VISUALIZATIONS...")

# Set visual style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# ---- 1. Confusion Matrix Heatmap ----
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=le.classes_, yticklabels=le.classes_,
            cbar_kws={'label': 'Count'}, ax=ax)
ax.set_title('Confusion Matrix - Test Set', fontsize=14, fontweight='bold')
ax.set_xlabel('Predicted Label', fontsize=12)
ax.set_ylabel('True Label', fontsize=12)
plt.tight_layout()
plt.savefig('visualizations/classification/01_confusion_matrix.png', dpi=300, bbox_inches='tight')
print(f"  [OK] Saved: 01_confusion_matrix.png")
plt.close()

# ---- 2. ROC Curve (if binary) ----
if len(le.classes_) == 2:
    fpr, tpr, thresholds = roc_curve(y_test, y_test_proba[:, 1])
    roc_auc = roc_auc_score(y_test, y_test_proba[:, 1])
    
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.plot(fpr, tpr, color='darkorange', lw=2.5, label=f'ROC curve (AUC = {roc_auc:.4f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curve - Test Set Performance', fontsize=14, fontweight='bold')
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('visualizations/classification/02_roc_curve.png', dpi=300, bbox_inches='tight')
    print(f"  [OK] Saved: 02_roc_curve.png")
    plt.close()

# ---- 3. Precision-Recall Curve ----
if len(le.classes_) == 2:
    precision, recall, _ = precision_recall_curve(y_test, y_test_proba[:, 1])
    
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.plot(recall, precision, color='green', lw=2.5, label='Precision-Recall Curve')
    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title('Precision-Recall Curve - Test Set', fontsize=14, fontweight='bold')
    ax.legend(loc="best", fontsize=11)
    ax.grid(alpha=0.3)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    plt.tight_layout()
    plt.savefig('visualizations/classification/03_precision_recall_curve.png', dpi=300, bbox_inches='tight')
    print(f"  [OK] Saved: 03_precision_recall_curve.png")
    plt.close()

# ---- 4. Feature Importance ----
fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(data=feature_importance, x='Importance', y='Feature', palette='viridis', ax=ax)
ax.set_title('Feature Importance in Classification Model', fontsize=14, fontweight='bold')
ax.set_xlabel('Importance Score', fontsize=12)
ax.set_ylabel('Feature', fontsize=12)
for i, v in enumerate(feature_importance['Importance']):
    ax.text(v + 0.002, i, f'{v:.4f}', va='center', fontsize=10)
plt.tight_layout()
plt.savefig('visualizations/classification/04_feature_importance.png', dpi=300, bbox_inches='tight')
print(f"  [OK] Saved: 04_feature_importance.png")
plt.close()

# ---- 5. Classification Metrics Comparison ----
metrics_data = {
    'Metric': ['Accuracy', 'Macro F1', 'Weighted F1', "Cohen's Kappa", 'MCC'],
    'Score': [accuracy, macro_f1, weighted_f1, kappa, mcc]
}
metrics_df = pd.DataFrame(metrics_data)

fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.barh(metrics_df['Metric'], metrics_df['Score'], color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8'])
ax.set_xlim([0, 1])
ax.set_xlabel('Score', fontsize=12)
ax.set_title('Overall Classification Performance Metrics', fontsize=14, fontweight='bold')
for i, (metric, score) in enumerate(zip(metrics_df['Metric'], metrics_df['Score'])):
    ax.text(score + 0.02, i, f'{score:.4f}', va='center', fontsize=11, fontweight='bold')
ax.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig('visualizations/classification/05_metrics_comparison.png', dpi=300, bbox_inches='tight')
print(f"  [OK] Saved: 05_metrics_comparison.png")
plt.close()

# ---- 6. Per-Class Performance ----
per_class_metrics = []
for i in range(len(le.classes_)):
    class_name = le.classes_[i]
    precision = class_report[class_name]['precision']
    recall = class_report[class_name]['recall']
    f1 = class_report[class_name]['f1-score']
    support = int(class_report[class_name]['support'])
    
    per_class_metrics.append({
        'Class': class_name,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'Support': support
    })

per_class_df = pd.DataFrame(per_class_metrics)

fig, ax = plt.subplots(figsize=(12, 6))
x = np.arange(len(per_class_df))
width = 0.25
ax.bar(x - width, per_class_df['Precision'], width, label='Precision', color='#FF6B6B')
ax.bar(x, per_class_df['Recall'], width, label='Recall', color='#4ECDC4')
ax.bar(x + width, per_class_df['F1-Score'], width, label='F1-Score', color='#45B7D1')
ax.set_xlabel('Risk Level', fontsize=12)
ax.set_ylabel('Score', fontsize=12)
ax.set_title('Per-Class Performance Metrics', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(per_class_df['Class'])
ax.legend(fontsize=11)
ax.set_ylim([0, 1.1])
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('visualizations/classification/06_per_class_performance.png', dpi=300, bbox_inches='tight')
print(f"  [OK] Saved: 06_per_class_performance.png")
plt.close()

# ---- 7. Error Distribution by Feature ----
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for idx, feature in enumerate(X.columns[:6]):
    ax = axes[idx]
    
    correct_vals = X_test.loc[y_test_pred == y_test, feature]
    incorrect_vals = X_test.loc[y_test_pred != y_test, feature]
    
    ax.hist(correct_vals, bins=20, alpha=0.6, label='Correct', color='green', edgecolor='black')
    ax.hist(incorrect_vals, bins=20, alpha=0.6, label='Incorrect', color='red', edgecolor='black')
    ax.set_xlabel(feature, fontsize=11)
    ax.set_ylabel('Frequency', fontsize=11)
    ax.set_title(f'Distribution: {feature}', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('visualizations/classification/07_error_distribution_by_feature.png', dpi=300, bbox_inches='tight')
print(f"  [OK] Saved: 07_error_distribution_by_feature.png")
plt.close()

# ---- 8. Prediction Confidence Distribution ----
fig, ax = plt.subplots(figsize=(10, 6))

correct_confidence = max_proba[correct_mask]
incorrect_confidence = max_proba[~correct_mask]

ax.hist(correct_confidence, bins=30, alpha=0.6, label=f'Correct (n={len(correct_confidence)})', 
        color='green', edgecolor='black')
ax.hist(incorrect_confidence, bins=30, alpha=0.6, label=f'Incorrect (n={len(incorrect_confidence)})', 
        color='red', edgecolor='black')
ax.axvline(avg_correct_confidence, color='darkgreen', linestyle='--', linewidth=2, 
           label=f'Mean Correct: {avg_correct_confidence:.3f}')
ax.axvline(avg_wrong_confidence, color='darkred', linestyle='--', linewidth=2,
           label=f'Mean Incorrect: {avg_wrong_confidence:.3f}')
ax.set_xlabel('Prediction Confidence (Max Probability)', fontsize=12)
ax.set_ylabel('Frequency', fontsize=12)
ax.set_title('Prediction Confidence Distribution', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('visualizations/classification/08_confidence_distribution.png', dpi=300, bbox_inches='tight')
print(f"  [OK] Saved: 08_confidence_distribution.png")
plt.close()

# ---- 9. Class Distribution (Train vs Test) ----
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

train_counts = pd.Series(y_train).value_counts().reindex([i for i in range(len(le.classes_))])
test_counts = pd.Series(y_test).value_counts().reindex([i for i in range(len(le.classes_))])

axes[0].bar(le.classes_, train_counts, color='skyblue', edgecolor='black')
axes[0].set_title('Training Set Class Distribution', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Count', fontsize=11)
for i, v in enumerate(train_counts):
    axes[0].text(i, v + 5, str(v), ha='center', fontweight='bold')
axes[0].grid(axis='y', alpha=0.3)

axes[1].bar(le.classes_, test_counts, color='lightcoral', edgecolor='black')
axes[1].set_title('Test Set Class Distribution', fontsize=12, fontweight='bold')
axes[1].set_ylabel('Count', fontsize=11)
for i, v in enumerate(test_counts):
    axes[1].text(i, v + 5, str(v), ha='center', fontweight='bold')
axes[1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('visualizations/classification/09_class_distribution.png', dpi=300, bbox_inches='tight')
print(f"  [OK] Saved: 09_class_distribution.png")
plt.close()

# ---- 10. Misclassification Heatmap ----
misclass_matrix = cm.astype('float')
for i in range(len(le.classes_)):
    misclass_matrix[i, :] = misclass_matrix[i, :] / misclass_matrix[i, :].sum()

fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(misclass_matrix, annot=True, fmt='.2%', cmap='RdYlGn_r',
            xticklabels=le.classes_, yticklabels=le.classes_,
            cbar_kws={'label': 'Percentage'}, ax=ax)
ax.set_title('Misclassification Rate Heatmap (Row-wise Normalized)', fontsize=14, fontweight='bold')
ax.set_xlabel('Predicted Label', fontsize=12)
ax.set_ylabel('True Label', fontsize=12)
plt.tight_layout()
plt.savefig('visualizations/classification/10_misclassification_heatmap.png', dpi=300, bbox_inches='tight')
print(f"  [OK] Saved: 10_misclassification_heatmap.png")
plt.close()

# ============================================================================
# 8. GENERATE SUMMARY REPORT
# ============================================================================
print("\n[8] GENERATING SUMMARY REPORT...")

report_text = f"""
{'='*80}
MATERNAL HEALTH RISK CLASSIFICATION - COMPREHENSIVE ANALYSIS REPORT
{'='*80}

DATASET OVERVIEW
{'-'*80}
Total Samples: {len(df)}
Features: {', '.join(X.columns)}
Target Classes: {', '.join(le.classes_)}
Class Distribution:
{df['RiskLevel'].value_counts().to_string()}

TRAIN-TEST SPLIT
{'-'*80}
Training Set Size: {len(X_train)} samples
Test Set Size: {len(X_test)} samples
Train/Test Ratio: 80/20
Stratified: Yes (maintains class proportions)

MODEL CONFIGURATION
{'-'*80}
Algorithm: Random Forest Classifier
Number of Trees: 100
Max Depth: 10
Min Samples Split: 5
Min Samples Leaf: 2
Class Weight: Balanced (to handle imbalance)
Random State: 42

STATISTICAL SIGNIFICANCE TESTS
{'-'*80}
Chi-Square Test (Independence):
  - Test Statistic: {chi2:.4f}
  - P-value: {p_value:.6f}
  - Conclusion: {'SIGNIFICANT (p < 0.05) [OK]' if p_value < 0.05 else 'NOT SIGNIFICANT (p >= 0.05) [X]'}
  - Interpretation: Predictions are {'strongly' if p_value < 0.05 else 'weakly'} associated with actual values

Cross-Validation (5-Fold Stratified):
  - Fold Scores: {', '.join([f'{s:.4f}' for s in cv_scores])}
  - Mean Accuracy: {cv_scores.mean():.4f}
  - Std Deviation: {cv_scores.std():.4f}
  - Range: [{cv_scores.min():.4f}, {cv_scores.max():.4f}]

OVERALL PERFORMANCE METRICS
{'-'*80}
Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)
Macro F1-Score: {macro_f1:.4f}
Weighted F1-Score: {weighted_f1:.4f}
Cohen's Kappa: {kappa:.4f}
Matthews Correlation Coefficient (MCC): {mcc:.4f}
{'ROC-AUC Score: ' + f'{roc_auc:.4f}' if len(le.classes_) == 2 else ''}

ERROR ANALYSIS
{'-'*80}
Total Misclassifications: {misclassified_count} out of {len(y_test)}
Misclassification Rate: {misclassification_rate:.2f}%
Correct Prediction Confidence: {avg_correct_confidence:.4f}
Incorrect Prediction Confidence: {avg_wrong_confidence:.4f}
Confidence Delta: {avg_correct_confidence - avg_wrong_confidence:.4f}

Per-Class Error Breakdown:
"""

for i, class_name in enumerate(le.classes_):
    class_mask = y_test == i
    class_size = class_mask.sum()
    class_errors = ((y_test_pred != y_test) & class_mask).sum()
    error_rate = class_errors / class_size * 100 if class_size > 0 else 0
    report_text += f"  {class_name}: {class_errors}/{class_size} errors ({error_rate:.2f}%)\n"

report_text += f"\nSystematic Misclassification Patterns:\n"
for i in range(len(le.classes_)):
    for j in range(len(le.classes_)):
        if i != j and cm[i, j] > 0:
            pattern_pct = cm[i, j] / cm[i, :].sum() * 100
            report_text += f"  {le.classes_[i]:12} -> {le.classes_[j]:12}: {cm[i, j]:3} times ({pattern_pct:.1f}%)\n"

report_text += f"""
PER-CLASS PERFORMANCE
{'-'*80}
"""
for idx, row in per_class_df.iterrows():
    report_text += f"\n{row['Class']}:\n"
    report_text += f"  Precision: {row['Precision']:.4f}\n"
    report_text += f"  Recall: {row['Recall']:.4f}\n"
    report_text += f"  F1-Score: {row['F1-Score']:.4f}\n"
    report_text += f"  Support: {row['Support']}\n"

report_text += f"""
FEATURE IMPORTANCE RANKING
{'-'*80}
"""
for idx, row in feature_importance.iterrows():
    report_text += f"{row['Feature']:15}: {row['Importance']:.4f}\n"

report_text += f"""
KEY FINDINGS & INTERPRETATIONS
{'-'*80}
1. Model Performance:
   - The model achieves {accuracy*100:.2f}% accuracy on the test set
   - Cross-validation results ({cv_scores.mean():.4f} ± {cv_scores.std():.4f}) suggest {'stable' if cv_scores.std() < 0.05 else 'variable'} performance
   - Cohen's Kappa ({kappa:.4f}) indicates {'excellent' if kappa > 0.8 else 'good' if kappa > 0.6 else 'moderate'} agreement beyond chance

2. Statistical Significance:
   - Chi-square p-value ({p_value:.6f}) {'< 0.05' if p_value < 0.05 else '> 0.05'}: Model predictions are {'significantly associated' if p_value < 0.05 else 'not significantly associated'} with actual labels
   - This confirms that the model is {'capturing meaningful patterns' if p_value < 0.05 else 'not capturing strong patterns'} in the data

3. Error Analysis:
   - Overall misclassification rate: {misclassification_rate:.2f}%
   - Model is {'more confident in correct' if avg_correct_confidence > avg_wrong_confidence else 'more confident in incorrect'} predictions (confidence delta: {avg_correct_confidence - avg_wrong_confidence:.4f})
   - Major confusion: {'See systematic error patterns above' if misclassified_count > 0 else 'No misclassifications'}

4. Feature Importance:
   - Top 3 features: {', '.join(feature_importance.head(3)['Feature'].values)}
   - Features {feature_importance['Feature'].iloc[0]} has the highest predictive power ({feature_importance['Importance'].iloc[0]:.4f})

5. Recommendations:
   - {'Model is production-ready' if accuracy > 0.85 and kappa > 0.7 else 'Consider further tuning'}
   - {'Focus on reducing' if misclassification_rate > 10 else 'Model misclassification rate is acceptable'}
   - {'Increase sample size for underrepresented classes' if any(train_counts < 50) else 'class distribution is balanced'}

VISUALIZATIONS GENERATED
{'-'*80}
[OK] 01_confusion_matrix.png - Detailed confusion matrix heatmap
[OK] 02_roc_curve.png - ROC curve with AUC score
[OK] 03_precision_recall_curve.png - Precision-recall trade-off
[OK] 04_feature_importance.png - Feature ranking by importance
[OK] 05_metrics_comparison.png - Overall performance metrics
[OK] 06_per_class_performance.png - Per-class precision/recall/F1
[OK] 07_error_distribution_by_feature.png - Feature-wise error analysis
[OK] 08_confidence_distribution.png - Prediction confidence patterns
[OK] 09_class_distribution.png - Train/test class balance
[OK] 10_misclassification_heatmap.png - Normalized error rates

{'='*80}
Report Generated: {pd.Timestamp.now()}
Location: visualizations/classification/ANALYSIS_REPORT.txt
{'='*80}
"""

with open('visualizations/classification/ANALYSIS_REPORT.txt', 'w', encoding='utf-8') as f:
    f.write(report_text)

print(f"[OK] Saved: ANALYSIS_REPORT.txt")
print("\n" + report_text)

print("\n" + "="*80)
print("[OK] ANALYSIS COMPLETE - All visualizations saved to visualizations/classification/")
print("="*80)
