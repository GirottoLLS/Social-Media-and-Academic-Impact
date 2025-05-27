#%% Step 1: Imports and global settings

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['figure.dpi'] = 300
pd.set_option('display.max_columns', None)

#%% Step 2: Load and preprocess data

# Load dataset, drop unnecessary columns, and rescale hours
df = pd.read_excel('socialmedia.xlsx')
df.drop(columns=['Country'], errors='ignore', inplace=True)
df['SleepHours'] = df['Sleep_Hours_Per_Night'] / 10
df['DailyUsageHours'] = df['Avg_Daily_Usage_Hours'] / 10
df.drop(columns=['Sleep_Hours_Per_Night', 'Avg_Daily_Usage_Hours'], inplace=True)

# Quick data overview
print("--> Data Info:")
df.info()
print("\n--> Descriptive stats for numerical columns:")
df.select_dtypes(include=[np.number]).describe().T

#%% Step 4: Exploratory plots - categorical variables

categorical = ['Gender', 'Academic_Level', 'Most_Used_Platform', 'Relationship_Status', 'Affects_Academic_Performance']
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
for ax, col in zip(axes.flatten(), categorical):
    sns.countplot(x=col, data=df, ax=ax)
    ax.set_title(f"{col} Distribution")
    ax.tick_params(axis='x', rotation=45)
fig.delaxes(axes.flatten()[-1])
plt.tight_layout(); plt.show()

#%% Step 5: Exploratory plots - numeric variables

numeric = ['Age', 'DailyUsageHours', 'SleepHours', 'Mental_Health_Score', 'Conflicts_Over_Social_Media', 'Addicted_Score']
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
for ax, col in zip(axes.flatten(), numeric):
    sns.histplot(df[col], kde=True, bins=20, ax=ax)
    ax.set_title(f"{col} Distribution")
plt.tight_layout(); plt.show()

#%% Step 6: Platform impact on sleep and mental health

# Sleep hours as target variable for Favorite Platform
order_sleep = df.groupby('Most_Used_Platform')['SleepHours'].mean().sort_values().index
sns.boxplot(x='Most_Used_Platform', y='SleepHours', data=df, order=order_sleep)
plt.xticks(rotation=45); plt.title('Sleep Hours by Platform'); plt.show()

# Mental health score as target variable for Favorite Platform
order_mh = df.groupby('Most_Used_Platform')['Mental_Health_Score'].mean().sort_values().index
sns.boxplot(x='Most_Used_Platform', y='Mental_Health_Score', data=df, order=order_mh)
plt.xticks(rotation=45); plt.title('Mental Health by Platform'); plt.show()

#%% Step 7: Age vs addiction and mental health

# Addiction score as functon of Age
sns.stripplot(x='Age', y='Addicted_Score', data=df, jitter=0.2, alpha=0.5)
sns.pointplot(x='Age', y='Addicted_Score', data=df, color='red', capsize=0.1)
plt.title('Addiction Score by Age'); plt.show()

# Mental health score as functon of Age
sns.stripplot(x='Age', y='Mental_Health_Score', data=df, jitter=0.2, alpha=0.5)
sns.pointplot(x='Age', y='Mental_Health_Score', data=df, color='red', capsize=0.1)
plt.title('Mental Health by Age'); plt.show()

#%% Step 8: Relationship status impact

# Daily usage hours impact on relationship
sns.boxplot(x='Relationship_Status', y='DailyUsageHours', data=df)
plt.title('Usage Hours by Relationship Status'); plt.show()

# Mental health score impact on relationship
sns.boxplot(x='Relationship_Status', y='Mental_Health_Score', data=df)
plt.title('Mental Health by Relationship Status'); plt.show()

#%% Step 9: Correlation heatmap

plt.figure(figsize=(8, 6))
sns.heatmap(df[numeric].corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Matrix'); plt.show()

#%% Step 10: Prepare features and target

y = df['Affects_Academic_Performance'].map({'Yes': 1, 'No': 0})
X = df.drop(columns=['Affects_Academic_Performance', 'Student_ID'], errors='ignore')
X = pd.get_dummies(X, drop_first=True).astype(float)

#%% Step 11: Check multicollinearity with VIF

def compute_vif(df_input):
    vif_df = sm.add_constant(df_input)
    return pd.Series(
        [variance_inflation_factor(vif_df.values, i) for i in range(vif_df.shape[1])],
        index=vif_df.columns,
        name='VIF')

vif_scores = compute_vif(X)
print("\nVIF scores:\n", vif_scores)

# Multicollinearity heatmap for key features
key_feats = ['Addicted_Score', 'DailyUsageHours', 'Conflicts_Over_Social_Media',
             'Mental_Health_Score', 'Relationship_Status_In Relationship',
             'Relationship_Status_Single']
plt.figure(figsize=(7, 6))
sns.heatmap(X[key_feats].corr(), annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Correlation with Relationship Status'); plt.show()

#%% Step 12: Train-test split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

#%% Step 14: ROC/AUC plotting function
def plot_roc(y_true, y_score, label=None):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"{label} (AUC = {roc_auc:.3f})")
    return roc_auc

#%% Step 15: Baseline (null) and full logistic regression models

# Null logistic model
X0 = sm.add_constant(pd.DataFrame(np.zeros((len(y_train), 0)), index=y_train.index))
model_null = sm.Logit(y_train, X0).fit(disp=False)
print(f"Null Model AIC: {model_null.aic:.2f}")

# Full logistic model, includes all variables
X1 = sm.add_constant(X_train)
model_full = sm.Logit(y_train, X1).fit(disp=False)
print(f"Full Model AIC: {model_full.aic:.2f}")

#%% Step 16: LASSO Logistic Regression

pipeline_lasso = Pipeline([
    ('scaler', StandardScaler()),
    ('lasso', LogisticRegression(penalty='l1', solver='liblinear', random_state=42))])

pipeline_lasso.fit(X_train, y_train)
coefs = pd.Series(pipeline_lasso.named_steps['lasso'].coef_[0], index=X_train.columns)
print("\nNon-zero LASSO coefficients:")
print(coefs[coefs.abs() > 1e-6].sort_values(ascending=False))

#%% Step 17: Compare ROC curves

plt.figure(figsize=(6, 6), dpi=300)

# Null
y_null_score = np.full_like(y_test, fill_value=y_train.mean(), dtype=float)
auc_null = plot_roc(y_test, y_null_score, label='Null')

# LASSO
y_lasso_score = pipeline_lasso.predict_proba(X_test)[:, 1]
auc_lasso = plot_roc(y_test, y_lasso_score, label='LASSO')
plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
plt.title('ROC Curve Comparison'); plt.legend(loc='lower right'); plt.tight_layout(); plt.show()

#%% Step 18: Confusion matrices and classification reports
print("Confusion Matrix & Report: LASSO")
y_pred = (y_lasso_score >= 0.5).astype(int)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=['No', 'Yes']))

#%% Step 19: Cross-validated AUC for LASSO

cv_auc = cross_val_score(pipeline_lasso, X, y, cv=5, scoring='roc_auc')
print(f"Cross-validated AUC (5-fold): {cv_auc.mean():.3f} ± {cv_auc.std():.3f}")

#%% Step 20: Alternative LASSO variants via cross-validation

def evaluate_variants(X_base, y, variants_dict, cv=15):
    results = {}
    for name, X_var in variants_dict.items():
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('lasso', LogisticRegression(penalty='l1', solver='liblinear', random_state=42))])
        scores = cross_val_score(pipeline, X_var, y, cv=cv, scoring='roc_auc')
        results[name] = scores
        print(f"{name} — AUC per fold: {np.round(scores,3)}")
        print(f"{name} — Mean AUC = {scores.mean():.3f} ± {scores.std():.3f}\n")
    return results

variants = {
    'No Addicted_Score': X.drop(columns=['Addicted_Score']),
    'No Conflicts': X.drop(columns=['Conflicts_Over_Social_Media']),
    'No Both': X.drop(columns=['Addicted_Score','Conflicts_Over_Social_Media'])}

evaluate_variants(X, y, variants)

#%% Step 21: Alternative LASSO "No Both" predictions and ROC

# Romove the variables from X
X_nb = variants['No Both']
X_train_nb, X_test_nb, y_train_nb, y_test_nb = train_test_split(
    X_nb, y, test_size=0.2, random_state=42, stratify=y)

# re-writing the pipeline to fit de model
pipeline_nb = Pipeline([
    ('scaler', StandardScaler()),
    ('lasso', LogisticRegression(penalty='l1', solver='liblinear', random_state=42))])
pipeline_nb.fit(X_train_nb, y_train_nb)
coefs_nb = pd.Series(pipeline_nb.named_steps['lasso'].coef_[0], index=X_train_nb.columns)
print("Variables retained in 'No Both' model:")
print(coefs_nb[coefs_nb.abs()>1e-6].sort_values(ascending=False))

# Ploting ROC
y_score_nb = pipeline_nb.predict_proba(X_test_nb)[:,1]
plt.figure(figsize=(6,6), dpi=300)
plot_roc(y_test_nb, y_score_nb, label='LASSO No Both')
plt.plot([0,1],[0,1],'k--',alpha=0.5)
plt.xlabel('FPR'); plt.ylabel('TPR'); plt.title('ROC — LASSO No Both'); plt.legend(loc='lower right'); plt.tight_layout(); plt.show()

# Prediction, confusion matrix and classification report
print("Confusion and report — LASSO No Both")
y_pred_nb = (y_score_nb>=0.5).astype(int)
print(confusion_matrix(y_test_nb, y_pred_nb))
print(classification_report(y_test_nb, y_pred_nb, target_names=['No','Yes']))

#%% Step 22: LASSO only demographic variables

# Remove selected variables from X
X_clean = X.drop(columns=['Addicted_Score','Conflicts_Over_Social_Media','Mental_Health_Score'])
X_train_clean, X_test_clean, y_train_clean, y_test_clean = train_test_split(
    X_clean, y, test_size=0.2, random_state=42, stratify=y)

# re-writing the pipeline to fit de model
pipe_clean = Pipeline([
    ('scaler', StandardScaler()),
    ('lasso', LogisticRegression(penalty='l1', solver='liblinear', random_state=42))])
cv_auc_clean = cross_val_score(pipe_clean, X_clean, y, cv=15, scoring='roc_auc')
print(f"Demographics-only AUC (15-fold): {cv_auc_clean.mean():.3f} ± {cv_auc_clean.std():.3f}")

pipe_clean.fit(X_train_clean, y_train_clean)
y_score_clean = pipe_clean.predict_proba(X_test_clean)[:, 1]

# Ploting ROC
plt.figure(figsize=(6,6), dpi=300)
plot_roc(train_test_split(X_clean, y, test_size=0.2, random_state=42, stratify=y)[3], y_score_clean, label='LASSO Demographics')
plt.plot([0,1],[0,1],'k--',alpha=0.5)
plt.xlabel('FPR'); plt.ylabel('TPR'); plt.title('ROC — LASSO only Demographics'); plt.legend(loc='lower right'); plt.tight_layout(); plt.show()

print("Non-zero coefs — Demographics-only")
coefs_clean = pd.Series(pipe_clean.named_steps['lasso'].coef_[0], index=X_clean.columns)
print(coefs_clean[coefs_clean.abs()>1e-6].sort_values(ascending=False))

# Prediction, confusion matrix and classification report
print("Confusion and report — LASSO only Demographics")
y_pred_clean = (y_score_clean>=0.5).astype(int)
print(confusion_matrix(y_test_clean, y_pred_clean))
print(classification_report(y_test_clean, y_pred_clean, target_names=['No','Yes']))

#%% Step 23: Random Forest Evaluation Function

def evaluate_rf(name, X_tr, y_tr, X_te, y_te):
    print(f"{name} - Random Forest Evaluation")
    # Initialize and train model
    rf = RandomForestClassifier(random_state=42)
    rf.fit(X_tr, y_tr)
    # Predict probabilities
    y_score = rf.predict_proba(X_te)[:, 1]
    
    # ROC Curve and AUC
    plt.figure(figsize=(6, 6), dpi=300)
    roc_auc = plot_roc(y_te, y_score, label=name)
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {name} (AUC = {roc_auc:.3f})')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.show()

    # Confusion Matrix and Classification Report
    y_pred = (y_score >= 0.5).astype(int)
    print("Confusion Matrix:")
    print(confusion_matrix(y_te, y_pred))
    print(classification_report(y_te, y_pred, target_names=['No', 'Yes']))

    # Feature Importances
    importances = pd.Series(rf.feature_importances_, index=X_tr.columns)
    importances = importances.sort_values(ascending=False)
    plt.figure(figsize=(8, 6), dpi=300)
    importances.head(10).plot(kind='bar')
    plt.title(f'Top 10 Feature Importances - {name}')
    plt.ylabel('Importance')
    plt.tight_layout()
    plt.show()

#%% Step 24: Execute Random Forest for each dataset variant

evaluate_rf("RF Full", X_train, y_train, X_test, y_test)
evaluate_rf("RF No Both", X_train_nb, y_train_nb, X_test_nb, y_test_nb)
evaluate_rf("RF Demographics Only", X_train_clean, y_train_clean, X_test_clean, y_test_clean)

#%% # Step 26: XGBoost Evaluation Function
def evaluate_xgb(name, X_tr, y_tr, X_te, y_te):
    print(f"{name} - XGBoost Evaluation")
    # Initialize and train model
    xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    xgb.fit(X_tr, y_tr)
    # Predict probabilities
    y_score = xgb.predict_proba(X_te)[:, 1]
    
    # ROC Curve and AUC
    plt.figure(figsize=(6, 6), dpi=300)
    roc_auc = plot_roc(y_te, y_score, label=name)
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {name} (AUC = {roc_auc:.3f})')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.show()

    # Confusion Matrix and Classification Report
    y_pred = (y_score >= 0.5).astype(int)
    print("Confusion Matrix:")
    print(confusion_matrix(y_te, y_pred))
    print(classification_report(y_te, y_pred, target_names=['No', 'Yes']))

    # Feature Importances
    importances = pd.Series(xgb.feature_importances_, index=X_tr.columns)
    importances = importances.sort_values(ascending=False)
    plt.figure(figsize=(8, 6), dpi=300)
    importances.head(10).plot(kind='bar')
    plt.title(f'Top 10 Feature Importances - {name}')
    plt.ylabel('Importance')
    plt.tight_layout()
    plt.show()

#%% Step 27: Execute XGBoost for each dataset variant

evaluate_xgb("XGB Full", X_train, y_train, X_test, y_test)
evaluate_xgb("XGB No Both", X_train_nb, y_train_nb, X_test_nb, y_test_nb)
evaluate_xgb("XGB Demographics Only", X_train_clean, y_train_clean, X_test_clean, y_test_clean)
