import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import classification_report, roc_auc_score
import xgboost as xgb
import gc

# File paths
train_file = '38_Training_Data_Set_V2/training.csv'
test_file = '38_Public_Test_Set_and_Submmision_Template_V2/public_x.csv'
output_file = 'predictions_XGBoost_gpu_weighted_top1000.csv'

# Step 1: Load training data
print("Loading training data...")
train_df = pd.read_csv(train_file)
label_column = train_df.columns[-1]

# Step 2: Prepare features and labels
X = train_df.drop(columns=['ID', label_column]).astype(np.float32)
y = train_df[label_column]

# Step 3: Clean data with mean
print("Cleaning data with mean imputation...")
X = X.replace([np.inf, -np.inf], np.nan)
X = X.fillna(X.mean())

# Step 4: Split into train/val
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)

# Step 5: Feature selection (top 1000)
print("Selecting top 1000 important features...")
selector_model = xgb.XGBClassifier(tree_method='hist', device='cuda', use_label_encoder=False, eval_metric='auc')
selector_model.fit(X_train, y_train)
selector = SelectFromModel(selector_model, prefit=True, max_features=1000, threshold=-np.inf)
X_train = selector.transform(X_train)
X_val = selector.transform(X_val)

# Step 6: Standardize
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

# Step 7: class weight
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
print(f"scale_pos_weight: {scale_pos_weight:.2f}")

# Step 8: Train final model
print("Training XGBoost model on GPU...")
model = xgb.XGBClassifier(
    objective='binary:logistic',
    eval_metric='auc',
    use_label_encoder=False,
    random_state=42,
    n_estimators=1000,
    max_depth=6,
    learning_rate=0.01,
    scale_pos_weight=scale_pos_weight,
    subsample=0.8,
    tree_method='hist',
    device='cuda'
)
model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=True)

# Step 9: Evaluate
print("Evaluating model...")
y_val_proba = model.predict_proba(X_val)[:, 1]
y_val_pred = (y_val_proba > 0.3).astype(int)
print(classification_report(y_val, y_val_pred))
print(f"AUC-ROC Score: {roc_auc_score(y_val, y_val_proba):.4f}")

# Step 10: Predict test data
print("Loading test data...")
test_df = pd.read_csv(test_file)
X_test = test_df.drop(columns=['ID']).astype(np.float32)
X_test = X_test.replace([np.inf, -np.inf], np.nan)
X_test = X_test.fillna(X_test.mean())
X_test = selector.transform(X_test)
X_test = scaler.transform(X_test)

print("Predicting test data...")
y_test_pred = (model.predict_proba(X_test)[:, 1] > 0.3).astype(int)

# Step 11: Save results
output_df = pd.DataFrame({'ID': test_df['ID'], label_column: y_test_pred})
output_df.to_csv(output_file, index=False)
print(f"Predictions saved to '{output_file}'.")

# Step 12: Cleanup
del model
gc.collect()
print("Model deleted and memory cleaned.")