import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
import re

# ============================================================================
# STEP 1: LOAD AND PARSE RAW LOG DATA
# ============================================================================

data = pd.read_csv('sessions_flattened_ready.csv')
print("Data Loaded Successfully!")
print(f"   Shape: {data.shape[0]} rows × {data.shape[1]} columns")
print(f"\n   First 5 rows:")
print(data.head())
print(f"\n   Data Info:")
print(data.info())

# Extract structured information from strings using regex
# Pattern matches: [timestamp] IP_address HTTP_METHOD /api/endpoint
LOG_PATTERN = r'\[(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z)\]\s+([\d\.]+)\s+(GET|POST|PUT|DELETE|PATCH)\s+(\/universe\/api\/v1\/[^\s,]+)'

# Extract timestamp, IP address, HTTP method, and API endpoint into separate columns
data[['call_timestamp', 'ip_address', 'http_method', 'api_endpoint']] = \
    data['callStack_log_raw'].str.extract(LOG_PATTERN)

print("Extracted log details into new columns")

# Convert string timestamps to datetime objects for time calculations
data['call_timestamp'] = pd.to_datetime(data['call_timestamp'])
data['session_startedAt'] = pd.to_datetime(data['session_startedAt'])
data['session_endedAt'] = pd.to_datetime(data['session_endedAt'])

print("Converted timestamps to datetime objects")

# ============================================================================
# STEP 2: EXTRACT API PATH COMPONENTS
# ============================================================================

# Remove query parameters (everything after ?) to get clean API paths
data['base_api_path'] = data['api_endpoint'].str.split('?').str[0]

# Extract the resource name (e.g., 'club', 'event', 'user') from path
# Path format: /universe/api/v1/resource/function
# Resource is at index 4 after splitting by '/'
data['api_resource'] = data['base_api_path'].str.split('/').str[4]

print("Extracted API path components")

# ============================================================================
# STEP 3: EXTRACT DATE AND TIME COMPONENTS
# ============================================================================

# Separate date and time for easier analysis
data['session_date'] = data['session_startedAt'].dt.date
data['call_time'] = data['call_timestamp'].dt.time
data['session_start_time'] = data['session_startedAt'].dt.time

print("Separated date and time components")

# ============================================================================
# STEP 4: CALCULATE SESSION DURATION AND CLEAN UP COLUMNS
# ============================================================================

# Calculate actual session duration in seconds
data['session_duration_sec'] = (data['session_endedAt'] - data['session_startedAt']).dt.total_seconds()

# Remove empty/redundant columns
data.drop(columns=['session_session_duration_sec'], inplace=True, errors='ignore')
data.drop(columns=['call_timestamp', 'session_startedAt', 'session_endedAt'], inplace=True, errors='ignore')

print("Calculated session duration and cleaned up columns")

# ============================================================================
# STEP 5: EXTRACT API FUNCTION NAMES
# ============================================================================

# Extract the function name (e.g., 'getStatus', 'getFastNativeFeed') from API path
# Function name is at index 5 after splitting by '/'
data['api_function_name'] = data['base_api_path'].str.split('/').str[5]
data['api_function_name'] = data['api_function_name'].fillna('N/A')

print("Extracted API function names")

# ============================================================================
# STEP 6: AGGREGATE LOG-LEVEL DATA TO SESSION-LEVEL FEATURES
# ============================================================================

# Group by session ID to create one row per session
GROUP_COLUMN = 'session__id'

# Keep these static session metadata columns
SESSION_STATIC_COLS = [
    GROUP_COLUMN, 
    'session_userId', 
    'session_date', 
    'session_start_time',
    'session_duration_sec' 
]

# Get unique session metadata (one row per session)
session_metadata = data[SESSION_STATIC_COLS].drop_duplicates().copy()

# Filter to only valid log entries (where regex extraction worked)
valid_logs = data[data['http_method'].notnull()].copy()

# Aggregate log features per session
df_agg = valid_logs.groupby(GROUP_COLUMN).agg(
    num_api_calls=('api_endpoint', 'count'),              # Total API calls in session
    endpoint_diversity=('base_api_path', 'nunique'),      # Number of unique endpoints
    first_call_time=('call_time', 'min'),                 # Time of first API call
    most_frequent_endpoint=('base_api_path', lambda x: x.mode()[0] if not x.empty else 'N/A'),
    most_frequent_resource=('api_resource', lambda x: x.mode()[0] if not x.empty else 'N/A')
).reset_index()

# Merge aggregated features back to session metadata
# Left merge keeps ALL sessions, even those with no valid logs
df_final_features = session_metadata.merge(df_agg, on=GROUP_COLUMN, how='left')

# Fill missing values: sessions with no logs get 0 for counts
df_final_features['num_api_calls'] = df_final_features['num_api_calls'].fillna(0)
df_final_features['endpoint_diversity'] = df_final_features['endpoint_diversity'].fillna(0)
df_final_features['most_frequent_endpoint'] = df_final_features['most_frequent_endpoint'].fillna('N/A')
df_final_features['most_frequent_resource'] = df_final_features['most_frequent_resource'].fillna('N/A')

print("Aggregated log data to session-level features")

# ============================================================================
# STEP 7: COUNT HTTP METHODS (GET vs POST)
# ============================================================================

# Count GET and POST requests per session
method_counts = pd.crosstab(
    valid_logs[GROUP_COLUMN], 
    valid_logs['http_method']
).reset_index()

method_counts.rename(columns={'GET': 'get_count', 'POST': 'post_count'}, inplace=True)

# Merge method counts
df_final_features = df_final_features.merge(method_counts, on=GROUP_COLUMN, how='left')
df_final_features['get_count'] = df_final_features['get_count'].fillna(0)
df_final_features['post_count'] = df_final_features['post_count'].fillna(0)

print("Counted HTTP methods per session")

# ============================================================================
# STEP 8: CALCULATE DERIVED FEATURES
# ============================================================================

# API call rate: calls per second
df_final_features['api_call_rate'] = (
    df_final_features['num_api_calls'] / df_final_features['session_duration_sec']
).replace([np.inf, -np.inf], 0).fillna(0)

# Time to first call: how long after session start until first API call
# Reconstruct full datetime from date + time
df_final_features['start_dt'] = pd.to_datetime(
    df_final_features['session_date'].astype(str) + ' ' + df_final_features['session_start_time'].astype(str),
    format='mixed'
)

# Calculate time difference for sessions that had calls
valid_first_call_time = df_final_features['first_call_time'].dropna()
valid_session_date_series = df_final_features.loc[valid_first_call_time.index, 'session_date']

first_call_dt_temp = pd.to_datetime(
    valid_session_date_series.astype(str) + ' ' + valid_first_call_time.astype(str),
    format='mixed'
)

time_diff = (first_call_dt_temp - df_final_features.loc[first_call_dt_temp.index, 'start_dt']).dt.total_seconds()

# For sessions with no calls, use full session duration as time to first call
df_final_features['time_to_first_call_sec'] = time_diff.fillna(df_final_features['session_duration_sec'])

# Clean up temporary columns
df_final_features.drop(columns=['first_call_time', 'start_dt'], inplace=True)

print("Calculated derived features (call rate, time to first call)")

# ============================================================================
# STEP 9: CREATE CLASSIFICATION TARGET
# ============================================================================

# Define high engagement as sessions lasting 2+ minutes (120 seconds)
HIGH_ENGAGEMENT_THRESHOLD_SEC = 120

# Create binary target: 1 = high engagement, 0 = low engagement
df_final_features['is_highly_engaged'] = (
    df_final_features['session_duration_sec'] >= HIGH_ENGAGEMENT_THRESHOLD_SEC
).astype(int)

engagement_counts = df_final_features['is_highly_engaged'].value_counts(normalize=True) * 100
print("\nClassification target created")
print(f"Target distribution: {engagement_counts[0]:.1f}% Low Engagement, {engagement_counts[1]:.1f}% High Engagement")

# ============================================================================
# STEP 10: ENCODE CATEGORICAL FEATURES
# ============================================================================

# Convert categorical text features to numerical format using one-hot encoding
categorical_features = ['most_frequent_endpoint', 'most_frequent_resource']

df_encoded = pd.get_dummies(
    df_final_features, 
    columns=categorical_features, 
    drop_first=True,  # Avoid multicollinearity
    prefix=categorical_features
)

print(f"Encoded categorical features. New shape: {df_encoded.shape}")

# ============================================================================
# STEP 11: SCALE NUMERICAL FEATURES
# ============================================================================

# List of numerical features to scale (normalize to mean=0, std=1)
numerical_features = [
    'session_duration_sec', 
    'num_api_calls', 
    'endpoint_diversity', 
    'get_count', 
    'post_count', 
    'api_call_rate', 
    'time_to_first_call_sec'
]

# Scale features so all are on similar scale (important for distance-based models)
scaler = StandardScaler()
scaler.fit(df_encoded[numerical_features])
df_encoded[numerical_features] = scaler.transform(df_encoded[numerical_features])

print("Scaled numerical features")

# ============================================================================
# STEP 12: PREPARE DATA FOR REGRESSION (PREDICT SESSION DURATION)
# ============================================================================

# Define which columns to exclude from features
COLUMNS_TO_DROP = [
    'session_duration_sec',    # This is what we're predicting (target)
    'is_highly_engaged',       # Classification target (not used for regression)
    'session__id',             # ID column (not a feature)
    'session_userId',          # ID column (not a feature)
    'session_date',            # Date metadata (not a feature)
    'session_start_time'       # Time metadata (not a feature)
]

# Create feature matrix X and target vector y
X = df_encoded.drop(columns=COLUMNS_TO_DROP)
X = X.fillna(0)
y = df_encoded['session_duration_sec']

# Split into training (70%) and testing (30%) sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

print(f"Prepared regression data. Train: {X_train.shape[0]} samples, Test: {X_test.shape[0]} samples")

# ============================================================================
# STEP 13: TRAIN RIDGE REGRESSION MODEL
# ============================================================================

print("\n--- Training Ridge Regression Model ---")

# Ridge Regression: Linear model with L2 regularization (prevents overfitting)
ridge_model = Ridge(random_state=42)
ridge_model.fit(X_train, y_train)

# Make predictions on test set
y_pred_ridge = ridge_model.predict(X_test)

# Calculate performance metrics
mae_ridge = mean_absolute_error(y_test, y_pred_ridge)
rmse_ridge = np.sqrt(mean_squared_error(y_test, y_pred_ridge))
r2_ridge = r2_score(y_test, y_pred_ridge)

print(f"MAE: {mae_ridge:.4f} seconds")
print(f"RMSE: {rmse_ridge:.4f} seconds")
print(f"R² Score: {r2_ridge:.4f}")

# Visualize predictions vs actual values
plt.figure(figsize=(8, 8))
plt.scatter(y_test, y_pred_ridge, alpha=0.3, color='blue', label='Predictions')
max_val = max(y_test.max(), y_pred_ridge.max())
plt.plot([0, max_val], [0, max_val], color='red', linestyle='--', label='Perfect Prediction')
plt.title('Ridge Regression: Actual vs Predicted Session Duration')
plt.xlabel('Actual Session Duration (Scaled)')
plt.ylabel('Predicted Session Duration (Scaled)')
plt.legend()
plt.grid(True, linestyle=':')
plt.show()

# ============================================================================
# STEP 14: TRAIN RANDOM FOREST REGRESSION MODEL
# ============================================================================

print("\n--- Training Random Forest Regression Model ---")

# Random Forest: Ensemble of decision trees (non-linear, more powerful)
rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)

# Make predictions
y_pred_rf = rf_model.predict(X_test)

# Calculate metrics
mae_rf = mean_absolute_error(y_test, y_pred_rf)
rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
r2_rf = r2_score(y_test, y_pred_rf)

print(f"MAE: {mae_rf:.4f} seconds")
print(f"RMSE: {rmse_rf:.4f} seconds")
print(f"R² Score: {r2_rf:.4f}")

# Visualize predictions
plt.figure(figsize=(8, 8))
plt.scatter(y_test, y_pred_rf, alpha=0.3, color='darkgreen', label='Predictions')
max_val = max(y_test.max(), y_pred_rf.max())
plt.plot([0, max_val], [0, max_val], color='red', linestyle='--', label='Perfect Prediction')
plt.title('Random Forest: Actual vs Predicted Session Duration')
plt.xlabel('Actual Session Duration (Scaled)')
plt.ylabel('Predicted Session Duration (Scaled)')
plt.legend()
plt.grid(True, linestyle=':')
plt.show()

# Compare regression models
print("\n--- Regression Model Comparison ---")
print(f"Ridge R²: {r2_ridge:.4f}")
print(f"Random Forest R²: {r2_rf:.4f}")

# ============================================================================
# STEP 15: PREPARE DATA FOR CLASSIFICATION (PREDICT HIGH ENGAGEMENT)
# ============================================================================

# Use the same feature matrix X, but different target (engagement classification)
Y_cls = df_encoded['is_highly_engaged']

# Split for classification
X_train_cls, X_test_cls, Y_train_cls, Y_test_cls = train_test_split(
    X, Y_cls, test_size=0.3, random_state=42
)

print(f"\nPrepared classification data. Train: {X_train_cls.shape[0]} samples, Test: {X_test_cls.shape[0]} samples")

# ============================================================================
# STEP 16: TRAIN KNN CLASSIFIER
# ============================================================================

print("\n--- Training K-Nearest Neighbors (KNN) Classifier ---")

# KNN: Classifies based on similarity to k nearest neighbors
knn_model = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
knn_model.fit(X_train_cls, Y_train_cls)

# Make predictions
Y_pred_knn = knn_model.predict(X_test_cls)

# Calculate classification metrics
accuracy_knn = accuracy_score(Y_test_cls, Y_pred_knn)
precision_knn = precision_score(Y_test_cls, Y_pred_knn)
recall_knn = recall_score(Y_test_cls, Y_pred_knn)
f1_knn = f1_score(Y_test_cls, Y_pred_knn)

print(f"Accuracy: {accuracy_knn:.4f}")
print(f"Precision: {precision_knn:.4f}")
print(f"Recall: {recall_knn:.4f}")
print(f"F1 Score: {f1_knn:.4f}")

# Confusion Matrix: Shows correct vs incorrect predictions
cm_knn = confusion_matrix(Y_test_cls, Y_pred_knn)
plt.figure(figsize=(6, 6))
disp_knn = ConfusionMatrixDisplay(confusion_matrix=cm_knn, display_labels=['Low Engagement (0)', 'High Engagement (1)'])
disp_knn.plot(cmap=plt.cm.Blues, values_format='d')
plt.title('KNN Confusion Matrix')
plt.show()

# Scatter plot: Actual vs Predicted (similar to regression plots)
plt.figure(figsize=(8, 8))
plt.scatter(Y_test_cls, Y_pred_knn, alpha=0.5, color='purple', s=50)
plt.plot([0, 1], [0, 1], color='red', linestyle='--', label='Perfect Prediction')
plt.title('KNN Classifier: Actual vs Predicted Engagement')
plt.xlabel('Actual Engagement (0=Low, 1=High)')
plt.ylabel('Predicted Engagement (0=Low, 1=High)')
plt.xticks([0, 1])
plt.yticks([0, 1])
plt.legend()
plt.grid(True, linestyle=':')
plt.show()

# ============================================================================
# STEP 17: TRAIN DECISION TREE CLASSIFIER
# ============================================================================

print("\n--- Training Decision Tree Classifier ---")

# Decision Tree: Makes decisions based on feature thresholds
dt_model = DecisionTreeClassifier(max_depth=5, random_state=42)
dt_model.fit(X_train_cls, Y_train_cls)

# Make predictions
Y_pred_dt = dt_model.predict(X_test_cls)

# Calculate metrics
accuracy_dt = accuracy_score(Y_test_cls, Y_pred_dt)
precision_dt = precision_score(Y_test_cls, Y_pred_dt)
recall_dt = recall_score(Y_test_cls, Y_pred_dt)
f1_dt = f1_score(Y_test_cls, Y_pred_dt)

print(f"Accuracy: {accuracy_dt:.4f}")
print(f"Precision: {precision_dt:.4f}")
print(f"Recall: {recall_dt:.4f}")
print(f"F1 Score: {f1_dt:.4f}")

# Visualize the decision tree structure
class_names = ["Low Engagement (0)", "High Engagement (1)"]
plt.figure(figsize=(30, 15))
plot_tree(
    dt_model, 
    filled=True, 
    feature_names=X.columns.tolist(), 
    class_names=class_names, 
    rounded=True, 
    fontsize=10 
)
plt.title(f"Decision Tree Structure (Max Depth: {dt_model.max_depth})", fontsize=16)
plt.show()

# Confusion Matrix
cm_dt = confusion_matrix(Y_test_cls, Y_pred_dt)
plt.figure(figsize=(6, 6))
disp_dt = ConfusionMatrixDisplay(confusion_matrix=cm_dt, display_labels=['Low Engagement (0)', 'High Engagement (1)'])
disp_dt.plot(cmap=plt.cm.Greens, values_format='d')
plt.title('Decision Tree Confusion Matrix')
plt.show()

# ============================================================================
# STEP 18: MODEL COMPARISON VISUALIZATIONS
# ============================================================================

# Compare regression models by R² score
regression_models = {
    'Ridge Regression': r2_ridge,
    'Random Forest': r2_rf
}

reg_df = pd.Series(regression_models).sort_values(ascending=False)
plt.figure(figsize=(8, 6))
reg_df.plot(kind='bar', color=['darkred', 'darkblue'])
plt.title('Regression Model Performance Comparison (R² Score)')
plt.ylabel('R² Score (Higher is Better)')
plt.ylim(0, 1.0)
plt.xticks(rotation=0)
plt.grid(axis='y', linestyle='--')
plt.show()

# Compare classification models by F1 score
classification_models = {
    'KNN Classifier': f1_knn,
    'Decision Tree': f1_dt
}

cls_df = pd.Series(classification_models).sort_values(ascending=False)
plt.figure(figsize=(8, 6))
cls_df.plot(kind='bar', color=['darkgreen', 'darkred'])
plt.title('Classification Model Performance Comparison (F1 Score)')
plt.ylabel('F1 Score (Higher is Better)')
plt.ylim(0.7, 1.0)
plt.xticks(rotation=0)
plt.grid(axis='y', linestyle='--')
plt.show()

print("\n Project analysis on Sessions data completed! Adios and all the best")