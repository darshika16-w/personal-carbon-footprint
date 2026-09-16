import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
# Load dataset
df = pd.read_csv("personal_carbon_footprint_behavior.csv")

# Features and target
X = df.drop(columns=["carbon_footprint_kg", "carbon_impact_level"])
y = df["carbon_footprint_kg"] 

# Categorical and numerical columns
categorical_features = [
    "day_type",
    "transport_mode",
    "food_type"
]

numerical_features = [
    "distance_km",
    "electricity_kwh",
    "renewable_usage_pct",
    "screen_time_hours",
    "waste_generated_kg",
    "eco_actions"
]

# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ("num", "passthrough", numerical_features)
    ]
)

# Model
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

# Pipeline
pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ]
)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train model
pipeline.fit(X_train, y_train)

# Prediction
y_pred = pipeline.predict(X_test)

# Evaluation
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation:")
print("Mean Absolute Error:", mae)
print("Mean Squared Error:", mse)
print("R² Score:", r2)
import joblib

joblib.dump(pipeline, "carbon_footprint_model.pkl")

print("\nModel saved successfully!")
# Linear Regression Model
lr_model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", LinearRegression())
])

# Train Linear Regression
lr_model.fit(X_train, y_train)

# Prediction
lr_pred = lr_model.predict(X_test)

# Evaluation
lr_mae = mean_absolute_error(y_test, lr_pred)
lr_mse = mean_squared_error(y_test, lr_pred)
lr_r2 = r2_score(y_test, lr_pred)

print("\nLinear Regression Results:")
print("MAE:", lr_mae)
print("MSE:", lr_mse)
print("R2 Score:", lr_r2)
print("\n===== MODEL COMPARISON =====")
print("Linear Regression R2 Score: 0.9482")
print("Random Forest R2 Score: 0.9423")

if 0.9482 > 0.9423:
    print("Linear Regression performs better.")
else:
    print("Random Forest performs better.")

# XGBoost Model
xgb_model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", XGBRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        random_state=42
    ))
])

# Train XGBoost
xgb_model.fit(X_train, y_train)

# Prediction
xgb_pred = xgb_model.predict(X_test)

# Evaluation
xgb_mae = mean_absolute_error(y_test, xgb_pred)
xgb_mse = mean_squared_error(y_test, xgb_pred)
xgb_r2 = r2_score(y_test, xgb_pred)

print("\nXGBoost Results:")
print("MAE:", xgb_mae)
print("MSE:", xgb_mse)
print("R2 Score:", xgb_r2)
# XGBoost Model
xgb_model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", XGBRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        random_state=42
    ))
])

# Train XGBoost
xgb_model.fit(X_train, y_train)

# Prediction
xgb_pred = xgb_model.predict(X_test)

# Evaluation
xgb_mae = mean_absolute_error(y_test, xgb_pred)
xgb_mse = mean_squared_error(y_test, xgb_pred)
xgb_r2 = r2_score(y_test, xgb_pred)


# Model Comparison Graph

import matplotlib.pyplot as plt

models = ["Random Forest", "Linear Regression", "XGBoost"]
r2_scores = [0.9423, 0.9482, 0.9834]

plt.figure(figsize=(8, 5))
plt.bar(models, r2_scores)

plt.title("Model Comparison - R² Score")
plt.xlabel("Models")
plt.ylabel("R² Score")
plt.ylim(0, 1)

plt.tight_layout()
plt.savefig("graphs/model_comparison_r2.png")
plt.show()
# XGBoost Feature Importance

importance = xgb_model.named_steps["model"].feature_importances_

# Get feature names after preprocessing
preprocessor = xgb_model.named_steps["preprocessor"]
features = preprocessor.get_feature_names_out()

plt.figure(figsize=(10, 6))
plt.barh(features, importance)

plt.title("XGBoost Feature Importance")
plt.xlabel("Importance")
plt.ylabel("Features")

plt.tight_layout()
plt.savefig("graphs/xgboost_feature_importance.png")
plt.show()