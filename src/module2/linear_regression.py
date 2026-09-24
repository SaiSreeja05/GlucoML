import os
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# 1. PROJECT PATH
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)


# ============================================================
# 2. DATASET PATH
# ============================================================

DATA_PATH = r"C:\Users\psais\PycharmProjects\MLProject\Data\GlucoBench_benchmark_dataset.csv"


# ============================================================
# 3. LOAD DATASET
# ============================================================

df = pd.read_csv(DATA_PATH)

print("\n========================================")
print("LINEAR REGRESSION")
print("========================================")

print("Original dataset shape:", df.shape)


# ============================================================
# 4. DROP UNNECESSARY COLUMNS
# ============================================================

df = df.drop(
    columns=[
        "user_id",
        "device_id",
        "notes"
    ],
    errors="ignore"
)


# ============================================================
# 5. CONVERT TIMESTAMP
# ============================================================

if "timestamp" in df.columns:

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        errors="coerce"
    )

    df["hour"] = df["timestamp"].dt.hour
    df["day"] = df["timestamp"].dt.day
    df["month"] = df["timestamp"].dt.month
    df["day_of_week"] = df["timestamp"].dt.dayofweek

    df = df.drop(
        columns=["timestamp"]
    )


# ============================================================
# 6. REMOVE ROWS WHERE TARGET IS MISSING
# ============================================================

df = df.dropna(
    subset=["glucose"]
)


# ============================================================
# 7. SEPARATE FEATURES AND TARGET
# ============================================================

X = df.drop(
    columns=["glucose"]
)

y = df["glucose"]


print("\nNumber of features:", X.shape[1])
print("Target: glucose")


# ============================================================
# 8. IDENTIFY COLUMN TYPES
# ============================================================

numeric_features = X.select_dtypes(
    include="number"
).columns.tolist()

categorical_features = X.select_dtypes(
    exclude="number"
).columns.tolist()


print("\nNumerical features:")
print(numeric_features)

print("\nCategorical features:")
print(categorical_features)


# ============================================================
# 9. NUMERICAL PREPROCESSING
# ============================================================

numeric_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="median"
            )
        ),
        (
            "scaler",
            StandardScaler()
        )
    ]
)


# ============================================================
# 10. CATEGORICAL PREPROCESSING
# ============================================================

categorical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="most_frequent"
            )
        ),
        (
            "onehot",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            )
        )
    ]
)


# ============================================================
# 11. COMBINE PREPROCESSING
# ============================================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            numeric_pipeline,
            numeric_features
        ),
        (
            "categorical",
            categorical_pipeline,
            categorical_features
        )
    ]
)


# ============================================================
# 12. CREATE LINEAR REGRESSION MODEL
# ============================================================

model = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "regressor",
            LinearRegression()
        )
    ]
)


# ============================================================
# 13. TRAIN TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


print("\n========================================")
print("TRAIN TEST SPLIT")
print("========================================")

print("Training samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])


# ============================================================
# 14. TRAIN MODEL
# ============================================================

model.fit(
    X_train,
    y_train
)


# ============================================================
# 15. PREDICTION
# ============================================================

y_pred = model.predict(
    X_test
)


# ============================================================
# 16. EVALUATION
# ============================================================

mae = mean_absolute_error(
    y_test,
    y_pred
)

mse = mean_squared_error(
    y_test,
    y_pred
)

rmse = np.sqrt(
    mse
)

r2 = r2_score(
    y_test,
    y_pred
)


print("\n========================================")
print("MODEL PERFORMANCE")
print("========================================")

print(
    "Mean Absolute Error (MAE):",
    round(mae, 4)
)

print(
    "Mean Squared Error (MSE):",
    round(mse, 4)
)

print(
    "Root Mean Squared Error (RMSE):",
    round(rmse, 4)
)

print(
    "R² Score:",
    round(r2, 4)
)


# ============================================================
# 17. ACTUAL VS PREDICTED
# ============================================================

results = pd.DataFrame(
    {
        "Actual Glucose": y_test.values,
        "Predicted Glucose": y_pred
    }
)

print("\n========================================")
print("ACTUAL VS PREDICTED")
print("========================================")

print(
    results.head(10)
)


# ============================================================
# 18. SAVE RESULTS
# ============================================================

OUTPUT_PATH = os.path.join(
    BASE_DIR,
    "Output"
)

os.makedirs(
    OUTPUT_PATH,
    exist_ok=True
)

results.to_csv(
    os.path.join(
        OUTPUT_PATH,
        "linear_regression_predictions.csv"
    ),
    index=False
)


# ============================================================
# 19. COMPLETION
# ============================================================

print("\n========================================")
print("LINEAR REGRESSION COMPLETED")
print("========================================")

print(
    "Prediction file saved in:",
    OUTPUT_PATH
)