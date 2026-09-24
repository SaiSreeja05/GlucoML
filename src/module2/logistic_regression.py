import os
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report


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
print("LOGISTIC REGRESSION")
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
# 6. CREATE BINARY TARGET
# ============================================================

df["glucose_class"] = (
    df["glucose"] >= 126
).astype(int)


print("\n========================================")
print("TARGET DISTRIBUTION")
print("========================================")

print(
    df["glucose_class"].value_counts()
)


# ============================================================
# 7. SEPARATE FEATURES AND TARGET
# ============================================================

X = df.drop(
    columns=[
        "glucose",
        "glucose_class"
    ]
)

y = df["glucose_class"]


# ============================================================
# 8. IDENTIFY FEATURE TYPES
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
# 12. CREATE LOGISTIC REGRESSION MODEL
# ============================================================

model = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=1000
            )
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
    random_state=42,
    stratify=y
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

accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    zero_division=0
)

cm = confusion_matrix(
    y_test,
    y_pred
)


# ============================================================
# 17. DISPLAY PERFORMANCE
# ============================================================

print("\n========================================")
print("LOGISTIC REGRESSION PERFORMANCE")
print("========================================")

print(
    "Accuracy:",
    round(accuracy, 4)
)

print(
    "Precision:",
    round(precision, 4)
)

print(
    "Recall:",
    round(recall, 4)
)

print(
    "F1 Score:",
    round(f1, 4)
)


print("\n========================================")
print("CONFUSION MATRIX")
print("========================================")

print(cm)


print("\n========================================")
print("CLASSIFICATION REPORT")
print("========================================")

print(
    classification_report(
        y_test,
        y_pred,
        zero_division=0
    )
)


# ============================================================
# 18. SAVE PREDICTIONS
# ============================================================

results = pd.DataFrame(
    {
        "Actual Class": y_test.values,
        "Predicted Class": y_pred
    }
)


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
        "logistic_regression_predictions.csv"
    ),
    index=False
)


# ============================================================
# 19. COMPLETION
# ============================================================

print("\n========================================")
print("LOGISTIC REGRESSION COMPLETED")
print("========================================")

print(
    "Prediction file saved in:",
    OUTPUT_PATH
)