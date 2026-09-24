import os
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler


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
# 3. LOAD DATA
# ============================================================

df = pd.read_csv(DATA_PATH)

print("\n========================================")
print("DATA LOADED")
print("========================================")

print("Original shape:", df.shape)


# ============================================================
# 4. DROP UNNECESSARY COLUMNS
# ============================================================

columns_to_drop = [
    "user_id",
    "device_id",
    "notes"
]

df = df.drop(
    columns=columns_to_drop,
    errors="ignore"
)


# ============================================================
# 5. TIMESTAMP FEATURE EXTRACTION
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
# 6. IDENTIFY NUMERICAL COLUMNS
# ============================================================

numeric_columns = df.select_dtypes(
    include="number"
).columns.tolist()


# ============================================================
# 7. IDENTIFY CATEGORICAL COLUMNS
# ============================================================

categorical_columns = df.select_dtypes(
    exclude="number"
).columns.tolist()


# ============================================================
# 8. HANDLE MISSING NUMERICAL VALUES
# ============================================================

if numeric_columns:

    numeric_imputer = SimpleImputer(
        strategy="median"
    )

    df[numeric_columns] = numeric_imputer.fit_transform(
        df[numeric_columns]
    )


# ============================================================
# 9. HANDLE MISSING CATEGORICAL VALUES
# ============================================================

if categorical_columns:

    categorical_imputer = SimpleImputer(
        strategy="most_frequent"
    )

    df[categorical_columns] = categorical_imputer.fit_transform(
        df[categorical_columns]
    )


# ============================================================
# 10. STANDARD SCALER
# ============================================================

standard_scaler = StandardScaler()

df_standard = df.copy()

if numeric_columns:

    df_standard[numeric_columns] = standard_scaler.fit_transform(
        df_standard[numeric_columns]
    )


# ============================================================
# 11. MIN-MAX SCALER
# ============================================================

minmax_scaler = MinMaxScaler()

df_minmax = df.copy()

if numeric_columns:

    df_minmax[numeric_columns] = minmax_scaler.fit_transform(
        df_minmax[numeric_columns]
    )


# ============================================================
# 12. ROBUST SCALER
# ============================================================

robust_scaler = RobustScaler()

df_robust = df.copy()

if numeric_columns:

    df_robust[numeric_columns] = robust_scaler.fit_transform(
        df_robust[numeric_columns]
    )


# ============================================================
# 13. DISPLAY RESULTS
# ============================================================

print("\n========================================")
print("AFTER PREPROCESSING")
print("========================================")

print("Shape:", df.shape)

print("\nNumerical columns:")
print(numeric_columns)

print("\nCategorical columns:")
print(categorical_columns)

print("\nMissing values after preprocessing:")

print(
    df.isnull().sum()[
        df.isnull().sum() > 0
    ]
)


# ============================================================
# 14. DISPLAY SAMPLE
# ============================================================

print("\n========================================")
print("PREPROCESSED DATA")
print("========================================")

print(df.head())


# ============================================================
# 15. SCALING SAMPLE
# ============================================================

print("\n========================================")
print("STANDARD SCALED DATA")
print("========================================")

print(
    df_standard.head()
)


print("\n========================================")
print("MIN-MAX SCALED DATA")
print("========================================")

print(
    df_minmax.head()
)


print("\n========================================")
print("ROBUST SCALED DATA")
print("========================================")

print(
    df_robust.head()
)


# ============================================================
# 16. SAVE PREPROCESSED DATA
# ============================================================

OUTPUT_PATH = os.path.join(
    BASE_DIR,
    "Output"
)

os.makedirs(
    OUTPUT_PATH,
    exist_ok=True
)

df.to_csv(
    os.path.join(
        OUTPUT_PATH,
        "preprocessed_data.csv"
    ),
    index=False
)

df_standard.to_csv(
    os.path.join(
        OUTPUT_PATH,
        "standard_scaled_data.csv"
    ),
    index=False
)

df_minmax.to_csv(
    os.path.join(
        OUTPUT_PATH,
        "minmax_scaled_data.csv"
    ),
    index=False
)

df_robust.to_csv(
    os.path.join(
        OUTPUT_PATH,
        "robust_scaled_data.csv"
    ),
    index=False
)


# ============================================================
# 17. COMPLETION
# ============================================================

print("\n========================================")
print("PREPROCESSING COMPLETED")
print("========================================")

print(
    "Files saved inside:",
    OUTPUT_PATH
)