import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)


DATA_PATH = r"C:\Users\psais\PycharmProjects\MLProject\Data\GlucoBench_benchmark_dataset.csv"


OUTPUT_PATH = os.path.join(
    BASE_DIR,
    "Output",
    "plots"
)


os.makedirs(
    OUTPUT_PATH,
    exist_ok=True
)


# ============================================================
# REMOVE OLD EDA PLOTS
# ============================================================

for filename in os.listdir(OUTPUT_PATH):

    if filename.endswith(".png"):

        os.remove(
            os.path.join(
                OUTPUT_PATH,
                filename
            )
        )


# ============================================================
# LOAD DATASET
# ============================================================

df = pd.read_csv(DATA_PATH)

print("Dataset shape:", df.shape)


# ============================================================
# FIRST 5 ROWS
# ============================================================

print("\nFIRST 5 ROWS")
print("========================================")
print(df.head())


# ============================================================
# LAST 5 ROWS
# ============================================================

print("\nLAST 5 ROWS")
print("========================================")
print(df.tail())


# ============================================================
# COLUMN NAMES
# ============================================================

print("\nCOLUMN NAMES")
print("========================================")
print(df.columns.tolist())


# ============================================================
# DATA TYPES
# ============================================================

print("\nDATA TYPES")
print("========================================")
print(df.dtypes)


# ============================================================
# STATISTICAL SUMMARY
# ============================================================

print("\nSTATISTICAL SUMMARY")
print("========================================")
print(df.describe(include="all").T)


# ============================================================
# MISSING VALUES
# ============================================================

print("\nMISSING VALUES")
print("========================================")

missing_values = df.isnull().sum()

print(
    missing_values[
        missing_values > 0
    ]
)


# ============================================================
# DUPLICATE ROWS
# ============================================================

print("\nDUPLICATE ROWS")
print("========================================")

print(
    "Number of duplicate rows:",
    df.duplicated().sum()
)


# ============================================================
# NUMERICAL COLUMNS
# ============================================================

numeric_columns = df.select_dtypes(
    include="number"
).columns.tolist()


print("\nNUMERICAL COLUMNS")
print("========================================")

print(numeric_columns)


# ============================================================
# CATEGORICAL COLUMNS
# ============================================================

categorical_columns = df.select_dtypes(
    exclude="number"
).columns.tolist()


print("\nCATEGORICAL COLUMNS")
print("========================================")

print(categorical_columns)


# ============================================================
# 1. GLUCOSE DISTRIBUTION
# ============================================================

if "glucose" in df.columns:

    plt.figure(
        figsize=(10, 6)
    )

    sns.histplot(
        df["glucose"].dropna(),
        kde=True
    )

    plt.title(
        "Glucose Distribution"
    )

    plt.xlabel(
        "Glucose"
    )

    plt.ylabel(
        "Frequency"
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_PATH,
            "glucose_distribution.png"
        ),
        dpi=150
    )

    plt.close()


# ============================================================
# 2. GLUCOSE BOXPLOT
# ============================================================

if "glucose" in df.columns:

    plt.figure(
        figsize=(10, 5)
    )

    sns.boxplot(
        x=df["glucose"]
    )

    plt.title(
        "Glucose Boxplot"
    )

    plt.xlabel(
        "Glucose"
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_PATH,
            "glucose_boxplot.png"
        ),
        dpi=150
    )

    plt.close()


# ============================================================
# 3. MISSING VALUES
# ============================================================

missing_plot = (
    df.isnull()
    .sum()
    .sort_values(
        ascending=False
    )
)


missing_plot = missing_plot[
    missing_plot > 0
]


if len(missing_plot) > 0:

    plt.figure(
        figsize=(10, 6)
    )

    sns.barplot(
        x=missing_plot.values,
        y=missing_plot.index
    )

    plt.title(
        "Missing Values by Column"
    )

    plt.xlabel(
        "Number of Missing Values"
    )

    plt.ylabel(
        "Column"
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_PATH,
            "missing_values.png"
        ),
        dpi=150
    )

    plt.close()


# ============================================================
# 4. CORRELATION HEATMAP
# ============================================================

if len(numeric_columns) > 1:

    correlation = df[
        numeric_columns
    ].corr()

    plt.figure(
        figsize=(16, 12)
    )

    sns.heatmap(
        correlation,
        annot=False,
        cmap="coolwarm",
        linewidths=0.5
    )

    plt.title(
        "Correlation Heatmap"
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_PATH,
            "correlation_heatmap.png"
        ),
        dpi=150
    )

    plt.close()


# ============================================================
# 5. CARBS VS GLUCOSE
# ============================================================

if (
    "carbs" in df.columns
    and
    "glucose" in df.columns
):

    plt.figure(
        figsize=(10, 6)
    )

    sns.scatterplot(
        data=df,
        x="carbs",
        y="glucose"
    )

    plt.title(
        "Carbs vs Glucose"
    )

    plt.xlabel(
        "Carbs"
    )

    plt.ylabel(
        "Glucose"
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_PATH,
            "carbs_vs_glucose.png"
        ),
        dpi=150
    )

    plt.close()


# ============================================================
# 6. INSULIN BOLUS VS GLUCOSE
# ============================================================

if (
    "insulin_bolus" in df.columns
    and
    "glucose" in df.columns
):

    plt.figure(
        figsize=(10, 6)
    )

    sns.scatterplot(
        data=df,
        x="insulin_bolus",
        y="glucose"
    )

    plt.title(
        "Insulin Bolus vs Glucose"
    )

    plt.xlabel(
        "Insulin Bolus"
    )

    plt.ylabel(
        "Glucose"
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_PATH,
            "insulin_bolus_vs_glucose.png"
        ),
        dpi=150
    )

    plt.close()


# ============================================================
# 7. EXERCISE STEPS VS GLUCOSE
# ============================================================

if (
    "exercise_steps" in df.columns
    and
    "glucose" in df.columns
):

    plt.figure(
        figsize=(10, 6)
    )

    sns.scatterplot(
        data=df,
        x="exercise_steps",
        y="glucose"
    )

    plt.title(
        "Exercise Steps vs Glucose"
    )

    plt.xlabel(
        "Exercise Steps"
    )

    plt.ylabel(
        "Glucose"
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_PATH,
            "exercise_steps_vs_glucose.png"
        ),
        dpi=150
    )

    plt.close()


# ============================================================
# 8. STRESS LEVEL VS GLUCOSE
# ============================================================

if (
    "stress_level" in df.columns
    and
    "glucose" in df.columns
):

    plt.figure(
        figsize=(10, 6)
    )

    sns.scatterplot(
        data=df,
        x="stress_level",
        y="glucose"
    )

    plt.title(
        "Stress Level vs Glucose"
    )

    plt.xlabel(
        "Stress Level"
    )

    plt.ylabel(
        "Glucose"
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_PATH,
            "stress_level_vs_glucose.png"
        ),
        dpi=150
    )

    plt.close()


# ============================================================
# SKEWNESS
# ============================================================

print("\n========================================")
print("SKEWNESS")
print("========================================")

print(
    df[numeric_columns].skew()
)


# ============================================================
# KURTOSIS
# ============================================================

print("\n========================================")
print("KURTOSIS")
print("========================================")

print(
    df[numeric_columns].kurtosis()
)


# ============================================================
# UNIQUE VALUES
# ============================================================

print("\n========================================")
print("UNIQUE VALUES")
print("========================================")

for column in categorical_columns:

    print(
        column,
        ":",
        df[column].nunique()
    )


# ============================================================
# FINISHED
# ============================================================

print("\n========================================")
print("EDA COMPLETED")
print("========================================")

print("Total EDA plots generated: 8")

print("\nGenerated plots:")

print("1. glucose_distribution.png")
print("2. glucose_boxplot.png")
print("3. missing_values.png")
print("4. correlation_heatmap.png")
print("5. carbs_vs_glucose.png")
print("6. insulin_bolus_vs_glucose.png")
print("7. exercise_steps_vs_glucose.png")
print("8. stress_level_vs_glucose.png")