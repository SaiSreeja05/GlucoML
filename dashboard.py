import os
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st


# ============================================================
# 1. PROJECT PATH
# ============================================================

BASE_DIR = Path(__file__).resolve().parent


# ============================================================
# 2. PATHS
# ============================================================

DATA_PATH = BASE_DIR / "Data" / "GlucoBench_benchmark_dataset.csv"

OUTPUT_PATH = BASE_DIR / "Output"


# ============================================================
# 3. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="GlucoBench ML Dashboard",
    page_icon="📊",
    layout="wide"
)


# ============================================================
# 4. TITLE
# ============================================================

st.title("📊 GlucoBench Machine Learning Dashboard")
st.caption("GlucoBench ML Dashboard — data analysis and model insights")
st.write(
    "Analysis, preprocessing, machine learning models, "
    "clustering and PCA results."
)


# ============================================================
# 5. LOAD DATASET
# ============================================================

@st.cache_data
def load_data():

    return pd.read_csv(DATA_PATH)


df = load_data()
# ============================================================
# 6. SIDEBAR
# ============================================================

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Select Section",
    [
        "Dataset Overview",
        "EDA",
        "Preprocessing",
        "Regression Models",
        "Classification",
        "Tree & Boosting Models",
        "Clustering",
        "PCA",
        "Model Comparison"
    ]
)


# ============================================================
# 7. DATASET OVERVIEW
# ============================================================

if page == "Dataset Overview":

    st.header("📋 Dataset Overview")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Rows",
        df.shape[0]
    )

    col2.metric(
        "Columns",
        df.shape[1]
    )

    col3.metric(
        "Missing Values",
        int(df.isnull().sum().sum())
    )

    col4.metric(
        "Duplicate Rows",
        int(df.duplicated().sum())
    )

    st.subheader("Dataset Preview")

    st.dataframe(
        df.head(20),
        use_container_width=True
    )

    st.subheader("Dataset Information")

    info_df = pd.DataFrame(
        {
            "Column": df.columns,
            "Data Type": df.dtypes.astype(str),
            "Missing Values": df.isnull().sum().values,
            "Unique Values": df.nunique().values
        }
    )

    st.dataframe(
        info_df,
        use_container_width=True
    )


# ============================================================
# 8. EDA
# ============================================================

elif page == "EDA":

    st.header("📈 Exploratory Data Analysis")

    plot_path = os.path.join(
        OUTPUT_PATH,
        "plots"
    )

    if not os.path.exists(plot_path):

        st.warning(
            "EDA plots not found. Run src/module1/eda.py first."
        )

    else:

        plot_files = [
            file
            for file in os.listdir(plot_path)
            if file.endswith(".png")
        ]

        if len(plot_files) == 0:

            st.warning(
                "No EDA plots found."
            )

        else:

            selected_plot = st.selectbox(
                "Select EDA Plot",
                plot_files
            )

            st.image(
                os.path.join(
                    plot_path,
                    selected_plot
                ),
                use_container_width=True
            )


# ============================================================
# 9. PREPROCESSING
# ============================================================

elif page == "Preprocessing":

    st.header("🧹 Preprocessing Results")

    preprocessing_files = [
        "preprocessed_data.csv",
        "standard_scaled_data.csv",
        "minmax_scaled_data.csv",
        "robust_scaled_data.csv",
        "one_hot_encoded_data.csv"
    ]

    selected_file = st.selectbox(
        "Select preprocessing output",
        preprocessing_files
    )

    file_path = os.path.join(
        OUTPUT_PATH,
        selected_file
    )

    if os.path.exists(file_path):

        processed_df = pd.read_csv(
            file_path
        )

        st.write(
            "Shape:",
            processed_df.shape
        )

        st.dataframe(
            processed_df.head(20),
            use_container_width=True
        )

    else:

        st.warning(
            f"{selected_file} not found. Run the preprocessing module first."
        )


# ============================================================
# 10. REGRESSION MODELS
# ============================================================

elif page == "Regression Models":

    st.header("📉 Regression Models")

    regression_files = {
        "Linear Regression": "linear_regression_predictions.csv",
        "Ridge Regression": "ridge_regression_predictions.csv",
        "Lasso Regression": "lasso_regression_predictions.csv"
    }

    selected_model = st.selectbox(
        "Select Regression Model",
        list(regression_files.keys())
    )

    file_path = os.path.join(
        OUTPUT_PATH,
        regression_files[selected_model]
    )

    if os.path.exists(file_path):

        results = pd.read_csv(
            file_path
        )

        st.subheader(
            selected_model
        )

        st.dataframe(
            results.head(20),
            use_container_width=True
        )

        st.line_chart(
            results.head(100)
        )

    else:

        st.warning(
            "Prediction file not found. Run the corresponding model first."
        )


# ============================================================
# 11. CLASSIFICATION
# ============================================================

elif page == "Classification":

    st.header("🎯 Logistic Regression")

    file_path = os.path.join(
        OUTPUT_PATH,
        "logistic_regression_predictions.csv"
    )

    if os.path.exists(file_path):

        results = pd.read_csv(
            file_path
        )

        st.dataframe(
            results.head(20),
            use_container_width=True
        )

        if "Actual Class" in results.columns:

            st.subheader(
                "Class Distribution"
            )

            st.bar_chart(
                results["Actual Class"].value_counts()
            )

    else:

        st.warning(
            "Logistic Regression results not found. "
            "Run logistic_regression.py first."
        )


# ============================================================
# 12. TREE & BOOSTING MODELS
# ============================================================

elif page == "Tree & Boosting Models":

    st.header("🌳 Tree & Boosting Models")

    model_files = {
        "Decision Tree": "decision_tree_predictions.csv",
        "Random Forest": "random_forest_predictions.csv",
        "AdaBoost": "adaboost_predictions.csv",
        "Gradient Boosting": "gradient_boosting_predictions.csv",
        "XGBoost": "xgboost_predictions.csv",
        "LightGBM": "lightgbm_predictions.csv"
    }

    selected_model = st.selectbox(
        "Select Model",
        list(model_files.keys())
    )

    file_path = os.path.join(
        OUTPUT_PATH,
        model_files[selected_model]
    )

    if os.path.exists(file_path):

        results = pd.read_csv(
            file_path
        )

        st.subheader(
            selected_model
        )

        st.dataframe(
            results.head(20),
            use_container_width=True
        )

        st.line_chart(
            results.head(100)
        )

    else:

        st.warning(
            "Prediction file not found. "
            "Run the selected model first."
        )


# ============================================================
# 13. CLUSTERING
# ============================================================

elif page == "Clustering":

    st.header("🔵 Clustering")

    clustering_type = st.selectbox(
        "Select Clustering Method",
        [
            "K-Means",
            "DBSCAN"
        ]
    )

    if clustering_type == "K-Means":

        file_path = os.path.join(
            OUTPUT_PATH,
            "kmeans_clustered_data.csv"
        )

        image_path = os.path.join(
            OUTPUT_PATH,
            "kmeans_glucose_stress_clusters.png"
        )

        summary_path = os.path.join(
            OUTPUT_PATH,
            "kmeans_cluster_centers.csv"
        )

    else:

        file_path = os.path.join(
            OUTPUT_PATH,
            "dbscan_clustered_data.csv"
        )

        image_path = os.path.join(
            OUTPUT_PATH,
            "dbscan_glucose_stress_clusters.png"
        )

        summary_path = os.path.join(
            OUTPUT_PATH,
            "dbscan_cluster_summary.csv"
        )

    if os.path.exists(file_path):

        cluster_df = pd.read_csv(
            file_path
        )

        st.subheader(
            f"{clustering_type} Results"
        )

        st.dataframe(
            cluster_df.head(20),
            use_container_width=True
        )

        if "cluster" in cluster_df.columns:

            st.subheader(
                "Cluster Distribution"
            )

            st.bar_chart(
                cluster_df["cluster"].value_counts().sort_index()
            )

        if os.path.exists(image_path):

            st.subheader(
                "Cluster Visualization"
            )

            st.image(
                image_path,
                use_container_width=True
            )

        if os.path.exists(summary_path):

            st.subheader(
                "Cluster Summary"
            )

            summary_df = pd.read_csv(
                summary_path
            )

            st.dataframe(
                summary_df,
                use_container_width=True
            )

    else:

        st.warning(
            f"{clustering_type} results not found. "
            "Run the corresponding clustering file first."
        )


# ============================================================
# 14. PCA
# ============================================================

elif page == "PCA":

    st.header("📐 Principal Component Analysis")

    pca_file = os.path.join(
        OUTPUT_PATH,
        "pca_transformed_data.csv"
    )

    components_file = os.path.join(
        OUTPUT_PATH,
        "pca_components.csv"
    )

    scatter_file = os.path.join(
        OUTPUT_PATH,
        "pca_scatter_plot.png"
    )

    variance_file = os.path.join(
        OUTPUT_PATH,
        "pca_explained_variance.png"
    )

    if os.path.exists(pca_file):

        pca_df = pd.read_csv(
            pca_file
        )

        st.subheader(
            "PCA Transformed Data"
        )

        st.dataframe(
            pca_df.head(20),
            use_container_width=True
        )

        if os.path.exists(scatter_file):

            st.subheader(
                "PCA Scatter Plot"
            )

            st.image(
                scatter_file,
                use_container_width=True
            )

        if os.path.exists(variance_file):

            st.subheader(
                "Explained Variance"
            )

            st.image(
                variance_file,
                use_container_width=True
            )

        if os.path.exists(components_file):

            st.subheader(
                "PCA Components"
            )

            components_df = pd.read_csv(
                components_file
            )

            st.dataframe(
                components_df,
                use_container_width=True
            )

    else:

        st.warning(
            "PCA results not found. Run pca.py first."
        )


# ============================================================
# 15. MODEL COMPARISON
# ============================================================

elif page == "Model Comparison":

    st.header("📊 Model Comparison")

    model_files = {
        "Linear Regression": "linear_regression_predictions.csv",
        "Ridge Regression": "ridge_regression_predictions.csv",
        "Lasso Regression": "lasso_regression_predictions.csv",
        "Decision Tree": "decision_tree_predictions.csv",
        "Random Forest": "random_forest_predictions.csv",
        "AdaBoost": "adaboost_predictions.csv",
        "Gradient Boosting": "gradient_boosting_predictions.csv",
        "XGBoost": "xgboost_predictions.csv",
        "LightGBM": "lightgbm_predictions.csv"
    }

    comparison = []

    for model_name, filename in model_files.items():

        file_path = os.path.join(
            OUTPUT_PATH,
            filename
        )

        if os.path.exists(file_path):

            results = pd.read_csv(
                file_path
            )

            if (
                "Actual Glucose" in results.columns
                and "Predicted Glucose" in results.columns
            ):

                actual = results[
                    "Actual Glucose"
                ]

                predicted = results[
                    "Predicted Glucose"
                ]

                mae = np.mean(
                    np.abs(
                        actual - predicted
                    )
                )

                mse = np.mean(
                    (
                        actual - predicted
                    ) ** 2
                )

                rmse = np.sqrt(
                    mse
                )

                comparison.append(
                    {
                        "Model": model_name,
                        "MAE": round(mae, 4),
                        "MSE": round(mse, 4),
                        "RMSE": round(rmse, 4)
                    }
                )

    if len(comparison) > 0:

        comparison_df = pd.DataFrame(
            comparison
        )

        st.dataframe(
            comparison_df,
            use_container_width=True
        )

        st.subheader(
            "RMSE Comparison"
        )

        chart_df = comparison_df.set_index(
            "Model"
        )[
            ["RMSE"]
        ]

        st.bar_chart(
            chart_df
        )

    else:

        st.warning(
            "No model prediction files were found."
        )


# ============================================================
# 16. FOOTER
# ============================================================

st.sidebar.markdown("---")

st.sidebar.info(
    "GlucoBench ML Project"
)
