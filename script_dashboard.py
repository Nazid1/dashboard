
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from mlxtend.frequent_patterns import apriori, association_rules

st.set_page_config(page_title="Student Analytics Dashboard", layout="wide")
st.title("Student Academic Performance Analytics System")
st.write("Interactive dashboard with filters, at-risk analysis, classification, clustering, association rules, and regression.")

def encode_features(X):
    X = X.copy()
    for col in X.columns:
        if not pd.api.types.is_numeric_dtype(X[col]):
            X[col] = X[col].astype(str)
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col])
    return X

def remove_id_like_columns(X):
    X = X.copy()
    bad_cols = [col for col in X.columns if "id" in col.lower() or "student" in col.lower()]
    X = X.drop(columns=bad_cols, errors="ignore")
    return X

def fill_missing_values(df):
    df = df.copy()
    for col in df.columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            if df[col].isnull().sum() > 0:
                mode_vals = df[col].mode()
                if len(mode_vals) > 0:
                    df[col] = df[col].fillna(mode_vals[0])
                else:
                    df[col] = df[col].fillna("Unknown")
        else:
            if df[col].isnull().sum() > 0:
                df[col] = df[col].fillna(df[col].median())
    return df

uploaded_file = st.sidebar.file_uploader("Upload CSV dataset", type=["csv"])

if uploaded_file is None:
    st.info("Please upload a CSV file from the sidebar to begin.")
    st.stop()

df = pd.read_csv(uploaded_file)

if "final_exam_score" in df.columns:
    if "pass_fail" not in df.columns:
        df["pass_fail"] = np.where(df["final_exam_score"] >= 50, "Pass", "Fail")
    if "grade_category" not in df.columns:
        bins = [-1, 39, 49, 59, 69, 79, 100]
        labels = ["F", "E", "D", "C", "B", "A"]
        df["grade_category"] = pd.cut(df["final_exam_score"], bins=bins, labels=labels)

if "pass_fail" in df.columns:
    df["at_risk"] = np.where(df["pass_fail"].astype(str) == "Fail", "Yes", "No")

df = fill_missing_values(df)

st.sidebar.header("Filters")
filtered_df = df.copy()

if "gender" in filtered_df.columns:
    gender_options = sorted(filtered_df["gender"].astype(str).dropna().unique().tolist())
    selected_gender = st.sidebar.multiselect("Gender", gender_options, default=gender_options)
    if selected_gender:
        filtered_df = filtered_df[filtered_df["gender"].astype(str).isin(selected_gender)]

if "at_risk" in filtered_df.columns:
    risk_filter = st.sidebar.selectbox("Risk Group", ["All Students", "At-Risk Only", "Not At-Risk"])
    if risk_filter == "At-Risk Only":
        filtered_df = filtered_df[filtered_df["at_risk"] == "Yes"]
    elif risk_filter == "Not At-Risk":
        filtered_df = filtered_df[filtered_df["at_risk"] == "No"]

for col in ["attendance_rate", "study_hours", "social_media_hours", "sleep_hours", "final_exam_score"]:
    if col in filtered_df.columns and pd.api.types.is_numeric_dtype(filtered_df[col]):
        min_val = float(filtered_df[col].min())
        max_val = float(filtered_df[col].max())
        if min_val != max_val:
            selected_range = st.sidebar.slider(
                f"{col} range",
                min_value=min_val,
                max_value=max_val,
                value=(min_val, max_val)
            )
            filtered_df = filtered_df[
                (filtered_df[col] >= selected_range[0]) &
                (filtered_df[col] <= selected_range[1])
            ]

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Overview",
    "At-Risk Insights",
    "Classification",
    "Clustering",
    "Regression & Rules"
])

with tab1:
    st.header("Dataset Overview")
    st.dataframe(filtered_df.head(20), use_container_width=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", len(filtered_df))
    c2.metric("Columns", filtered_df.shape[1])
    c3.metric("Missing Values", int(filtered_df.isnull().sum().sum()))

    st.subheader("Missing Values by Column")
    st.write(filtered_df.isnull().sum())

    numeric_cols = filtered_df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) > 0:
        st.subheader("Summary Statistics")
        st.dataframe(filtered_df[numeric_cols].describe(), use_container_width=True)

        st.subheader("Correlation Matrix")
        st.dataframe(filtered_df[numeric_cols].corr(), use_container_width=True)

    if "study_hours" in filtered_df.columns and "final_exam_score" in filtered_df.columns:
        fig, ax = plt.subplots()
        ax.scatter(filtered_df["study_hours"], filtered_df["final_exam_score"])
        ax.set_xlabel("Study Hours")
        ax.set_ylabel("Final Exam Score")
        ax.set_title("Study Hours vs Final Exam Score")
        st.pyplot(fig)

    if "attendance_rate" in filtered_df.columns and "final_exam_score" in filtered_df.columns:
        fig, ax = plt.subplots()
        ax.scatter(filtered_df["attendance_rate"], filtered_df["final_exam_score"])
        ax.set_xlabel("Attendance Rate")
        ax.set_ylabel("Final Exam Score")
        ax.set_title("Attendance Rate vs Final Exam Score")
        st.pyplot(fig)

with tab2:
    st.header("At-Risk Student Analysis")

    if "at_risk" in filtered_df.columns:
        total_students = len(filtered_df)
        at_risk_count = int((filtered_df["at_risk"] == "Yes").sum())
        not_at_risk_count = int((filtered_df["at_risk"] == "No").sum())
        risk_pct = (at_risk_count / total_students * 100) if total_students > 0 else 0

        c1, c2, c3 = st.columns(3)
        c1.metric("Total Students", total_students)
        c2.metric("At-Risk Students", at_risk_count)
        c3.metric("At-Risk %", f"{risk_pct:.2f}%")

        risk_summary = pd.DataFrame({
            "Category": ["At-Risk", "Not At-Risk"],
            "Count": [at_risk_count, not_at_risk_count]
        })
        st.subheader("Risk Summary")
        st.dataframe(risk_summary, use_container_width=True)

        if "gender" in filtered_df.columns:
            st.subheader("At-Risk Rate by Gender")
            risk_by_gender = (
                filtered_df.assign(at_risk_flag=np.where(filtered_df["at_risk"] == "Yes", 1, 0))
                .groupby("gender")["at_risk_flag"]
                .mean()
                .mul(100)
                .round(2)
                .reset_index()
                .rename(columns={"at_risk_flag": "At-Risk Percentage"})
            )
            st.dataframe(risk_by_gender, use_container_width=True)

        st.subheader("At-Risk Students Table")
        st.dataframe(filtered_df[filtered_df["at_risk"] == "Yes"], use_container_width=True)
    else:
        st.warning("At-risk analysis is not available because pass/fail information is missing.")

with tab3:
    st.header("Classification: Predict Pass/Fail")

    if "pass_fail" in filtered_df.columns:
        model_choice = st.selectbox("Choose classification model", ["Decision Tree", "Random Forest"])

        X = filtered_df.drop(columns=["pass_fail", "grade_category", "at_risk"], errors="ignore")
        X = X.drop(columns=["final_exam_score"], errors="ignore")
        X = remove_id_like_columns(X)
        X = encode_features(X)

        y = filtered_df["pass_fail"].astype(str)
        y = LabelEncoder().fit_transform(y)

        if len(filtered_df) >= 10 and len(np.unique(y)) > 1:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )

            if model_choice == "Decision Tree":
                clf = DecisionTreeClassifier(random_state=42, max_depth=5)
            else:
                clf = RandomForestClassifier(random_state=42, n_estimators=100)

            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)

            st.write(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")

            fig, ax = plt.subplots()
            ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred)).plot(ax=ax)
            ax.set_title("Confusion Matrix")
            st.pyplot(fig)
        else:
            st.warning("Not enough filtered data or only one class remains after filtering.")
    else:
        st.warning("Classification cannot run because pass/fail is not available.")

with tab4:
    st.header("Clustering: Student Grouping")

    numeric_cols = filtered_df.select_dtypes(include=[np.number]).columns.tolist()
    default_cluster_cols = [
        c for c in ["study_hours", "attendance_rate", "social_media_hours", "sleep_hours", "final_exam_score"]
        if c in filtered_df.columns
    ]

    cluster_cols = st.multiselect(
        "Select clustering columns",
        numeric_cols,
        default=default_cluster_cols
    )

    if len(cluster_cols) >= 2:
        k = st.slider("Choose number of clusters (k)", 2, 5, 3)

        cluster_data = filtered_df[cluster_cols].copy()
        for col in cluster_data.columns:
            if cluster_data[col].isnull().sum() > 0:
                cluster_data[col] = cluster_data[col].fillna(cluster_data[col].median())

        scaler = StandardScaler()
        scaled = scaler.fit_transform(cluster_data)

        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        cluster_labels = km.fit_predict(scaled)

        cluster_output = filtered_df.copy()
        cluster_output["cluster"] = cluster_labels

        st.subheader("Cluster Counts")
        st.write(cluster_output["cluster"].value_counts().sort_index())

        st.subheader("Cluster Profiles")
        st.dataframe(cluster_output.groupby("cluster")[cluster_cols].mean().round(2), use_container_width=True)

        fig, ax = plt.subplots()
        ax.scatter(cluster_output[cluster_cols[0]], cluster_output[cluster_cols[1]], c=cluster_output["cluster"])
        ax.set_xlabel(cluster_cols[0])
        ax.set_ylabel(cluster_cols[1])
        ax.set_title("Student Clusters")
        st.pyplot(fig)
    else:
        st.info("Select at least two numeric columns for clustering.")

with tab5:
    st.header("Regression and Association Rules")

    left, right = st.columns(2)

    with left:
        st.subheader("Regression: Predict Final Exam Score")

        if "final_exam_score" in filtered_df.columns:
            X = filtered_df.drop(columns=["final_exam_score", "grade_category", "at_risk"], errors="ignore")
            X = remove_id_like_columns(X)
            X = encode_features(X)

            y = filtered_df["final_exam_score"]

            if len(filtered_df) >= 10:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42
                )

                reg = LinearRegression()
                reg.fit(X_train, y_train)
                y_pred = reg.predict(X_test)

                st.write(f"MAE: {mean_absolute_error(y_test, y_pred):.2f}")
                st.write(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")
                st.write(f"R²: {r2_score(y_test, y_pred):.3f}")

                fig, ax = plt.subplots()
                ax.scatter(y_test, y_pred)
                ax.set_xlabel("Actual Final Exam Score")
                ax.set_ylabel("Predicted Final Exam Score")
                ax.set_title("Actual vs Predicted Scores")
                st.pyplot(fig)
            else:
                st.warning("Not enough filtered data for regression.")
        else:
            st.warning("Regression cannot run because final_exam_score is missing.")

    with right:
        st.subheader("Association Rule Mining")

        if st.checkbox("Run association rules"):
            assoc_df = pd.DataFrame()
            selected_cols = [c for c in ["study_hours", "attendance_rate", "social_media_hours", "pass_fail"] if c in filtered_df.columns]

            for col in selected_cols:
                if col == "pass_fail":
                    assoc_df["Pass"] = filtered_df[col].astype(str) == "Pass"
                    assoc_df["Fail"] = filtered_df[col].astype(str) == "Fail"
                else:
                    if pd.api.types.is_numeric_dtype(filtered_df[col]):
                        median_val = filtered_df[col].median()
                        assoc_df[f"{col}_High"] = filtered_df[col] >= median_val
                        assoc_df[f"{col}_Low"] = filtered_df[col] < median_val

            if assoc_df.shape[1] > 0:
                freq_items = apriori(assoc_df.astype(bool), min_support=0.1, use_colnames=True)

                if not freq_items.empty:
                    rules = association_rules(freq_items, metric="confidence", min_threshold=0.6)
                    if not rules.empty:
                        rules["antecedents"] = rules["antecedents"].apply(lambda x: ", ".join(list(x)))
                        rules["consequents"] = rules["consequents"].apply(lambda x: ", ".join(list(x)))
                        st.dataframe(
                            rules[["antecedents", "consequents", "support", "confidence", "lift"]].head(10),
                            use_container_width=True
                        )
                    else:
                        st.warning("No rules found.")
                else:
                    st.warning("No frequent itemsets found.")
            else:
                st.warning("No suitable columns available for association rules.")
