# =========================
# SAFE IMPORTS
# =========================
try:
    from mlxtend.frequent_patterns import apriori, association_rules
    mlxtend_available = True
except ImportError:
    mlxtend_available = False

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

# =========================
# PAGE SETUP
# =========================
st.set_page_config(page_title="Student Analytics Dashboard", layout="wide")
st.title("Student Academic Performance Analytics System")

# =========================
# HELPER FUNCTIONS
# =========================
def encode_features(X):
    X = X.copy()
    for col in X.columns:
        if not pd.api.types.is_numeric_dtype(X[col]):
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
    return X

def remove_id_like_columns(X):
    return X.drop(
        columns=[col for col in X.columns if "id" in col.lower() or "student" in col.lower()],
        errors="ignore"
    )

def fill_missing_values(df):
    df = df.copy()
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].fillna(df[col].median())
            else:
                df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else "Unknown")
    return df

# =========================
# DATA LOADING
# =========================
uploaded_file = st.sidebar.file_uploader("Upload CSV dataset", type=["csv"])

if uploaded_file is None:
    st.info("Upload a dataset to begin.")
    st.stop()

df = pd.read_csv(uploaded_file)
df = fill_missing_values(df)

# Create derived columns
if "final_exam_score" in df.columns:
    df["pass_fail"] = np.where(df["final_exam_score"] >= 50, "Pass", "Fail")
    df["at_risk"] = np.where(df["pass_fail"] == "Fail", "Yes", "No")

# =========================
# FILTERS
# =========================
st.sidebar.header("Filters")
filtered_df = df.copy()

if "gender" in df.columns:
    options = df["gender"].dropna().unique()
    selected = st.sidebar.multiselect("Gender", options, default=options)
    filtered_df = filtered_df[filtered_df["gender"].isin(selected)]

if "at_risk" in df.columns:
    risk = st.sidebar.selectbox("Risk", ["All", "At-Risk", "Not At-Risk"])
    if risk == "At-Risk":
        filtered_df = filtered_df[filtered_df["at_risk"] == "Yes"]
    elif risk == "Not At-Risk":
        filtered_df = filtered_df[filtered_df["at_risk"] == "No"]

# =========================
# TABS
# =========================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Overview", "At-Risk", "Classification", "Clustering", "Regression & Rules"
])

# =========================
# TAB 1: OVERVIEW
# =========================
with tab1:
    st.subheader("Dataset Preview")
    st.dataframe(filtered_df.head(), use_container_width=True)

    st.write("Summary Stats")
    st.dataframe(filtered_df.describe(), use_container_width=True)

# =========================
# TAB 2: AT RISK
# =========================
with tab2:
    if "at_risk" in filtered_df.columns:
        st.write(filtered_df["at_risk"].value_counts())

# =========================
# TAB 3: CLASSIFICATION
# =========================
with tab3:
    if "pass_fail" in filtered_df.columns:
        X = filtered_df.drop(columns=["pass_fail", "final_exam_score"], errors="ignore")
        X = encode_features(remove_id_like_columns(X))
        y = LabelEncoder().fit_transform(filtered_df["pass_fail"])

        if len(set(y)) > 1:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

            model = DecisionTreeClassifier(max_depth=5)
            model.fit(X_train, y_train)

            preds = model.predict(X_test)

            st.write("Accuracy:", accuracy_score(y_test, preds))

            fig, ax = plt.subplots()
            ConfusionMatrixDisplay(confusion_matrix(y_test, preds)).plot(ax=ax)
            st.pyplot(fig)

# =========================
# TAB 4: CLUSTERING
# =========================
with tab4:
    numeric = filtered_df.select_dtypes(include=np.number)

    if numeric.shape[1] >= 2:
        k = st.slider("Clusters", 2, 5, 3)
        scaled = StandardScaler().fit_transform(numeric)

        km = KMeans(n_clusters=k, n_init=10)
        labels = km.fit_predict(scaled)

        numeric["cluster"] = labels
        st.write(numeric.groupby("cluster").mean())

# =========================
# TAB 5: REGRESSION + RULES
# =========================
with tab5:

    # REGRESSION
    if "final_exam_score" in filtered_df.columns:
        X = filtered_df.drop(columns=["final_exam_score"], errors="ignore")
        X = encode_features(remove_id_like_columns(X))
        y = filtered_df["final_exam_score"]

        X_train, X_test, y_train, y_test = train_test_split(X, y)

        model = LinearRegression()
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        st.write("R²:", r2_score(y_test, preds))

    # =========================
    # ASSOCIATION RULES
    # =========================
    st.subheader("Association Rule Mining")

    if not mlxtend_available:
        st.warning("mlxtend not installed — rules unavailable.")
    else:
        try:
            if all(col in filtered_df.columns for col in ["attendance_rate", "social_media_hours", "final_exam_score"]):

                df_rules = filtered_df.copy()

                df_rules["attendance_level"] = pd.cut(df_rules["attendance_rate"], [0,70,85,100], labels=["Low","Medium","High"])
                df_rules["social_level"] = pd.cut(df_rules["social_media_hours"], [0,2,5,10], labels=["Low","Medium","High"])
                df_rules["result"] = np.where(df_rules["final_exam_score"] >= 50, "Pass", "Fail")

                df_rules = df_rules[["attendance_level","social_level","result"]].dropna()

                encoded = pd.get_dummies(df_rules)

                freq = apriori(encoded, min_support=0.1, use_colnames=True)

                if not freq.empty:
                    rules = association_rules(freq, metric="confidence", min_threshold=0.6)

                    if not rules.empty:
                        st.dataframe(rules[["antecedents","consequents","support","confidence","lift"]])
                    else:
                        st.info("No strong rules found.")
                else:
                    st.info("No frequent patterns found.")

        except Exception as e:
            st.error(f"Rules failed: {e}")
