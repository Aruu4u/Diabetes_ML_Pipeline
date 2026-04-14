import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from sklearn.model_selection import train_test_split, StratifiedKFold, KFold, cross_validate
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, r2_score, mean_squared_error, mean_absolute_error
)

# ─────────────────────────── CONFIG ───────────────────────────
st.set_page_config(page_title="ML Pipeline Dashboard", layout="wide")

st.markdown("""
<style>
.main { background-color: #f5f7f9; }
[data-testid="stMetric"] {
    background-color: #ffffff;
    padding: 15px;
    border-radius: 10px;
    box-shadow: 0px 2px 10px rgba(0,0,0,0.05);
}
.info-box {
    background-color: #eef4ff;
    border-left: 4px solid #4a7cff;
    padding: 12px 16px;
    border-radius: 6px;
    margin-bottom: 16px;
    font-size: 0.93em;
    color: #1a1a2e;
}
</style>
""", unsafe_allow_html=True)

def info_box(text):
    st.markdown(f'<div class="info-box">ℹ️ {text}</div>', unsafe_allow_html=True)

# ─────────────────────────── HELPERS ───────────────────────────
def detect_task_type(series, threshold=10):
    if series.dtype == object or str(series.dtype) == "category":
        return "classification"
    return "classification" if series.nunique() <= threshold else "regression"

def encode_target(y):
    if y.dtype == object or str(y.dtype) == "category":
        le = LabelEncoder()
        return pd.Series(le.fit_transform(y), name=y.name), le
    return y, None

def get_per_col_outliers(df_numeric):
    Q1 = df_numeric.quantile(0.25)
    Q3 = df_numeric.quantile(0.75)
    IQR = Q3 - Q1
    counts = {}
    for col in df_numeric.columns:
        mask = (df_numeric[col] < (Q1[col] - 1.5 * IQR[col])) | \
               (df_numeric[col] > (Q3[col] + 1.5 * IQR[col]))
        counts[col] = int(mask.sum())
    return pd.Series(counts)

# ─────────────────────────── SIDEBAR ───────────────────────────
st.sidebar.title("🛠️ ML Workspace")
uploaded_file = st.sidebar.file_uploader("📁 Upload your CSV dataset", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.sidebar.success("Dataset loaded ✅")
else:
    st.title("👋 Welcome to the ML Pipeline Dashboard")
    st.markdown("""
    Upload a **CSV file** using the sidebar on the left to begin your machine learning workflow.

    **The pipeline covers:**
    1. 📊 **Dashboard** — Dataset overview & statistics
    2. 🔍 **EDA** — Distributions & correlations
    3. 🧹 **Data Cleaning** — Handle missing values & outliers
    4. 🎯 **Feature Selection** — Pick the most useful columns
    5. 🤖 **Model Training** — Train, evaluate & validate your model
    """)
    st.stop()

page = st.sidebar.selectbox("Navigate", [
    "Dashboard",
    "EDA",
    "Data Cleaning",
    "Feature Selection",
    "Model Training",
])

# ═══════════════════════════ DASHBOARD ═══════════════════════════
if page == "Dashboard":
    st.title("📊 Dataset Overview")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Rows", len(df))
    col2.metric("Total Columns", len(df.columns))
    col3.metric("Missing Values", int(df.isnull().sum().sum()))
    col4.metric("Numeric Columns", len(df.select_dtypes(include=np.number).columns))

    st.subheader("Raw Data Preview")
    st.dataframe(df.head(10), use_container_width=True)

    st.subheader("Summary Statistics")
    st.dataframe(df.describe().round(3), use_container_width=True)


# ═══════════════════════════ EDA ═══════════════════════════
elif page == "EDA":
    st.title("🔍 Exploratory Data Analysis")

    numeric_df = df.select_dtypes(include=np.number)

    if numeric_df.empty:
        st.warning("No numeric columns found in this dataset.")
        st.stop()

    st.subheader("Distribution Plot")
    info_box(
        "Select any numeric column to see how its values are spread out. "
        "The histogram shows frequency of value ranges; the box plot above shows median, quartiles, and outliers."
    )
    feature = st.selectbox("Select a column to inspect", numeric_df.columns)
    fig = px.histogram(numeric_df, x=feature, marginal="box",
                       color_discrete_sequence=["#FF4B4B"],
                       title=f"Distribution of {feature}")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Correlation Matrix")
    info_box(
        "Correlation measures how strongly two columns move together. "
        "Values near +1 = strong positive relationship, near -1 = strong negative, near 0 = little relationship."
    )
    fig2 = px.imshow(numeric_df.corr().round(2), text_auto=True,
                     color_continuous_scale="RdBu_r", aspect="auto",
                     title="Correlation Heatmap")
    st.plotly_chart(fig2, use_container_width=True)


# # ═══════════════════════════ DATA CLEANING ═══════════════════════════
# elif page == "Data Cleaning":
#     st.title("🧹 Data Cleaning")

#     info_box(
#         "Before training any model, we need clean data. "
#         "This step lets you handle missing values (blank cells) and outliers (extreme values that can distort learning). "
#         "Choose a strategy for each, then click Apply Cleaning to see the result."
#     )

#     numeric_df_raw = df.select_dtypes(include=np.number)

#     # ── Before-cleaning stats ──
#     st.subheader("Before Cleaning — Current Data Issues")

#     col_mv, col_out = st.columns(2)

#     with col_mv:
#         st.markdown("**Missing Values per Column**")
#         missing = df.isnull().sum()
#         missing = missing[missing > 0]
#         if missing.empty:
#             st.success("No missing values found!")
#         else:
#             st.dataframe(
#                 missing.rename("Missing Count")
#                        .reset_index()
#                        .rename(columns={"index": "Column"}),
#                 use_container_width=True
#             )

#     with col_out:
#         st.markdown("**Outliers per Numeric Column** *(IQR method)*")
#         if numeric_df_raw.empty:
#             st.info("No numeric columns to check.")
#         else:
#             outlier_counts = get_per_col_outliers(numeric_df_raw)
#             outlier_counts = outlier_counts[outlier_counts > 0]
#             if outlier_counts.empty:
#                 st.success("No outliers detected!")
#             else:
#                 st.dataframe(
#                     outlier_counts.rename("Outlier Count")
#                                   .reset_index()
#                                   .rename(columns={"index": "Column"}),
#                     use_container_width=True
#                 )

#     st.markdown("---")
#     st.subheader("Cleaning Options")

#     col1, col2 = st.columns(2)
#     with col1:
#         missing_option = st.selectbox(
#             "Missing Value Strategy",
#             ["None", "Drop Rows with Missing", "Fill with Mean", "Fill with Median", "Fill with Zero"],
#             help="Choose how to handle blank/null cells."
#         )
#     with col2:
#         outlier_option = st.selectbox(
#             "Outlier Handling Strategy",
#             ["None", "Remove Outlier Rows (IQR)"],
#             help="IQR removes rows where any numeric value falls far outside the typical range."
#         )

#     if st.button("✅ Apply Cleaning"):
#         df_clean = df.copy()

#         # Missing values
#         if missing_option == "Drop Rows with Missing":
#             df_clean = df_clean.dropna()
#         elif missing_option == "Fill with Mean":
#             num_cols = df_clean.select_dtypes(include=np.number).columns
#             df_clean[num_cols] = df_clean[num_cols].fillna(df_clean[num_cols].mean())
#         elif missing_option == "Fill with Median":
#             num_cols = df_clean.select_dtypes(include=np.number).columns
#             df_clean[num_cols] = df_clean[num_cols].fillna(df_clean[num_cols].median())
#         elif missing_option == "Fill with Zero":
#             num_cols = df_clean.select_dtypes(include=np.number).columns
#             df_clean[num_cols] = df_clean[num_cols].fillna(0)

#         # Outliers
#         if outlier_option == "Remove Outlier Rows (IQR)":
#             numeric_part = df_clean.select_dtypes(include=np.number)
#             Q1 = numeric_part.quantile(0.25)
#             Q3 = numeric_part.quantile(0.75)
#             IQR = Q3 - Q1
#             mask = ~((numeric_part < (Q1 - 1.5 * IQR)) | (numeric_part > (Q3 + 1.5 * IQR))).any(axis=1)
#             df_clean = df_clean[mask]

#         # Keep numeric only for ML
#         df_clean_num = df_clean.select_dtypes(include=np.number)
#         st.session_state["df_clean"] = df_clean_num

#         # ── After-cleaning summary ──
#         st.subheader("After Cleaning")

#         c1, c2, c3, c4 = st.columns(4)
#         c1.metric("Original Rows", len(df))
#         c2.metric("Cleaned Rows", len(df_clean_num),
#                   delta=f"{len(df_clean_num) - len(df)} rows", delta_color="off")
#         c3.metric("Remaining Missing", int(df_clean_num.isnull().sum().sum()))
#         c4.metric("Numeric Features", len(df_clean_num.columns))

#         st.markdown("**Cleaned Dataset Preview**")
#         st.dataframe(df_clean_num.head(10), use_container_width=True)

#         st.success("✅ Cleaning applied and saved. Proceed to Feature Selection.")

#     else:
#         if "df_clean" in st.session_state:
#             st.info("A previously cleaned dataset is saved. Change options and click **Apply Cleaning** to update.")
#         else:
#             st.info("Select your strategies above and click **Apply Cleaning**.")
# ═══════════════════════════ DATA CLEANING ═══════════════════════════
elif page == "Data Cleaning":
    st.title("🧹 Data Cleaning")

    info_box(
        "Before training any model, we need clean data. "
        "This step handles missing values, outliers, and invalid zero values (like 0 glucose, 0 BMI)."
    )

    numeric_df_raw = df.select_dtypes(include=np.number)

    # ── ZERO VALUE DETECTION ──
    st.subheader("⚠️ Biologically Invalid Zero Values Detection")

    zero_counts = {}
    for col in numeric_df_raw.columns:
        zero_counts[col] = int((numeric_df_raw[col] == 0).sum())

    zero_series = pd.Series(zero_counts)
    zero_series = zero_series[zero_series > 0]

    if zero_series.empty:
        st.success("No suspicious zero values found!")
    else:
        st.dataframe(
            zero_series.rename("Zero Count")
                       .reset_index()
                       .rename(columns={"index": "Column"}),
            use_container_width=True
        )

    # ── Before-cleaning stats ──
    st.subheader("Before Cleaning — Current Data Issues")

    col_mv, col_out = st.columns(2)

    with col_mv:
        st.markdown("**Missing Values per Column**")
        missing = df.isnull().sum()
        missing = missing[missing > 0]
        if missing.empty:
            st.success("No missing values found!")
        else:
            st.dataframe(
                missing.rename("Missing Count")
                       .reset_index()
                       .rename(columns={"index": "Column"}),
                use_container_width=True
            )

    with col_out:
        st.markdown("**Outliers per Numeric Column (IQR)**")
        if numeric_df_raw.empty:
            st.info("No numeric columns.")
        else:
            outlier_counts = get_per_col_outliers(numeric_df_raw)
            outlier_counts = outlier_counts[outlier_counts > 0]
            if outlier_counts.empty:
                st.success("No outliers detected!")
            else:
                st.dataframe(
                    outlier_counts.rename("Outlier Count")
                                  .reset_index()
                                  .rename(columns={"index": "Column"}),
                    use_container_width=True
                )

    st.markdown("---")
    st.subheader("Cleaning Options")

    col1, col2, col3 = st.columns(3)

    with col1:
        missing_option = st.selectbox(
            "Missing Value Strategy",
            ["None", "Drop Rows", "Mean", "Median"]
        )

    with col2:
        outlier_option = st.selectbox(
            "Outlier Handling",
            ["None", "Remove IQR"]
        )

    with col3:
        zero_option = st.selectbox(
            "Zero Handling",
            ["None", "Convert Zero → NaN (then fill)"]
        )

    # ── APPLY CLEANING ──
    if st.button("✅ Apply Cleaning"):
        df_clean = df.copy()

        # -------- ZERO HANDLING --------
        if zero_option == "Convert Zero → NaN (then fill)":
            for col in df_clean.select_dtypes(include=np.number).columns:
                df_clean[col] = df_clean[col].replace(0, np.nan)

        # -------- MISSING VALUES --------
        if missing_option == "Drop Rows":
            df_clean = df_clean.dropna()

        elif missing_option == "Mean":
            num_cols = df_clean.select_dtypes(include=np.number).columns
            df_clean[num_cols] = df_clean[num_cols].fillna(df_clean[num_cols].mean())

        elif missing_option == "Median":
            num_cols = df_clean.select_dtypes(include=np.number).columns
            df_clean[num_cols] = df_clean[num_cols].fillna(df_clean[num_cols].median())

        # -------- OUTLIER HANDLING --------
        if outlier_option == "Remove IQR":
            numeric_part = df_clean.select_dtypes(include=np.number)
            Q1 = numeric_part.quantile(0.25)
            Q3 = numeric_part.quantile(0.75)
            IQR = Q3 - Q1
            mask = ~((numeric_part < (Q1 - 1.5 * IQR)) | (numeric_part > (Q3 + 1.5 * IQR))).any(axis=1)
            df_clean = df_clean[mask]

        # -------- FINAL CLEAN DATA --------
        df_clean_num = df_clean.select_dtypes(include=np.number)
        st.session_state["df_clean"] = df_clean_num

        # ── AFTER CLEANING ──
        st.subheader("After Cleaning")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Original Rows", len(df))
        c2.metric("Cleaned Rows", len(df_clean_num))
        c3.metric("Remaining Missing", int(df_clean_num.isnull().sum().sum()))
        c4.metric("Features", len(df_clean_num.columns))

        st.dataframe(df_clean_num.head(), use_container_width=True)

        st.success("✅ Data cleaned successfully!")



# ═══════════════════════════ FEATURE SELECTION ═══════════════════════════
elif page == "Feature Selection":
    st.title("🎯 Feature Selection")

    info_box(
        "A dataset can have many columns, but not all of them are useful for prediction. "
        "Feature Selection identifies and keeps only the columns that matter most — "
        "this improves model accuracy, reduces training time, and cuts out noise."
    )

    df_used = st.session_state.get("df_clean", df.select_dtypes(include=np.number))

    if df_used.shape[1] < 2:
        st.error("Need at least 2 numeric columns. Please complete Data Cleaning first.")
        st.stop()

    # Step 1 — Target
    st.subheader("Step 1 — Choose Your Target Column")
    info_box(
        "The target column is what you want the model to learn to predict — for example, "
        "'Survived' in Titanic, 'Outcome' in a diabetes dataset, or 'Price' in a housing dataset. "
        "All other columns become inputs (features) the model uses to make that prediction."
    )
    target = st.selectbox("Select Target Column", df_used.columns)
    task = detect_task_type(df_used[target])

    task_icon = "🟢" if task == "classification" else "🔵"
    task_explain = (
        f"Since `{target}` has **{df_used[target].nunique()} unique values**, "
        "this is treated as a **Classification** problem — the model will predict a category/class."
        if task == "classification"
        else
        f"Since `{target}` has **{df_used[target].nunique()} unique values**, "
        "this is treated as a **Regression** problem — the model will predict a continuous number."
    )
    st.markdown(f"{task_icon} **Detected Task: {task.upper()}** — {task_explain}")

    X = df_used.drop(columns=[target])
    y = df_used[target]

    # Step 2 — Method
    st.subheader("Step 2 — Choose a Feature Importance Method")
    info_box(
        "**None** — All columns are kept as features with no ranking. \n\n"
        "**Random Forest** — A temporary forest model is trained to score each column by how much "
        "it reduces prediction error. Columns used at early/many decision splits score higher. "
        "Best when relationships between features and target are complex or non-linear.\n\n"
        "**Correlation** — Measures the linear relationship between each feature and the target. "
        "Simple and fast — best for datasets where relationships are roughly linear."
    )
    method = st.selectbox("Feature Importance Method", ["None", "Random Forest", "Correlation"])

    selected_features = list(X.columns)

    if method == "Random Forest":
        try:
            with st.spinner("Computing feature importances..."):
                if task == "classification":
                    y_enc, _ = encode_target(y)
                    rf = RandomForestClassifier(n_estimators=100, random_state=42)
                    rf.fit(X.fillna(0), y_enc)
                else:
                    rf = RandomForestRegressor(n_estimators=100, random_state=42)
                    rf.fit(X.fillna(0), y)

            importance = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=True)
            fig = px.bar(importance, orientation="h", color=importance,
                         color_continuous_scale="Reds",
                         labels={"value": "Importance Score", "index": "Feature"},
                         title="Random Forest Feature Importances (higher = more useful)")
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("Step 3 — How Many Top Features to Keep?")
            info_box(
                "From the ranked list above, slide to choose how many of the top-ranked features to keep. "
                "Using fewer but more relevant features often produces a faster, more accurate model "
                "by removing columns that add noise rather than signal."
            )
            top_n = st.slider("Top N Features to Keep", 1, len(X.columns), min(5, len(X.columns)))
            selected_features = importance.sort_values(ascending=False).head(top_n).index.tolist()

        except Exception as e:
            st.error(f"Random Forest feature selection failed: {e}")

    elif method == "Correlation":
        corr = df_used.corr()[target].abs().drop(target).sort_values(ascending=True)
        fig = px.bar(corr, orientation="h", color=corr,
                     color_continuous_scale="Blues",
                     labels={"value": "Absolute Correlation", "index": "Feature"},
                     title="Absolute Correlation with Target (higher = more related)")
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Step 3 — How Many Top Features to Keep?")
        info_box(
            "Keep the N columns most correlated with the target. "
            "Columns with near-zero correlation contribute very little and can safely be dropped."
        )
        top_n = st.slider("Top N Features to Keep", 1, len(X.columns), min(5, len(X.columns)))
        selected_features = corr.sort_values(ascending=False).head(top_n).index.tolist()

    else:
        st.info("No method selected — all features will be passed to the model.")

    st.markdown("---")
    st.success(f"✅ Features selected for training ({len(selected_features)}): {selected_features}")
    st.session_state["selected_features"] = selected_features
    st.session_state["target"] = target
    st.session_state["task"] = task


# ═══════════════════════════ MODEL TRAINING ═══════════════════════════
elif page == "Model Training":
    st.title("🤖 Model Training")

    info_box(
        "This is where the model is built and rigorously evaluated. "
        "The pipeline: split data → scale features → train model → evaluate on held-out test data → "
        "validate stability with K-Fold cross-validation."
    )

    df_used = st.session_state.get("df_clean", df.select_dtypes(include=np.number))
    features = st.session_state.get("selected_features", list(df_used.columns[:-1]))
    target   = st.session_state.get("target", df_used.columns[-1])
    task     = st.session_state.get("task", detect_task_type(df_used[target]))

    # Configuration summary
    st.subheader("Current Pipeline Configuration")
    info_box(
        f"Task: <b>{task.upper()}</b> &nbsp;|&nbsp; "
        f"Target (what we predict): <b>{target}</b> &nbsp;|&nbsp; "
        f"Input features (what we use): <b>{len(features)} columns</b> → {features}"
    )

    X = df_used[features].fillna(0)
    y = df_used[target]

    if task == "classification":
        y, le = encode_target(y)
        model_options = ["Logistic Regression", "Random Forest"]
    else:
        le = None
        model_options = ["Linear Regression", "Random Forest"]

    st.markdown("---")
    st.subheader("Training Settings")

    col1, col2, col3 = st.columns(3)
    with col1:
        test_size = st.slider(
            "Test Size",
            0.1, 0.4, 0.2, step=0.05,
            help="Fraction of the data held back for testing. 0.2 = 80% train, 20% test."
        )
    with col2:
        k_folds = st.slider(
            "K-Fold Splits",
            2, 10, 5,
            help="Number of cross-validation folds. More folds = more reliable estimate but slower."
        )
    with col3:
        model_type = st.selectbox(
            "Model",
            model_options,
            help="Logistic/Linear Regression: simpler, faster. Random Forest: more powerful, handles complex patterns."
        )

    info_box(
        f"<b>Test Size {int(test_size*100)}%</b>: model trains on "
        f"{int((1-test_size)*len(X))} rows, tested on {int(test_size*len(X))} unseen rows. &nbsp;|&nbsp; "
        f"<b>K-Fold ({k_folds})</b>: the dataset is split into {k_folds} parts; "
        f"the model trains {k_folds} times, each time using a different part as test, "
        f"giving an averaged score that is more reliable than a single train/test split."
    )

    if st.button("🚀 Train Model"):
        with st.spinner("Training model..."):
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42
            )
            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_test_s  = scaler.transform(X_test)
            X_all_s   = scaler.fit_transform(X)

            if task == "classification":
                model = (LogisticRegression(max_iter=1000)
                         if model_type == "Logistic Regression"
                         else RandomForestClassifier(n_estimators=100, random_state=42))
            else:
                model = (LinearRegression()
                         if model_type == "Linear Regression"
                         else RandomForestRegressor(n_estimators=100, random_state=42))

            try:
                model.fit(X_train_s, y_train)
                y_pred = model.predict(X_test_s)

                st.markdown("---")
                st.subheader("📈 Performance Metrics")

                # ── CLASSIFICATION ──
                if task == "classification":
                    avg_strategy = "binary" if y.nunique() == 2 else "weighted"

                    acc  = accuracy_score(y_test, y_pred)
                    prec = precision_score(y_test, y_pred, average=avg_strategy, zero_division=0)
                    rec  = recall_score(y_test, y_pred, average=avg_strategy, zero_division=0)
                    f1   = f1_score(y_test, y_pred, average=avg_strategy, zero_division=0)

                    cv = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
                    cv_res = cross_validate(
                        model, X_all_s, y, cv=cv,
                        scoring=["accuracy", "precision_weighted", "recall_weighted", "f1_weighted"]
                    )
                    avg_acc  = cv_res["test_accuracy"].mean()
                    avg_prec = cv_res["test_precision_weighted"].mean()
                    avg_rec  = cv_res["test_recall_weighted"].mean()
                    avg_f1   = cv_res["test_f1_weighted"].mean()

                    st.markdown("**Test Set Scores** *(single held-out test split)*")
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Accuracy",  f"{acc:.4f}")
                    c2.metric("Precision", f"{prec:.4f}")
                    c3.metric("Recall",    f"{rec:.4f}")
                    c4.metric("F1 Score",  f"{f1:.4f}")

                    info_box(
                        "<b>Accuracy</b> = % of all predictions that are correct. &nbsp;"
                        "<b>Precision</b> = of all positive predictions, how many were truly positive (avoids false alarms). &nbsp;"
                        "<b>Recall</b> = of all actual positives, how many did we correctly identify (avoids missed cases). &nbsp;"
                        "<b>F1 Score</b> = balance between Precision and Recall — most useful when classes are imbalanced."
                    )

                    st.markdown(f"**{k_folds}-Fold Cross-Validation Averages** *(more reliable, averaged across {k_folds} runs)*")
                    k1, k2, k3, k4 = st.columns(4)
                    k1.metric("Avg Accuracy",  f"{avg_acc:.4f}")
                    k2.metric("Avg Precision", f"{avg_prec:.4f}")
                    k3.metric("Avg Recall",    f"{avg_rec:.4f}")
                    k4.metric("Avg F1 Score",  f"{avg_f1:.4f}")

                    # Grouped bar chart
                    metrics_df = pd.DataFrame({
                        "Metric":        ["Accuracy", "Precision", "Recall", "F1 Score"],
                        "Test Score":    [acc,      prec,      rec,    f1],
                        "CV Avg Score":  [avg_acc,  avg_prec,  avg_rec, avg_f1],
                    })
                    fig_bar = px.bar(
                        metrics_df.melt(id_vars="Metric", var_name="Evaluation", value_name="Score"),
                        x="Metric", y="Score", color="Evaluation", barmode="group",
                        color_discrete_sequence=["#FF4B4B", "#4a7cff"],
                        range_y=[0, 1],
                        title="Performance Metrics — Test Set vs K-Fold Average"
                    )
                    st.plotly_chart(fig_bar, use_container_width=True)

                    # Confusion Matrix
                    st.subheader("Confusion Matrix")
                    info_box(
                        "Each cell shows how many samples were predicted as a given class vs their actual class. "
                        "Diagonal = correct predictions. Off-diagonal = mistakes."
                    )
                    cm = confusion_matrix(y_test, y_pred)
                    fig_cm = px.imshow(cm, text_auto=True, color_continuous_scale="Reds",
                                       labels={"x": "Predicted", "y": "Actual"},
                                       title="Confusion Matrix")
                    st.plotly_chart(fig_cm, use_container_width=True)

                # ── REGRESSION ──
                else:
                    r2   = r2_score(y_test, y_pred)
                    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                    mae  = mean_absolute_error(y_test, y_pred)

                    cv = KFold(n_splits=k_folds, shuffle=True, random_state=42)
                    cv_res = cross_validate(
                        model, X_all_s, y, cv=cv,
                        scoring=["r2", "neg_mean_squared_error", "neg_mean_absolute_error"]
                    )
                    avg_r2   = cv_res["test_r2"].mean()
                    avg_rmse = np.sqrt((-cv_res["test_neg_mean_squared_error"]).mean())
                    avg_mae  = (-cv_res["test_neg_mean_absolute_error"]).mean()

                    st.markdown("**Test Set Scores**")
                    c1, c2, c3 = st.columns(3)
                    c1.metric("R² Score", f"{r2:.4f}")
                    c2.metric("RMSE",     f"{rmse:.4f}")
                    c3.metric("MAE",      f"{mae:.4f}")

                    st.markdown(f"**{k_folds}-Fold Cross-Validation Averages**")
                    k1, k2, k3 = st.columns(3)
                    k1.metric("Avg R²",   f"{avg_r2:.4f}")
                    k2.metric("Avg RMSE", f"{avg_rmse:.4f}")
                    k3.metric("Avg MAE",  f"{avg_mae:.4f}")

                    info_box(
                        "<b>R²</b> = proportion of target variance explained by the model (1.0 = perfect). &nbsp;"
                        "<b>RMSE</b> = average prediction error in the target's units — penalises large errors more. &nbsp;"
                        "<b>MAE</b> = average absolute error — treats all errors equally regardless of size."
                    )

                    metrics_df = pd.DataFrame({
                        "Metric":       ["R²",    "RMSE",     "MAE"],
                        "Test Score":   [r2,      rmse,       mae],
                        "CV Avg Score": [avg_r2,  avg_rmse,   avg_mae],
                    })
                    fig_bar = px.bar(
                        metrics_df.melt(id_vars="Metric", var_name="Evaluation", value_name="Score"),
                        x="Metric", y="Score", color="Evaluation", barmode="group",
                        color_discrete_sequence=["#FF4B4B", "#4a7cff"],
                        title="Regression Metrics — Test Set vs K-Fold Average"
                    )
                    st.plotly_chart(fig_bar, use_container_width=True)

                st.success("✅ Training complete!")

            except Exception as e:
                st.error(f"Training failed: {e}")
                st.info("Make sure you have completed Data Cleaning and Feature Selection before training.")


# import streamlit as st
# import pandas as pd
# import numpy as np
# import plotly.express as px

# from sklearn.model_selection import train_test_split, cross_val_score, KFold
# from sklearn.preprocessing import StandardScaler, LabelEncoder
# from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
# from sklearn.linear_model import LogisticRegression, LinearRegression
# from sklearn.metrics import (
#     accuracy_score, classification_report, confusion_matrix,
#     r2_score, mean_squared_error, mean_absolute_error
# )

# # ------------------ CONFIG ------------------
# st.set_page_config(page_title="ML Pipeline Dashboard", layout="wide")

# st.markdown("""
# <style>
# .main { background-color: #f5f7f9; }
# .stMetric {
#     background-color: #ffffff;
#     padding: 15px;
#     border-radius: 10px;
#     box-shadow: 0px 2px 10px rgba(0,0,0,0.05);
# }
# </style>
# """, unsafe_allow_html=True)

# # ------------------ HELPERS ------------------
# def detect_task_type(series, threshold=10):
#     """Auto-detect if target is classification or regression."""
#     if series.dtype == object or str(series.dtype) == 'category':
#         return "classification"
#     n_unique = series.nunique()
#     if n_unique <= threshold:
#         return "classification"
#     return "regression"

# def get_model(model_type, task):
#     if task == "classification":
#         return LogisticRegression(max_iter=1000) if model_type == "Logistic Regression" else RandomForestClassifier(n_estimators=100, random_state=42)
#     else:
#         return LinearRegression() if model_type == "Linear Regression" else RandomForestRegressor(n_estimators=100, random_state=42)

# def encode_target(y):
#     """Encode string/object target to numeric for classification."""
#     if y.dtype == object or str(y.dtype) == 'category':
#         le = LabelEncoder()
#         return pd.Series(le.fit_transform(y), name=y.name), le
#     return y, None

# # ------------------ LOAD DATA ------------------
# @st.cache_data
# def load_default_data():
#     url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
#     return pd.read_csv(url)

# # ------------------ SIDEBAR ------------------
# st.sidebar.title("🛠️ ML Workspace")

# uploaded_file = st.sidebar.file_uploader("📁 Upload your CSV", type=["csv"])

# if uploaded_file is not None:
#     df = pd.read_csv(uploaded_file)
#     st.sidebar.success("Custom dataset loaded ✅")
# else:
#     df = load_default_data()
#     st.sidebar.info("Using Titanic dataset")

# page = st.sidebar.selectbox("Navigate", [
#     "Dashboard",
#     "EDA",
#     "Data Cleaning",
#     "Feature Selection",
#     "Model Training",
#     "Prediction"
# ])

# # ------------------ DASHBOARD ------------------
# if page == "Dashboard":
#     st.title("📊 Dataset Overview")

#     col1, col2, col3, col4 = st.columns(4)
#     col1.metric("Samples", len(df))
#     col2.metric("Features", len(df.columns))
#     col3.metric("Missing Values", int(df.isnull().sum().sum()))
#     col4.metric("Numeric Columns", len(df.select_dtypes(include=np.number).columns))

#     st.subheader("Raw Data Preview")
#     st.dataframe(df.head(10))

#     st.subheader("Data Types")
#     dtype_df = pd.DataFrame({"Column": df.dtypes.index, "Type": df.dtypes.values.astype(str)})
#     st.dataframe(dtype_df, use_container_width=True)

#     st.subheader("Summary Statistics")
#     st.dataframe(df.describe(), use_container_width=True)

# # ------------------ EDA ------------------
# elif page == "EDA":
#     st.title("📊 Exploratory Data Analysis")

#     numeric_df = df.select_dtypes(include=np.number)

#     if numeric_df.empty:
#         st.warning("No numeric columns found.")
#     else:
#         st.subheader("Distribution Plot")
#         feature = st.selectbox("Select Feature", numeric_df.columns)
#         fig = px.histogram(numeric_df, x=feature, marginal="box",
#                            color_discrete_sequence=["#FF4B4B"])
#         st.plotly_chart(fig, use_container_width=True)

#         st.subheader("Correlation Matrix")
#         fig2 = px.imshow(numeric_df.corr().round(2), text_auto=True,
#                          color_continuous_scale="RdBu_r", aspect="auto")
#         st.plotly_chart(fig2, use_container_width=True)

#         st.subheader("Pairplot (top 5 numeric columns)")
#         top_cols = numeric_df.columns[:5].tolist()
#         fig3 = px.scatter_matrix(numeric_df[top_cols], color_discrete_sequence=["#FF4B4B"])
#         st.plotly_chart(fig3, use_container_width=True)

# # ------------------ DATA CLEANING ------------------
# elif page == "Data Cleaning":
#     st.title("🧹 Data Cleaning")

#     df_clean = df.copy()

#     st.subheader("Missing Values Before Cleaning")
#     missing = df_clean.isnull().sum()
#     missing = missing[missing > 0]
#     if missing.empty:
#         st.success("No missing values found!")
#     else:
#         st.dataframe(missing.rename("Missing Count").reset_index(), use_container_width=True)

#     col1, col2 = st.columns(2)

#     with col1:
#         missing_option = st.selectbox("Missing Value Handling",
#                                       ["None", "Drop Rows", "Fill Mean", "Fill Median", "Fill Zero"])

#     with col2:
#         outlier_option = st.selectbox("Outlier Handling", ["None", "Remove via IQR"])

#     # Apply missing value strategy
#     if missing_option == "Drop Rows":
#         df_clean = df_clean.dropna()
#     elif missing_option == "Fill Mean":
#         num_cols = df_clean.select_dtypes(include=np.number).columns
#         df_clean[num_cols] = df_clean[num_cols].fillna(df_clean[num_cols].mean())
#     elif missing_option == "Fill Median":
#         num_cols = df_clean.select_dtypes(include=np.number).columns
#         df_clean[num_cols] = df_clean[num_cols].fillna(df_clean[num_cols].median())
#     elif missing_option == "Fill Zero":
#         num_cols = df_clean.select_dtypes(include=np.number).columns
#         df_clean[num_cols] = df_clean[num_cols].fillna(0)

#     # Apply outlier strategy (numeric only)
#     if outlier_option == "Remove via IQR":
#         numeric_part = df_clean.select_dtypes(include=np.number)
#         Q1 = numeric_part.quantile(0.25)
#         Q3 = numeric_part.quantile(0.75)
#         IQR = Q3 - Q1
#         mask = ~((numeric_part < (Q1 - 1.5 * IQR)) | (numeric_part > (Q3 + 1.5 * IQR))).any(axis=1)
#         df_clean = df_clean[mask]

#     # Keep numeric columns only for ML
#     df_clean_num = df_clean.select_dtypes(include=np.number)
#     st.session_state["df_clean"] = df_clean_num
#     st.session_state["df_clean_full"] = df_clean  # keep full for column reference

#     col3, col4, col5 = st.columns(3)
#     col3.metric("Original Rows", len(df))
#     col4.metric("Cleaned Rows", len(df_clean_num))
#     col5.metric("Numeric Features", len(df_clean_num.columns))

#     st.subheader("After Cleaning Preview")
#     st.dataframe(df_clean_num.head(), use_container_width=True)

# # ------------------ FEATURE SELECTION ------------------
# elif page == "Feature Selection":
#     st.title("🎯 Feature Selection")

#     df_used = st.session_state.get("df_clean", df.select_dtypes(include=np.number))

#     if df_used.shape[1] < 2:
#         st.error("Need at least 2 numeric columns. Please run Data Cleaning first.")
#         st.stop()

#     target = st.selectbox("Select Target Column", df_used.columns)
#     task = detect_task_type(df_used[target])
#     st.info(f"🔍 Detected task type: **{task.upper()}** "
#             f"(target has {df_used[target].nunique()} unique values)")

#     X = df_used.drop(columns=[target])
#     y = df_used[target]

#     method = st.selectbox("Feature Importance Method", ["None", "Random Forest", "Correlation"])

#     selected_features = list(X.columns)

#     if method == "Random Forest":
#         try:
#             if task == "classification":
#                 y_enc, _ = encode_target(y)
#                 # Only use if looks like classification after encoding
#                 rf = RandomForestClassifier(n_estimators=50, random_state=42)
#                 rf.fit(X.fillna(0), y_enc)
#             else:
#                 rf = RandomForestRegressor(n_estimators=50, random_state=42)
#                 rf.fit(X.fillna(0), y)

#             importance = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=True)
#             fig = px.bar(importance, orientation='h', color=importance,
#                          color_continuous_scale="Reds",
#                          labels={"value": "Importance", "index": "Feature"},
#                          title="Feature Importances")
#             st.plotly_chart(fig, use_container_width=True)

#             top_n = st.slider("Top N Features to Keep", 1, len(X.columns), min(5, len(X.columns)))
#             selected_features = importance.sort_values(ascending=False).head(top_n).index.tolist()

#         except Exception as e:
#             st.error(f"Random Forest feature selection failed: {e}")

#     elif method == "Correlation":
#         corr = df_used.corr()[target].abs().drop(target).sort_values(ascending=True)
#         fig = px.bar(corr, orientation='h', color=corr,
#                      color_continuous_scale="Blues",
#                      labels={"value": "Correlation", "index": "Feature"},
#                      title="Absolute Correlation with Target")
#         st.plotly_chart(fig, use_container_width=True)

#         top_n = st.slider("Top N Features to Keep", 1, len(X.columns), min(5, len(X.columns)))
#         selected_features = corr.sort_values(ascending=False).head(top_n).index.tolist()

#     st.success(f"✅ Selected Features: {selected_features}")
#     st.session_state["selected_features"] = selected_features
#     st.session_state["target"] = target
#     st.session_state["task"] = task

# # ------------------ MODEL TRAINING ------------------
# elif page == "Model Training":
#     st.title("🤖 Model Training")

#     df_used = st.session_state.get("df_clean", df.select_dtypes(include=np.number))
#     features = st.session_state.get("selected_features", list(df_used.columns[:-1]))
#     target = st.session_state.get("target", df_used.columns[-1])

#     # Re-detect task type in case session state missing
#     task = st.session_state.get("task", detect_task_type(df_used[target]))
#     st.info(f"🔍 Task: **{task.upper()}** | Target: **{target}** | Features: **{len(features)}**")

#     X = df_used[features].fillna(0)
#     y = df_used[target]

#     if task == "classification":
#         y, le = encode_target(y)
#         model_options = ["Logistic Regression", "Random Forest"]
#     else:
#         le = None
#         model_options = ["Linear Regression", "Random Forest"]

#     col1, col2 = st.columns(2)
#     with col1:
#         test_size = st.slider("Test Size", 0.1, 0.4, 0.2, step=0.05)
#     with col2:
#         model_type = st.selectbox("Model", model_options)

#     cv_folds = st.slider("K-Fold Cross-Validation Folds", 2, 10, 5)

#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, test_size=test_size, random_state=42
#     )

#     scaler = StandardScaler()
#     X_train_s = scaler.fit_transform(X_train)
#     X_test_s = scaler.transform(X_test)
#     X_all_s = scaler.fit_transform(X)

#     if st.button("🚀 Train Model"):
#         with st.spinner("Training..."):
#             model = get_model(model_type, task)

#             try:
#                 model.fit(X_train_s, y_train)
#                 y_pred = model.predict(X_test_s)

#                 st.subheader("📈 Performance Metrics")

#                 if task == "classification":
#                     acc = accuracy_score(y_test, y_pred)
#                     scores = cross_val_score(model, X_all_s, y, cv=KFold(n_splits=cv_folds, shuffle=True, random_state=42))

#                     c1, c2, c3 = st.columns(3)
#                     c1.metric("Test Accuracy", f"{acc:.4f}")
#                     c2.metric(f"Avg CV Score ({cv_folds}-Fold)", f"{scores.mean():.4f}")
#                     c3.metric("CV Std Dev", f"{scores.std():.4f}")

#                     st.subheader(f"📊 {cv_folds}-Fold Cross-Validation Scores")
#                     cv_df = pd.DataFrame({"Fold": [f"Fold {i+1}" for i in range(len(scores))], "Accuracy": scores})
#                     fig = px.bar(cv_df, x="Fold", y="Accuracy", color="Accuracy",
#                                  color_continuous_scale="Reds", range_y=[0, 1],
#                                  title="CV Scores per Fold")
#                     st.plotly_chart(fig, use_container_width=True)

#                     st.subheader("Classification Report")
#                     report = classification_report(y_test, y_pred, output_dict=True)
#                     st.dataframe(pd.DataFrame(report).T.round(3), use_container_width=True)

#                     st.subheader("Confusion Matrix")
#                     cm = confusion_matrix(y_test, y_pred)
#                     fig_cm = px.imshow(cm, text_auto=True, color_continuous_scale="Reds",
#                                        labels={"x": "Predicted", "y": "Actual"},
#                                        title="Confusion Matrix")
#                     st.plotly_chart(fig_cm, use_container_width=True)

#                 else:  # regression
#                     r2 = r2_score(y_test, y_pred)
#                     rmse = np.sqrt(mean_squared_error(y_test, y_pred))
#                     mae = mean_absolute_error(y_test, y_pred)
#                     scores = cross_val_score(model, X_all_s, y, cv=KFold(n_splits=cv_folds, shuffle=True, random_state=42),
#                                              scoring="r2")

#                     c1, c2, c3, c4 = st.columns(4)
#                     c1.metric("R² Score", f"{r2:.4f}")
#                     c2.metric("RMSE", f"{rmse:.4f}")
#                     c3.metric("MAE", f"{mae:.4f}")
#                     c4.metric(f"Avg CV R² ({cv_folds}-Fold)", f"{scores.mean():.4f}")

#                     st.subheader(f"📊 {cv_folds}-Fold Cross-Validation R² Scores")
#                     cv_df = pd.DataFrame({"Fold": [f"Fold {i+1}" for i in range(len(scores))], "R²": scores})
#                     fig = px.bar(cv_df, x="Fold", y="R²", color="R²",
#                                  color_continuous_scale="Blues",
#                                  title="CV R² Scores per Fold")
#                     st.plotly_chart(fig, use_container_width=True)

#                     st.subheader("Actual vs Predicted")
#                     pred_df = pd.DataFrame({"Actual": y_test.values, "Predicted": y_pred})
#                     fig2 = px.scatter(pred_df, x="Actual", y="Predicted",
#                                       trendline="ols", title="Actual vs Predicted",
#                                       color_discrete_sequence=["#FF4B4B"])
#                     st.plotly_chart(fig2, use_container_width=True)

#                 # Save model artifacts
#                 st.session_state["model"] = model
#                 st.session_state["scaler"] = scaler
#                 st.session_state["le"] = le
#                 st.success("✅ Model trained and saved for Prediction!")

#             except Exception as e:
#                 st.error(f"Training failed: {e}")
#                 st.info("Tip: Make sure your target column is appropriate for the detected task type.")

# # ------------------ PREDICTION ------------------
# elif page == "Prediction":
#     st.title("🔮 Prediction")

#     df_used = st.session_state.get("df_clean", df.select_dtypes(include=np.number))
#     features = st.session_state.get("selected_features", list(df_used.columns[:-1]))
#     target = st.session_state.get("target", df_used.columns[-1])
#     task = st.session_state.get("task", detect_task_type(df_used[target]))

#     model = st.session_state.get("model", None)
#     scaler = st.session_state.get("scaler", None)
#     le = st.session_state.get("le", None)

#     if model is None or scaler is None:
#         st.warning("⚠️ No trained model found. Please go to **Model Training** and train a model first.")
#         st.stop()

#     st.info(f"Task: **{task.upper()}** | Target: **{target}** | Model: **{type(model).__name__}**")
#     st.subheader("Adjust Feature Values")

#     X = df_used[features].fillna(0)

#     user_input = []
#     cols = st.columns(min(3, len(features)))
#     for i, col in enumerate(features):
#         col_min = float(X[col].min())
#         col_max = float(X[col].max())
#         col_mean = float(X[col].mean())
#         if col_min == col_max:
#             col_max = col_min + 1.0
#         with cols[i % len(cols)]:
#             val = st.slider(col, col_min, col_max, col_mean,
#                             key=f"pred_{col}")
#         user_input.append(val)

#     if st.button("🔮 Predict"):
#         input_scaled = scaler.transform([user_input])
#         prediction = model.predict(input_scaled)[0]

#         if task == "classification" and le is not None:
#             prediction_label = le.inverse_transform([int(prediction)])[0]
#         else:
#             prediction_label = prediction

#         st.markdown("---")
#         if task == "classification":
#             st.success(f"🎯 Predicted Class: **{prediction_label}**")
#             if hasattr(model, "predict_proba"):
#                 proba = model.predict_proba(input_scaled)[0]
#                 proba_df = pd.DataFrame({
#                     "Class": le.classes_ if le else list(range(len(proba))),
#                     "Probability": proba
#                 })
#                 fig = px.bar(proba_df, x="Class", y="Probability",
#                              color="Probability", color_continuous_scale="Reds",
#                              title="Prediction Probabilities")
#                 st.plotly_chart(fig, use_container_width=True)
#         else:
#             st.success(f"📈 Predicted Value: **{prediction_label:.4f}**")


# import streamlit as st
# import pandas as pd
# import numpy as np
# import plotly.express as px

# from sklearn.model_selection import train_test_split, cross_val_score, KFold
# from sklearn.preprocessing import StandardScaler
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import accuracy_score

# # ------------------ CONFIG ------------------
# st.set_page_config(page_title="ML Pipeline Dashboard", layout="wide")

# st.markdown("""
# <style>
# .main { background-color: #f5f7f9; }
# .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0px 2px 10px rgba(0,0,0,0.05); }
# </style>
# """, unsafe_allow_html=True)

# # ------------------ LOAD DATA ------------------
# @st.cache_data
# def load_default_data():
#     url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
#     return pd.read_csv(url)

# # ------------------ SIDEBAR ------------------
# st.sidebar.title("🛠️ ML Workspace")

# # 🔥 FILE UPLOAD
# uploaded_file = st.sidebar.file_uploader("📁 Upload your CSV", type=["csv"])

# if uploaded_file is not None:
#     df = pd.read_csv(uploaded_file)
#     st.sidebar.success("Custom dataset loaded ✅")
# else:
#     df = load_default_data()
#     st.sidebar.info("Using Titanic dataset")

# # PAGE NAVIGATION
# page = st.sidebar.selectbox("Navigate", [
#     "Dashboard",
#     "EDA",
#     "Data Cleaning",
#     "Feature Selection",
#     "Model Training",
#     "Prediction"
# ])

# # ------------------ DASHBOARD ------------------
# if page == "Dashboard":
#     st.title("📊 Dataset Overview")

#     col1, col2, col3, col4 = st.columns(4)
#     col1.metric("Samples", len(df))
#     col2.metric("Features", len(df.columns))
#     col3.metric("Missing Values", df.isnull().sum().sum())
#     col4.metric("Columns", len(df.columns))

#     st.dataframe(df.head())

# # ------------------ EDA ------------------
# elif page == "EDA":
#     st.title("📊 Exploratory Data Analysis")

#     numeric_df = df.select_dtypes(include=np.number)

#     feature = st.selectbox("Select Feature", numeric_df.columns)

#     fig = px.histogram(
#         numeric_df, x=feature, marginal="box",
#         color_discrete_sequence=["#FF4B4B"]
#     )
#     st.plotly_chart(fig, use_container_width=True)

#     st.subheader("Correlation Matrix")

#     fig = px.imshow(
#         numeric_df.corr(),
#         text_auto=True,
#         color_continuous_scale="Reds"
#     )
#     st.plotly_chart(fig, use_container_width=True)

# # ------------------ DATA CLEANING ------------------
# elif page == "Data Cleaning":
#     st.title("🧹 Data Cleaning")

#     df_clean = df.copy()

#     st.write("Missing Values Before Cleaning:")
#     st.write(df_clean.isnull().sum())

#     col1, col2 = st.columns(2)

#     with col1:
#         missing_option = st.selectbox("Missing Handling", ["None", "Drop", "Mean", "Median"])

#     if missing_option == "Drop":
#         df_clean = df_clean.dropna()
#     elif missing_option == "Mean":
#         df_clean = df_clean.fillna(df_clean.mean(numeric_only=True))
#     elif missing_option == "Median":
#         df_clean = df_clean.fillna(df_clean.median(numeric_only=True))

#     with col2:
#         outlier_option = st.selectbox("Outlier Handling", ["None", "IQR"])

#     numeric_df = df_clean.select_dtypes(include=np.number)

#     if outlier_option == "IQR":
#         Q1 = numeric_df.quantile(0.25)
#         Q3 = numeric_df.quantile(0.75)
#         IQR = Q3 - Q1
#         mask = ~((numeric_df < (Q1 - 1.5 * IQR)) | (numeric_df > (Q3 + 1.5 * IQR))).any(axis=1)
#         df_clean = df_clean[mask]

#     df_clean = df_clean.select_dtypes(include=np.number)

#     st.session_state["df_clean"] = df_clean

#     st.subheader("After Cleaning")

#     col3, col4 = st.columns(2)
#     col3.metric("Original Rows", len(df))
#     col4.metric("Cleaned Rows", len(df_clean))

#     st.dataframe(df_clean.head())

# # ------------------ FEATURE SELECTION ------------------
# elif page == "Feature Selection":
#     st.title("🎯 Feature Selection")

#     df_used = st.session_state.get("df_clean", df.select_dtypes(include=np.number))

#     target = st.selectbox("Select Target Column", df_used.columns)

#     X = df_used.drop(target, axis=1)
#     y = df_used[target]

#     method = st.selectbox("Method", ["None", "Random Forest", "Correlation"])

#     selected_features = X.columns

#     if method == "Random Forest":
#         rf = RandomForestClassifier()
#         rf.fit(X, y)
#         importance = pd.Series(rf.feature_importances_, index=X.columns)

#         st.bar_chart(importance.sort_values())

#         top_n = st.slider("Top Features", 1, len(X.columns), 5)
#         selected_features = importance.sort_values(ascending=False).head(top_n).index

#     elif method == "Correlation":
#         corr = df_used.corr()[target].abs().drop(target)
#         st.bar_chart(corr)

#         top_n = st.slider("Top Features", 1, len(X.columns), 5)
#         selected_features = corr.sort_values(ascending=False).head(top_n).index

#     st.write("Selected Features:", list(selected_features))
#     st.session_state["selected_features"] = list(selected_features)
#     st.session_state["target"] = target

# # ------------------ MODEL TRAINING ------------------
# elif page == "Model Training":
#     st.title("🤖 Model Training")

#     df_used = st.session_state.get("df_clean", df.select_dtypes(include=np.number))
#     features = st.session_state.get("selected_features", df_used.columns[:-1])
#     target = st.session_state.get("target", df_used.columns[-1])

#     X = df_used[features]
#     y = df_used[target]

#     test_size = st.slider("Test Size", 0.1, 0.4, 0.2)
#     model_type = st.selectbox("Model", ["Logistic Regression", "Random Forest"])

#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

#     scaler = StandardScaler()
#     X_train = scaler.fit_transform(X_train)
#     X_test = scaler.transform(X_test)

#     if st.button("Train Model"):
#         model = LogisticRegression() if model_type == "Logistic Regression" else RandomForestClassifier()

#         model.fit(X_train, y_train)
#         y_pred = model.predict(X_test)

#         col1, col2 = st.columns(2)
#         col1.metric("Accuracy", round(accuracy_score(y_test, y_pred), 4))

#         scores = cross_val_score(model, X, y, cv=KFold(5))
#         col2.metric("Avg CV Score", round(scores.mean(), 4))

#         st.line_chart(scores)

# # ------------------ PREDICTION ------------------
# elif page == "Prediction":
#     st.title("🔮 Prediction")

#     df_used = st.session_state.get("df_clean", df.select_dtypes(include=np.number))
#     features = st.session_state.get("selected_features", df_used.columns[:-1])
#     target = st.session_state.get("target", df_used.columns[-1])

#     X = df_used[features]
#     y = df_used[target]

#     scaler = StandardScaler()
#     X_scaled = scaler.fit_transform(X)

#     model = RandomForestClassifier()
#     model.fit(X_scaled, y)

#     user_input = []
#     for col in features:
#         val = st.slider(col, float(X[col].min()), float(X[col].max()))
#         user_input.append(val)

#     input_scaled = scaler.transform([user_input])
#     prediction = model.predict(input_scaled)[0]

#     st.success(f"Prediction: {prediction}")

# import streamlit as st
# import pandas as pd
# import numpy as np
# import plotly.express as px

# from sklearn.model_selection import train_test_split, cross_val_score, KFold
# from sklearn.preprocessing import StandardScaler
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.linear_model import LogisticRegression, Lasso
# from sklearn.metrics import accuracy_score

# # ------------------ CONFIG ------------------
# st.set_page_config(page_title="ML Pipeline Dashboard", layout="wide")

# st.markdown("""
# <style>
# .main { background-color: #f5f7f9; }
# .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0px 2px 10px rgba(0,0,0,0.05); }
# </style>
# """, unsafe_allow_html=True)

# # ------------------ LOAD DATA ------------------
# @st.cache_data
# def load_data():
#     url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
#     return pd.read_csv(url)

# df = load_data()

# # ------------------ SIDEBAR ------------------
# st.sidebar.title("🛠️ ML Workspace")
# page = st.sidebar.selectbox("Navigate", [
#     "Dashboard",
#     "EDA",
#     "Data Cleaning",
#     "Feature Selection",
#     "Model Training",
#     "Prediction"
# ])

# # ------------------ DASHBOARD ------------------
# if page == "Dashboard":
#     st.title("🚢 Titanic Survival Prediction")

#     col1, col2, col3, col4 = st.columns(4)
#     col1.metric("Samples", len(df))
#     col2.metric("Features", len(df.columns)-1)
#     col3.metric("Missing Values", df.isnull().sum().sum())
#     col4.metric("Target", "Survived")

#     st.dataframe(df.head())

# # ------------------ EDA ------------------
# elif page == "EDA":
#     st.title("📊 Exploratory Data Analysis")

#     numeric_df = df.select_dtypes(include=np.number)

#     feature = st.selectbox("Select Feature", numeric_df.columns)

#     fig = px.histogram(
#         numeric_df, x=feature, marginal="box",
#         color_discrete_sequence=["#FF4B4B"]  # RED
#     )
#     st.plotly_chart(fig, use_container_width=True)

#     st.subheader("Correlation Matrix")

#     fig = px.imshow(
#         numeric_df.corr(),
#         text_auto=True,
#         color_continuous_scale="Reds"
#     )
#     st.plotly_chart(fig, use_container_width=True)

# # ------------------ DATA CLEANING ------------------
# elif page == "Data Cleaning":
#     st.title("🧹 Data Cleaning")

#     df_clean = df.copy()

#     st.write("Missing Values Before Cleaning:")
#     st.write(df_clean.isnull().sum())

#     col1, col2 = st.columns(2)

#     # -------- MISSING HANDLING --------
#     with col1:
#         missing_option = st.selectbox("Missing Handling", ["None", "Drop", "Mean", "Median"])

#     if missing_option == "Drop":
#         df_clean = df_clean.dropna()
#     elif missing_option == "Mean":
#         df_clean = df_clean.fillna(df_clean.mean(numeric_only=True))
#     elif missing_option == "Median":
#         df_clean = df_clean.fillna(df_clean.median(numeric_only=True))

#     # -------- OUTLIER HANDLING --------
#     with col2:
#         outlier_option = st.selectbox("Outlier Handling", ["None", "IQR"])

#     numeric_df = df_clean.select_dtypes(include=np.number)

#     if outlier_option == "IQR":
#         Q1 = numeric_df.quantile(0.25)
#         Q3 = numeric_df.quantile(0.75)
#         IQR = Q3 - Q1
#         mask = ~((numeric_df < (Q1 - 1.5 * IQR)) | (numeric_df > (Q3 + 1.5 * IQR))).any(axis=1)
#         df_clean = df_clean[mask]

#     # -------- DROP NON-NUMERIC --------
#     df_clean = df_clean.select_dtypes(include=np.number)

#     st.session_state["df_clean"] = df_clean

#     st.subheader("📊 After Cleaning")

#     col3, col4 = st.columns(2)
#     col3.metric("Original Rows", len(df))
#     col4.metric("Cleaned Rows", len(df_clean))

#     st.write("Missing Values After Cleaning:")
#     st.write(df_clean.isnull().sum())

#     st.dataframe(df_clean.head())

# # ------------------ FEATURE SELECTION ------------------
# elif page == "Feature Selection":
#     st.title("🎯 Feature Selection")

#     df_used = st.session_state.get("df_clean", df.select_dtypes(include=np.number))

#     X = df_used.drop('Survived', axis=1)
#     y = df_used['Survived']

#     method = st.selectbox("Method", ["None", "Random Forest", "Correlation"])

#     selected_features = X.columns

#     if method == "Random Forest":
#         rf = RandomForestClassifier()
#         rf.fit(X, y)
#         importance = pd.Series(rf.feature_importances_, index=X.columns)
#         st.bar_chart(importance.sort_values())

#         top_n = st.slider("Top Features", 1, len(X.columns), 5)
#         selected_features = importance.sort_values(ascending=False).head(top_n).index

#     elif method == "Correlation":
#         corr = df_used.corr()['Survived'].abs().drop('Survived')
#         st.bar_chart(corr)

#         top_n = st.slider("Top Features", 1, len(X.columns), 5)
#         selected_features = corr.sort_values(ascending=False).head(top_n).index

#     st.write("Selected Features:", list(selected_features))
#     st.session_state["selected_features"] = list(selected_features)

# # ------------------ MODEL TRAINING ------------------
# elif page == "Model Training":
#     st.title("🤖 Model Training")

#     df_used = st.session_state.get("df_clean", df.select_dtypes(include=np.number))
#     features = st.session_state.get("selected_features", df_used.columns.drop('Survived'))

#     X = df_used[features]
#     y = df_used['Survived']

#     test_size = st.slider("Test Size", 0.1, 0.4, 0.2)
#     model_type = st.selectbox("Model", ["Logistic Regression", "Random Forest"])

#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

#     scaler = StandardScaler()
#     X_train = scaler.fit_transform(X_train)
#     X_test = scaler.transform(X_test)

#     if st.button("Train Model"):
#         model = LogisticRegression() if model_type == "Logistic Regression" else RandomForestClassifier()

#         model.fit(X_train, y_train)
#         y_pred = model.predict(X_test)

#         col1, col2 = st.columns(2)
#         col1.metric("Accuracy", round(accuracy_score(y_test, y_pred), 4))

#         scores = cross_val_score(model, X, y, cv=KFold(5))
#         col2.metric("Avg CV Score", round(scores.mean(), 4))

#         st.line_chart(scores)

# # ------------------ PREDICTION ------------------
# elif page == "Prediction":
#     st.title("🔮 Prediction")

#     df_used = st.session_state.get("df_clean", df.select_dtypes(include=np.number))
#     features = st.session_state.get("selected_features", df_used.columns.drop('Survived'))

#     X = df_used[features]
#     y = df_used['Survived']

#     scaler = StandardScaler()
#     X_scaled = scaler.fit_transform(X)

#     model = RandomForestClassifier()
#     model.fit(X_scaled, y)

#     user_input = []
#     for col in features:
#         val = st.slider(col, float(X[col].min()), float(X[col].max()))
#         user_input.append(val)

#     input_scaled = scaler.transform([user_input])
#     prediction = model.predict(input_scaled)[0]

#     st.success("Survived" if prediction == 1 else "Not Survived")











# import streamlit as st
# import pandas as pd
# import numpy as np
# import plotly.express as px

# from sklearn.model_selection import train_test_split, cross_val_score, KFold
# from sklearn.preprocessing import StandardScaler
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import accuracy_score

# # ------------------ CONFIG ------------------
# st.set_page_config(page_title="ML Pipeline Dashboard", layout="wide")

# st.markdown("""
# <style>
# .main { background-color: #f5f7f9; }
# .stMetric {
#     background-color: #ffffff;
#     padding: 15px;
#     border-radius: 10px;
#     box-shadow: 0px 2px 10px rgba(0,0,0,0.05);
# }
# </style>
# """, unsafe_allow_html=True)

# # ------------------ SIDEBAR ------------------
# st.sidebar.title("🛠️ ML Workspace")

# uploaded_file = st.sidebar.file_uploader("📂 Upload CSV File", type=["csv"])

# if uploaded_file is None:
#     st.warning("⚠️ Please upload a CSV file to continue")
#     st.stop()

# @st.cache_data
# def load_data(file):
#     return pd.read_csv(file)

# df = load_data(uploaded_file)

# target_column = st.sidebar.selectbox("🎯 Select Target Column", df.columns)

# page = st.sidebar.selectbox("Navigate", [
#     "Dashboard",
#     "EDA",
#     "Data Cleaning",
#     "Feature Selection",
#     "Model Training",
#     "Prediction"
# ])

# # ------------------ DASHBOARD ------------------
# if page == "Dashboard":
#     st.title("📊 ML Dataset Dashboard")

#     col1, col2, col3, col4 = st.columns(4)
#     col1.metric("Samples", len(df))
#     col2.metric("Features", len(df.columns)-1)
#     col3.metric("Missing Values", df.isnull().sum().sum())
#     col4.metric("Target", target_column)

#     st.dataframe(df.head())

# # ------------------ EDA ------------------
# elif page == "EDA":
#     st.title("📊 Exploratory Data Analysis")

#     numeric_df = df.select_dtypes(include=np.number)

#     feature = st.selectbox("Select Feature", numeric_df.columns)

#     fig = px.histogram(
#         numeric_df, x=feature, marginal="box",
#         color_discrete_sequence=["#FF4B4B"]
#     )
#     st.plotly_chart(fig, use_container_width=True)

#     st.subheader("Correlation Matrix")

#     fig = px.imshow(
#         numeric_df.corr(),
#         text_auto=True,
#         color_continuous_scale="Reds"
#     )
#     st.plotly_chart(fig, use_container_width=True)

# # ------------------ DATA CLEANING ------------------
# elif page == "Data Cleaning":
#     st.title("🧹 Data Cleaning")

#     df_clean = df.copy()

#     st.write("Missing Values Before Cleaning:")
#     st.write(df_clean.isnull().sum())

#     col1, col2 = st.columns(2)

#     with col1:
#         missing_option = st.selectbox("Missing Handling", ["None", "Drop", "Mean", "Median"])

#     if missing_option == "Drop":
#         df_clean = df_clean.dropna()
#     elif missing_option == "Mean":
#         df_clean = df_clean.fillna(df_clean.mean(numeric_only=True))
#     elif missing_option == "Median":
#         df_clean = df_clean.fillna(df_clean.median(numeric_only=True))

#     with col2:
#         outlier_option = st.selectbox("Outlier Handling", ["None", "IQR"])

#     numeric_df = df_clean.select_dtypes(include=np.number)

#     if outlier_option == "IQR":
#         Q1 = numeric_df.quantile(0.25)
#         Q3 = numeric_df.quantile(0.75)
#         IQR = Q3 - Q1
#         mask = ~((numeric_df < (Q1 - 1.5 * IQR)) |
#                  (numeric_df > (Q3 + 1.5 * IQR))).any(axis=1)
#         df_clean = df_clean[mask]

#     # Keep only numeric
#     df_clean = df_clean.select_dtypes(include=np.number)

#     st.session_state["df_clean"] = df_clean

#     st.subheader("After Cleaning")

#     col3, col4 = st.columns(2)
#     col3.metric("Original Rows", len(df))
#     col4.metric("Cleaned Rows", len(df_clean))

#     st.write("Missing Values After Cleaning:")
#     st.write(df_clean.isnull().sum())

#     st.dataframe(df_clean.head())

# # ------------------ FEATURE SELECTION ------------------
# elif page == "Feature Selection":
#     st.title("🎯 Feature Selection")

#     df_used = st.session_state.get("df_clean", df.select_dtypes(include=np.number))

#     if target_column not in df_used.columns:
#         st.error("Target column removed during cleaning!")
#         st.stop()

#     X = df_used.drop(target_column, axis=1)
#     y = df_used[target_column]

#     method = st.selectbox("Method", ["None", "Random Forest", "Correlation"])

#     selected_features = X.columns

#     if method == "Random Forest":
#         rf = RandomForestClassifier()
#         rf.fit(X, y)

#         importance = pd.Series(rf.feature_importances_, index=X.columns)
#         st.bar_chart(importance.sort_values())

#         top_n = st.slider("Top Features", 1, len(X.columns), 5)
#         selected_features = importance.sort_values(ascending=False).head(top_n).index

#     elif method == "Correlation":
#         corr = df_used.corr()[target_column].abs().drop(target_column)
#         st.bar_chart(corr)

#         top_n = st.slider("Top Features", 1, len(X.columns), 5)
#         selected_features = corr.sort_values(ascending=False).head(top_n).index

#     st.write("Selected Features:", list(selected_features))
#     st.session_state["selected_features"] = list(selected_features)

# # ------------------ MODEL TRAINING ------------------
# elif page == "Model Training":
#     st.title("🤖 Model Training")

#     df_used = st.session_state.get("df_clean", df.select_dtypes(include=np.number))
#     features = st.session_state.get("selected_features", df_used.columns.drop(target_column))

#     X = df_used[features]
#     y = df_used[target_column]

#     test_size = st.slider("Test Size", 0.1, 0.4, 0.2)
#     model_type = st.selectbox("Model", ["Logistic Regression", "Random Forest"])

#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

#     scaler = StandardScaler()
#     X_train = scaler.fit_transform(X_train)
#     X_test = scaler.transform(X_test)

#     if st.button("Train Model"):
#         model = LogisticRegression() if model_type == "Logistic Regression" else RandomForestClassifier()

#         model.fit(X_train, y_train)
#         y_pred = model.predict(X_test)

#         col1, col2 = st.columns(2)
#         col1.metric("Accuracy", round(accuracy_score(y_test, y_pred), 4))

#         scores = cross_val_score(model, X, y, cv=KFold(5))
#         col2.metric("Avg CV Score", round(scores.mean(), 4))

#         st.line_chart(scores)

# # ------------------ PREDICTION ------------------
# elif page == "Prediction":
#     st.title("🔮 Prediction")

#     df_used = st.session_state.get("df_clean", df.select_dtypes(include=np.number))
#     features = st.session_state.get("selected_features", df_used.columns.drop(target_column))

#     X = df_used[features]
#     y = df_used[target_column]

#     scaler = StandardScaler()
#     X_scaled = scaler.fit_transform(X)

#     model = RandomForestClassifier()
#     model.fit(X_scaled, y)

#     user_input = []
#     for col in features:
#         val = st.slider(col, float(X[col].min()), float(X[col].max()))
#         user_input.append(val)

#     input_scaled = scaler.transform([user_input])
#     prediction = model.predict(input_scaled)[0]

#     st.success(f"Prediction: {prediction}")




# """
# ╔══════════════════════════════════════════════════════════╗
# ║         ML PIPELINE DASHBOARD — Streamlit App           ║
# ║  Run: streamlit run ml_pipeline_app.py                  ║
# ╚══════════════════════════════════════════════════════════╝
# """

# import streamlit as st
# import pandas as pd
# import numpy as np
# import plotly.express as px
# import plotly.graph_objects as go
# from plotly.subplots import make_subplots
# import warnings
# warnings.filterwarnings("ignore")

# # ── Page config ────────────────────────────────────────────
# st.set_page_config(
#     page_title="ML Pipeline",
#     page_icon="🧬",
#     layout="wide",
#     initial_sidebar_state="collapsed",
# )

# # ── Global CSS ─────────────────────────────────────────────
# st.markdown("""
# <style>
# @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

# :root {
#     --bg:        #0d0f14;
#     --surface:   #151820;
#     --panel:     #1c2030;
#     --border:    #2a2f3e;
#     --accent:    #6c63ff;
#     --accent2:   #00d4aa;
#     --accent3:   #ff6584;
#     --text:      #e2e8f0;
#     --muted:     #7a85a0;
#     --mono:      'Space Mono', monospace;
#     --sans:      'DM Sans', sans-serif;
# }

# html, body, [data-testid="stAppViewContainer"] {
#     background: var(--bg) !important;
#     color: var(--text) !important;
#     font-family: var(--sans) !important;
# }

# [data-testid="stMain"], [data-testid="block-container"] {
#     background: var(--bg) !important;
#     padding-top: 1rem !important;
# }

# /* ── Sidebar ── */
# [data-testid="stSidebar"] {
#     background: var(--surface) !important;
#     border-right: 1px solid var(--border) !important;
# }

# /* ── Headers ── */
# h1, h2, h3 { font-family: var(--mono) !important; color: var(--text) !important; }

# /* ── Stepper bar ── */
# .stepper-container {
#     display: flex;
#     align-items: center;
#     justify-content: center;
#     gap: 0;
#     padding: 1.5rem 2rem;
#     background: var(--surface);
#     border: 1px solid var(--border);
#     border-radius: 16px;
#     margin-bottom: 2rem;
#     overflow-x: auto;
# }

# .step-item {
#     display: flex;
#     align-items: center;
#     flex-shrink: 0;
# }

# .step-bubble {
#     width: 40px;
#     height: 40px;
#     border-radius: 50%;
#     display: flex;
#     align-items: center;
#     justify-content: center;
#     font-family: var(--mono);
#     font-size: 13px;
#     font-weight: 700;
#     border: 2px solid var(--border);
#     background: var(--panel);
#     color: var(--muted);
#     transition: all .3s;
#     position: relative;
#     z-index: 1;
# }

# .step-bubble.active {
#     background: var(--accent);
#     border-color: var(--accent);
#     color: #fff;
#     box-shadow: 0 0 20px rgba(108,99,255,0.5);
# }

# .step-bubble.done {
#     background: var(--accent2);
#     border-color: var(--accent2);
#     color: #0d0f14;
# }

# .step-label {
#     font-size: 10px;
#     color: var(--muted);
#     text-align: center;
#     margin-top: 4px;
#     font-family: var(--sans);
#     max-width: 72px;
#     line-height: 1.3;
# }

# .step-label.active { color: var(--accent); font-weight: 600; }
# .step-label.done   { color: var(--accent2); }

# .step-wrap {
#     display: flex;
#     flex-direction: column;
#     align-items: center;
#     gap: 4px;
# }

# .step-connector {
#     height: 2px;
#     width: 40px;
#     background: var(--border);
#     flex-shrink: 0;
#     margin-bottom: 20px;
# }

# .step-connector.done { background: var(--accent2); }
# .step-connector.active { background: var(--accent); }

# /* ── Cards ── */
# .card {
#     background: var(--surface);
#     border: 1px solid var(--border);
#     border-radius: 12px;
#     padding: 1.5rem;
#     margin-bottom: 1rem;
# }

# .card-title {
#     font-family: var(--mono);
#     font-size: 13px;
#     color: var(--accent);
#     text-transform: uppercase;
#     letter-spacing: 2px;
#     margin-bottom: 1rem;
#     display: flex;
#     align-items: center;
#     gap: 8px;
# }

# /* ── Metric pill ── */
# .metric-pill {
#     background: var(--panel);
#     border: 1px solid var(--border);
#     border-radius: 8px;
#     padding: .6rem 1rem;
#     text-align: center;
#     display: inline-block;
# }

# .metric-pill .val {
#     font-family: var(--mono);
#     font-size: 1.4rem;
#     color: var(--accent2);
#     display: block;
# }

# .metric-pill .lbl {
#     font-size: 11px;
#     color: var(--muted);
# }

# /* ── Section header ── */
# .section-header {
#     font-family: var(--mono);
#     font-size: 1.3rem;
#     color: var(--text);
#     border-left: 4px solid var(--accent);
#     padding-left: 1rem;
#     margin-bottom: 1.5rem;
# }

# /* ── Tag ── */
# .tag {
#     background: rgba(108,99,255,.15);
#     color: var(--accent);
#     border: 1px solid rgba(108,99,255,.3);
#     border-radius: 20px;
#     padding: 2px 12px;
#     font-size: 12px;
#     font-family: var(--mono);
#     display: inline-block;
#     margin: 2px;
# }

# .tag.green {
#     background: rgba(0,212,170,.12);
#     color: var(--accent2);
#     border-color: rgba(0,212,170,.3);
# }

# .tag.red {
#     background: rgba(255,101,132,.12);
#     color: var(--accent3);
#     border-color: rgba(255,101,132,.3);
# }

# /* ── Streamlit element overrides ── */
# div[data-testid="stSelectbox"] > div,
# div[data-testid="stMultiSelect"] > div,
# div[data-testid="stNumberInput"] > div > div,
# div[data-testid="stTextInput"] > div > div {
#     background: var(--panel) !important;
#     border-color: var(--border) !important;
#     color: var(--text) !important;
#     border-radius: 8px !important;
# }

# .stButton > button {
#     background: var(--accent) !important;
#     color: #fff !important;
#     border: none !important;
#     border-radius: 8px !important;
#     font-family: var(--mono) !important;
#     font-size: 13px !important;
#     padding: .5rem 1.5rem !important;
#     transition: all .2s !important;
# }

# .stButton > button:hover {
#     background: #8b85ff !important;
#     box-shadow: 0 0 20px rgba(108,99,255,0.4) !important;
#     transform: translateY(-1px) !important;
# }

# .stButton > button[kind="secondary"] {
#     background: var(--panel) !important;
#     border: 1px solid var(--border) !important;
#     color: var(--text) !important;
# }

# div[data-testid="stExpander"] {
#     background: var(--surface) !important;
#     border: 1px solid var(--border) !important;
#     border-radius: 10px !important;
# }

# div[data-testid="stDataFrame"] {
#     background: var(--surface) !important;
# }

# .stSlider > div > div > div { background: var(--accent) !important; }

# [data-testid="stCheckbox"] span { color: var(--text) !important; }

# [data-testid="stTab"] button {
#     font-family: var(--mono) !important;
#     color: var(--muted) !important;
#     background: transparent !important;
# }

# [data-testid="stTab"] button[aria-selected="true"] {
#     color: var(--accent) !important;
#     border-bottom: 2px solid var(--accent) !important;
# }

# /* scrollbar */
# ::-webkit-scrollbar { width: 6px; height: 6px; }
# ::-webkit-scrollbar-track { background: var(--bg); }
# ::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }

# hr { border-color: var(--border) !important; }

# p, label, .stMarkdown { color: var(--text) !important; }
# small, .caption { color: var(--muted) !important; }

# div[data-testid="stAlert"] { border-radius: 8px !important; }

# </style>
# """, unsafe_allow_html=True)

# # ══════════════════════════════════════════════════════════
# #  STEP DEFINITIONS
# # ══════════════════════════════════════════════════════════
# STEPS = [
#     ("01", "Problem\nType"),
#     ("02", "Data\nInput"),
#     ("03", "EDA"),
#     ("04", "Engineering"),
#     ("05", "Features"),
#     ("06", "Split"),
#     ("07", "Model\nSelect"),
#     ("08", "Training"),
#     ("09", "Metrics"),
#     ("10", "Tuning"),
# ]

# def render_stepper(current: int):
#     """Render horizontal stepper with animated states."""
#     html = '<div class="stepper-container">'
#     for i, (num, label) in enumerate(STEPS):
#         step_num = i + 1
#         if step_num < current:
#             bubble_cls = "done"
#             label_cls  = "done"
#             conn_cls   = "done"
#             icon = "✓"
#         elif step_num == current:
#             bubble_cls = "active"
#             label_cls  = "active"
#             conn_cls   = "active"
#             icon = num
#         else:
#             bubble_cls = ""
#             label_cls  = ""
#             conn_cls   = ""
#             icon = num

#         html += f"""
#         <div class="step-item">
#             <div class="step-wrap">
#                 <div class="step-bubble {bubble_cls}">{icon}</div>
#                 <div class="step-label {label_cls}">{label}</div>
#             </div>
#         </div>"""
#         if i < len(STEPS) - 1:
#             html += f'<div class="step-connector {conn_cls}"></div>'
#     html += '</div>'
#     st.markdown(html, unsafe_allow_html=True)


# def section_header(title: str):
#     st.markdown(f'<div class="section-header">{title}</div>', unsafe_allow_html=True)


# def card_open(title: str, icon: str = "◈"):
#     st.markdown(f"""
#     <div class="card">
#     <div class="card-title">{icon} {title}</div>
#     """, unsafe_allow_html=True)


# def card_close():
#     st.markdown("</div>", unsafe_allow_html=True)


# # ══════════════════════════════════════════════════════════
# #  PLOTLY THEME HELPER
# # ══════════════════════════════════════════════════════════
# PLOTLY_LAYOUT = dict(
#     paper_bgcolor="rgba(0,0,0,0)",
#     plot_bgcolor="rgba(28,32,48,0.6)",
#     font=dict(color="#e2e8f0", family="DM Sans"),
#     colorway=["#6c63ff", "#00d4aa", "#ff6584", "#ffa040", "#40c8ff"],
#     xaxis=dict(gridcolor="#2a2f3e", linecolor="#2a2f3e"),
#     yaxis=dict(gridcolor="#2a2f3e", linecolor="#2a2f3e"),
#     margin=dict(l=40, r=20, t=40, b=40),
#     legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="#2a2f3e"),
# )

# def styled_fig(fig):
#     fig.update_layout(**PLOTLY_LAYOUT)
#     return fig


# # ══════════════════════════════════════════════════════════
# #  SESSION STATE INIT
# # ══════════════════════════════════════════════════════════
# def init_state():
#     defaults = dict(
#         step=1,
#         problem_type=None,
#         df=None,
#         target=None,
#         selected_features=None,
#         df_clean=None,
#         outlier_indices=[],
#         feature_selected=None,
#         X_train=None, X_test=None, y_train=None, y_test=None,
#         model_name=None,
#         model_obj=None,
#         k_folds=5,
#         cv_results=None,
#         best_params=None,
#     )
#     for k, v in defaults.items():
#         if k not in st.session_state:
#             st.session_state[k] = v

# init_state()
# S = st.session_state
# def get_df():
#     return S.df_clean if S.df_clean is not None else S.df

# # ══════════════════════════════════════════════════════════
# #  NAVIGATION HELPERS
# # ══════════════════════════════════════════════════════════
# def go_next(): S.step += 1
# def go_back(): S.step = max(1, S.step - 1)

# def nav_buttons(back=True, next_label="Continue →", next_key=None, disabled=False):
#     cols = st.columns([1, 4, 1])
#     with cols[0]:
#         if back and S.step > 1:
#             st.button("← Back", on_click=go_back, key=f"back_{S.step}")
#     with cols[2]:
#         st.button(next_label, on_click=go_next, key=next_key or f"next_{S.step}",
#                   disabled=disabled)


# # ══════════════════════════════════════════════════════════
# #  TOP BANNER
# # ══════════════════════════════════════════════════════════
# col_logo, col_title = st.columns([1, 8])
# with col_logo:
#     st.markdown("""
#     <div style="font-size:2.8rem;margin-top:4px;text-align:center">🧬</div>
#     """, unsafe_allow_html=True)
# with col_title:
#     st.markdown("""
#     <div style="padding:4px 0">
#     <div style="font-family:'Space Mono',monospace;font-size:1.6rem;color:#e2e8f0;letter-spacing:2px">
#         ML PIPELINE <span style="color:#6c63ff">STUDIO</span>
#     </div>
#     <div style="font-size:13px;color:#7a85a0;margin-top:2px">
#         End-to-end machine learning • from raw data to tuned model
#     </div>
#     </div>
#     """, unsafe_allow_html=True)

# st.markdown("<hr style='margin:.5rem 0 1rem'>", unsafe_allow_html=True)
# render_stepper(S.step)


# # ══════════════════════════════════════════════════════════
# #  STEP 1 — PROBLEM TYPE
# # ══════════════════════════════════════════════════════════
# if S.step == 1:
#     section_header("Step 1 · Choose Problem Type")

#     c1, c2 = st.columns(2)
#     with c1:
#         st.markdown("""
#         <div style="background:#1c2030;border:2px solid #6c63ff;border-radius:16px;padding:2rem;text-align:center;cursor:pointer">
#             <div style="font-size:3rem">📊</div>
#             <div style="font-family:'Space Mono',monospace;font-size:1.1rem;color:#6c63ff;margin:.5rem 0">CLASSIFICATION</div>
#             <div style="font-size:13px;color:#7a85a0">Predict discrete categories or classes.<br>Binary or multi-class supported.</div>
#         </div>
#         """, unsafe_allow_html=True)
#         if st.button("Select Classification", key="cls_btn"):
#             S.problem_type = "Classification"
#             go_next()

#     with c2:
#         st.markdown("""
#         <div style="background:#1c2030;border:2px solid #00d4aa;border-radius:16px;padding:2rem;text-align:center;cursor:pointer">
#             <div style="font-size:3rem">📈</div>
#             <div style="font-family:'Space Mono',monospace;font-size:1.1rem;color:#00d4aa;margin:.5rem 0">REGRESSION</div>
#             <div style="font-size:13px;color:#7a85a0">Predict continuous numeric values.<br>Linear and non-linear models.</div>
#         </div>
#         """, unsafe_allow_html=True)
#         if st.button("Select Regression", key="reg_btn"):
#             S.problem_type = "Regression"
#             go_next()


# # ══════════════════════════════════════════════════════════
# #  STEP 2 — DATA INPUT
# # ══════════════════════════════════════════════════════════
# elif S.step == 2:
#     section_header(f"Step 2 · Data Input  ·  [{S.problem_type}]")

#     tab_upload, tab_sample = st.tabs(["📁  Upload CSV", "🎲  Use Sample Dataset"])

#     with tab_upload:
#         uploaded = st.file_uploader("Drop your CSV here", type=["csv"])
#         if uploaded:
#             S.df = pd.read_csv(uploaded)
#             st.success(f"✓ Loaded {S.df.shape[0]:,} rows × {S.df.shape[1]} columns")

#     with tab_sample:
#         sample_choice = st.selectbox("Pick a sample dataset", [
#             "Iris (Classification)",
#             "Breast Cancer (Classification)",
#             "California Housing (Regression)",
#             "Diabetes (Regression)",
#         ])
#         if st.button("Load Sample", key="load_sample"):
#             from sklearn import datasets
#             if "Iris" in sample_choice:
#                 d = datasets.load_iris(as_frame=True)
#             elif "Cancer" in sample_choice:
#                 d = datasets.load_breast_cancer(as_frame=True)
#             elif "Housing" in sample_choice:
#                 d = datasets.fetch_california_housing(as_frame=True)
#             else:
#                 d = datasets.load_diabetes(as_frame=True)
#             S.df = pd.concat([d.data, d.target.rename("target")], axis=1)
#             st.success(f"✓ Loaded {S.df.shape[0]:,} rows × {S.df.shape[1]} columns")

#     if S.df is not None:
#         df = S.df
#         st.markdown("---")
#         # Shape metrics
#         mc = st.columns(4)
#         for col, val, lbl in zip(mc,
#             [df.shape[0], df.shape[1], int(df.isnull().sum().sum()), df.select_dtypes(include=np.number).shape[1]],
#             ["Rows", "Columns", "Missing Values", "Numeric Cols"]):
#             with col:
#                 st.markdown(f"""
#                 <div class="metric-pill" style="width:100%">
#                     <span class="val">{val:,}</span>
#                     <span class="lbl">{lbl}</span>
#                 </div>""", unsafe_allow_html=True)

#         st.markdown("<br>", unsafe_allow_html=True)

#         tab_prev, tab_pca = st.tabs(["📋  Data Preview", "🔮  PCA Shape Explorer"])

#         with tab_prev:
#             # Target selection
#             S.target = st.selectbox("Select TARGET feature", df.columns.tolist(),
#                                      index=len(df.columns)-1)

#             # Feature selection
#             feat_options = [c for c in df.columns if c != S.target]
#             S.selected_features = st.multiselect("Select INPUT features (default = all)",
#                                                    feat_options, default=feat_options)
#             if S.selected_features:
#                 st.dataframe(df[S.selected_features + [S.target]].head(20),
#                              use_container_width=True, height=300)

#         with tab_pca:
#             if S.selected_features and len(S.selected_features) >= 2:
#                 from sklearn.preprocessing import StandardScaler
#                 from sklearn.decomposition import PCA

#                 num_feats = [f for f in S.selected_features
#                              if df[f].dtype in [np.float64, np.float32, np.int64, np.int32]]
#                 if len(num_feats) >= 2:
#                     scaler = StandardScaler()
#                     X_scaled = scaler.fit_transform(df[num_feats].dropna())
#                     pca = PCA(n_components=min(3, len(num_feats)))
#                     comps = pca.fit_transform(X_scaled)
#                     var_exp = pca.explained_variance_ratio_ * 100

#                     n_comp = st.radio("PCA dimensions", [2, 3] if len(num_feats) >= 3 else [2],
#                                       horizontal=True)

#                     pca_df = pd.DataFrame(comps[:, :n_comp],
#                                           columns=[f"PC{i+1}" for i in range(n_comp)])
#                     y_vals = df[S.target].iloc[:len(pca_df)].astype(str)

#                     if n_comp == 2:
#                         fig = px.scatter(pca_df, x="PC1", y="PC2", color=y_vals,
#                                          labels={"color": S.target},
#                                          title=f"PCA 2D · Var explained: PC1={var_exp[0]:.1f}% PC2={var_exp[1]:.1f}%",
#                                          template="plotly_dark")
#                     else:
#                         fig = px.scatter_3d(pca_df, x="PC1", y="PC2", z="PC3",
#                                              color=y_vals, labels={"color": S.target},
#                                              title="PCA 3D",
#                                              template="plotly_dark")
#                     fig = styled_fig(fig)
#                     st.plotly_chart(fig, use_container_width=True)

#                     # Explained variance bar
#                     fig2 = go.Figure(go.Bar(
#                         x=[f"PC{i+1}" for i in range(len(var_exp))],
#                         y=var_exp, marker_color="#6c63ff",
#                         text=[f"{v:.1f}%" for v in var_exp], textposition="outside",
#                     ))
#                     fig2.update_layout(title="Explained Variance per Component",
#                                        yaxis_title="%", **PLOTLY_LAYOUT)
#                     st.plotly_chart(fig2, use_container_width=True)

#         if S.target and S.selected_features:
#             nav_buttons(next_label="Run EDA →", next_key="to_eda")


# # ══════════════════════════════════════════════════════════
# #  STEP 3 — EDA
# # ══════════════════════════════════════════════════════════
# elif S.step == 3:
#     section_header("Step 3 · Exploratory Data Analysis")
#     df = S.df
#     feats = S.selected_features or [c for c in df.columns if c != S.target]
#     all_cols = feats + [S.target]
#     sub = df[all_cols].copy()

#     tab1, tab2, tab3, tab4, tab5 = st.tabs([
#         "📊 Distributions", "🔗 Correlations", "🎯 Target Analysis",
#         "📦 Box Plots", "📋 Summary Stats"
#     ])

#     num_cols = sub.select_dtypes(include=np.number).columns.tolist()
#     cat_cols = sub.select_dtypes(exclude=np.number).columns.tolist()

#     with tab1:
#         if num_cols:
#             sel = st.multiselect("Choose features to plot", num_cols,
#                                   default=num_cols[:min(4, len(num_cols))])
#             if sel:
#                 rows = (len(sel) + 1) // 2
#                 fig = make_subplots(rows=rows, cols=2,
#                                     subplot_titles=[f"Distribution: {c}" for c in sel])
#                 for i, col in enumerate(sel):
#                     r, c_ = divmod(i, 2)
#                     fig.add_trace(go.Histogram(
#                         x=sub[col].dropna(), name=col,
#                         marker_color=["#6c63ff","#00d4aa","#ff6584","#ffa040"][i % 4],
#                         opacity=0.8,
#                     ), row=r+1, col=c_+1)
#                 fig.update_layout(height=300*rows, showlegend=False, **PLOTLY_LAYOUT)
#                 st.plotly_chart(fig, use_container_width=True)

#     with tab2:
#         if len(num_cols) >= 2:
#             corr = sub[num_cols].corr()
#             fig = px.imshow(corr, color_continuous_scale="RdBu_r",
#                             zmin=-1, zmax=1, aspect="auto",
#                             title="Pearson Correlation Matrix",
#                             template="plotly_dark")
#             fig.update_layout(**PLOTLY_LAYOUT)
#             st.plotly_chart(fig, use_container_width=True)

#             # Top correlations with target
#             if S.target in num_cols:
#                 target_corr = corr[S.target].drop(S.target).abs().sort_values(ascending=False)
#                 fig2 = go.Figure(go.Bar(
#                     x=target_corr.values,
#                     y=target_corr.index,
#                     orientation="h",
#                     marker_color=["#6c63ff" if v > 0.5 else "#00d4aa" if v > 0.3 else "#7a85a0"
#                                   for v in target_corr.values],
#                 ))
#                 fig2.update_layout(title=f"Feature Correlation with {S.target}",
#                                    xaxis_title="|Pearson r|", **PLOTLY_LAYOUT)
#                 st.plotly_chart(fig2, use_container_width=True)

#     with tab3:
#         if S.target:
#             if S.problem_type == "Classification":
#                 vc = sub[S.target].value_counts()
#                 fig = px.pie(values=vc.values, names=vc.index.astype(str),
#                               title=f"Class distribution: {S.target}",
#                               template="plotly_dark",
#                               color_discrete_sequence=["#6c63ff","#00d4aa","#ff6584","#ffa040"])
#                 fig.update_layout(**PLOTLY_LAYOUT)
#             else:
#                 fig = px.histogram(sub, x=S.target, nbins=40,
#                                     title=f"Target distribution: {S.target}",
#                                     template="plotly_dark",
#                                     color_discrete_sequence=["#6c63ff"])
#                 fig.update_layout(**PLOTLY_LAYOUT)
#             st.plotly_chart(fig, use_container_width=True)

#             if len(num_cols) >= 2:
#                 feat_x = st.selectbox("Feature (X-axis)", [c for c in num_cols if c != S.target])
#                 fig2 = px.scatter(sub, x=feat_x, y=S.target,
#                                    color=S.target if S.problem_type=="Classification" else None,
#                                    trendline="ols" if S.problem_type=="Regression" else None,
#                                    template="plotly_dark",
#                                    title=f"{feat_x} vs {S.target}")
#                 fig2.update_layout(**PLOTLY_LAYOUT)
#                 st.plotly_chart(fig2, use_container_width=True)

#     with tab4:
#         if num_cols:
#             sel2 = st.multiselect("Select features for box plots", num_cols,
#                                    default=num_cols[:min(6, len(num_cols))], key="box_sel")
#             if sel2:
#                 fig = go.Figure()
#                 for i, col in enumerate(sel2):
#                     color = ["#6c63ff","#00d4aa","#ff6584","#ffa040","#40c8ff","#c084fc"][i % 6]
#                     fig.add_trace(go.Box(y=sub[col].dropna(), name=col,
#                                          marker_color=color, line_color=color))
#                 fig.update_layout(title="Box Plots — Feature Spread",
#                                    showlegend=False, **PLOTLY_LAYOUT)
#                 st.plotly_chart(fig, use_container_width=True)

#     with tab5:
#         st.dataframe(sub.describe().T.style.background_gradient(
#             cmap="Blues", axis=0), use_container_width=True)
#         missing = sub.isnull().sum()
#         if missing.sum() > 0:
#             st.warning(f"⚠ {missing.sum()} missing values detected")
#             st.dataframe(missing[missing > 0].rename("Missing Count").to_frame(),
#                          use_container_width=True)

#     nav_buttons(next_label="Data Engineering →")


# # ══════════════════════════════════════════════════════════
# #  STEP 4 — DATA ENGINEERING & CLEANING
# # ══════════════════════════════════════════════════════════
# elif S.step == 4:
#     section_header("Step 4 · Data Engineering & Cleaning")

#     df = S.df.copy()
#     feats = S.selected_features or [c for c in df.columns if c != S.target]
#     num_cols = df[feats].select_dtypes(include=np.number).columns.tolist()

#     # ── Imputation
#     with st.expander("🔧 Missing Value Imputation", expanded=True):
#         imp_method = st.selectbox("Imputation strategy",
#                                    ["None", "Mean", "Median", "Mode", "Constant (0)"])
#         if imp_method != "None" and st.button("Apply Imputation"):
#             for col in num_cols:
#                 if df[col].isnull().sum() > 0:
#                     if imp_method == "Mean":
#                         df[col].fillna(df[col].mean(), inplace=True)
#                     elif imp_method == "Median":
#                         df[col].fillna(df[col].median(), inplace=True)
#                     elif imp_method == "Mode":
#                         df[col].fillna(df[col].mode()[0], inplace=True)
#                     else:
#                         df[col].fillna(0, inplace=True)
#             st.success("✓ Imputation applied")

#     # ── Outlier Detection
#     with st.expander("🔍 Outlier Detection", expanded=True):
#         outlier_method = st.selectbox("Detection method",
#                                        ["IQR", "Isolation Forest", "DBSCAN", "OPTICS"])
#         feat_for_outlier = st.multiselect("Features to check", num_cols,
#                                            default=num_cols[:min(3, len(num_cols))])

#         if feat_for_outlier and st.button("Detect Outliers"):
#             sub = df[feat_for_outlier].dropna()

#             if outlier_method == "IQR":
#                 mask = pd.Series([False] * len(sub), index=sub.index)
#                 for col in feat_for_outlier:
#                     q1, q3 = sub[col].quantile([.25, .75])
#                     iqr = q3 - q1
#                     mask |= (sub[col] < q1 - 1.5*iqr) | (sub[col] > q3 + 1.5*iqr)
#                 outlier_idx = sub[mask].index.tolist()

#             elif outlier_method == "Isolation Forest":
#                 from sklearn.ensemble import IsolationForest
#                 clf = IsolationForest(contamination=0.05, random_state=42)
#                 preds = clf.fit_predict(sub)
#                 outlier_idx = sub[preds == -1].index.tolist()

#             elif outlier_method == "DBSCAN":
#                 from sklearn.cluster import DBSCAN
#                 from sklearn.preprocessing import StandardScaler
#                 X_s = StandardScaler().fit_transform(sub)
#                 labels = DBSCAN(eps=0.5, min_samples=5).fit_predict(X_s)
#                 outlier_idx = sub[labels == -1].index.tolist()

#             else:  # OPTICS
#                 from sklearn.cluster import OPTICS
#                 from sklearn.preprocessing import StandardScaler
#                 X_s = StandardScaler().fit_transform(sub)
#                 labels = OPTICS(min_samples=5).fit_predict(X_s)
#                 outlier_idx = sub[labels == -1].index.tolist()

#             S.outlier_indices = outlier_idx
#             st.session_state["outlier_df_preview"] = df.loc[outlier_idx]

#             pct = len(outlier_idx) / len(df) * 100
#             col1, col2 = st.columns(2)
#             with col1:
#                 st.markdown(f"""
#                 <div class="metric-pill" style="width:100%;margin-bottom:1rem">
#                     <span class="val" style="color:{'#ff6584' if pct>5 else '#00d4aa'}">{len(outlier_idx)}</span>
#                     <span class="lbl">Outliers detected ({pct:.1f}%)</span>
#                 </div>""", unsafe_allow_html=True)

#             if len(feat_for_outlier) >= 2:
#                 fig = px.scatter(df, x=feat_for_outlier[0], y=feat_for_outlier[1],
#                                   template="plotly_dark",
#                                   title=f"Outlier Visualization — {outlier_method}",
#                                   color=pd.Series(df.index.isin(outlier_idx),
#                                                   name="Is Outlier").astype(str),
#                                   color_discrete_map={"True": "#ff6584", "False": "#6c63ff"})
#                 fig.update_layout(**PLOTLY_LAYOUT)
#                 st.plotly_chart(fig, use_container_width=True)

#         if S.outlier_indices:
#             st.warning(f"⚠ {len(S.outlier_indices)} outliers found. Remove them?")
#             c1, c2 = st.columns(2)
#             with c1:
#                 if st.button("🗑 Remove Outliers", key="rm_out"):
#                     df = df.drop(index=S.outlier_indices).reset_index(drop=True)
#                     S.outlier_indices = []
#                     S.df = df
#                     st.success("✓ Outliers removed")
#             with c2:
#                 if st.button("Keep Outliers", key="keep_out"):
#                     S.outlier_indices = []
#                     st.info("Outliers retained.")

#     # ── Scaling
#     with st.expander("⚖ Feature Scaling"):
#         scale_method = st.selectbox("Scaling method",
#                                      ["None", "StandardScaler", "MinMaxScaler", "RobustScaler"])
#         if scale_method != "None" and st.button("Apply Scaling"):
#             from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
#             scalers = {"StandardScaler": StandardScaler(),
#                        "MinMaxScaler": MinMaxScaler(),
#                        "RobustScaler": RobustScaler()}
#             scaler = scalers[scale_method]
#             df_scaled = df.copy()
#             df_scaled[num_cols] = scaler.fit_transform(df[num_cols])
#             S.df = df_scaled
#             st.success(f"✓ {scale_method} applied to {len(num_cols)} numeric features")

#     S.df_clean = S.df
#     nav_buttons(next_label="Feature Selection →")


# # ══════════════════════════════════════════════════════════
# #  STEP 5 — FEATURE SELECTION
# # ══════════════════════════════════════════════════════════
# elif S.step == 5:
#     section_header("Step 5 · Feature Selection")

#     df = get_df()
#     feats = S.selected_features or [c for c in df.columns if c != S.target]
#     num_cols = df[feats].select_dtypes(include=np.number).columns.tolist()

#     method = st.selectbox("Selection method", [
#         "Variance Threshold",
#         "Correlation with Target",
#         "Information Gain (Mutual Information)",
#         "All Features (no selection)",
#     ])

#     threshold_val = None
#     if method == "Variance Threshold":
#         threshold_val = st.slider("Variance threshold", 0.0, 1.0, 0.01, 0.01)
#     elif method == "Correlation with Target":
#         threshold_val = st.slider("Min |correlation| with target", 0.0, 1.0, 0.1, 0.01)
#     elif method == "Information Gain (Mutual Information)":
#         threshold_val = st.slider("Keep top N features", 1, len(num_cols), min(10, len(num_cols)))

#     if st.button("Apply Feature Selection"):
#         if method == "All Features (no selection)":
#             S.feature_selected = num_cols
#         elif method == "Variance Threshold":
#             from sklearn.feature_selection import VarianceThreshold
#             sel = VarianceThreshold(threshold=threshold_val)
#             X = df[num_cols].fillna(0)
#             sel.fit(X)
#             S.feature_selected = [c for c, s in zip(num_cols, sel.get_support()) if s]
#         elif method == "Correlation with Target":
#             if S.target in df.columns:
#                 corr = df[num_cols + [S.target]].corr()[S.target].drop(S.target).abs()
#                 S.feature_selected = corr[corr >= threshold_val].index.tolist()
#             else:
#                 S.feature_selected = num_cols
#         else:
#             from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
#             X = df[num_cols].fillna(0)
#             y = df[S.target].fillna(0)
#             fn = mutual_info_classif if S.problem_type == "Classification" else mutual_info_regression
#             scores = fn(X, y)
#             top_idx = np.argsort(scores)[::-1][:threshold_val]
#             S.feature_selected = [num_cols[i] for i in top_idx]

#         st.success(f"✓ Selected {len(S.feature_selected)} features: {', '.join(S.feature_selected[:8])}{'...' if len(S.feature_selected) > 8 else ''}")

#         # Visualize selected vs removed
#         removed = [c for c in num_cols if c not in S.feature_selected]
#         col1, col2 = st.columns(2)
#         with col1:
#             st.markdown(f'<div class="metric-pill" style="width:100%"><span class="val" style="color:#00d4aa">{len(S.feature_selected)}</span><span class="lbl">Selected Features</span></div>', unsafe_allow_html=True)
#         with col2:
#             st.markdown(f'<div class="metric-pill" style="width:100%"><span class="val" style="color:#ff6584">{len(removed)}</span><span class="lbl">Removed Features</span></div>', unsafe_allow_html=True)

#         if method in ["Information Gain (Mutual Information)", "Correlation with Target"]:
#             if S.target in df.columns:
#                 if method.startswith("Information"):
#                     from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
#                     X = df[num_cols].fillna(0)
#                     y = df[S.target].fillna(0)
#                     fn = mutual_info_classif if S.problem_type == "Classification" else mutual_info_regression
#                     scores = fn(X, y)
#                     score_df = pd.DataFrame({"feature": num_cols, "score": scores}).sort_values("score", ascending=False)
#                 else:
#                     corr = df[num_cols + [S.target]].corr()[S.target].drop(S.target).abs().reset_index()
#                     corr.columns = ["feature", "score"]
#                     score_df = corr.sort_values("score", ascending=False)

#                 fig = go.Figure(go.Bar(
#                     x=score_df["score"],
#                     y=score_df["feature"],
#                     orientation="h",
#                     marker_color=["#6c63ff" if f in S.feature_selected else "#2a2f3e"
#                                   for f in score_df["feature"]],
#                 ))
#                 fig.update_layout(title="Feature Importance Scores",
#                                    xaxis_title="Score", **PLOTLY_LAYOUT, height=max(300, len(num_cols)*28))
#                 st.plotly_chart(fig, use_container_width=True)

#     if not S.feature_selected:
#         S.feature_selected = num_cols

#     nav_buttons(next_label="Split Data →", disabled=not S.feature_selected)


# # ══════════════════════════════════════════════════════════
# #  STEP 6 — DATA SPLIT
# # ══════════════════════════════════════════════════════════
# elif S.step == 6:
#     section_header("Step 6 · Train / Test Split")

#     test_size = st.slider("Test set size (%)", 10, 40, 20, 5)
#     stratify_opt = st.checkbox("Stratified split (classification)", value=S.problem_type == "Classification")
#     random_seed = st.number_input("Random seed", 0, 9999, 42)

#     if st.button("Apply Split"):
#         from sklearn.model_selection import train_test_split
#         df = get_df()
#         feats = S.feature_selected or S.selected_features
#         feats = [f for f in feats if f in df.columns]

#         X = df[feats].fillna(0)
#         y = df[S.target]

#         strat = y if stratify_opt and S.problem_type == "Classification" else None
#         S.X_train, S.X_test, S.y_train, S.y_test = train_test_split(
#             X, y, test_size=test_size/100, random_state=random_seed, stratify=strat)

#         c1, c2, c3, c4 = st.columns(4)
#         for col_, val, lbl in zip([c1,c2,c3,c4],
#             [len(S.X_train), len(S.X_test), S.X_train.shape[1], f"{100-test_size}/{test_size}"],
#             ["Train Samples", "Test Samples", "Features", "Train/Test %"]):
#             with col_:
#                 st.markdown(f'<div class="metric-pill" style="width:100%"><span class="val">{val}</span><span class="lbl">{lbl}</span></div>', unsafe_allow_html=True)

#         # Sanity bar
#         fig = go.Figure()
#         fig.add_trace(go.Bar(name="Train", x=["Split"], y=[len(S.X_train)],
#                               marker_color="#6c63ff"))
#         fig.add_trace(go.Bar(name="Test", x=["Split"], y=[len(S.X_test)],
#                               marker_color="#00d4aa"))
#         fig.update_layout(barmode="stack", title="Train / Test Distribution",
#                            **PLOTLY_LAYOUT, height=250)
#         st.plotly_chart(fig, use_container_width=True)

#         st.success("✓ Data split complete")

#     nav_buttons(next_label="Select Model →", disabled=S.X_train is None)


# # ══════════════════════════════════════════════════════════
# #  STEP 7 — MODEL SELECTION
# # ══════════════════════════════════════════════════════════
# elif S.step == 7:
#     section_header("Step 7 · Model Selection")

#     if S.problem_type == "Classification":
#         models_available = ["Logistic Regression", "SVM (Classifier)", "Random Forest (Classifier)", "K-Nearest Neighbors"]
#     else:
#         models_available = ["Linear Regression", "SVM (Regressor)", "Random Forest (Regressor)", "K-Nearest Neighbors"]

#     S.model_name = st.selectbox("Choose a model", models_available)

#     # Model-specific options
#     model_params = {}
#     st.markdown("**Model Configuration**")

#     if "SVM" in S.model_name:
#         kernel = st.selectbox("Kernel", ["rbf", "linear", "poly", "sigmoid"])
#         C_val = st.slider("Regularization C", 0.01, 100.0, 1.0, step=0.1)
#         model_params = {"kernel": kernel, "C": C_val}

#     elif "Random Forest" in S.model_name:
#         n_trees = st.slider("Number of trees", 10, 300, 100, 10)
#         max_depth = st.slider("Max depth (0 = unlimited)", 0, 30, 0)
#         model_params = {"n_estimators": n_trees,
#                          "max_depth": max_depth if max_depth > 0 else None}

#     elif "K-Nearest" in S.model_name:
#         k_n = st.slider("K neighbors", 1, 20, 5)
#         model_params = {"n_neighbors": k_n}

#     elif "Logistic" in S.model_name:
#         C_val2 = st.slider("Regularization C", 0.01, 100.0, 1.0, 0.1)
#         model_params = {"C": C_val2, "max_iter": 1000}

#     if st.button("Confirm Model"):
#         from sklearn.linear_model import LogisticRegression, LinearRegression
#         from sklearn.svm import SVC, SVR
#         from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
#         from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

#         model_map = {
#             "Logistic Regression": LogisticRegression,
#             "Linear Regression": LinearRegression,
#             "SVM (Classifier)": SVC,
#             "SVM (Regressor)": SVR,
#             "Random Forest (Classifier)": RandomForestClassifier,
#             "Random Forest (Regressor)": RandomForestRegressor,
#             "K-Nearest Neighbors": KNeighborsClassifier if S.problem_type=="Classification" else KNeighborsRegressor,
#         }
#         S.model_obj = model_map[S.model_name](**model_params)
#         st.success(f"✓ {S.model_name} configured with {model_params}")

#     nav_buttons(next_label="Train Model →", disabled=S.model_obj is None)


# # ══════════════════════════════════════════════════════════
# #  STEP 8 — TRAINING + KFOLD
# # ══════════════════════════════════════════════════════════
# elif S.step == 8:
#     section_header("Step 8 · Model Training & K-Fold Validation")

#     S.k_folds = st.slider("Number of folds (K)", 2, 15, 5)
#     scoring_metric = ("accuracy" if S.problem_type == "Classification" else "r2")
#     st.info(f"Primary metric: **{scoring_metric}**")

#     if st.button("🚀 Train & Validate"):
#         from sklearn.model_selection import cross_validate, KFold, StratifiedKFold
#         import time

#         progress = st.progress(0, text="Initializing...")
#         time.sleep(0.2)

#         X_all = pd.concat([S.X_train, S.X_test])
#         y_all = pd.concat([S.y_train, S.y_test])

#         cv = (StratifiedKFold(n_splits=S.k_folds, shuffle=True, random_state=42)
#               if S.problem_type == "Classification"
#               else KFold(n_splits=S.k_folds, shuffle=True, random_state=42))

#         progress.progress(20, text="Running cross-validation…")
#         metrics_to_score = (["accuracy", "f1_macro", "roc_auc_ovr_weighted"]
#                              if S.problem_type == "Classification"
#                              else ["r2", "neg_mean_squared_error", "neg_mean_absolute_error"])

#         try:
#             cv_result = cross_validate(S.model_obj, X_all, y_all,
#                                         cv=cv, scoring=metrics_to_score,
#                                         return_train_score=True)
#         except Exception:
#             cv_result = cross_validate(S.model_obj, X_all, y_all,
#                                         cv=cv, scoring=scoring_metric,
#                                         return_train_score=True)

#         progress.progress(70, text="Fitting final model…")
#         S.model_obj.fit(S.X_train, S.y_train)
#         S.cv_results = cv_result
#         progress.progress(100, text="Done!")

#         st.success("✓ Training complete!")

#         # CV fold chart
#         train_scores = cv_result.get(f"train_{scoring_metric}",
#                                       cv_result.get("train_score", np.zeros(S.k_folds)))
#         test_scores  = cv_result.get(f"test_{scoring_metric}",
#                                       cv_result.get("test_score", np.zeros(S.k_folds)))

#         fig = go.Figure()
#         folds = [f"Fold {i+1}" for i in range(S.k_folds)]
#         fig.add_trace(go.Scatter(x=folds, y=train_scores, name="Train",
#                                   line=dict(color="#6c63ff", width=2),
#                                   mode="lines+markers", marker_size=8))
#         fig.add_trace(go.Scatter(x=folds, y=test_scores, name="Validation",
#                                   line=dict(color="#00d4aa", width=2),
#                                   mode="lines+markers", marker_size=8))
#         fig.update_layout(title=f"K-Fold CV — {scoring_metric}",
#                            yaxis_title="Score", **PLOTLY_LAYOUT)
#         st.plotly_chart(fig, use_container_width=True)

#         c1, c2, c3 = st.columns(3)
#         with c1:
#             st.markdown(f'<div class="metric-pill" style="width:100%"><span class="val">{np.mean(test_scores):.4f}</span><span class="lbl">Mean CV Score</span></div>', unsafe_allow_html=True)
#         with c2:
#             st.markdown(f'<div class="metric-pill" style="width:100%"><span class="val">{np.std(test_scores):.4f}</span><span class="lbl">CV Std Dev</span></div>', unsafe_allow_html=True)
#         with c3:
#             gap = np.mean(train_scores) - np.mean(test_scores)
#             lbl = "🔴 Overfit" if gap > 0.1 else ("🟡 Borderline" if gap > 0.05 else "🟢 Good Fit")
#             st.markdown(f'<div class="metric-pill" style="width:100%"><span class="val" style="font-size:1.1rem">{lbl}</span><span class="lbl">Train–Val gap: {gap:.4f}</span></div>', unsafe_allow_html=True)

#     nav_buttons(next_label="View Metrics →", disabled=S.cv_results is None)


# # ══════════════════════════════════════════════════════════
# #  STEP 9 — PERFORMANCE METRICS
# # ══════════════════════════════════════════════════════════
# elif S.step == 9:
#     section_header("Step 9 · Performance Metrics & Fit Diagnostics")

#     if S.model_obj is None or S.X_test is None:
#         st.warning("No trained model found. Please go back and train.")
#     else:
#         y_pred = S.model_obj.predict(S.X_test)
#         y_pred_train = S.model_obj.predict(S.X_train)

#         if S.problem_type == "Classification":
#             from sklearn.metrics import (accuracy_score, f1_score, precision_score,
#                                           recall_score, confusion_matrix, roc_auc_score,
#                                           classification_report)

#             acc  = accuracy_score(S.y_test, y_pred)
#             f1   = f1_score(S.y_test, y_pred, average="macro", zero_division=0)
#             prec = precision_score(S.y_test, y_pred, average="macro", zero_division=0)
#             rec  = recall_score(S.y_test, y_pred, average="macro", zero_division=0)
#             acc_train = accuracy_score(S.y_train, y_pred_train)

#             c1, c2, c3, c4 = st.columns(4)
#             for col_, val, lbl, clr in zip([c1,c2,c3,c4],
#                 [f"{acc:.4f}", f"{f1:.4f}", f"{prec:.4f}", f"{rec:.4f}"],
#                 ["Test Accuracy", "F1 Score", "Precision", "Recall"],
#                 ["#6c63ff","#00d4aa","#ffa040","#ff6584"]):
#                 with col_:
#                     st.markdown(f'<div class="metric-pill" style="width:100%"><span class="val" style="color:{clr}">{val}</span><span class="lbl">{lbl}</span></div>', unsafe_allow_html=True)

#             st.markdown("<br>", unsafe_allow_html=True)

#             # Confusion matrix
#             cm = confusion_matrix(S.y_test, y_pred)
#             labels = sorted(S.y_test.unique())
#             fig = px.imshow(cm, x=[str(l) for l in labels], y=[str(l) for l in labels],
#                              color_continuous_scale="Purples",
#                              labels=dict(x="Predicted", y="Actual", color="Count"),
#                              title="Confusion Matrix", text_auto=True, template="plotly_dark")
#             fig.update_layout(**PLOTLY_LAYOUT)
#             st.plotly_chart(fig, use_container_width=True)

#             # Over/underfitting
#             gap = acc_train - acc
#             st.markdown("### Fit Diagnostics")
#             fig2 = go.Figure(go.Bar(
#                 x=["Train Accuracy", "Test Accuracy"],
#                 y=[acc_train, acc],
#                 marker_color=["#6c63ff", "#00d4aa"],
#                 text=[f"{acc_train:.4f}", f"{acc:.4f}"],
#                 textposition="outside",
#             ))
#             fig2.add_shape(type="line", x0=-0.5, x1=1.5, y0=acc_train, y1=acc_train,
#                             line=dict(dash="dot", color="#7a85a0", width=1))
#             fig2.update_layout(title=f"Train vs Test — Gap: {gap:.4f}",
#                                 yaxis_range=[0, 1.1], **PLOTLY_LAYOUT)
#             st.plotly_chart(fig2, use_container_width=True)

#             if gap > 0.15:
#                 st.error("🔴 **Overfitting detected** — large train/test gap. Consider regularization, more data, or simpler model.")
#             elif acc < 0.6:
#                 st.warning("🟡 **Possible underfitting** — low accuracy. Consider more features or a more complex model.")
#             else:
#                 st.success("🟢 Model appears well-fitted.")

#         else:
#             from sklearn.metrics import (r2_score, mean_squared_error,
#                                           mean_absolute_error)
#             r2   = r2_score(S.y_test, y_pred)
#             rmse = np.sqrt(mean_squared_error(S.y_test, y_pred))
#             mae  = mean_absolute_error(S.y_test, y_pred)
#             r2_tr = r2_score(S.y_train, y_pred_train)

#             c1,c2,c3,c4 = st.columns(4)
#             for col_, val, lbl, clr in zip([c1,c2,c3,c4],
#                 [f"{r2:.4f}", f"{rmse:.4f}", f"{mae:.4f}", f"{r2_tr:.4f}"],
#                 ["R² (Test)","RMSE","MAE","R² (Train)"],
#                 ["#6c63ff","#ff6584","#ffa040","#00d4aa"]):
#                 with col_:
#                     st.markdown(f'<div class="metric-pill" style="width:100%"><span class="val" style="color:{clr}">{val}</span><span class="lbl">{lbl}</span></div>', unsafe_allow_html=True)

#             st.markdown("<br>", unsafe_allow_html=True)

#             # Actual vs Predicted
#             fig = go.Figure()
#             fig.add_trace(go.Scatter(x=S.y_test, y=y_pred, mode="markers",
#                                       name="Predictions",
#                                       marker=dict(color="#6c63ff", opacity=0.6, size=5)))
#             lim = [min(S.y_test.min(), min(y_pred)), max(S.y_test.max(), max(y_pred))]
#             fig.add_trace(go.Scatter(x=lim, y=lim, mode="lines", name="Perfect fit",
#                                       line=dict(color="#00d4aa", dash="dot", width=2)))
#             fig.update_layout(title="Actual vs Predicted", xaxis_title="Actual",
#                                yaxis_title="Predicted", **PLOTLY_LAYOUT)
#             st.plotly_chart(fig, use_container_width=True)

#             # Residuals
#             residuals = np.array(S.y_test) - np.array(y_pred)
#             fig2 = go.Figure()
#             fig2.add_trace(go.Scatter(x=y_pred, y=residuals, mode="markers",
#                                        marker=dict(color="#ff6584", opacity=0.6, size=5),
#                                        name="Residuals"))
#             fig2.add_hline(y=0, line_dash="dot", line_color="#00d4aa")
#             fig2.update_layout(title="Residuals Plot", xaxis_title="Predicted",
#                                 yaxis_title="Residual", **PLOTLY_LAYOUT)
#             st.plotly_chart(fig2, use_container_width=True)

#             gap = r2_tr - r2
#             if gap > 0.2:
#                 st.error("🔴 **Overfitting detected** — R² gap is large.")
#             elif r2 < 0.3:
#                 st.warning("🟡 **Underfitting** — R² is low. Try a more complex model.")
#             else:
#                 st.success("🟢 Model generalizes well.")

#     nav_buttons(next_label="Hyperparameter Tuning →")


# # ══════════════════════════════════════════════════════════
# #  STEP 10 — HYPERPARAMETER TUNING
# # ══════════════════════════════════════════════════════════
# elif S.step == 10:
#     section_header("Step 10 · Hyperparameter Tuning")

#     model_name = S.model_name or "Unknown"
#     st.markdown(f"Tuning: `{model_name}`")

#     search_method = st.radio("Search strategy", ["Grid Search", "Random Search"], horizontal=True)
#     n_iter = st.slider("Random Search iterations", 5, 50, 20) if search_method == "Random Search" else None
#     cv_k   = st.slider("CV folds for tuning", 2, 10, 3)

#     # Build param grid based on model
#     param_grid = {}
#     if "SVM" in model_name:
#         param_grid = {"C": [0.1, 1, 10, 100], "kernel": ["rbf", "linear"],
#                        "gamma": ["scale", "auto"]}
#     elif "Random Forest" in model_name:
#         param_grid = {"n_estimators": [50, 100, 200], "max_depth": [None, 5, 10, 20],
#                        "min_samples_split": [2, 5, 10]}
#     elif "Logistic" in model_name or "Linear" in model_name:
#         param_grid = {"C": [0.01, 0.1, 1, 10, 100]} if "Logistic" in model_name else {}
#     elif "K-Nearest" in model_name:
#         param_grid = {"n_neighbors": [3, 5, 7, 9, 11, 15], "weights": ["uniform", "distance"]}

#     if param_grid:
#         st.markdown("**Parameter grid:**")
#         for k, v in param_grid.items():
#             st.markdown(f"<span class='tag'>{k}</span> → {v}", unsafe_allow_html=True)

#     if param_grid and st.button("🔬 Start Tuning"):
#         from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
#         scoring = "accuracy" if S.problem_type == "Classification" else "r2"

#         progress2 = st.progress(0, text="Tuning in progress…")

#         if search_method == "Grid Search":
#             searcher = GridSearchCV(S.model_obj, param_grid, cv=cv_k,
#                                      scoring=scoring, n_jobs=-1, return_train_score=True)
#         else:
#             searcher = RandomizedSearchCV(S.model_obj, param_grid, cv=cv_k,
#                                            scoring=scoring, n_jobs=-1,
#                                            n_iter=n_iter, random_state=42,
#                                            return_train_score=True)

#         X_all = pd.concat([S.X_train, S.X_test])
#         y_all = pd.concat([S.y_train, S.y_test])
#         searcher.fit(X_all, y_all)
#         progress2.progress(100, text="Tuning complete!")

#         S.best_params = searcher.best_params_
#         best_score    = searcher.best_score_

#         st.success(f"✓ Best score: **{best_score:.4f}**")
#         st.markdown("**Best parameters:**")
#         for k, v in S.best_params.items():
#             st.markdown(f"<span class='tag green'>{k} = {v}</span>", unsafe_allow_html=True)

#         # Results table
#         results = pd.DataFrame(searcher.cv_results_)
#         if "param_C" in results.columns:
#             disp_cols = ["mean_test_score", "std_test_score", "mean_train_score"] + \
#                         [c for c in results.columns if c.startswith("param_")]
#             disp_cols = [c for c in disp_cols if c in results.columns]
#             st.dataframe(results[disp_cols].sort_values("mean_test_score", ascending=False).head(15),
#                          use_container_width=True)

#         # Score chart
#         fig = go.Figure()
#         fig.add_trace(go.Scatter(
#             x=list(range(len(results))),
#             y=results["mean_test_score"],
#             mode="markers+lines",
#             name="CV Test Score",
#             line=dict(color="#6c63ff", width=1),
#             marker=dict(size=6),
#             error_y=dict(array=results["std_test_score"].tolist(), color="#7a85a0"),
#         ))
#         fig.add_trace(go.Scatter(
#             x=list(range(len(results))),
#             y=results["mean_train_score"],
#             mode="lines",
#             name="Train Score",
#             line=dict(color="#00d4aa", dash="dot", width=1.5),
#         ))
#         best_idx = results["mean_test_score"].idxmax()
#         fig.add_vline(x=best_idx, line_dash="dot", line_color="#ff6584",
#                        annotation_text="Best", annotation_position="top right")
#         fig.update_layout(title="Tuning — All Candidates",
#                            xaxis_title="Candidate #", yaxis_title="Score", **PLOTLY_LAYOUT)
#         st.plotly_chart(fig, use_container_width=True)

#         # Retrain with best
#         if st.button("🔄 Retrain model with best params"):
#             S.model_obj = searcher.best_estimator_
#             S.model_obj.fit(S.X_train, S.y_train)
#             S.cv_results = None
#             st.success("✓ Model retrained with optimized hyperparameters! Go to Metrics tab to re-evaluate.")
#             if st.button("← View Metrics", key="go_metrics"):
#                 S.step = 9
#                 st.rerun()

#     elif not param_grid:
#         st.info("No tunable parameters defined for this model. Training is already complete.")

#     st.markdown("<br>", unsafe_allow_html=True)
#     st.markdown("""
#     <div style="background:#1c2030;border:1px solid #2a2f3e;border-radius:12px;padding:1.5rem;text-align:center">
#         <div style="font-family:'Space Mono',monospace;font-size:1.1rem;color:#00d4aa;margin-bottom:.5rem">🎉 Pipeline Complete!</div>
#         <div style="font-size:14px;color:#7a85a0">Your model has been trained, evaluated, and tuned end-to-end.</div>
#     </div>
#     """, unsafe_allow_html=True)

#     col1, col2 = st.columns(2)
#     with col1:
#         if st.button("← Back"):
#             go_back()
#     with col2:
#         if st.button("🔁 Start Over"):
#             for key in list(st.session_state.keys()):
#                 del st.session_state[key]
#             st.rerun()


# # ══════════════════════════════════════════════════════════
# #  FOOTER
# # ══════════════════════════════════════════════════════════
# st.markdown("<br><hr>", unsafe_allow_html=True)
# st.markdown("""
# <div style="text-align:center;font-size:12px;color:#7a85a0;font-family:'Space Mono',monospace;padding:.5rem 0">
#     ML PIPELINE STUDIO · Built with Streamlit + Plotly + scikit-learn
# </div>
# """, unsafe_allow_html=True)

