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
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.ensemble import IsolationForest, RandomForestClassifier, RandomForestRegressor
from sklearn.cluster import DBSCAN, OPTICS, KMeans
from sklearn.feature_selection import VarianceThreshold, mutual_info_classif, mutual_info_regression
from sklearn.model_selection import train_test_split, cross_val_score, KFold, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVC, SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score
import warnings
warnings.filterwarnings('ignore')

# --- Page Configuration ---
st.set_page_config(page_title="End-to-End ML Pipeline", layout="wide", page_icon="🚀")

# --- Custom CSS for UI styling ---
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0px 2px 10px rgba(0,0,0,0.05); }
    .stTabs [data-baseweb="tab-list"] { gap: 10px; }
    .stTabs [data-baseweb="tab"] { padding: 10px 20px; border-radius: 5px 5px 0px 0px; background-color: #eef2f5; }
    .stTabs [aria-selected="true"] { background-color: #ffffff; border-bottom: 3px solid #ff4b4b; }
    </style>
    """, unsafe_allow_html=True)

st.title("🚀 Interactive End-to-End ML Pipeline")

# --- Session State Initialization ---
if 'df' not in st.session_state:
    st.session_state.df = None
if 'target' not in st.session_state:
    st.session_state.target = None

# --- Horizontal Steps using Tabs ---
tabs = st.tabs([
    "1. Setup", "2. Data & PCA", "3. EDA", "4. Clean & Outliers", 
    "5. Feature Selection", "6. Data Split", "7. Model & K-Fold", 
    "8. Metrics", "9. Tuning"
])

# --- Step 1: Problem Type ---
with tabs[0]:
    st.header("1. Problem Formulation")
    st.session_state.problem_type = st.radio("Select Problem Type:", ["Regression", "Classification"])
    st.success(f"Pipeline configured for **{st.session_state.problem_type}**.")

# --- Step 2: Data Input & PCA ---
with tabs[1]:
    st.header("2. Data Input & Shape Analysis")
    uploaded_file = st.file_uploader("Upload Dataset (CSV)", type=["csv"])
    
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.session_state.df = df
        st.dataframe(df.head(), use_container_width=True)
        
        st.session_state.target = st.selectbox("Select Target Feature:", df.columns)
        
        st.subheader("PCA Data Shape Visualization")
        features = st.multiselect("Select features for PCA:", [c for c in df.columns if c != st.session_state.target], default=[c for c in df.columns if c != st.session_state.target][:4])
        
        if len(features) >= 2:
            temp_df = df[features].dropna()
            if not temp_df.empty:
                pca = PCA(n_components=2)
                components = pca.fit_transform(StandardScaler().fit_transform(temp_df))
                
                fig = px.scatter(
                    x=components[:, 0], y=components[:, 1], 
                    color=df.loc[temp_df.index, st.session_state.target] if st.session_state.target else None,
                    title="2D PCA Projection",
                    labels={'x': 'Principal Component 1', 'y': 'Principal Component 2'}
                )
                st.plotly_chart(fig, use_container_width=True)

# --- Step 3: EDA ---
with tabs[2]:
    st.header("3. Exploratory Data Analysis (EDA)")
    if st.session_state.df is not None:
        df = st.session_state.df
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Rows", df.shape[0])
        c2.metric("Columns", df.shape[1])
        c3.metric("Total Missing", df.isnull().sum().sum())
        
        t1, t2 = st.tabs(["Distributions", "Correlations"])
        with t1:
            numeric_cols = df.select_dtypes(include=np.number).columns
            if len(numeric_cols) > 0:
                feature = st.selectbox("Select Feature", numeric_cols)
                fig = px.histogram(df, x=feature, marginal="box", color_discrete_sequence=['#ff4b4b'])
                st.plotly_chart(fig, use_container_width=True)
        with t2:
            if len(numeric_cols) > 1:
                corr = df[numeric_cols].corr()
                fig = px.imshow(corr, text_auto=True, aspect="auto", color_continuous_scale='RdBu_r')
                st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Upload data in Step 2.")

# --- Step 4: Clean & Outliers ---
with tabs[3]:
    st.header("4. Data Engineering & Cleaning")
    if st.session_state.df is not None:
        df = st.session_state.df.copy()
        
        # Imputation
        st.subheader("Missing Value Imputation")
        missing_cols = df.columns[df.isnull().any()].tolist()
        if missing_cols:
            impute_cols = st.multiselect("Columns to impute:", missing_cols, default=missing_cols)
            impute_strategy = st.selectbox("Strategy:", ["mean", "median", "most_frequent"])
            if st.button("Apply Imputation"):
                imputer = SimpleImputer(strategy=impute_strategy)
                df[impute_cols] = imputer.fit_transform(df[impute_cols])
                st.session_state.df = df
                st.success("Imputation applied!")
        else:
            st.success("No missing values found!")

        # Outliers
        st.subheader("Outlier Detection")
        num_cols = df.select_dtypes(include=np.number).columns
        outlier_method = st.selectbox("Method:", ["IQR", "Isolation Forest", "DBSCAN", "OPTICS"])
        
        if st.button("Detect Outliers"):
            outliers_mask = np.zeros(len(df), dtype=bool)
            clean_num_df = df[num_cols].fillna(df[num_cols].median()) # Handle NaNs for outlier detection
            
            if outlier_method == "IQR":
                Q1 = clean_num_df.quantile(0.25)
                Q3 = clean_num_df.quantile(0.75)
                IQR = Q3 - Q1
                outliers_mask = ((clean_num_df < (Q1 - 1.5 * IQR)) | (clean_num_df > (Q3 + 1.5 * IQR))).any(axis=1)
            elif outlier_method == "Isolation Forest":
                iso = IsolationForest(contamination=0.05, random_state=42)
                outliers_mask = iso.fit_predict(clean_num_df) == -1
            elif outlier_method == "DBSCAN":
                dbscan = DBSCAN(eps=3, min_samples=2)
                outliers_mask = dbscan.fit_predict(StandardScaler().fit_transform(clean_num_df)) == -1
            elif outlier_method == "OPTICS":
                optics = OPTICS(min_samples=5)
                outliers_mask = optics.fit_predict(StandardScaler().fit_transform(clean_num_df)) == -1

            st.session_state.outliers_mask = outliers_mask
            st.warning(f"Detected {outliers_mask.sum()} outliers.")
            
        if 'outliers_mask' in st.session_state and st.session_state.outliers_mask.sum() > 0:
            if st.button("Remove Outliers"):
                st.session_state.df = df[~st.session_state.outliers_mask].reset_index(drop=True)
                st.session_state.outliers_mask = np.zeros(len(st.session_state.df), dtype=bool)
                st.success("Outliers removed!")
    else:
        st.info("Upload data in Step 2.")

# --- Step 5: Feature Selection ---
with tabs[4]:
    st.header("5. Feature Selection")
    if st.session_state.df is not None and st.session_state.target:
        df = st.session_state.df.dropna()
        X = df.drop(columns=[st.session_state.target]).select_dtypes(include=np.number)
        y = df[st.session_state.target]
        
        method = st.selectbox("Selection Method:", ["Variance Threshold", "Information Gain", "Correlation with Target"])
        
        if st.button("Evaluate Features"):
            if method == "Variance Threshold":
                vt = VarianceThreshold(threshold=0.05)
                vt.fit(X)
                selected = X.columns[vt.get_support()].tolist()
                st.write(f"**Features passing threshold:** {selected}")
            
            elif method == "Information Gain":
                if st.session_state.problem_type == "Classification":
                    # Encode y if categorical for info gain
                    if y.dtype == 'object': y = LabelEncoder().fit_transform(y)
                    scores = mutual_info_classif(X, y)
                else:
                    scores = mutual_info_regression(X, y)
                
                res = pd.DataFrame({'Feature': X.columns, 'Score': scores}).sort_values(by='Score', ascending=False)
                fig = px.bar(res, x='Score', y='Feature', orientation='h', title="Information Gain")
                st.plotly_chart(fig, use_container_width=True)
                
            elif method == "Correlation with Target":
                if pd.api.types.is_numeric_dtype(y):
                    corr = X.apply(lambda col: col.corr(y)).abs().sort_values(ascending=False)
                    fig = px.bar(x=corr.values, y=corr.index, orientation='h', title="Absolute Correlation with Target")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.error("Target must be numeric for Pearson correlation.")
    else:
        st.info("Upload data and select a target.")

# --- Step 6: Data Split ---
with tabs[5]:
    st.header("6. Train/Test Split")
    if st.session_state.df is not None:
        test_size = st.slider("Test Set Size (%)", 10, 50, 20, step=5) / 100.0
        if st.button("Split Data"):
            df = st.session_state.df.dropna()
            X = df.drop(columns=[st.session_state.target])
            # Basic dummy encoding for modeling
            X = pd.get_dummies(X, drop_first=True) 
            y = df[st.session_state.target]
            
            if st.session_state.problem_type == "Classification" and y.dtype == 'object':
                y = LabelEncoder().fit_transform(y)
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
            
            # Scaling
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
            
            st.session_state.split = (X_train, X_test, y_train, y_test)
            st.success(f"Split completed! Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")
    else:
        st.info("Upload data.")

# --- Step 7: Model & K-Fold ---
with tabs[6]:
    st.header("7. Model Configuration")
    models_list = ["Linear Model", "Random Forest", "SVM", "K-Means (Unsupervised)"]
    model_choice = st.selectbox("Select Algorithm:", models_list)
    
    if model_choice == "SVM":
        st.session_state.svm_kernel = st.selectbox("SVM Kernel:", ["linear", "poly", "rbf"])
        
    st.session_state.k_folds = st.slider("K-Fold Splits:", 2, 10, 5)
    st.session_state.model_choice = model_choice

# --- Step 8: Metrics & Overfitting Check ---
with tabs[7]:
    st.header("8. Model Training & Evaluation")
    if st.button("Train Model"):
        if 'split' in st.session_state:
            X_train, X_test, y_train, y_test = st.session_state.split
            is_class = st.session_state.problem_type == "Classification"
            
            # Setup Model
            if st.session_state.model_choice == "Linear Model":
                model = LogisticRegression() if is_class else LinearRegression()
            elif st.session_state.model_choice == "Random Forest":
                model = RandomForestClassifier() if is_class else RandomForestRegressor()
            elif st.session_state.model_choice == "SVM":
                kernel = st.session_state.get('svm_kernel', 'rbf')
                model = SVC(kernel=kernel) if is_class else SVR(kernel=kernel)
            elif st.session_state.model_choice == "K-Means (Unsupervised)":
                n_clusters = len(np.unique(y_train)) if is_class else 3
                model = KMeans(n_clusters=n_clusters, random_state=42)
            
            # K-Fold CV
            if st.session_state.model_choice != "K-Means (Unsupervised)":
                kf = KFold(n_splits=st.session_state.k_folds, shuffle=True, random_state=42)
                scoring = 'accuracy' if is_class else 'r2'
                cv_scores = cross_val_score(model, X_train, y_train, cv=kf, scoring=scoring)
                st.write(f"**Mean K-Fold {scoring.capitalize()}:** {cv_scores.mean():.4f}")
            
            # Train and Predict
            model.fit(X_train, y_train)
            st.session_state.trained_model = model
            
            if st.session_state.model_choice != "K-Means (Unsupervised)":
                y_pred_train = model.predict(X_train)
                y_pred_test = model.predict(X_test)
                
                # Metrics Display
                col1, col2 = st.columns(2)
                if is_class:
                    train_score = accuracy_score(y_train, y_pred_train)
                    test_score = accuracy_score(y_test, y_pred_test)
                    col1.metric("Train Accuracy", f"{train_score:.4f}")
                    col2.metric("Test Accuracy", f"{test_score:.4f}")
                else:
                    train_score = r2_score(y_train, y_pred_train)
                    test_score = r2_score(y_test, y_pred_test)
                    col1.metric("Train R²", f"{train_score:.4f}")
                    col2.metric("Test R²", f"{test_score:.4f}")
                    st.write(f"**Test MSE:** {mean_squared_error(y_test, y_pred_test):.4f}")
                
                # Overfit/Underfit Logic
                diff = train_score - test_score
                if diff > 0.15:
                    st.error("⚠️ Overfitting Detected: Train score is much higher than Test score.")
                elif test_score < 0.50:
                    st.warning("⚠️ Underfitting Detected: Model is performing poorly on both sets.")
                else:
                    st.success("✅ Model generalized well!")
                    
                # Chart
                if not is_class:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(y=y_test[:50], mode='lines+markers', name='Actual'))
                    fig.add_trace(go.Scatter(y=y_pred_test[:50], mode='lines+markers', name='Predicted'))
                    fig.update_layout(title="Actual vs Predicted (First 50 Samples)")
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.error("Split data in Step 6 first.")

# --- Step 9: Hyperparameter Tuning ---
with tabs[8]:
    st.header("9. Hyperparameter Tuning")
    st.write("Tune Random Forest hyperparameters using GridSearchCV.")
    
    if st.button("Run GridSearch"):
        if 'split' in st.session_state:
            X_train, X_test, y_train, y_test = st.session_state.split
            is_class = st.session_state.problem_type == "Classification"
            
            model = RandomForestClassifier() if is_class else RandomForestRegressor()
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20]
            }
            
            with st.spinner("Searching for best parameters..."):
                grid = GridSearchCV(model, param_grid, cv=3)
                grid.fit(X_train, y_train)
                
            st.success("Tuning Finished!")
            st.write("**Best Parameters:**", grid.best_params_)
            
            best_model = grid.best_estimator_
            y_pred = best_model.predict(X_test)
            
            if is_class:
                st.metric("Tuned Test Accuracy", f"{accuracy_score(y_test, y_pred):.4f}")
            else:
                st.metric("Tuned Test R²", f"{r2_score(y_test, y_pred):.4f}")
        else:
            st.error("Split data in Step 6 first.")
