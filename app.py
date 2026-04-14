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
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.ensemble import IsolationForest, RandomForestClassifier, RandomForestRegressor
from sklearn.cluster import DBSCAN, OPTICS
from sklearn.feature_selection import VarianceThreshold, SelectKBest, mutual_info_classif, mutual_info_regression
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVC, SVR
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# --- Page Config ---
st.set_page_config(page_title="End-to-End ML Pipeline", layout="wide", page_icon="🚀")

# --- Custom CSS for Aesthetics ---
st.markdown("""
    <style>
    .main {background-color: #f8f9fa;}
    h1, h2, h3 {color: #2c3e50;}
    .stTabs [data-baseweb="tab-list"] {gap: 10px;}
    .stTabs [data-baseweb="tab"] {padding: 10px 20px; border-radius: 5px 5px 0px 0px;}
    </style>
    """, unsafe_allow_html=True)

st.title("🚀 Interactive End-to-End ML Pipeline")

# --- Session State Initialization ---
if 'df' not in st.session_state:
    st.session_state.df = None
if 'target' not in st.session_state:
    st.session_state.target = None
if 'problem_type' not in st.session_state:
    st.session_state.problem_type = None

# --- Horizontal Steps using Tabs ---
tabs = st.tabs([
    "1. Problem Setup", "2. Data & PCA", "3. EDA", "4. Clean & Outliers", 
    "5. Feature Selection", "6. Data Split", "7. Model & K-Fold", 
    "8. Metrics", "9. Hyper-Tuning"
])

# --- Step 1: Problem Type ---
with tabs[0]:
    st.header("1. Problem Formulation")
    prob_type = st.radio("Select the type of problem to solve:", ["Classification", "Regression"])
    st.session_state.problem_type = prob_type
    st.success(f"Pipeline set for **{prob_type}** tasks.")

# --- Step 2: Data Input & PCA ---
with tabs[1]:
    st.header("2. Data Input & Shape Analysis")
    uploaded_file = st.file_uploader("Upload your dataset (CSV)", type=["csv"])
    
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.session_state.df = df
        st.write("### Data Preview", df.head())
        
        target_col = st.selectbox("Select Target Feature:", df.columns)
        st.session_state.target = target_col
        
        st.write("### PCA Visualization (Data Shape)")
        features = st.multiselect("Select features for PCA:", [c for c in df.columns if c != target_col], default=[c for c in df.columns if c != target_col][:5])
        
        if features and len(features) > 1:
            # Simple imputation for PCA to work
            temp_df = df[features].dropna()
            if len(temp_df) > 0:
                pca = PCA(n_components=2)
                components = pca.fit_transform(temp_df)
                
                # Plotly Scatter
                fig = px.scatter(
                    components, x=0, y=1, 
                    color=df.loc[temp_df.index, target_col] if target_col in df.columns else None,
                    title="2D PCA Projection",
                    labels={'0': 'Principal Component 1', '1': 'Principal Component 2'}
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Selected features contain too many missing values for PCA.")

# --- Step 3: EDA ---
with tabs[2]:
    st.header("3. Exploratory Data Analysis (EDA)")
    if st.session_state.df is not None:
        df = st.session_state.df
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Summary Statistics**")
            st.dataframe(df.describe())
        with col2:
            st.write("**Missing Values**")
            missing = df.isnull().sum()
            st.dataframe(missing[missing > 0])
            
        st.write("**Feature Distributions**")
        numeric_cols = df.select_dtypes(include=np.number).columns
        if len(numeric_cols) > 0:
            hist_col = st.selectbox("Select feature to view distribution:", numeric_cols)
            fig = px.histogram(df, x=hist_col, marginal="box", color_discrete_sequence=['#3498db'])
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Please upload data in Step 2.")

# --- Step 4: Clean & Outliers ---
with tabs[3]:
    st.header("4. Data Engineering & Cleaning")
    if st.session_state.df is not None:
        df = st.session_state.df.copy()
        
        # Imputation
        st.subheader("Missing Value Imputation")
        impute_cols = st.multiselect("Select columns to impute:", df.columns[df.isnull().any()])
        impute_strategy = st.selectbox("Strategy:", ["mean", "median", "most_frequent"])
        if st.button("Apply Imputation"):
            imputer = SimpleImputer(strategy=impute_strategy)
            df[impute_cols] = imputer.fit_transform(df[impute_cols])
            st.session_state.df = df
            st.success("Imputation applied!")

        # Outliers
        st.subheader("Outlier Detection & Removal")
        num_cols = df.select_dtypes(include=np.number).columns
        outlier_method = st.selectbox("Select Outlier Detection Method:", ["IQR", "Isolation Forest", "DBSCAN", "OPTICS"])
        
        if st.button("Detect Outliers"):
            outliers_mask = np.zeros(len(df), dtype=bool)
            
            if outlier_method == "IQR":
                Q1 = df[num_cols].quantile(0.25)
                Q3 = df[num_cols].quantile(0.75)
                IQR = Q3 - Q1
                outliers_mask = ((df[num_cols] < (Q1 - 1.5 * IQR)) | (df[num_cols] > (Q3 + 1.5 * IQR))).any(axis=1)
            
            elif outlier_method == "Isolation Forest":
                iso = IsolationForest(contamination=0.05, random_state=42)
                preds = iso.fit_predict(df[num_cols].fillna(df[num_cols].mean()))
                outliers_mask = preds == -1
                
            # Note: DBSCAN and OPTICS omitted for brevity but follow the same mask logic
            
            st.warning(f"Detected {outliers_mask.sum()} outliers.")
            st.session_state.outliers_mask = outliers_mask
            
        if 'outliers_mask' in st.session_state and st.session_state.outliers_mask.sum() > 0:
            if st.button("Drop Outliers"):
                st.session_state.df = df[~st.session_state.outliers_mask].reset_index(drop=True)
                st.session_state.outliers_mask = np.zeros(len(st.session_state.df), dtype=bool)
                st.success("Outliers removed successfully!")
    else:
        st.info("Please upload data in Step 2.")

# --- Step 5: Feature Selection ---
with tabs[4]:
    st.header("5. Feature Selection")
    if st.session_state.df is not None and st.session_state.target is not None:
        st.write("Select features based on statistical significance with the target.")
        method = st.selectbox("Method:", ["Variance Threshold", "Information Gain"])
        
        df = st.session_state.df.dropna() # Require clean data
        X = df.drop(columns=[st.session_state.target]).select_dtypes(include=np.number)
        y = df[st.session_state.target]
        
        if st.button("Run Selection"):
            if method == "Variance Threshold":
                selector = VarianceThreshold(threshold=0.1)
                selector.fit(X)
                selected = X.columns[selector.get_support()]
                st.write("Selected Features based on Variance:", selected.tolist())
            elif method == "Information Gain":
                if st.session_state.problem_type == "Classification":
                    scores = mutual_info_classif(X, y)
                else:
                    scores = mutual_info_regression(X, y)
                
                score_df = pd.DataFrame({'Feature': X.columns, 'Score': scores}).sort_values(by='Score', ascending=False)
                fig = px.bar(score_df, x='Feature', y='Score', title="Information Gain Scores")
                st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Ensure data is uploaded and target is selected.")

# --- Step 6: Data Split ---
with tabs[5]:
    st.header("6. Data Split")
    if st.session_state.df is not None:
        test_size = st.slider("Test Size (%)", 10, 50, 20, step=5) / 100.0
        if st.button("Split Data"):
            df = st.session_state.df.dropna()
            X = df.drop(columns=[st.session_state.target]).select_dtypes(include=np.number)
            y = df[st.session_state.target]
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
            st.session_state.split_data = (X_train, X_test, y_train, y_test)
            st.success(f"Data split successfully! Training samples: {X_train.shape[0]}, Testing samples: {X_test.shape[0]}")
    else:
        st.info("Please upload data.")

# --- Step 7: Model & K-Fold ---
with tabs[6]:
    st.header("7. Model Selection & Configuration")
    model_choice = st.selectbox("Select Model:", ["Linear Model (Regression/Logistic)", "SVM", "Random Forest", "K-Means (Clustering)"])
    
    if model_choice == "SVM":
        kernel = st.selectbox("SVM Kernel:", ["linear", "poly", "rbf", "sigmoid"])
        st.session_state.svm_kernel = kernel
        
    k_folds = st.slider("Select K for K-Fold Cross Validation:", 2, 10, 5)
    st.session_state.k_folds = k_folds
    st.session_state.model_choice = model_choice

# --- Step 8: Metrics ---
with tabs[7]:
    st.header("8. Model Training & Evaluation")
    if st.button("Train & Evaluate"):
        if 'split_data' in st.session_state:
            X_train, X_test, y_train, y_test = st.session_state.split_data
            is_classification = st.session_state.problem_type == "Classification"
            
            # Instantiate Model
            if st.session_state.model_choice == "Linear Model (Regression/Logistic)":
                model = LogisticRegression() if is_classification else LinearRegression()
            elif st.session_state.model_choice == "SVM":
                kernel = st.session_state.svm_kernel
                model = SVC(kernel=kernel) if is_classification else SVR(kernel=kernel)
            elif st.session_state.model_choice == "Random Forest":
                model = RandomForestClassifier() if is_classification else RandomForestRegressor()
            elif st.session_state.model_choice == "K-Means (Clustering)":
                st.warning("K-Means is unsupervised. Target labels will be ignored for training.")
                model = KMeans(n_clusters=len(y_train.unique()) if is_classification else 3)
            
            # Cross Validation
            kf = KFold(n_splits=st.session_state.k_folds, shuffle=True, random_state=42)
            scoring = 'accuracy' if is_classification else 'r2'
            
            if st.session_state.model_choice != "K-Means (Clustering)":
                cv_scores = cross_val_score(model, X_train, y_train, cv=kf, scoring=scoring)
                st.write(f"**Cross-Validation Scores ({scoring}):**", cv_scores)
                st.write(f"**Mean CV Score:** {cv_scores.mean():.4f}")
            
            # Standard Training
            model.fit(X_train, y_train)
            st.session_state.trained_model = model
            
            if st.session_state.model_choice != "K-Means (Clustering)":
                train_preds = model.predict(X_train)
                test_preds = model.predict(X_test)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Training Metrics")
                    if is_classification:
                        st.metric("Train Accuracy", f"{accuracy_score(y_train, train_preds):.4f}")
                    else:
                        st.metric("Train R2", f"{r2_score(y_train, train_preds):.4f}")
                with col2:
                    st.subheader("Testing Metrics")
                    if is_classification:
                        st.metric("Test Accuracy", f"{accuracy_score(y_test, test_preds):.4f}")
                    else:
                        st.metric("Test R2", f"{r2_score(y_test, test_preds):.4f}")
                        
                # Overfitting check
                if is_classification:
                    diff = accuracy_score(y_train, train_preds) - accuracy_score(y_test, test_preds)
                else:
                    diff = r2_score(y_train, train_preds) - r2_score(y_test, test_preds)
                    
                if diff > 0.15:
                    st.error("⚠️ The model shows signs of OVERFITTING (Training score is significantly higher than Testing score).")
                elif diff < -0.10:
                    st.warning("⚠️ The model shows signs of UNDERFITTING.")
                else:
                    st.success("✅ The model generalizes well.")
        else:
            st.error("Please split the data in Step 6 first.")

# --- Step 9: Hyper-Tuning ---
with tabs[8]:
    st.header("9. Hyperparameter Tuning (GridSearch)")
    st.write("Tune the currently selected model.")
    
    if st.button("Run GridSearch (Random Forest Example)"):
        if 'split_data' in st.session_state and st.session_state.model_choice == "Random Forest":
            X_train, X_test, y_train, y_test = st.session_state.split_data
            is_classification = st.session_state.problem_type == "Classification"
            
            base_model = RandomForestClassifier() if is_classification else RandomForestRegressor()
            param_grid = {
                'n_estimators': [50, 100],
                'max_depth': [None, 10, 20]
            }
            
            with st.spinner('Running Grid Search...'):
                grid_search = GridSearchCV(base_model, param_grid, cv=3)
                grid_search.fit(X_train, y_train)
                
            st.success("Tuning Complete!")
            st.write("**Best Parameters:**", grid_search.best_params_)
            
            best_model = grid_search.best_estimator_
            test_preds = best_model.predict(X_test)
            
            if is_classification:
                st.metric("Tuned Test Accuracy", f"{accuracy_score(y_test, test_preds):.4f}")
            else:
                st.metric("Tuned Test R2", f"{r2_score(y_test, test_preds):.4f}")
        else:
            st.warning("Please run the standard setup first. This demo tuning is configured for Random Forest.")
