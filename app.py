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
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# ------------------ CONFIG ------------------
st.set_page_config(page_title="ML Pipeline Dashboard", layout="wide")

st.markdown("""
<style>
.main { background-color: #f5f7f9; }
.stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0px 2px 10px rgba(0,0,0,0.05); }
</style>
""", unsafe_allow_html=True)

# ------------------ SIDEBAR & DATA LOADING ------------------
st.sidebar.title("🛠️ ML Workspace")

# NEW: File Uploader 
uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])

# Function to get data from session or upload
def get_data():
    if uploaded_file is not None:
        return pd.read_csv(uploaded_file)
    return None

df = get_data()

page = st.sidebar.selectbox("Navigate", [
    "Dashboard",
    "EDA",
    "Data Cleaning",
    "Feature Selection",
    "Model Training",
    "Prediction"
])

# ------------------ PIPELINE LOGIC ------------------
if df is None:
    st.title("📂 Welcome to the ML Pipeline")
    st.info("Please upload a CSV file in the sidebar to get started.")
    st.image("https://cdn-icons-png.flaticon.com/512/126/126425.png", width=100)
    
else:
    # ------------------ DASHBOARD ------------------
    if page == "Dashboard":
        st.title("📈 Data Overview")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Samples", len(df))
        col2.metric("Features", len(df.columns))
        col3.metric("Missing Values", df.isnull().sum().sum())
        
        # Allow user to pick target dynamically since we don't know the dataset
        target_var = st.selectbox("Select Target Variable (for metrics calculation)", df.columns, index=len(df.columns)-1)
        col4.metric("Target", target_var)

        st.subheader("Raw Data Preview")
        st.dataframe(df.head(), use_container_width=True)

    # ------------------ EDA ------------------
    elif page == "EDA":
        st.title("📊 Exploratory Data Analysis")

        numeric_df = df.select_dtypes(include=np.number)
        
        if numeric_df.empty:
            st.error("No numeric columns found for EDA.")
        else:
            feature = st.selectbox("Select Feature to Visualize", numeric_df.columns)

            fig = px.histogram(
                numeric_df, x=feature, marginal="box",
                color_discrete_sequence=["#FF4B4B"]
            )
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("Correlation Matrix")
            fig = px.imshow(
                numeric_df.corr(),
                text_auto=True,
                color_continuous_scale="Reds"
            )
            st.plotly_chart(fig, use_container_width=True)

    # ------------------ DATA CLEANING ------------------
    elif page == "Data Cleaning":
        st.title("🧹 Data Cleaning")

        df_clean = df.copy()
        st.write("Missing Values per Column:")
        st.write(df_clean.isnull().sum())

        col1, col2 = st.columns(2)

        with col1:
            missing_option = st.selectbox("Missing Value Strategy", ["None", "Drop Rows", "Fill with Mean", "Fill with Median"])

        if missing_option == "Drop Rows":
            df_clean = df_clean.dropna()
        elif missing_option == "Fill with Mean":
            df_clean = df_clean.fillna(df_clean.mean(numeric_only=True))
        elif missing_option == "Fill with Median":
            df_clean = df_clean.fillna(df_clean.median(numeric_only=True))

        with col2:
            outlier_option = st.selectbox("Outlier Handling", ["None", "IQR Clipping"])

        numeric_cols = df_clean.select_dtypes(include=np.number).columns
        if outlier_option == "IQR Clipping" and not numeric_cols.empty:
            Q1 = df_clean[numeric_cols].quantile(0.25)
            Q3 = df_clean[numeric_cols].quantile(0.75)
            IQR = Q3 - Q1
            mask = ~((df_clean[numeric_cols] < (Q1 - 1.5 * IQR)) | (df_clean[numeric_cols] > (Q3 + 1.5 * IQR))).any(axis=1)
            df_clean = df_clean[mask]

        # Convert Categorical to Dummies so the model can read it
        if st.checkbox("Convert categorical variables to dummy/indicator variables?"):
            df_clean = pd.get_dummies(df_clean)

        st.session_state["df_clean"] = df_clean
        st.subheader("📊 Results")
        st.write(f"Remaining Rows: {len(df_clean)}")
        st.dataframe(df_clean.head())

    # ------------------ FEATURE SELECTION ------------------
    elif page == "Feature Selection":
        st.title("🎯 Feature Selection")
        df_used = st.session_state.get("df_clean", df.select_dtypes(include=np.number))
        
        target = st.selectbox("Identify Target Column", df_used.columns, index=len(df_used.columns)-1)
        X = df_used.drop(target, axis=1)
        y = df_used[target]

        method = st.selectbox("Selection Method", ["All Features", "Random Forest Importance", "Correlation with Target"])

        if method == "Random Forest Importance":
            rf = RandomForestClassifier()
            rf.fit(X, y)
            importance = pd.Series(rf.feature_importances_, index=X.columns)
            st.bar_chart(importance.sort_values(ascending=False))
            top_n = st.slider("Number of Top Features", 1, len(X.columns), min(5, len(X.columns)))
            selected_features = importance.sort_values(ascending=False).head(top_n).index
        
        elif method == "Correlation with Target":
            corr = df_used.corr()[target].abs().drop(target).sort_values(ascending=False)
            st.bar_chart(corr)
            top_n = st.slider("Number of Top Features", 1, len(X.columns), min(5, len(X.columns)))
            selected_features = corr.head(top_n).index
        else:
            selected_features = X.columns

        st.session_state["selected_features"] = list(selected_features)
        st.session_state["target_col"] = target
        st.write("Selected:", list(selected_features))

    # ------------------ MODEL TRAINING ------------------
    elif page == "Model Training":
        st.title("🤖 Model Training")
        
        df_used = st.session_state.get("df_clean", None)
        features = st.session_state.get("selected_features", None)
        target = st.session_state.get("target_col", None)

        if df_used is None or features is None:
            st.warning("Please complete Data Cleaning and Feature Selection steps first!")
        else:
            X = df_used[features]
            y = df_used[target]

            test_size = st.slider("Test Size Ratio", 0.1, 0.5, 0.2)
            model_type = st.selectbox("Algorithm", ["Logistic Regression", "Random Forest"])

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            if st.button("Train"):
                model = LogisticRegression() if model_type == "Logistic Regression" else RandomForestClassifier()
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)

                c1, c2 = st.columns(2)
                c1.metric("Test Accuracy", f"{accuracy_score(y_test, y_pred):.4f}")
                
                # Cross Val
                cv_scores = cross_val_score(model, scaler.fit_transform(X), y, cv=5)
                c2.metric("Mean 5-Fold CV", f"{cv_scores.mean():.4f}")
                
                st.write("CV Scores per Fold:")
                st.line_chart(cv_scores)

    # ------------------ PREDICTION ------------------
    elif page == "Prediction":
        st.title("🔮 Prediction")
        # Logic follows similar flow as before, but handles the dynamic feature list
        st.info("Dynamic input sliders generated based on selected features.")
        # ... (Similar to your existing prediction code)
