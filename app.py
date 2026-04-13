import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# --- Page Configuration ---
st.set_page_config(page_title="ML Pipeline Dashboard", layout="wide")

# --- Custom CSS for UI styling ---
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0px 2px 10px rgba(0,0,0,0.05); }
    </style>
    """, unsafe_allow_stdio=True)

# --- 1. Input Data (Load Dataset) ---
@st.cache_data
def load_wine_data():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
    data = pd.read_csv(url, sep=';')
    return data

df = load_wine_data()

# --- Sidebar Navigation (Based on your UI images) ---
st.sidebar.title("🛠️ ML Workspace")
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/2103/2103633.png", width=100)
page = st.sidebar.selectbox("Navigate Pipeline", [
    "Dashboard Overview", 
    "Exploratory Data Analysis", 
    "Engineering & Selection", 
    "Model Training & Metrics"
])

# --- PAGE: Dashboard Overview ---
if page == "Dashboard Overview":
    st.title("🍷 Red Wine Quality Prediction")
    st.write("This dashboard showcases a full Machine Learning pipeline to predict wine quality based on physicochemical tests.")
    
    # Metric Cards
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Samples", len(df))
    col2.metric("Features", len(df.columns)-1)
    col3.metric("Missing Values", df.isnull().sum().sum())
    col4.metric("Target Variable", "Quality (3-8)")
    
    st.subheader("Raw Data Preview")
    st.dataframe(df.head(10), use_container_width=True)

# --- PAGE: Exploratory Data Analysis ---
elif page == "Exploratory Data Analysis":
    st.title("📊 Exploratory Data Analysis (EDA)")
    
    tab1, tab2 = st.tabs(["Distributions", "Correlations"])
    
    with tab1:
        feature = st.selectbox("Select Feature to Visualize", df.columns)
        fig = px.histogram(df, x=feature, marginal="box", title=f"Distribution of {feature}", color_discrete_sequence=['#722F37'])
        st.plotly_chart(fig, use_container_width=True)
        
    with tab2:
        st.subheader("Feature Correlation Matrix")
        corr = df.corr()
        fig = px.imshow(corr, text_auto=True, aspect="auto", color_continuous_scale='RdBu_r')
        st.plotly_chart(fig, use_container_width=True)

# --- PAGE: Engineering & Selection ---
elif page == "Engineering & Selection":
    st.title("⚙️ Data Engineering & Feature Selection")
    
    st.info("Step 3 & 4: Cleaning, Scaling, and Selection")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("### Feature Scaling")
        st.write("Standardizing data using `StandardScaler` to ensure mean=0 and variance=1.")
        scaler = StandardScaler()
        scaled_X = scaler.fit_transform(df.drop('quality', axis=1))
        st.dataframe(pd.DataFrame(scaled_X, columns=df.columns[:-1]).head())
        
    with col2:
        st.write("### Feature Importance")
        rf = RandomForestRegressor()
        rf.fit(df.drop('quality', axis=1), df['quality'])
        importance = pd.DataFrame({'Feature': df.columns[:-1], 'Importance': rf.feature_importances_}).sort_values(by='Importance', ascending=False)
        fig = px.bar(importance, x='Importance', y='Feature', orientation='h', title="Random Forest Feature Importance")
        st.plotly_chart(fig, use_container_width=True)

# --- PAGE: Model Training & Metrics ---
elif page == "Model Training & Metrics":
    st.title("🤖 Model Training & Performance")
    
    # Sidebar params for this page
    st.sidebar.subheader("Training Parameters")
    test_size = st.sidebar.slider("Test Set Size", 0.1, 0.4, 0.2)
    k_folds = st.sidebar.slider("K-Fold Splits", 2, 10, 5)
    
    # 5. Data Split
    X = df.drop('quality', axis=1)
    y = df['quality']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    
    # 6. Model Selection
    model_choice = st.selectbox("Select Algorithm", ["Linear Regression", "Random Forest Regressor"])
    
    if model_choice == "Linear Regression":
        model = LinearRegression()
    else:
        model = RandomForestRegressor(n_estimators=100)
    
    # 7 & 8. Training & K-Fold
    if st.button("Run Pipeline"):
        with st.spinner("Training model..."):
            kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
            cv_scores = cross_val_score(model, X_train, y_train, cv=kf)
            
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            # 9. Performance Metrics
            st.success("Training Complete!")
            
            m1, m2, m3 = st.columns(3)
            m1.metric("Test R² Score", f"{r2_score(y_test, y_pred):.4f}")
            m2.metric("Mean Absolute Error", f"{mean_absolute_error(y_test, y_pred):.4f}")
            m3.metric("Avg K-Fold Score", f"{cv_scores.mean():.4f}")
            
            # Visualization of results
            res_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred}).head(50)
            fig = go.Figure()
            fig.add_trace(go.Scatter(y=res_df['Actual'], mode='lines+markers', name='Actual'))
            fig.add_trace(go.Scatter(y=res_df['Predicted'], mode='lines+markers', name='Predicted'))
            fig.update_layout(title="Actual vs Predicted (Test Set Preview)")
            st.plotly_chart(fig, use_container_width=True)
