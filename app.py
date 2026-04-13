import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# ------------------ PAGE CONFIG ------------------
st.set_page_config(page_title="ML Pipeline Dashboard", layout="wide")

# ------------------ FIXED CSS ------------------
st.markdown("""
<style>
.main { background-color: #f5f7f9; }
.stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0px 2px 10px rgba(0,0,0,0.05); }
</style>
""", unsafe_allow_html=True)

# ------------------ LOAD DATA ------------------
@st.cache_data
def load_data():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
    return pd.read_csv(url, sep=';')

df = load_data()

# ------------------ SIDEBAR ------------------
st.sidebar.title("🛠️ ML Workspace")
page = st.sidebar.selectbox("Navigate Pipeline", [
    "Dashboard Overview", 
    "Exploratory Data Analysis", 
    "Engineering & Selection", 
    "Model Training & Metrics",
    "Prediction"
])

# ------------------ DASHBOARD ------------------
if page == "Dashboard Overview":
    st.title("🍷 Red Wine Quality Prediction")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Samples", len(df))
    col2.metric("Features", len(df.columns)-1)
    col3.metric("Missing Values", df.isnull().sum().sum())
    col4.metric("Target Variable", "Quality")

    st.dataframe(df.head(10), use_container_width=True)

# ------------------ EDA ------------------
elif page == "Exploratory Data Analysis":
    st.title("📊 EDA")

    feature = st.selectbox("Select Feature", df.columns)
    fig = px.histogram(df, x=feature, marginal="box", color_discrete_sequence=['#722F37'])
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Correlation Matrix")
    fig = px.imshow(df.corr(), text_auto=True, color_continuous_scale='RdBu_r')
    st.plotly_chart(fig, use_container_width=True)

# ------------------ ENGINEERING ------------------
elif page == "Engineering & Selection":
    st.title("⚙️ Feature Engineering")

    X = df.drop('quality', axis=1)
    y = df['quality']

    scaler = StandardScaler()
    scaled = scaler.fit_transform(X)

    st.write("Scaled Data Preview")
    st.dataframe(pd.DataFrame(scaled, columns=X.columns).head())

    rf = RandomForestRegressor()
    rf.fit(X, y)

    importance = pd.Series(rf.feature_importances_, index=X.columns).sort_values()

    st.subheader("Feature Importance")
    st.bar_chart(importance)

# ------------------ MODEL TRAINING ------------------
elif page == "Model Training & Metrics":
    st.title("🤖 Model Training")

    test_size = st.slider("Test Size", 0.1, 0.4, 0.2)
    k_folds = st.slider("K-Folds", 2, 10, 5)

    X = df.drop('quality', axis=1)
    y = df['quality']

    # CORRECT FLOW
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model_choice = st.selectbox("Model", ["Linear Regression", "Random Forest"])

    if st.button("Run Pipeline"):
        model = LinearRegression() if model_choice == "Linear Regression" else RandomForestRegressor()

        kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
        cv_scores = cross_val_score(model, X_train, y_train, cv=kf)

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        col1, col2, col3 = st.columns(3)
        col1.metric("R2 Score", f"{r2_score(y_test, y_pred):.4f}")
        col2.metric("MAE", f"{mean_absolute_error(y_test, y_pred):.4f}")
        col3.metric("CV Score", f"{cv_scores.mean():.4f}")

        fig = go.Figure()
        fig.add_trace(go.Scatter(y=y_test.values[:50], mode='lines', name='Actual'))
        fig.add_trace(go.Scatter(y=y_pred[:50], mode='lines', name='Predicted'))
        st.plotly_chart(fig)

# ------------------ PREDICTION ------------------
elif page == "Prediction":
    st.title("🔮 Predict Wine Quality")

    X = df.drop('quality', axis=1)
    y = df['quality']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = RandomForestRegressor()
    model.fit(X_scaled, y)

    st.sidebar.subheader("Enter Features")

    user_input = []
    for col in X.columns:
        val = st.sidebar.slider(col, float(X[col].min()), float(X[col].max()))
        user_input.append(val)

    input_array = np.array([user_input])
    input_scaled = scaler.transform(input_array)

    prediction = model.predict(input_scaled)[0]

    st.success(f"Predicted Wine Quality: {round(prediction,2)}")
