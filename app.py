import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# ------------------ PAGE CONFIG ------------------
st.set_page_config(page_title="Diabetes ML Dashboard", layout="wide")
st.title("🚀 Diabetes Progression Prediction Dashboard")

# ------------------ LOAD DATA ------------------
@st.cache_data
def load_data():
    data = load_diabetes(as_frame=True)
    return data.frame

df = load_data()

# ------------------ SIDEBAR ------------------
st.sidebar.title("Navigation")
step = st.sidebar.radio("Go to", [
    "1. Data Overview",
    "2. EDA",
    "3. Model Training",
    "4. Prediction"
])

# ------------------ STEP 1 ------------------
if step == "1. Data Overview":
    st.header("📊 Dataset Overview")
    st.write(df.head())
    st.write("Shape:", df.shape)

# ------------------ STEP 2 ------------------
elif step == "2. EDA":
    st.header("📈 Exploratory Data Analysis")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Correlation Heatmap")
        fig, ax = plt.subplots()
        sns.heatmap(df.corr(), annot=False, cmap="coolwarm", ax=ax)
        st.pyplot(fig)

    with col2:
        st.subheader("Target Distribution")
        fig, ax = plt.subplots()
        sns.histplot(df["target"], kde=True, ax=ax)
        st.pyplot(fig)

# ------------------ STEP 3 ------------------
elif step == "3. Model Training":
    st.header("🤖 Model Training & Evaluation")

    # Feature selection toggle
    use_top = st.checkbox("Use Top 3 Features Only (bmi, bp, s5)")

    if use_top:
        X = df[['bmi', 'bp', 's5']]
    else:
        X = df.drop('target', axis=1)

    y = df['target']

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Scaling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Model selection
    model_name = st.selectbox("Select Model", ["Linear Regression", "Random Forest"])

    if model_name == "Linear Regression":
        model = LinearRegression()
    else:
        model = RandomForestRegressor(n_estimators=100, random_state=42)

    # Train
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)

    # Metrics
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    st.subheader("📊 Performance Metrics")
    col1, col2 = st.columns(2)
    col1.metric("R2 Score", round(r2, 4))
    col2.metric("MAE", round(mae, 2))

    # Cross Validation
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X, y, cv=kf, scoring='r2')

    st.subheader("📉 K-Fold Cross Validation (R2)")
    st.line_chart(cv_scores)

    # Feature importance (only for RF)
    if model_name == "Random Forest":
        st.subheader("🔥 Feature Importance")
        importances = model.feature_importances_
        st.bar_chart(pd.Series(importances, index=X.columns))

# ------------------ STEP 4 ------------------
elif step == "4. Prediction":
    st.header("🔮 Predict Diabetes Progression")

    st.sidebar.subheader("Enter Patient Data")

    # Use top features for prediction (simpler UI)
    bmi = st.sidebar.slider("BMI", float(df.bmi.min()), float(df.bmi.max()))
    bp = st.sidebar.slider("BP", float(df.bp.min()), float(df.bp.max()))
    s5 = st.sidebar.slider("S5", float(df.s5.min()), float(df.s5.max()))

    input_data = np.array([[bmi, bp, s5]])

    # Train model on top features
    X = df[['bmi', 'bp', 's5']]
    y = df['target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)

    model = RandomForestRegressor()
    model.fit(X_train, y_train)

    # Scale input
    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)

    st.success(f"Predicted Disease Progression Score: {prediction[0]:.2f}")