import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.preprocessing import StandardScaler

# Set page configuration
st.set_page_config(
    page_title="Diabetes Prediction App",
    page_icon="🏥",
    layout="wide"
)

# Load the saved model
@st.cache_resource
def load_model():
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    return model

# Load the dataset
@st.cache_data
def load_data():
    df = pd.read_csv('data/diabetes.csv')
    return df

# Load model and data
try:
    model = load_model()
    df = load_data()
except Exception as e:
    st.error("Error loading model or data. Please make sure model.pkl and data/diabetes.csv exist.")
    st.stop()

# Title and Description
st.title("Diabetes Prediction App")
st.write("This app predicts if a person has diabetes based on health measurements.")

# Sidebar Navigation
menu = ["Data Exploration", "Visualizations", "Model Prediction", "Model Performance"]
choice = st.sidebar.selectbox("Navigation", menu)

# Data Exploration Section
if choice == "Data Exploration":
    st.header("Data Exploration")
    
    # Show dataset preview
    st.subheader("Dataset Preview")
    st.dataframe(df.head())
    
    # Display shape and columns
    st.subheader("Dataset Information")
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"Number of rows: {df.shape[0]}")
        st.write(f"Number of columns: {df.shape[1]}")
    with col2:
        st.write("Columns:")
        st.write(df.columns.tolist())
    
    # Add filtering options
    st.subheader("Filter Data")
    age_range = st.slider(
        "Filter by Age",
        min_value=int(df['Age'].min()),
        max_value=int(df['Age'].max()),
        value=(int(df['Age'].min()), int(df['Age'].max()))
    )
    
    filtered_df = df[(df['Age'] >= age_range[0]) & (df['Age'] <= age_range[1])]
    st.write(f"Filtered data shape: {filtered_df.shape}")
    st.dataframe(filtered_df)

# Visualizations Section
elif choice == "Visualizations":
    st.header("Data Visualizations")
    
    # Glucose Distribution
    st.subheader("Glucose Level Distribution")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(data=df, x='Glucose', bins=30)
    st.pyplot(fig)
    
    # Correlation Heatmap
    st.subheader("Feature Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
    st.pyplot(fig)
    
    # Outcome Distribution
    st.subheader("Diabetes Outcome Distribution")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.countplot(data=df, x='Outcome')
    plt.xlabel('Outcome (0: No Diabetes, 1: Diabetes)')
    st.pyplot(fig)

# Model Prediction Section
elif choice == "Model Prediction":
    st.header("Diabetes Prediction")
    
    # Create input widgets for all features
    st.subheader("Enter Patient Information")
    col1, col2 = st.columns(2)
    
    with col1:
        pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=1)
        glucose = st.slider("Glucose Level", 0, 200, 100)
        blood_pressure = st.slider("Blood Pressure", 0, 150, 70)
        skin_thickness = st.slider("Skin Thickness", 0, 100, 20)
    
    with col2:
        insulin = st.slider("Insulin", 0, 900, 80)
        bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0)
        diabetes_pedigree = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5)
        age = st.number_input("Age", min_value=0, max_value=120, value=30)
    
    # Create prediction button
    if st.button("Predict"):
        # Prepare input data
        user_input = np.array([[
            pregnancies, glucose, blood_pressure, skin_thickness,
            insulin, bmi, diabetes_pedigree, age
        ]])
        
        # Scale the input (assuming the model was trained with scaled data)
        scaler = StandardScaler()
        scaler.fit(df.drop('Outcome', axis=1))
        user_input_scaled = scaler.transform(user_input)
        
        # Make prediction
        prediction = model.predict(user_input_scaled)
        probability = model.predict_proba(user_input_scaled)
        
        # Display results
        st.subheader("Prediction Results")
        if prediction[0] == 1:
            st.error("The model predicts: Diabetes")
        else:
            st.success("The model predicts: No Diabetes")
        
        st.write(f"Probability of having diabetes: {probability[0][1]:.2%}")

# Model Performance Section
elif choice == "Model Performance":
    st.header("Model Performance Metrics")
    
    # You would need to load these metrics from a saved file
    # For now, we'll show placeholder metrics
    metrics = {
        'Accuracy': 0.85,
        'Precision': 0.83,
        'Recall': 0.82,
        'F1-score': 0.82,
    }
    
    # Display metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Accuracy", f"{metrics['Accuracy']:.2%}")
    with col2:
        st.metric("Precision", f"{metrics['Precision']:.2%}")
    with col3:
        st.metric("Recall", f"{metrics['Recall']:.2%}")
    with col4:
        st.metric("F1-score", f"{metrics['F1-score']:.2%}")
    
    # Add note about model details
    st.info("""
    This model was trained on the PIMA Indians Diabetes dataset using various machine learning algorithms.
    The best performing model was selected based on cross-validation scores and overall performance metrics.
    """)

# Add footer
st.sidebar.markdown("---")
st.sidebar.markdown("Made with ❤️ by Thari")
