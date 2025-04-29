import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

# Set page config
st.set_page_config(
    page_title="Logistic Regression Classifier",
    page_icon="📊",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        background-color: #f5f5f5;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
        padding: 10px 20px;
    }
    .stSelectbox {
        background-color: white;
    }
    </style>
    """, unsafe_allow_html=True)

# Title and description
st.title("📊 Advanced Logistic Regression Classifier")
st.markdown("""
    This application demonstrates binary classification using Logistic Regression with interactive visualizations
    and model evaluation metrics.
""")

# Sidebar for dataset selection and parameters
st.sidebar.header("Model Configuration")

# Dataset selection
dataset_option = st.sidebar.selectbox(
    "Select Dataset",
    ["Breast Cancer", "Iris (Binary)", "Custom Dataset"]
)

# Model parameters
st.sidebar.subheader("Model Parameters")
C = st.sidebar.slider("Regularization Strength (C)", 0.01, 10.0, 1.0)
max_iter = st.sidebar.slider("Maximum Iterations", 100, 1000, 100)
test_size = st.sidebar.slider("Test Size", 0.1, 0.4, 0.2)

# Load dataset
@st.cache_data
def load_data(dataset_name):
    if dataset_name == "Breast Cancer":
        from sklearn.datasets import load_breast_cancer
        data = load_breast_cancer()
        X = pd.DataFrame(data.data, columns=data.feature_names)
        y = data.target
    elif dataset_name == "Iris (Binary)":
        from sklearn.datasets import load_iris
        data = load_iris()
        X = pd.DataFrame(data.data, columns=data.feature_names)
        y = (data.target == 0).astype(int)  # Binary classification
    else:
        st.error("Please upload your custom dataset")
        return None, None
    return X, y

# Main content
if dataset_option != "Custom Dataset":
    X, y = load_data(dataset_option)
    
    if X is not None:
        # Data preprocessing
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Train model
        model = LogisticRegression(C=C, max_iter=max_iter)
        model.fit(X_train_scaled, y_train)

        # Make predictions
        y_pred = model.predict(X_test_scaled)
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

        # Create two columns for metrics
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Model Performance Metrics")
            report = classification_report(y_test, y_pred, output_dict=True)
            metrics_df = pd.DataFrame(report).transpose()
            st.dataframe(metrics_df.style.background_gradient(cmap='RdYlGn'))

        with col2:
            st.subheader("Confusion Matrix")
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
            st.pyplot(fig)

        # ROC Curve
        st.subheader("ROC Curve")
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=fpr, y=tpr, name=f'ROC curve (AUC = {roc_auc:.2f})'))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], name='Random', line=dict(dash='dash')))
        fig.update_layout(title='Receiver Operating Characteristic (ROC) Curve',
                         xaxis_title='False Positive Rate',
                         yaxis_title='True Positive Rate')
        st.plotly_chart(fig)

        # Feature Importance
        st.subheader("Feature Importance")
        feature_importance = pd.DataFrame({
            'Feature': X.columns,
            'Importance': np.abs(model.coef_[0])
        })
        feature_importance = feature_importance.sort_values('Importance', ascending=False)
        
        fig = px.bar(feature_importance.head(10), 
                    x='Importance', 
                    y='Feature',
                    orientation='h',
                    title='Top 10 Most Important Features')
        st.plotly_chart(fig)

        # Sigmoid Function Visualization
        st.subheader("Sigmoid Function")
        x = np.linspace(-10, 10, 100)
        y = 1 / (1 + np.exp(-x))
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x, y=y, name='Sigmoid Function'))
        fig.update_layout(title='Sigmoid Function',
                         xaxis_title='Input',
                         yaxis_title='Probability')
        st.plotly_chart(fig)

        # Model Coefficients
        st.subheader("Model Coefficients")
        coef_df = pd.DataFrame({
            'Feature': X.columns,
            'Coefficient': model.coef_[0]
        })
        st.dataframe(coef_df.style.background_gradient(cmap='RdYlGn'))

else:
    st.info("Please upload your custom dataset in CSV format")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.write("Preview of uploaded data:")
        st.dataframe(data.head()) 