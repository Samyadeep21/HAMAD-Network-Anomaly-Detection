"""
HAMAD Network Anomaly Detection Dashboard
"""
#import streamlit as st

#st.title("Test: Streamlit Dashboard is Running!")
#st.write("If you see this message, Streamlit is working!")

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix, roc_curve, auc
import sys
sys.path.append('.')
from src.models.hamad import HAMADFramework

# Page config
st.set_page_config(
    page_title="HAMAD - Network Anomaly Detection",
    page_icon="🔒",
    layout="wide"
)

# Load model
@st.cache_resource
def load_model():
    try:
        return HAMADFramework.load_model('models/hamad_model.pkl')
    except:
        st.error("Model not found! Please train the model first.")
        return None

# Load data
@st.cache_data
def load_data():
    try:
        X_test = pd.read_csv('data/processed/X_test.csv')
        y_test = pd.read_csv('data/processed/y_test.csv').values.ravel()
        return X_test, y_test
    except:
        return None, None

# Main app
def main():
    st.title("🔒 HAMAD: Network Anomaly Detection System")
    st.markdown("*Hybrid Attention-based Multi-scale Anomaly Detection Framework*")
    
    # Sidebar
    st.sidebar.header("Navigation")
    page = st.sidebar.selectbox(
        "Select Page",
        ["System Overview", "Live Detection", "Performance Analytics", "Model Insights"]
    )
    
    # Load resources
    hamad = load_model()
    X_test, y_test = load_data()
    
    if hamad is None or X_test is None:
        st.warning("⚠️ Please ensure model is trained and data is processed!")
        return
    
    # Page routing
    if page == "System Overview":
        show_overview(hamad, X_test, y_test)
    elif page == "Live Detection":
        show_detection(hamad, X_test)
    elif page == "Performance Analytics":
        show_analytics(hamad, X_test, y_test)
    elif page == "Model Insights":
        show_insights(hamad)

def show_overview(hamad, X_test, y_test):
    """System overview page"""
    st.header("📊 System Overview")
    
    # Key metrics
    metrics = hamad.evaluate(X_test, y_test)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Accuracy", f"{metrics['accuracy']:.2%}")
    with col2:
        st.metric("Precision", f"{metrics['precision']:.2%}")
    with col3:
        st.metric("Recall", f"{metrics['recall']:.2%}")
    with col4:
        st.metric("F1-Score", f"{metrics['f1_score']:.2%}")
    
    # Dataset info
    st.subheader("Dataset Information")
    col1, col2 = st.columns(2)
    
    with col1:
        st.info(f"**Total Samples**: {len(X_test):,}")
        st.info(f"**Features**: {X_test.shape[1]}")
    
    with col2:
        st.info(f"**Normal Traffic**: {(y_test == 0).sum():,}")
        st.info(f"**Attack Traffic**: {(y_test == 1).sum():,}")

def show_detection(hamad, X_test):
    """Live detection page"""
    st.header("🎯 Live Detection")
    
    num_samples = st.slider("Select number of samples to predict", 10, 500, 100)
    
    if st.button("Run Detection"):
        with st.spinner("Detecting anomalies..."):
            sample_data = X_test.iloc[:num_samples]
            predictions = hamad.predict(sample_data)
            probabilities = hamad.predict_proba(sample_data)
            
            results_df = pd.DataFrame({
                'Sample ID': range(1, num_samples + 1),
                'Prediction': ['Attack' if p == 1 else 'Normal' for p in predictions],
                'Confidence': probabilities
            })
            
            st.success(f"✓ Analyzed {num_samples} samples")
            
            # Show results
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Attacks Detected", (predictions == 1).sum())
            with col2:
                st.metric("Normal Traffic", (predictions == 0).sum())
            
            # Results table
            st.dataframe(results_df, use_container_width=True)
            
            # Visualization
            fig = px.histogram(results_df, x='Confidence', color='Prediction',
                             title="Prediction Confidence Distribution")
            st.plotly_chart(fig, use_container_width=True)

def show_analytics(hamad, X_test, y_test):
    """Performance analytics page"""
    st.header("📈 Performance Analytics")
    
    y_pred = hamad.predict(X_test)
    y_proba = hamad.predict_proba(X_test)
    
    # Confusion Matrix
    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    
    fig = px.imshow(cm, 
                    labels=dict(x="Predicted", y="Actual", color="Count"),
                    x=['Normal', 'Attack'],
                    y=['Normal', 'Attack'],
                    text_auto=True,
                    color_continuous_scale='Blues')
    st.plotly_chart(fig, use_container_width=True)
    
    # ROC Curve
    st.subheader("ROC Curve")
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr, y=tpr, name=f'ROC (AUC = {roc_auc:.3f})', mode='lines'))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', line=dict(dash='dash'), name='Random'))
    fig.update_layout(xaxis_title='False Positive Rate', yaxis_title='True Positive Rate')
    st.plotly_chart(fig, use_container_width=True)

def show_insights(hamad):
    """Model insights page"""
    st.header("🔍 Model Insights")
    
    st.subheader("Ensemble Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("**Random Forest**")
        st.write(f"- Estimators: {hamad.rf_classifier.n_estimators}")
        st.write(f"- Max Depth: {hamad.rf_classifier.max_depth}")
    
    with col2:
        st.info("**XGBoost**")
        st.write(f"- Estimators: {hamad.xgb_classifier.n_estimators}")
        st.write(f"- Max Depth: {hamad.xgb_classifier.max_depth}")
    
    if hamad.attention_weights:
        st.subheader("Attention Weights")
        weights_df = pd.DataFrame([hamad.attention_weights])
        fig = px.bar(weights_df.T, title="Model Contribution Weights")
        st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
