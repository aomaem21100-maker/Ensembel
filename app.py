import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go

# ============================================
# Page Configuration
# ============================================
st.set_page_config(
    page_title="Iris Flower Predictor",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# Custom CSS (Minimalist & Beautiful)
# ============================================
st.markdown("""
<style>
    /* Main background */
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #e8ecf1 100%);
    }
    /* Title styling */
    .main-title {
        font-family: 'Helvetica Neue', sans-serif;
        font-size: 2.5rem;
        font-weight: 700;
        color: #2c3e50;
        text-align: center;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        text-align: center;
        color: #7f8c8d;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    /* Metric card */
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.05);
        text-align: center;
        transition: transform 0.3s ease;
    }
    .metric-card:hover {
        transform: translateY(-3px);
    }
    .metric-label {
        color: #95a5a6;
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #2c3e50;
        margin: 0.5rem 0;
    }
    /* Prediction result */
    .prediction-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 20px;
        text-align: center;
        color: white;
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);
    }
    .prediction-label {
        font-size: 0.9rem;
        opacity: 0.9;
        letter-spacing: 2px;
    }
    .prediction-value {
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0.5rem 0;
    }
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: white;
    }
    .stSlider > div > div {
        color: #2c3e50 !important;
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# Load Model
# ============================================
@st.cache_resource
def load_artifacts():
    model = joblib.load('rf_model.pkl')
    scaler = joblib.load('scaler.pkl')
    features = joblib.load('feature_names.pkl')
    targets = joblib.load('target_names.pkl')
    return model, scaler, features, targets

model, scaler, features, targets = load_artifacts()

# ============================================
# Header
# ============================================
st.markdown('<p class="main-title">🌸 Iris Flower Predictor</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Predict the species of iris flower using Random Forest</p>', unsafe_allow_html=True)

# ============================================
# Sidebar - Input Features
# ============================================
with st.sidebar:
    st.header("🎛️ Flower Parameters")
    st.markdown("---")
    
    sepal_length = st.slider("Sepal Length (cm)", 4.0, 8.0, 5.5, 0.1)
    sepal_width = st.slider("Sepal Width (cm)", 2.0, 4.5, 3.0, 0.1)
    petal_length = st.slider("Petal Length (cm)", 1.0, 7.0, 4.0, 0.1)
    petal_width = st.slider("Petal Width (cm)", 0.1, 2.5, 1.3, 0.1)
    
    st.markdown("---")
    predict_btn = st.button("🔮 Predict Now", use_container_width=True, type="primary")

# ============================================
# Prediction
# ============================================
if predict_btn:
    # Prepare input
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    input_scaled = scaler.transform(input_data)
    
    # Predict
    prediction = model.predict(input_scaled)[0]
    probabilities = model.predict_proba(input_scaled)[0]
    predicted_species = targets[prediction]
    confidence = probabilities[prediction] * 100
    
    # Display Prediction Result
    st.markdown("### 🎯 Prediction Result")
    st.markdown(f"""
    <div class="prediction-box">
        <div class="prediction-label">PREDICTED SPECIES</div>
        <div class="prediction-value">{predicted_species.replace('_', ' ').title()}</div>
        <div>Confidence: {confidence:.2f}%</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Metrics Row
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Sepal</div>
            <div class="metric-value">{sepal_length} × {sepal_width}</div>
            <div style="color:#95a5a6;font-size:0.85rem;">length × width (cm)</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Petal</div>
            <div class="metric-value">{petal_length} × {petal_width}</div>
            <div style="color:#95a5a6;font-size:0.85rem;">length × width (cm)</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Confidence</div>
            <div class="metric-value">{confidence:.1f}%</div>
            <div style="color:#95a5a6;font-size:0.85rem;">model certainty</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Probability Chart
    col_a, col_b = st.columns([1.3, 1])
    
    with col_a:
        st.markdown("### 📊 Probability Distribution")
        fig = go.Figure(data=[
            go.Bar(
                x=[t.replace('_', ' ').title() for t in targets],
                y=probabilities * 100,
                marker_color=['#667eea' if i == prediction else '#dfe6e9' for i in range(len(targets))],
                text=[f"{p*100:.1f}%" for p in probabilities],
                textposition='auto'
            )
        ])
        fig.update_layout(
            yaxis_title="Probability (%)",
            xaxis_title="Species",
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            height=350,
            margin=dict(l=20, r=20, t=20, b=20)
        )
        fig.update_yaxes(showgrid=True, gridcolor='#ecf0f1')
        st.plotly_chart(fig, use_container_width=True)
    
    with col_b:
        st.markdown("### 🌟 Feature Importance")
        importances = model.feature_importances_
        fig2 = go.Figure(data=[
            go.Bar(
                y=[f.replace(' (cm)', '') for f in features],
                x=importances,
                orientation='h',
                marker_color='#764ba2',
                text=[f"{i*100:.1f}%" for i in importances],
                textposition='auto'
            )
        ])
        fig2.update_layout(
            xaxis_title="Importance",
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            height=350,
            margin=dict(l=20, r=20, t=20, b=20)
        )
        fig2.update_xaxes(showgrid=True, gridcolor='#ecf0f1')
        st.plotly_chart(fig2, use_container_width=True)

else:
    # Welcome screen
    st.markdown("<br><br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.info("""
        👈 **Adjust the flower parameters** in the sidebar and click **"Predict Now"** to see the prediction.
        
        **Features:**
        - 🌿 Sepal Length & Width
        - 🌸 Petal Length & Width
        - 🎯 Species Classification (Setosa / Versicolor / Virginica)
        """)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:#95a5a6;font-size:0.85rem;'>"
    "Built with ❤️ using Streamlit + Random Forest | © 2026"
    "</div>",
    unsafe_allow_html=True
)