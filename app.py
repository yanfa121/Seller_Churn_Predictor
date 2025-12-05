import streamlit as st
import pandas as pd
import numpy as np
import pickle
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="Olist Seller Predictor",
    page_icon="🤖",
    layout="wide"
)

# Title and description
st.title("🤖 Olist Seller Predictor")
st.markdown("### Predict whether a seller will churn or stay")

# Load the model
@st.cache_resource
def load_model():
    model_path = Path("bestmodel.mdl")
    if not model_path.exists():
        st.error("Model file 'bestmodel.mdl' not found!")
        st.stop()
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model

try:
    model = load_model()
    st.success("✅ Model loaded successfully!")
except Exception as e:
    st.error(f"Error loading model: {str(e)}")
    st.stop()

# Create two columns for input
st.markdown("---")
st.header("📝 Enter Data Values")

col1, col2 = st.columns(2)

# Categorical features with their options
categorical_options = {
    'seller_state': ['RS', 'SP', 'MG', 'PR', 'SC', 'RJ', 'BA', 'MS', 'ES', 'DF', 'SE', 'PB', 'CE', 'GO', 'AM', 'PE', 'RN', 'RO', 'MT', 'PA', 'MA', 'PI'],
    'top_main_category': ['home', 'automotive', 'gifts_market', 'fashion', 'pet', 'toys_leisure', 'garden', 'beauty_health', 'electronics', 'baby', 'office', 'others', 'food_drink', 'books_media', 'tools', 'small_appliances', 'industry_business', 'security_services', 'Others']
}

# Numerical features
numerical_features = [
    "total_orders",
    "cumulative_orders",
    "monetary",
    "cumulative_monetary",
    "recency_days",
    "active_months_count",
    "active_days_count",
    "tenure_months",
    "median_review_score",
    "total_orders_late_to_carrier",
    "total_orders_late_to_customer",
    "median_approval_time_hours"
]

# Store input values
input_data = {}

# Categorical inputs in first column
with col1:
    st.subheader("Categorical Data")
    
    input_data['seller_state'] = st.selectbox(
        "Seller State",
        options=categorical_options['seller_state']
    )
    
    input_data['top_main_category'] = st.selectbox(
        "Top Main Category",
        options=categorical_options['top_main_category']
    )

# Numerical inputs in second column
with col2:
    st.subheader("Numerical Data")
    
    input_data['total_orders'] = st.number_input(
        "Total Orders",
        value=0,
        step=1,
        format="%d"
    )
    
    input_data['cumulative_orders'] = st.number_input(
        "Cumulative Orders",
        value=0,
        step=1,
        format="%d"
    )
    
    input_data['monetary'] = st.number_input(
        "Monetary",
        value=0.0,
        step=0.01
    )
    
    input_data['cumulative_monetary'] = st.number_input(
        "Cumulative Monetary",
        value=0.0,
        step=0.01
    )
    
    input_data['recency_days'] = st.number_input(
        "Recency Days",
        value=0,
        step=1,
        format="%d"
    )
    
    input_data['active_months_count'] = st.number_input(
        "Active Months Count",
        value=0,
        step=1,
        format="%d"
    )
    
    input_data['active_days_count'] = st.number_input(
        "Active Days Count",
        value=0,
        step=1,
        format="%d"
    )
    
    input_data['tenure_months'] = st.number_input(
        "Tenure Months",
        value=0,
        step=1,
        format="%d"
    )
    
    input_data['median_review_score'] = st.number_input(
        "Median Review Score",
        value=0.0,
        min_value=0.0,
        max_value=5.0,
        step=0.1
    )
    
    input_data['total_orders_late_to_carrier'] = st.number_input(
        "Total Orders Late to Carrier",
        value=0,
        step=1,
        format="%d"
    )
    
    input_data['total_orders_late_to_customer'] = st.number_input(
        "Total Orders Late to Customer",
        value=0,
        step=1,
        format="%d"
    )
    
    input_data['median_approval_time_hours'] = st.number_input(
        "Median Approval Time Hours",
        value=0.0,
        step=0.1
    )

# Prediction button
st.markdown("---")
col_pred1, col_pred2, col_pred3 = st.columns([1, 2, 1])

with col_pred2:
    if st.button("🔮 Make Prediction", use_container_width=True):
        try:
            # Create DataFrame from input
            input_df = pd.DataFrame([input_data])
            
            # Make prediction
            prediction = model.predict(input_df)[0]
            prediction_proba = model.predict_proba(input_df)[0]
            confidence = max(prediction_proba) * 100
            
            # Display results
            st.markdown("---")
            st.header("📊 Prediction Results")
            
            # Determine prediction message
            if prediction == 1:
                st.error("⚠️ **This seller will potentially CHURN**")
            else:
                st.success("✅ **This seller will potentially STAY**")
            
            # Display confidence
            st.info(f"🎯 **Confidence:** {confidence:.2f}%")
            
            # Show detailed probabilities
            st.markdown("### Probability Breakdown")
            col_prob1, col_prob2 = st.columns(2)
            
            with col_prob1:
                st.metric("Probability of Staying (Class 0)", f"{prediction_proba[0]:.4f}")
            
            with col_prob2:
                st.metric("Probability of Churning (Class 1)", f"{prediction_proba[1]:.4f}")
            
            # Show input summary
            with st.expander("📋 View Input Summary"):
                st.dataframe(input_df)
                
        except Exception as e:
            st.error(f"❌ Error making prediction: {str(e)}")
            st.exception(e)



# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>Built with Streamlit • XGBoost Model v1.0</p>
</div>
""", unsafe_allow_html=True)