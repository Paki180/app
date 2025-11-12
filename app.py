"""
Customer Segmentation ML Application
A machine learning web app for segmenting customers into distinct groups
"""

import pickle
import streamlit as st
import pandas as pd
import numpy as np

# Page configuration
st.set_page_config(
    page_title='Customer Segmentation ML',
    page_icon='🎯',
    layout='wide',
    initial_sidebar_state='expanded'
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #FF4B4B;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    </style>
""", unsafe_allow_html=True)

# Load the model
@st.cache_resource
def load_model():
    try:
        return pickle.load(open('classifier.pkl', 'rb'))
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

classifier = load_model()

# Header
st.markdown('<h1 class="main-header">🎯 Customer Segmentation ML</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Intelligent customer segmentation using machine learning</p>', unsafe_allow_html=True)

# Information expander
with st.expander("ℹ️ About This Application"):
    st.markdown("""
    This application uses **machine learning** to segment customers into 4 distinct clusters based on:
    - **Demographics**: Age, income, education, family structure
    - **Purchasing Behavior**: Spending patterns across product categories
    - **Engagement**: Purchase channels and campaign responses
    
    **Model Performance:**
    - Logistic Regression: 98.9% accuracy
    - Random Forest: 95.5% accuracy  
    - SVM: 97.3% accuracy
    - Voting Ensemble: 98.2% accuracy
    
    **The 4 Customer Segments:**
    - 🟢 **Cluster 0**: Moderate spenders with average income and family size
    - 🔵 **Cluster 1**: Budget-conscious customers with lower income
    - 🟡 **Cluster 2**: High-value customers with strong purchasing power
    - 🔴 **Cluster 3**: Premium segment - highest income and spending
    """)

def segment_customers(input_data):
    """Predict customer segment"""
    if classifier is None:
        return None, "Model not loaded"
    
    try:
        df_input = pd.DataFrame([input_data])
        prediction = classifier.predict(df_input)
        
        cluster_info = {
            0: {
                "name": "Moderate Spenders",
                "icon": "🟢",
                "description": "Average income with balanced spending patterns and moderate family size",
                "strategy": "Target with mid-range products and family-oriented campaigns"
            },
            1: {
                "name": "Budget Conscious",
                "icon": "🔵",
                "description": "Lower income segment focusing on value and essential purchases",
                "strategy": "Focus on value deals, discounts, and budget-friendly options"
            },
            2: {
                "name": "High Value",
                "icon": "🟡",
                "description": "High income with strong purchasing power across categories",
                "strategy": "Promote premium products, exclusive offers, and loyalty programs"
            },
            3: {
                "name": "Premium Elite",
                "icon": "🔴",
                "description": "Top-tier customers with highest income and spending levels",
                "strategy": "VIP treatment, luxury products, and personalized service"
            }
        }
        
        cluster_num = prediction[0]
        info = cluster_info.get(cluster_num, {"name": f"Cluster {cluster_num}", "icon": "⚪", 
                                               "description": "Unknown", "strategy": "Standard approach"})
        
        return cluster_num, info
    except Exception as e:
        return None, {"name": "Error", "icon": "❌", "description": str(e), "strategy": ""}

def main():
    # Sidebar inputs
    st.sidebar.header("📋 Customer Information")
    st.sidebar.markdown("---")
    
    # Demographics
    st.sidebar.subheader("📊 Demographics")
    Income = st.sidebar.number_input(
        "Annual Income ($)", 
        min_value=0, 
        max_value=200000, 
        value=50000, 
        step=1000,
        help="Customer's annual household income"
    )
    
    Age = st.sidebar.slider(
        "Age", 
        18, 85, 45,
        help="Customer's age in years"
    )
    
    Education = st.sidebar.selectbox(
        "Education Level", 
        ["Basic", "2n Cycle", "Graduation", "Master", "PhD"],
        index=2,
        help="Highest education level achieved"
    )
    education_mapping = {'Basic': 1, '2n Cycle': 2, 'Graduation': 3, 'Master': 4, 'PhD': 5}
    Education_Encoded = education_mapping[Education]
    
    Marital_Status = st.sidebar.selectbox(
        "Marital Status", 
        ["Single", "Partner"],
        help="Current relationship status"
    )
    marital_status = 1 if Marital_Status == "Partner" else 0
    
    st.sidebar.markdown("---")
    
    # Family
    st.sidebar.subheader("👨‍👩‍👧‍👦 Family Structure")
    Kidhome = st.sidebar.number_input("Children at Home", 0, 5, 0)
    Teenhome = st.sidebar.number_input("Teenagers at Home", 0, 5, 0)
    ChildrenCount = Kidhome + Teenhome
    Family_Size = ChildrenCount + (2 if marital_status == 1 else 1)
    
    st.sidebar.markdown("---")
    
    # Customer History
    st.sidebar.subheader("📅 Customer History")
    Days_As_Customer = st.sidebar.number_input(
        "Days as Customer", 
        0, 5000, 365,
        help="Number of days since customer registration"
    )
    Recency = st.sidebar.slider(
        "Days Since Last Purchase", 
        0, 100, 30,
        help="Recency of last purchase"
    )
    
    st.sidebar.markdown("---")
    
    # Spending
    st.sidebar.subheader("💰 Annual Spending ($)")
    MntWines = st.sidebar.number_input("Wine", 0.0, 2000.0, 300.0, 10.0)
    MntFruits = st.sidebar.number_input("Fruits", 0.0, 200.0, 20.0, 5.0)
    MntMeatProducts = st.sidebar.number_input("Meat", 0.0, 2000.0, 150.0, 10.0)
    MntFishProducts = st.sidebar.number_input("Fish", 0.0, 500.0, 50.0, 5.0)
    MntSweetProducts = st.sidebar.number_input("Sweets", 0.0, 500.0, 30.0, 5.0)
    MntGoldProds = st.sidebar.number_input("Gold Products", 0.0, 500.0, 40.0, 5.0)
    
    st.sidebar.markdown("---")
    
    # Purchases
    st.sidebar.subheader("🛒 Purchase Channels")
    NumWebPurchases = st.sidebar.number_input("Web Purchases", 0, 30, 4)
    NumCatalogPurchases = st.sidebar.number_input("Catalog Purchases", 0, 30, 3)
    NumStorePurchases = st.sidebar.number_input("Store Purchases", 0, 30, 5)
    NumDealsPurchases = st.sidebar.number_input("Deals Purchases", 0, 20, 2)
    NumWebVisitsMonth = st.sidebar.number_input("Web Visits per Month", 0, 20, 5)
    
    st.sidebar.markdown("---")
    
    # Campaigns
    st.sidebar.subheader("📢 Campaign Engagement")
    Total_Campaign_Accepted = st.sidebar.slider("Total Campaigns Accepted", 0, 5, 0)
    Response = st.sidebar.selectbox("Last Campaign Response", ["No", "Yes"])
    Response_val = 1 if Response == "Yes" else 0
    
    # Calculate derived features
    Total_Spent = np.log1p(MntWines + MntFruits + MntMeatProducts + 
                           MntFishProducts + MntSweetProducts + MntGoldProds)
    Total_Purchases = np.log1p(NumWebPurchases + NumCatalogPurchases + NumStorePurchases)
    
    # Prepare input data with all 23 features
    input_data = {
        'Income': Income,
        'Recency': Recency,
        'MntWines': np.log1p(MntWines),
        'MntFruits': np.log1p(MntFruits),
        'MntMeatProducts': np.log1p(MntMeatProducts),
        'MntFishProducts': np.log1p(MntFishProducts),
        'MntSweetProducts': np.log1p(MntSweetProducts),
        'MntGoldProds': np.log1p(MntGoldProds),
        'NumDealsPurchases': np.log1p(NumDealsPurchases),
        'NumWebPurchases': np.log1p(NumWebPurchases),
        'NumCatalogPurchases': np.log1p(NumCatalogPurchases),
        'NumStorePurchases': np.log1p(NumStorePurchases),
        'NumWebVisitsMonth': NumWebVisitsMonth,
        'Response': np.log1p(Response_val),
        'Age': Age,
        'Days_As_Customer': Days_As_Customer,
        'ChildrenCount': ChildrenCount,
        'Family_Size': Family_Size,
        'marital_status': np.log1p(marital_status),
        'Total_Spent': Total_Spent,
        'Total_Purchases': Total_Purchases,
        'Total_Campaign_Accepted': np.log1p(Total_Campaign_Accepted),
        'Education_Encoded': Education_Encoded
    }
    
    # Display customer summary
    st.subheader("📊 Customer Profile Summary")
    
    # Metrics in a grid
    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
    
    with metric_col1:
        st.metric("💵 Income", f"${Income:,}")
        st.metric("👤 Age", f"{Age} years")
    
    with metric_col2:
        st.metric("👨‍👩‍👧‍👦 Family Size", Family_Size)
        st.metric("📅 Tenure", f"{Days_As_Customer} days")
    
    with metric_col3:
        total_raw_spending = MntWines + MntFruits + MntMeatProducts + MntFishProducts + MntSweetProducts + MntGoldProds
        st.metric("💰 Total Spent", f"${total_raw_spending:.0f}")
        st.metric("📚 Education", Education)
    
    with metric_col4:
        total_purchases = NumWebPurchases + NumCatalogPurchases + NumStorePurchases
        st.metric("🛍️ Purchases", total_purchases)
        st.metric("📢 Campaigns", Total_Campaign_Accepted)
    
    st.markdown("---")
    
    # Prediction button
    if st.button("🎯 Segment Customer", type="primary", use_container_width=True):
        with st.spinner("🔄 Analyzing customer profile..."):
            cluster_num, cluster_info = segment_customers(input_data)
            
            if cluster_num is not None:
                st.success("✅ Segmentation Complete!")
                
                # Display result in a nice card
                st.markdown(f"""
                <div style="
                    padding: 2rem;
                    border-radius: 10px;
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    text-align: center;
                    margin: 1rem 0;
                ">
                    <h2 style="margin: 0; font-size: 3rem;">{cluster_info['icon']}</h2>
                    <h3 style="margin: 0.5rem 0;">Cluster {cluster_num}: {cluster_info['name']}</h3>
                    <p style="margin: 0.5rem 0; font-size: 1.1rem;">{cluster_info['description']}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Marketing strategy
                st.subheader("📈 Recommended Marketing Strategy")
                st.info(f"**Strategy:** {cluster_info['strategy']}")
                
                # Additional insights
                st.subheader("🔍 Profile Insights")
                insights_col1, insights_col2 = st.columns(2)
                
                with insights_col1:
                    st.markdown("**Spending Pattern:**")
                    if total_raw_spending > 1000:
                        st.write("✅ High spender - Premium products")
                    elif total_raw_spending > 500:
                        st.write("📊 Moderate spender - Mid-range products")
                    else:
                        st.write("💡 Budget conscious - Value deals")
                
                with insights_col2:
                    st.markdown("**Engagement Level:**")
                    if Total_Campaign_Accepted >= 3:
                        st.write("🌟 Highly engaged - Responsive to campaigns")
                    elif Total_Campaign_Accepted >= 1:
                        st.write("📧 Moderately engaged - Selective response")
                    else:
                        st.write("💤 Low engagement - Needs activation")
            else:
                st.error(f"❌ {cluster_info['description']}")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    <p>Built with ❤️ using Streamlit | ML Model: Voting Ensemble (98.2% accuracy)</p>
    <p>Customer Segmentation using K-Means Clustering & Classification Models</p>
</div>
""", unsafe_allow_html=True)

if __name__ == '__main__':
    main()
