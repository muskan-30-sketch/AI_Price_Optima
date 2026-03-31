import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# PAGE CONFIG - PEHLE HI SET KARO
# ============================================================================
st.set_page_config(
    page_title="Dynamic Pricing System",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CUSTOM CSS
# ============================================================================
st.markdown("""
    <style>
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        margin: 10px 0;
    }
    .metric-value {
        font-size: 32px;
        font-weight: bold;
        margin: 10px 0;
    }
    </style>
    """, unsafe_allow_html=True)

# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================
if 'models_loaded' not in st.session_state:
    st.session_state.models_loaded = False
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False

# ============================================================================
# HELPER FUNCTIONS - OPTIMIZED
# ============================================================================

@st.cache_resource
def load_models():
    """Load pre-trained models"""
    try:
        import pickle
        import xgboost as xgb
        import lightgbm as lgb
        
        with open('xgboost_revenue_model.pkl', 'rb') as f:
            xgb_model = pickle.load(f)
        with open('lightgbm_revenue_model.pkl', 'rb') as f:
            lgb_model = pickle.load(f)
        with open('feature_columns.pkl', 'rb') as f:
            feature_columns = pickle.load(f)
        
        return xgb_model, lgb_model, feature_columns, True
    except Exception as e:
        st.error(f"⚠️ Model load error: {str(e)}")
        return None, None, None, False

@st.cache_data
def load_data(csv_path):
    """Load dataset efficiently"""
    try:
        df = pd.read_csv(csv_path)
        return df, True
    except Exception as e:
        st.error(f"⚠️ Data load error: {str(e)}")
        return None, False

def prepare_features(df):
    """Prepare features for prediction"""
    df_prep = df.copy()
    df_prep['OrderDate'] = pd.to_datetime(df_prep['OrderDate'], errors='coerce')
    df_prep['Year'] = df_prep['OrderDate'].dt.year
    df_prep['Month'] = df_prep['OrderDate'].dt.month
    df_prep['Quarter'] = df_prep['OrderDate'].dt.quarter
    df_prep['DayOfWeek'] = df_prep['OrderDate'].dt.dayofweek
    df_prep['DayOfMonth'] = df_prep['OrderDate'].dt.day
    df_prep['EffectivePrice'] = df_prep['UnitPrice'] * (1 - df_prep['Discount'])
    df_prep['PriceQuantityProduct'] = df_prep['UnitPrice'] * df_prep['Quantity']
    return df_prep

def predict_revenue(X, xgb_model, lgb_model, feature_columns):
    """Predict revenue using both models"""
    try:
        X_features = X[feature_columns].copy()
        xgb_pred = xgb_model.predict(X_features)
        lgb_pred = lgb_model.predict(X_features)
        ensemble_pred = (xgb_pred + lgb_pred) / 2
        return xgb_pred, lgb_pred, ensemble_pred
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return None, None, None

def calculate_metrics(actual, predicted):
    """Calculate evaluation metrics"""
    try:
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        mae = mean_absolute_error(actual, predicted)
        rmse = np.sqrt(mean_squared_error(actual, predicted))
        r2 = r2_score(actual, predicted)
        mape = np.mean(np.abs((actual - predicted) / actual)) * 100
        return {'MAE': mae, 'RMSE': rmse, 'R²': r2, 'MAPE': mape}
    except Exception as e:
        st.error(f"Metrics error: {str(e)}")
        return None

# ============================================================================
# MAIN APP
# ============================================================================

# Load models
xgb_model, lgb_model, feature_columns, models_ok = load_models()
st.session_state.models_loaded = models_ok

if not models_ok:
    st.error("❌ Models not loaded! Files needed:")
    st.code("• xgboost_revenue_model.pkl\n• lightgbm_revenue_model.pkl\n• feature_columns.pkl")
    st.stop()

# Main title
st.title("💰 Dynamic Pricing System - ML-Based Optimizer")
st.markdown("*Machine Learning Model for Revenue Optimization*")

# ============================================================================
# SIDEBAR NAVIGATION
# ============================================================================

with st.sidebar:
    st.title("🎯 Navigation")
    page = st.radio("Select Page:", [
        "📊 Dashboard",
        "🔮 Price Prediction",
        "⚡ Price Optimization",
        "📈 Performance Analysis",
        "💼 Batch Processing"
    ])

# ============================================================================
# PAGE 1: DASHBOARD
# ============================================================================

if page == "📊 Dashboard":
    st.subheader("📊 System Dashboard")
    
    # Load data
    df, data_ok = load_data('clean_dataset_numeric.csv')
    st.session_state.data_loaded = data_ok
    
    if data_ok and models_ok:
        with st.spinner("Loading dashboard..."):
            df_prep = prepare_features(df)
            
            # Sample subset
            sample_size = min(500, len(df_prep))
            df_sample = df_prep.sample(n=sample_size, random_state=42)
            
            # Make predictions
            X_sample = df_sample[feature_columns]
            xgb_pred, lgb_pred, ensemble_pred = predict_revenue(X_sample, xgb_model, lgb_model, feature_columns)
            
            if xgb_pred is not None:
                # Calculate metrics
                xgb_metrics = calculate_metrics(df_sample['TotalAmount'], xgb_pred)
                lgb_metrics = calculate_metrics(df_sample['TotalAmount'], lgb_pred)
                ensemble_metrics = calculate_metrics(df_sample['TotalAmount'], ensemble_pred)
                
                # Top metrics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("XGBoost R²", f"{xgb_metrics['R²']*100:.2f}%")
                
                with col2:
                    st.metric("LightGBM R²", f"{lgb_metrics['R²']*100:.2f}%")
                
                with col3:
                    st.metric("Ensemble MAE", f"${ensemble_metrics['MAE']:.2f}")
                
                with col4:
                    st.metric("Ensemble RMSE", f"${ensemble_metrics['RMSE']:.2f}")
                
                st.divider()
                
                # Model comparison table
                st.subheader("Model Performance Comparison")
                metrics_df = pd.DataFrame({
                    'Model': ['XGBoost', 'LightGBM', 'Ensemble'],
                    'R² Score': [f"{xgb_metrics['R²']*100:.2f}%", f"{lgb_metrics['R²']*100:.2f}%", f"{ensemble_metrics['R²']*100:.2f}%"],
                    'MAE': [f"${xgb_metrics['MAE']:.2f}", f"${lgb_metrics['MAE']:.2f}", f"${ensemble_metrics['MAE']:.2f}"],
                    'RMSE': [f"${xgb_metrics['RMSE']:.2f}", f"${lgb_metrics['RMSE']:.2f}", f"${ensemble_metrics['RMSE']:.2f}"]
                })
                st.dataframe(metrics_df, use_container_width=True)
                
                st.divider()
                
                # Charts
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Actual vs Predicted (XGBoost)")
                    fig1 = px.scatter(
                        x=df_sample['TotalAmount'],
                        y=xgb_pred,
                        labels={'x': 'Actual Revenue', 'y': 'Predicted Revenue'},
                        title="XGBoost Performance"
                    )
                    st.plotly_chart(fig1, use_container_width=True)
                
                with col2:
                    st.subheader("Actual vs Predicted (LightGBM)")
                    fig2 = px.scatter(
                        x=df_sample['TotalAmount'],
                        y=lgb_pred,
                        labels={'x': 'Actual Revenue', 'y': 'Predicted Revenue'},
                        title="LightGBM Performance"
                    )
                    st.plotly_chart(fig2, use_container_width=True)

# ============================================================================
# PAGE 2: PRICE PREDICTION
# ============================================================================

elif page == "🔮 Price Prediction":
    st.subheader("🔮 Single Product Revenue Prediction")
    
    st.write("Enter product details to get revenue predictions:")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        unit_price = st.number_input("Unit Price ($)", min_value=0.0, value=100.0, step=1.0)
        discount = st.slider("Discount (%)", 0.0, 50.0, 5.0, 0.5) / 100
        quantity = st.number_input("Quantity", min_value=1, value=5, step=1)
    
    with col2:
        year = st.number_input("Year", min_value=2020, max_value=2025, value=2024)
        month = st.selectbox("Month", range(1, 13), index=0)
        quarter = (month - 1) // 3 + 1
    
    with col3:
        day_of_month = st.number_input("Day of Month", min_value=1, max_value=31, value=15)
        day_of_week = st.selectbox("Day of Week", ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"], index=0)
        day_of_week_num = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"].index(day_of_week)
    
    if st.button("🔮 Predict Revenue", use_container_width=True):
        effective_price = unit_price * (1 - discount)
        price_qty_product = unit_price * quantity
        
        input_data = pd.DataFrame({
            'UnitPrice': [unit_price],
            'Discount': [discount],
            'Quantity': [quantity],
            'EffectivePrice': [effective_price],
            'PriceQuantityProduct': [price_qty_product],
            'Year': [year],
            'Month': [month],
            'Quarter': [quarter],
            'DayOfWeek': [day_of_week_num],
            'DayOfMonth': [day_of_month]
        })
        
        xgb_pred, lgb_pred, ensemble_pred = predict_revenue(input_data, xgb_model, lgb_model, feature_columns)
        
        if xgb_pred is not None:
            st.divider()
            st.subheader("📊 Prediction Results")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("XGBoost", f"${xgb_pred[0]:.2f}")
            with col2:
                st.metric("LightGBM", f"${lgb_pred[0]:.2f}")
            with col3:
                static_revenue = unit_price * quantity * (1 - discount)
                lift = ((ensemble_pred[0] - static_revenue) / static_revenue * 100) if static_revenue > 0 else 0
                st.metric("Ensemble", f"${ensemble_pred[0]:.2f}", delta=f"+{lift:.2f}%")
            
            st.divider()
            st.info(f"""
            **Input Summary:**
            - Unit Price: ${unit_price:.2f}
            - Discount: {discount*100:.1f}%
            - Quantity: {quantity}
            - Expected Static Revenue: ${static_revenue:.2f}
            - Ensemble Prediction: ${ensemble_pred[0]:.2f}
            - Potential Improvement: {lift:.2f}%
            """)

# ============================================================================
# PAGE 3: PRICE OPTIMIZATION
# ============================================================================

elif page == "⚡ Price Optimization":
    st.subheader("⚡ Automated Price Optimization")
    
    df, data_ok = load_data('clean_dataset_numeric.csv')
    
    if data_ok and models_ok:
        st.write("Configure optimization parameters:")
        
        col1, col2 = st.columns(2)
        
        with col1:
            price_min = st.slider("Min Price Multiplier", 0.5, 1.0, 0.8, 0.05)
            price_max = st.slider("Max Price Multiplier", 1.0, 2.0, 1.3, 0.05)
        
        with col2:
            discount_min = st.slider("Min Discount", 0.0, 0.5, 0.0, 0.05)
            discount_max = st.slider("Max Discount", 0.0, 0.5, 0.3, 0.05)
        
        sample_size = st.slider("Sample Size", 10, 100, 50, 10)
        
        if st.button("⚡ Run Optimization", use_container_width=True):
            with st.spinner("Optimizing prices..."):
                try:
                    df_prep = prepare_features(df)
                    df_sample = df_prep.sample(n=sample_size, random_state=42)
                    
                    results = []
                    
                    for idx, row in df_sample.iterrows():
                        best_revenue = 0
                        best_price_mult = 1.0
                        best_discount = 0
                        
                        # Simple optimization
                        for p_mult in np.linspace(price_min, price_max, 10):
                            for d in np.linspace(discount_min, discount_max, 10):
                                test_data = pd.DataFrame([row[feature_columns]])
                                test_data['UnitPrice'] = test_data['UnitPrice'] * p_mult
                                test_data['Discount'] = d
                                test_data['EffectivePrice'] = test_data['UnitPrice'] * (1 - d)
                                
                                pred = xgb_model.predict(test_data[feature_columns])[0]
                                
                                if pred > best_revenue:
                                    best_revenue = pred
                                    best_price_mult = p_mult
                                    best_discount = d
                        
                        original_revenue = row['TotalAmount']
                        results.append({
                            'Original Price': f"${row['UnitPrice']:.2f}",
                            'Optimized Price': f"${row['UnitPrice'] * best_price_mult:.2f}",
                            'Optimized Discount': f"{best_discount*100:.1f}%",
                            'Original Revenue': f"${original_revenue:.2f}",
                            'Optimized Revenue': f"${best_revenue:.2f}",
                            'Revenue Lift': f"{((best_revenue - original_revenue) / original_revenue * 100):.2f}%"
                        })
                    
                    st.success("✅ Optimization Complete!")
                    st.divider()
                    
                    results_df = pd.DataFrame(results)
                    st.dataframe(results_df, use_container_width=True)
                    
                    # Download
                    csv = results_df.to_csv(index=False)
                    st.download_button(
                        "📥 Download Results (CSV)",
                        csv,
                        "optimization_results.csv",
                        "text/csv",
                        use_container_width=True
                    )
                
                except Exception as e:
                    st.error(f"❌ Optimization error: {str(e)}")

# ============================================================================
# PAGE 4: PERFORMANCE ANALYSIS
# ============================================================================

elif page == "📈 Performance Analysis":
    st.subheader("📈 Advanced Performance Analysis")
    
    df, data_ok = load_data('clean_dataset_numeric.csv')
    
    if data_ok and models_ok:
        metric = st.selectbox("Select Metric", ["R² Score", "MAE", "RMSE"])
        group_by = st.selectbox("Group By", ["Month", "Quarter", "Day of Week"])
        
        if st.button("📊 Run Analysis", use_container_width=True):
            with st.spinner("Analyzing..."):
                try:
                    df_prep = prepare_features(df)
                    sample_size = min(1000, len(df_prep))
                    df_sample = df_prep.sample(n=sample_size, random_state=42)
                    
                    X_sample = df_sample[feature_columns]
                    xgb_pred, lgb_pred, ensemble_pred = predict_revenue(X_sample, xgb_model, lgb_model, feature_columns)
                    
                    if xgb_pred is not None:
                        st.success("✅ Analysis Complete!")
                        
                        # Group and calculate
                        if group_by == "Month":
                            df_sample['Group'] = df_sample['Month'].astype(str)
                        elif group_by == "Quarter":
                            df_sample['Group'] = 'Q' + df_sample['Quarter'].astype(str)
                        else:
                            day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
                            df_sample['Group'] = df_sample['DayOfWeek'].map(lambda x: day_names[x] if 0 <= x <= 6 else str(x))
                        
                        from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
                        
                        results = []
                        for group in sorted(df_sample['Group'].unique()):
                            mask = df_sample['Group'] == group
                            actual = df_sample[mask]['TotalAmount'].values
                            pred = ensemble_pred[mask]
                            
                            results.append({
                                group_by: group,
                                'Count': mask.sum(),
                                'R² Score': f"{r2_score(actual, pred):.4f}",
                                'MAE': f"${mean_absolute_error(actual, pred):.2f}",
                                'RMSE': f"${np.sqrt(mean_squared_error(actual, pred)):.2f}"
                            })
                        
                        results_df = pd.DataFrame(results)
                        st.dataframe(results_df, use_container_width=True)
                
                except Exception as e:
                    st.error(f"❌ Analysis error: {str(e)}")

# ============================================================================
# PAGE 5: BATCH PROCESSING
# ============================================================================

elif page == "💼 Batch Processing":
    st.subheader("💼 Batch CSV Processing")
    
    if models_ok:
        uploaded_file = st.file_uploader("Upload CSV File", type=['csv'])
        
        if uploaded_file is not None:
            try:
                batch_df = pd.read_csv(uploaded_file)
                st.write(f"Loaded {len(batch_df)} records")
                st.dataframe(batch_df.head(), use_container_width=True)
                
                if st.button("🚀 Process Batch", use_container_width=True):
                    with st.spinner("Processing..."):
                        batch_df_prep = prepare_features(batch_df)
                        X_batch = batch_df_prep[feature_columns]
                        xgb_pred, lgb_pred, ensemble_pred = predict_revenue(X_batch, xgb_model, lgb_model, feature_columns)
                        
                        if xgb_pred is not None:
                            results_df = batch_df.copy()
                            results_df['XGBoost_Pred'] = xgb_pred
                            results_df['LightGBM_Pred'] = lgb_pred
                            results_df['Ensemble_Pred'] = ensemble_pred
                            results_df['Error'] = results_df['TotalAmount'] - ensemble_pred
                            
                            st.success("✅ Processing Complete!")
                            st.divider()
                            
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Total Records", len(results_df))
                            with col2:
                                avg_error = np.abs(results_df['Error']).mean()
                                st.metric("Avg Error", f"${avg_error:.2f}")
                            with col3:
                                st.metric("Total Predicted", f"${results_df['Ensemble_Pred'].sum():,.2f}")
                            
                            st.dataframe(results_df, use_container_width=True)
                            
                            csv = results_df.to_csv(index=False)
                            st.download_button(
                                "📥 Download Results",
                                csv,
                                "batch_results.csv",
                                "text/csv",
                                use_container_width=True
                            )
            
            except Exception as e:
                st.error(f"❌ Error processing file: {str(e)}")

# ============================================================================
# FOOTER
# ============================================================================

st.divider()
st.caption("💰 Dynamic Pricing System v1.0 | Powered by XGBoost & LightGBM")
