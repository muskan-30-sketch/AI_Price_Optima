import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
from datetime import date
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="PriceOptima – Dynamic Pricing System",
    page_icon="💰",
    layout="wide"
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem; font-weight: 800;
        color: #1a1a2e; margin-bottom: 0;
    }
    .sub-title {
        font-size: 1rem; color: #555;
        margin-top: 0; margin-bottom: 2rem;
    }
    .kpi-card {
        background: #f8f9fa;
        border-left: 5px solid #4361ee;
        border-radius: 8px;
        padding: 16px 20px;
        margin-bottom: 10px;
    }
    .kpi-label { font-size: 0.8rem; color: #888; text-transform: uppercase; letter-spacing: 0.05em; }
    .kpi-value { font-size: 1.6rem; font-weight: 700; color: #1a1a2e; }
    .lift-positive { color: #2dc653; font-weight: 700; font-size: 1.1rem; }
    .lift-negative { color: #e63946; font-weight: 700; font-size: 1.1rem; }
    .section-header {
        font-size: 1.2rem; font-weight: 700; color: #1a1a2e;
        border-bottom: 2px solid #4361ee;
        padding-bottom: 6px; margin-bottom: 16px;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# MODEL LOADING / TRAINING
# ─────────────────────────────────────────────
@st.cache_resource
def load_or_train_models():
    feature_columns = ['UnitPrice', 'Discount', 'Quantity', 'Year', 'Month', 'Quarter', 'DayOfWeek']
    xgb_path  = "xgboost_model.pkl"
    lgb_path  = "lightgbm_model.pkl"
    data_path = "clean_dataset_numeric.csv"

    # ── Saved models + CSV ───────────────────────────────────────────────
    if all(os.path.exists(p) for p in [xgb_path, lgb_path, data_path]):
        with open(xgb_path, 'rb') as f:
            xgb_model = pickle.load(f)
        with open(lgb_path, 'rb') as f:
            lgb_model = pickle.load(f)
        df = pd.read_csv(data_path)
        df['OrderDate'] = pd.to_datetime(df['OrderDate'])
        df['Year']       = df['OrderDate'].dt.year
        df['Month']      = df['OrderDate'].dt.month
        df['Quarter']    = df['OrderDate'].dt.quarter
        df['DayOfWeek']  = df['OrderDate'].dt.dayofweek
        X = df[feature_columns]
        y = df['TotalAmount']
        _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        y_pred_xgb = xgb_model.predict(X_test)
        y_pred_lgb = lgb_model.predict(X_test)
        return xgb_model, lgb_model, feature_columns, X_test, y_test, \
               y_pred_xgb, y_pred_lgb, \
               r2_score(y_test, y_pred_xgb), r2_score(y_test, y_pred_lgb), True

    # ── CSV only: train on the fly ────────────────────────────────────────
    elif os.path.exists(data_path):
        df = pd.read_csv(data_path)
        df['OrderDate'] = pd.to_datetime(df['OrderDate'])
        df['Year']       = df['OrderDate'].dt.year
        df['Month']      = df['OrderDate'].dt.month
        df['Quarter']    = df['OrderDate'].dt.quarter
        df['DayOfWeek']  = df['OrderDate'].dt.dayofweek
        X = df[feature_columns]
        y = df['TotalAmount']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        xgb_model = xgb.XGBRegressor(
            n_estimators=200, learning_rate=0.05, max_depth=5,
            min_child_weight=5, subsample=0.8, colsample_bytree=0.8,
            gamma=1, reg_lambda=1.0, reg_alpha=0.5, random_state=42, verbosity=0)
        lgb_model = lgb.LGBMRegressor(
            n_estimators=200, learning_rate=0.05, max_depth=5,
            num_leaves=20, min_child_samples=10, subsample=0.8,
            colsample_bytree=0.8, reg_lambda=1.0, reg_alpha=0.5,
            random_state=42, verbosity=-1)
        xgb_model.fit(X_train, y_train)
        lgb_model.fit(X_train, y_train)
        y_pred_xgb = xgb_model.predict(X_test)
        y_pred_lgb = lgb_model.predict(X_test)
        return xgb_model, lgb_model, feature_columns, X_test, y_test, \
               y_pred_xgb, y_pred_lgb, \
               r2_score(y_test, y_pred_xgb), r2_score(y_test, y_pred_lgb), True

    # ── Demo mode: synthetic data ─────────────────────────────────────────
    else:
        np.random.seed(42)
        n = 1000
        X_test = pd.DataFrame({
            'UnitPrice':  np.random.uniform(10, 500, n),
            'Discount':   np.random.uniform(0, 0.3, n),
            'Quantity':   np.random.randint(1, 50, n),
            'Year':       np.random.choice([2022, 2023], n),
            'Month':      np.random.randint(1, 13, n),
            'Quarter':    np.random.randint(1, 5, n),
            'DayOfWeek':  np.random.randint(0, 7, n),
        })
        y_test = (X_test['UnitPrice'] * X_test['Quantity'] * (1 - X_test['Discount'])
                  + np.random.normal(0, 50, n))
        xgb_model = xgb.XGBRegressor(n_estimators=100, random_state=42, verbosity=0)
        lgb_model = lgb.LGBMRegressor(n_estimators=100, random_state=42, verbosity=-1)
        xgb_model.fit(X_test, y_test)
        lgb_model.fit(X_test, y_test)
        y_pred_xgb = xgb_model.predict(X_test)
        y_pred_lgb = lgb_model.predict(X_test)
        return xgb_model, lgb_model, feature_columns, X_test, y_test, \
               y_pred_xgb, y_pred_lgb, \
               r2_score(y_test, y_pred_xgb), r2_score(y_test, y_pred_lgb), False


# ─────────────────────────────────────────────
# BOOTSTRAP
# ─────────────────────────────────────────────
(xgb_model, lgb_model, feature_columns,
 X_test, y_test, y_pred_xgb, y_pred_lgb,
 r2_xgb, r2_lgb, real_data) = load_or_train_models()

if r2_xgb >= r2_lgb:
    best_model, best_name, y_pred_best = xgb_model, "XGBoost", y_pred_xgb
else:
    best_model, best_name, y_pred_best = lgb_model, "LightGBM", y_pred_lgb

# Pre-compute revenue strategies
y_test_arr      = np.array(y_test)
static_revenue  = float(y_test_arr.sum())
rule_revenue    = float((y_test_arr * 0.95).sum())
threshold       = np.percentile(y_pred_best, 70)
ml_adjusted     = np.where(y_pred_best >= threshold, y_test_arr * 1.05, y_test_arr)
ml_revenue      = float(ml_adjusted.sum())
lift_ml         = ((ml_revenue  - static_revenue) / static_revenue) * 100
lift_rule       = ((rule_revenue - static_revenue) / static_revenue) * 100


# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.markdown('<p class="main-title">💰 PriceOptima – Dynamic Pricing System</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">AI-Powered Revenue Optimization &nbsp;|&nbsp; Milestone 6 – Deployment & Dashboard</p>', unsafe_allow_html=True)

if not real_data:
    st.info("⚠️ **Demo Mode** — place `clean_dataset_numeric.csv` and `.pkl` model files in the same folder to use your real data.")

# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "🎯 Price Recommender",
    "📊 KPI Dashboard",
    "🔍 Model Comparison",
    "📈 Revenue Analysis"
])


# ══════════════════════════════════════════════
# TAB 1 — PRICE RECOMMENDER
# ══════════════════════════════════════════════
with tab1:
    st.markdown('<p class="section-header">🎯 Get a Price Recommendation</p>', unsafe_allow_html=True)
    st.write("Enter product details and the ML model will return an optimised price recommendation.")

    col_in, col_out = st.columns([1, 1], gap="large")

    with col_in:
        st.markdown("**Product & Order Details**")
        unit_price   = st.number_input("Current Unit Price ($)", min_value=1.0, max_value=10000.0, value=100.0, step=1.0)
        quantity     = st.number_input("Quantity",               min_value=1,   max_value=1000,    value=10)
        discount_pct = st.slider("Discount (%)",                 min_value=0,   max_value=50,      value=10)
        discount     = discount_pct / 100
        order_date   = st.date_input("Order Date", value=date.today())

        year        = order_date.year
        month       = order_date.month
        quarter     = (month - 1) // 3 + 1
        day_of_week = order_date.weekday()

        model_choice = st.selectbox("Model to Use", ["Best Model (Auto)", "XGBoost", "LightGBM"])
        predict_btn  = st.button("🚀 Get Price Recommendation", use_container_width=True, type="primary")

    with col_out:
        st.markdown("**Recommendation Output**")
        if predict_btn:
            input_df = pd.DataFrame([{
                'UnitPrice': unit_price, 'Discount': discount, 'Quantity': quantity,
                'Year': year, 'Month': month, 'Quarter': quarter, 'DayOfWeek': day_of_week
            }])

            if model_choice == "XGBoost":
                pred_revenue = float(xgb_model.predict(input_df)[0])
                used_model   = "XGBoost"
            elif model_choice == "LightGBM":
                pred_revenue = float(lgb_model.predict(input_df)[0])
                used_model   = "LightGBM"
            else:
                pred_revenue = float(best_model.predict(input_df)[0])
                used_model   = best_name

            current_revenue   = unit_price * quantity * (1 - discount)
            denom             = quantity * (1 - discount)
            recommended_price = pred_revenue / denom if denom > 0 else unit_price
            revenue_lift      = ((pred_revenue - current_revenue) / current_revenue) * 100
            demand_change     = -0.5 * ((recommended_price - unit_price) / unit_price) * 100
            expected_demand   = max(0, quantity * (1 + demand_change / 100))

            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-label">Recommended Price</div>
                <div class="kpi-value">${recommended_price:,.2f}</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Expected Revenue</div>
                <div class="kpi-value">${pred_revenue:,.2f}</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Current Revenue</div>
                <div class="kpi-value">${current_revenue:,.2f}</div>
            </div>
            """, unsafe_allow_html=True)

            lift_cls = "lift-positive" if revenue_lift >= 0 else "lift-negative"
            arrow    = "▲" if revenue_lift >= 0 else "▼"
            st.markdown(f'<p class="{lift_cls}">{arrow} Revenue Lift: {revenue_lift:+.2f}%</p>', unsafe_allow_html=True)
            st.markdown(f"**Model Used:** `{used_model}`  |  **Expected Demand:** ~{expected_demand:.1f} units")

            fig, ax = plt.subplots(figsize=(5, 2.8))
            ax.bar(["Current Revenue", "Predicted Revenue"],
                   [current_revenue, pred_revenue],
                   color=["#adb5bd", "#4361ee"], edgecolor="white", width=0.5)
            for rect in ax.patches:
                ax.text(rect.get_x() + rect.get_width()/2,
                        rect.get_height() + max(current_revenue, pred_revenue) * 0.01,
                        f"${rect.get_height():,.0f}", ha='center', va='bottom', fontsize=9, fontweight='bold')
            ax.set_ylabel("Revenue ($)", fontsize=9)
            ax.set_title("Current vs Predicted Revenue", fontsize=10, fontweight='bold')
            ax.spines[['top', 'right']].set_visible(False)
            ax.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        else:
            st.info("👈 Fill in the details on the left and click **Get Price Recommendation**.")


# ══════════════════════════════════════════════
# TAB 2 — KPI DASHBOARD
# ══════════════════════════════════════════════
with tab2:
    st.markdown('<p class="section-header">📊 Key Performance Indicators</p>', unsafe_allow_html=True)

    mae_best  = mean_absolute_error(y_test, y_pred_best)
    r2_best   = max(r2_xgb, r2_lgb)

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown(f'<div class="kpi-card"><div class="kpi-label">Best Model R²</div><div class="kpi-value">{r2_best*100:.2f}%</div></div>', unsafe_allow_html=True)
    with k2:
        st.markdown(f'<div class="kpi-card"><div class="kpi-label">Mean Abs Error</div><div class="kpi-value">${mae_best:,.2f}</div></div>', unsafe_allow_html=True)
    with k3:
        lc = "#2dc653" if lift_ml >= 0 else "#e63946"
        st.markdown(f'<div class="kpi-card" style="border-left-color:{lc}"><div class="kpi-label">ML Revenue Lift</div><div class="kpi-value" style="color:{lc}">{lift_ml:+.2f}%</div></div>', unsafe_allow_html=True)
    with k4:
        st.markdown(f'<div class="kpi-card"><div class="kpi-label">Test Samples</div><div class="kpi-value">{len(y_test):,}</div></div>', unsafe_allow_html=True)

    st.markdown("---")
    c1, c2 = st.columns(2)

    with c1:
        st.markdown("**Revenue by Pricing Strategy**")
        fig, ax = plt.subplots(figsize=(6, 4))
        strategies = ["Static\nPricing", "Rule-Based\nPricing", f"{best_name}\nPricing"]
        revenues   = [static_revenue, rule_revenue, ml_revenue]
        bars = ax.bar(strategies, revenues, color=["#adb5bd", "#f4a261", "#4361ee"], edgecolor="white", width=0.5)
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.01,
                    f"${bar.get_height():,.0f}", ha='center', va='bottom', fontsize=9, fontweight='bold')
        ax.set_ylabel("Total Revenue ($)")
        ax.set_title("Revenue Comparison – Three Strategies", fontweight='bold')
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with c2:
        st.markdown("**Revenue Lift vs Static Baseline**")
        fig, ax = plt.subplots(figsize=(6, 4))
        lifts   = [0, lift_rule, lift_ml]
        lcolors = ["#adb5bd",
                   "#2dc653" if lift_rule >= 0 else "#e63946",
                   "#2dc653" if lift_ml   >= 0 else "#e63946"]
        bars = ax.bar(strategies, lifts, color=lcolors, edgecolor="white", width=0.5)
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2,
                    h + (0.05 if h >= 0 else -0.25),
                    f"{h:+.2f}%", ha='center', va='bottom', fontsize=9, fontweight='bold')
        ax.axhline(0, color='black', linewidth=0.8)
        ax.set_ylabel("Revenue Lift (%)")
        ax.set_title("Revenue Lift vs Static Pricing", fontweight='bold')
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # Monthly trend
    st.markdown("**Monthly Revenue Trend (Test Set)**")
    try:
        monthly = pd.DataFrame({
            'Month':     X_test['Month'].values,
            'Actual':    y_test_arr,
            'Predicted': y_pred_best
        }).groupby('Month').mean().reset_index()
        fig, ax = plt.subplots(figsize=(10, 3.5))
        ax.plot(monthly['Month'], monthly['Actual'],    marker='o', label='Actual Revenue',    color='#4361ee', linewidth=2)
        ax.plot(monthly['Month'], monthly['Predicted'], marker='s', label='Predicted Revenue', color='#f4a261', linewidth=2, linestyle='--')
        ax.set_xlabel("Month")
        ax.set_ylabel("Avg Revenue ($)")
        ax.set_title("Monthly Average Revenue – Actual vs Predicted", fontweight='bold')
        ax.set_xticks(range(1, 13))
        ax.set_xticklabels(['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'])
        ax.legend()
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    except Exception:
        st.info("Monthly trend unavailable for this dataset.")


# ══════════════════════════════════════════════
# TAB 3 — MODEL COMPARISON
# ══════════════════════════════════════════════
with tab3:
    st.markdown('<p class="section-header">🔍 XGBoost vs LightGBM</p>', unsafe_allow_html=True)

    mae_xgb  = mean_absolute_error(y_test, y_pred_xgb)
    mae_lgb  = mean_absolute_error(y_test, y_pred_lgb)
    rmse_xgb = np.sqrt(mean_squared_error(y_test, y_pred_xgb))
    rmse_lgb = np.sqrt(mean_squared_error(y_test, y_pred_lgb))

    comp_df = pd.DataFrame({
        "Metric":   ["R² Score", "MAE ($)", "RMSE ($)"],
        "XGBoost":  [f"{r2_xgb:.4f}", f"${mae_xgb:,.2f}", f"${rmse_xgb:,.2f}"],
        "LightGBM": [f"{r2_lgb:.4f}", f"${mae_lgb:,.2f}", f"${rmse_lgb:,.2f}"],
    })
    st.table(comp_df)
    winner = "XGBoost" if r2_xgb >= r2_lgb else "LightGBM"
    st.success(f"✅  **Winner: {winner}** — Higher R² Score on the test set.")
    st.markdown("---")

    c1, c2 = st.columns(2)
    lims = [min(y_test_arr.min(), y_pred_xgb.min(), y_pred_lgb.min()),
            max(y_test_arr.max(), y_pred_xgb.max(), y_pred_lgb.max())]

    with c1:
        st.markdown("**Actual vs Predicted – XGBoost**")
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.scatter(y_test_arr[:300], y_pred_xgb[:300], alpha=0.4, color='#4361ee', s=20)
        ax.plot(lims, lims, 'r--', lw=1.5)
        ax.set_xlabel("Actual Revenue")
        ax.set_ylabel("Predicted Revenue")
        ax.set_title(f"XGBoost  R²={r2_xgb:.4f}", fontweight='bold')
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with c2:
        st.markdown("**Actual vs Predicted – LightGBM**")
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.scatter(y_test_arr[:300], y_pred_lgb[:300], alpha=0.4, color='#2a9d8f', s=20)
        ax.plot(lims, lims, 'r--', lw=1.5)
        ax.set_xlabel("Actual Revenue")
        ax.set_ylabel("Predicted Revenue")
        ax.set_title(f"LightGBM  R²={r2_lgb:.4f}", fontweight='bold')
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # Feature importance
    st.markdown("**Feature Importance**")
    fi1, fi2 = st.columns(2)
    xgb_imp = pd.DataFrame({'Feature': feature_columns, 'Importance': xgb_model.feature_importances_}).sort_values('Importance')
    lgb_imp = pd.DataFrame({'Feature': feature_columns, 'Importance': lgb_model.feature_importances_}).sort_values('Importance')

    with fi1:
        fig, ax = plt.subplots(figsize=(5, 3.5))
        ax.barh(xgb_imp['Feature'], xgb_imp['Importance'], color='#4361ee')
        ax.set_xlabel("Importance Score")
        ax.set_title("XGBoost – Feature Importance", fontweight='bold')
        ax.spines[['top', 'right']].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with fi2:
        fig, ax = plt.subplots(figsize=(5, 3.5))
        ax.barh(lgb_imp['Feature'], lgb_imp['Importance'], color='#2a9d8f')
        ax.set_xlabel("Importance Score")
        ax.set_title("LightGBM – Feature Importance", fontweight='bold')
        ax.spines[['top', 'right']].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()


# ══════════════════════════════════════════════
# TAB 4 — REVENUE ANALYSIS
# ══════════════════════════════════════════════
with tab4:
    st.markdown('<p class="section-header">📈 Revenue Analysis & Pricing Impact</p>', unsafe_allow_html=True)

    # Price vs Revenue scatter
    st.markdown("**Unit Price vs Revenue (coloured by Discount)**")
    try:
        fig, ax = plt.subplots(figsize=(10, 4))
        sc = ax.scatter(X_test['UnitPrice'][:500], y_test_arr[:500],
                        c=X_test['Discount'][:500], cmap='RdYlGn_r',
                        alpha=0.6, s=25, edgecolors='none')
        plt.colorbar(sc, ax=ax, label='Discount Rate')
        ax.set_xlabel("Unit Price ($)")
        ax.set_ylabel("Total Revenue ($)")
        ax.set_title("Unit Price vs Revenue (coloured by Discount Rate)", fontweight='bold')
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    except Exception:
        st.info("Scatter chart unavailable.")

    st.markdown("---")
    r1, r2_col = st.columns(2)

    with r1:
        st.markdown("**Revenue Distribution – Actual vs Predicted**")
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(y_test_arr, bins=40, alpha=0.6, color='#4361ee', label='Actual')
        ax.hist(y_pred_best, bins=40, alpha=0.6, color='#f4a261', label='Predicted')
        ax.set_xlabel("Revenue ($)")
        ax.set_ylabel("Frequency")
        ax.set_title("Revenue Distribution", fontweight='bold')
        ax.legend()
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with r2_col:
        st.markdown("**Residual Analysis**")
        residuals = y_test_arr - y_pred_best
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.scatter(y_pred_best[:400], residuals[:400], alpha=0.4, color='#4361ee', s=20)
        ax.axhline(0, color='red', linestyle='--', lw=1.5)
        ax.set_xlabel("Predicted Revenue ($)")
        ax.set_ylabel("Residual ($)")
        ax.set_title(f"Residual Plot – {best_name}  (Mean: ${residuals.mean():.2f})", fontweight='bold')
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # Sample comparison table
    st.markdown("**Original vs Recommended Pricing – Sample of 10 Orders**")
    np.random.seed(0)
    idx = np.random.choice(len(y_test_arr), size=10, replace=False)
    sample_df = pd.DataFrame({
        "Unit Price ($)":         np.array(X_test['UnitPrice'])[idx].round(2),
        "Quantity":               np.array(X_test['Quantity'])[idx],
        "Discount (%)":           (np.array(X_test['Discount'])[idx] * 100).round(1),
        "Original Revenue ($)":   y_test_arr[idx].round(2),
        "Predicted Revenue ($)":  y_pred_best[idx].round(2),
        "Difference ($)":         (y_pred_best[idx] - y_test_arr[idx]).round(2),
    })
    sample_df["Lift (%)"] = ((sample_df["Predicted Revenue ($)"] - sample_df["Original Revenue ($)"]) /
                              sample_df["Original Revenue ($)"] * 100).round(2)
    st.dataframe(sample_df, use_container_width=True)


# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:#aaa; font-size:0.85rem;'>"
    "PriceOptima – AI Dynamic Pricing System &nbsp;|&nbsp; Milestone 6 &nbsp;|&nbsp; "
    "Built with Streamlit, XGBoost & LightGBM"
    "</p>",
    unsafe_allow_html=True
)
