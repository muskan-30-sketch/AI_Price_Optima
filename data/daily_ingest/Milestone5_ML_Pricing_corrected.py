import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

import xgboost as xgb
import lightgbm as lgb


def effective_price(unit_price: np.ndarray, discount: np.ndarray) -> np.ndarray:
    """Effective selling price after discount."""
    return unit_price * (1.0 - discount)


def deterministic_inventory(order_id_series: pd.Series) -> np.ndarray:
    """
    Simulate inventory deterministically (so results are reproducible),
    matching the spirit of Milestone 4 where inventory was simulated.
    """
    # Stable hashing for reproducibility across runs.
    h = pd.util.hash_pandas_object(order_id_series, index=False).astype("int64")
    return 10 + (h % 141)  # -> [10, 150]


def deterministic_demand_flag(order_id_series: pd.Series) -> np.ndarray:
    """Simulate high/low demand deterministically."""
    h = pd.util.hash_pandas_object(order_id_series, index=False).astype("int64")
    return np.where((h % 2) == 0, "low", "high")


def rule_based_adjust_unit_price(
    df_rows: pd.DataFrame,
) -> np.ndarray:
    """
    Milestone-4-style price adjustment:
    - Demand rule: +25% if high demand else -2%
    - Inventory rule: +20% if inventory < 20 else -5% if inventory > 100
    - Time rule: +10% on weekends
    """
    price = df_rows["UnitPrice"].to_numpy(dtype="float64").copy()

    demand_high = df_rows["demand_flag"].to_numpy() == "high"
    price[demand_high] *= 1.25
    price[~demand_high] *= 0.98

    inv = df_rows["Inventory"].to_numpy(dtype="float64")
    price[inv < 20] *= 1.20
    price[inv > 100] *= 0.95

    day_of_week = df_rows["DayOfWeek"].to_numpy(dtype="int64")
    weekend = (day_of_week == 5) | (day_of_week == 6)
    price[weekend] *= 1.10

    # Guardrails: unit price should remain positive.
    return np.clip(price, 1e-6, None)


def optimise_price_ml(
    model,
    X_test: pd.DataFrame,
    discount_test: np.ndarray,
    multipliers: np.ndarray,
) -> dict:
    """
    For each row, evaluate multiple price multipliers around the current UnitPrice
    and pick the one maximizing predicted revenue.
    """
    n = len(X_test)
    base_unit_price = X_test["UnitPrice"].to_numpy(dtype="float64")

    predicted_qty_by_multiplier = np.zeros((len(multipliers), n), dtype="float64")
    revenue_by_multiplier = np.zeros((len(multipliers), n), dtype="float64")

    for mi, m in enumerate(multipliers):
        X_candidate = X_test.copy()
        X_candidate["UnitPrice"] = base_unit_price * m

        qty_pred = model.predict(X_candidate)
        qty_pred = np.clip(qty_pred, 0.0, None)
        predicted_qty_by_multiplier[mi, :] = qty_pred

        price_eff = effective_price(X_candidate["UnitPrice"].to_numpy(dtype="float64"), discount_test)
        revenue_by_multiplier[mi, :] = price_eff * qty_pred

    best_idx = np.argmax(revenue_by_multiplier, axis=0)  # per row
    best_multipliers = multipliers[best_idx]
    best_unit_price = base_unit_price * best_multipliers
    best_pred_qty = predicted_qty_by_multiplier[best_idx, np.arange(n)]
    best_revenue = revenue_by_multiplier[best_idx, np.arange(n)]

    return {
        "best_multipliers": best_multipliers,
        "best_unit_price": best_unit_price,
        "best_pred_qty": best_pred_qty,
        "best_revenue_per_row": best_revenue,
        "best_total_revenue": float(best_revenue.sum()),
    }


def train_models_and_backtest(
    data_path: Path,
    output_dir: Path,
    test_size: float = 0.2,
    random_state: int = 42,
):
    df = pd.read_csv(data_path)

    # --- Feature engineering ---
    df["OrderDate"] = pd.to_datetime(df["OrderDate"], errors="coerce")
    df["Year"] = df["OrderDate"].dt.year
    df["Month"] = df["OrderDate"].dt.month
    df["Quarter"] = df["OrderDate"].dt.quarter
    df["DayOfWeek"] = df["OrderDate"].dt.dayofweek  # 0=Mon ... 6=Sun

    # Simulated inventory + demand flag (deterministic, for reproducibility)
    df["Inventory"] = deterministic_inventory(df["OrderID"])
    df["demand_flag"] = deterministic_demand_flag(df["OrderID"])

    # Encode product identifiers + geography as numeric features
    df["ProductCode"] = pd.factorize(df["ProductID"])[0].astype("int64")
    df["SellerCode"] = pd.factorize(df["SellerID"])[0].astype("int64")
    df["CityCode"] = pd.factorize(df["City"])[0].astype("int64")

    feature_columns = [
        "UnitPrice",
        "Discount",
        "Inventory",
        "ProductCode",
        "SellerCode",
        "CityCode",
        "Year",
        "Month",
        "Quarter",
        "DayOfWeek",
    ]
    target_column = "Quantity"

    # Defensive: ensure numeric
    df[feature_columns] = df[feature_columns].apply(pd.to_numeric, errors="coerce")
    df[target_column] = pd.to_numeric(df[target_column], errors="coerce")
    df = df.dropna(subset=feature_columns + [target_column]).reset_index(drop=True)

    # --- Train/test split ---
    train_idx, test_idx = train_test_split(
        np.arange(len(df)),
        test_size=test_size,
        random_state=random_state,
        shuffle=True,
    )
    train_df = df.iloc[train_idx].copy()
    test_df = df.iloc[test_idx].copy()

    X_train = train_df[feature_columns].copy()
    y_train = train_df[target_column].copy().astype("float64")
    X_test = test_df[feature_columns].copy()
    y_test = test_df[target_column].copy().astype("float64")

    discount_test = test_df["Discount"].to_numpy(dtype="float64")
    unit_price_test = test_df["UnitPrice"].to_numpy(dtype="float64")

    # Baseline (Static Pricing) revenue uses *original prices* and *actual quantities*
    static_revenue = float(
        (effective_price(unit_price_test, discount_test) * y_test.to_numpy(dtype="float64")).sum()
    )

    # Pre-compute rule-based adjusted prices for backtesting
    rule_unit_price = rule_based_adjust_unit_price(test_df)
    X_rule = X_test.copy()
    X_rule["UnitPrice"] = rule_unit_price
    rule_effective_price = effective_price(rule_unit_price, discount_test)

    # Price multipliers for ML-based optimization.
    # Milestone-4-style rules can shift price by multiple factors (demand * inventory * time),
    # so we search a wider range than +/-10%.
    multipliers = np.linspace(0.7, 1.6, 19)

    models = {
        "XGBoost": xgb.XGBRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=5,
            min_child_weight=5,
            subsample=0.8,
            colsample_bytree=0.8,
            gamma=1,
            reg_lambda=1.0,
            reg_alpha=0.5,
            objective="reg:squarederror",
            random_state=random_state,
            verbosity=0,
        ),
        "LightGBM": lgb.LGBMRegressor(
            n_estimators=400,
            learning_rate=0.05,
            max_depth=5,
            num_leaves=31,
            min_child_samples=10,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            reg_alpha=0.5,
            random_state=random_state,
            verbosity=-1,
        ),
    }

    metrics_rows = []
    revenue_rows = []
    backtest_rows = []

    for model_name, model in models.items():
        print(f"Training {model_name}...")
        model.fit(X_train, y_train)

        # --- Demand prediction metrics (MAE/RMSE/R²) ---
        y_pred = model.predict(X_test)
        y_pred = np.asarray(y_pred, dtype="float64")
        y_pred = np.clip(y_pred, 0.0, None)

        mae = mean_absolute_error(y_test, y_pred)
        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        r2 = float(r2_score(y_test, y_pred))

        metrics_rows.append(
            {
                "model": model_name,
                "test_mae": float(mae),
                "test_rmse": float(rmse),
                "test_r2": float(r2),
            }
        )

        # --- Backtesting: Rule-Based Pricing ---
        qty_rule = np.asarray(model.predict(X_rule), dtype="float64")
        qty_rule = np.clip(qty_rule, 0.0, None)
        rule_revenue = float((rule_effective_price * qty_rule).sum())

        # --- Backtesting: ML-Based Pricing (optimize UnitPrice) ---
        opt = optimise_price_ml(
            model=model,
            X_test=X_test,
            discount_test=discount_test,
            multipliers=multipliers,
        )
        ml_revenue = opt["best_total_revenue"]

        # --- Revenue lift ---
        rule_lift = (rule_revenue - static_revenue) / static_revenue * 100.0
        ml_lift = (ml_revenue - static_revenue) / static_revenue * 100.0

        revenue_rows.append(
            {
                "model": model_name,
                "static_revenue": static_revenue,
                "rule_based_revenue": rule_revenue,
                "ml_based_revenue": ml_revenue,
                "rule_lift_pct": float(rule_lift),
                "ml_lift_pct": float(ml_lift),
            }
        )

        # Store per-row backtest info for the best model later
        backtest_rows.append(
            pd.DataFrame(
                {
                    "model": model_name,
                    "OrderID": test_df["OrderID"].values,
                    "OrderDate": test_df["OrderDate"].dt.strftime("%Y-%m-%d").values,
                    "ProductID": test_df["ProductID"].values,
                    "UnitPrice_static": unit_price_test,
                    "UnitPrice_rule": rule_unit_price,
                    "UnitPrice_ml_opt": opt["best_unit_price"],
                    "Discount": test_df["Discount"].values,
                    "Inventory": test_df["Inventory"].values,
                    "DayOfWeek": test_df["DayOfWeek"].values,
                    "demand_flag": test_df["demand_flag"].values,
                    "Quantity_actual": y_test.values,
                    "Quantity_pred_rule": qty_rule,
                    "Quantity_pred_ml": opt["best_pred_qty"],
                    "Revenue_static_actual": effective_price(unit_price_test, discount_test) * y_test.values,
                    "Revenue_rule_pred": rule_effective_price * qty_rule,
                    "Revenue_ml_pred": opt["best_revenue_per_row"],
                }
            )
        )

        print(f"{model_name} demand metrics: MAE={mae:.3f}, RMSE={rmse:.3f}, R2={r2:.4f}")
        print(
            f"{model_name} revenue: static=${static_revenue:.2f}, rule=${rule_revenue:.2f}, ml=${ml_revenue:.2f}"
        )
        print(
            f"{model_name} revenue lift: rule={rule_lift:.2f}%, ml={ml_lift:.2f}%\n"
        )

    metrics_df = pd.DataFrame(metrics_rows)
    revenue_df = pd.DataFrame(revenue_rows)
    backtest_df = pd.concat(backtest_rows, ignore_index=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_csv = output_dir / "milestone5_model_metrics.csv"
    revenue_csv = output_dir / "milestone5_revenue_comparison.csv"
    backtest_csv = output_dir / "milestone5_backtest_predictions.csv"
    metrics_json = output_dir / "milestone5_model_metrics_summary.json"

    metrics_df.to_csv(metrics_csv, index=False)
    revenue_df.to_csv(revenue_csv, index=False)
    backtest_df.to_csv(backtest_csv, index=False)

    summary = {
        "static_revenue": static_revenue,
        "feature_columns": feature_columns,
        "test_size": int(len(test_df)),
        "metrics": metrics_df.to_dict(orient="records"),
        "revenue_comparison": revenue_df.to_dict(orient="records"),
    }
    metrics_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("Saved outputs:")
    print(f"- {metrics_csv}")
    print(f"- {revenue_csv}")
    print(f"- {backtest_csv}")
    print(f"- {metrics_json}")

    return {
        "metrics_df": metrics_df,
        "revenue_df": revenue_df,
        "backtest_df": backtest_df,
        "static_revenue": static_revenue,
        "feature_columns": feature_columns,
    }


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent
    data_path = (base_dir.parent / "processed" / "clean_dataset_numeric.csv").resolve()
    output_dir = base_dir.resolve()

    train_models_and_backtest(data_path=data_path, output_dir=output_dir)

