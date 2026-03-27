import os
import pandas as pd
from datetime import datetime

RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"
DAILY_DIR = "data/daily_ingest"

os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(DAILY_DIR, exist_ok=True)

today = datetime.today().strftime("%Y-%m-%d")
today_dir = os.path.join(DAILY_DIR, today)
os.makedirs(today_dir, exist_ok=True)

SALES_REQUIRED = {
    "OrderID", "OrderDate", "CustomerID", "ProductID", "Quantity",
    "UnitPrice", "Discount", "Tax", "ShippingCost", "TotalAmount",
    "PaymentMethod", "OrderStatus", "City", "State", "Country"
}

INVENTORY_REQUIRED = {
    "ProductID", "ProductName", "Category", "Brand", "SellerID"
}

def clean_sales(df: pd.DataFrame) -> pd.DataFrame:
    missing_cols = SALES_REQUIRED - set(df.columns)
    if missing_cols:
        raise ValueError(f"Sales data missing columns: {missing_cols}")

    df = df.drop_duplicates()

    # Type corrections
    df["OrderDate"] = pd.to_datetime(df["OrderDate"], errors="coerce")
    for col in ["Quantity", "UnitPrice", "Discount", "Tax", "ShippingCost", "TotalAmount"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Missing value handling (basic + safe)
    df["Quantity"] = df["Quantity"].fillna(1)
    df["UnitPrice"] = df["UnitPrice"].fillna(df["UnitPrice"].median())
    df["Discount"] = df["Discount"].fillna(0)
    df["Tax"] = df["Tax"].fillna(0)
    df["ShippingCost"] = df["ShippingCost"].fillna(0)
    df["TotalAmount"] = df["TotalAmount"].fillna(
        df["Quantity"] * df["UnitPrice"]
    )

    # Categorical fills
    for col in ["PaymentMethod", "OrderStatus", "City", "State", "Country"]:
        df[col] = df[col].fillna("Unknown")

    return df

def clean_inventory(df: pd.DataFrame) -> pd.DataFrame:
    missing_cols = INVENTORY_REQUIRED - set(df.columns)
    if missing_cols:
        raise ValueError(f"Inventory data missing columns: {missing_cols}")

    df = df.drop_duplicates()

    # Fill basic missing values
    for col in ["ProductName", "Category", "Brand", "SellerID"]:
        df[col] = df[col].fillna("Unknown")

    return df

def main():
    try:
        sales_path = os.path.join(RAW_DIR, "sales_data.csv")
        sales_df = pd.read_csv(sales_path)
        print("✅ Sales data loaded successfully")

        sales_cleaned = clean_sales(sales_df)
        sales_cleaned.to_csv(os.path.join(PROCESSED_DIR, "sales_cleaned.csv"), index=False)
        sales_cleaned.to_csv(os.path.join(today_dir, "sales_cleaned.csv"), index=False)
        print("✅ Sales data cleaned and saved")

    except Exception as e:
        print("❌ Error processing sales data:", e)

    try:
        inv_path = os.path.join(RAW_DIR, "inventory_data.csv")
        if os.path.exists(inv_path):
            inv_df = pd.read_csv(inv_path)
            print("✅ Inventory data loaded successfully")

            inv_cleaned = clean_inventory(inv_df)
            inv_cleaned.to_csv(os.path.join(PROCESSED_DIR, "inventory_cleaned.csv"), index=False)
            inv_cleaned.to_csv(os.path.join(today_dir, "inventory_cleaned.csv"), index=False)
            print("✅ Inventory data cleaned and saved")
        else:
            print("⚠️ inventory_data.csv not found — skipping inventory ingestion")

    except Exception as e:
        print("❌ Error processing inventory data:", e)

    print("🎯 Ingestion completed")

if __name__ == "__main__":
    main()
   