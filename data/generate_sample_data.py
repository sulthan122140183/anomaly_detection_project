"""
Generate sample transaction data for anomaly detection demo
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

np.random.seed(42)

# Configuration
N_CUSTOMERS = 100
N_PRODUCTS = 50
N_TRANSACTIONS = 2000
START_DATE = datetime(2024, 1, 1)

# Generate customer data
customers = pd.DataFrame({
    'Customer_ID': range(N_CUSTOMERS),
    'Customer_Type': np.random.choice(['Regular', 'VIP', 'Corporate'], N_CUSTOMERS, p=[0.7, 0.2, 0.1]),
    'Customer_Name': [f'Customer_{i:03d}' for i in range(N_CUSTOMERS)]
})

# Generate product data with realistic prices
product_categories = ['Obat Resep', 'Obat Bebas', 'Alat Kesehatan', 'Suplemen', 'Kosmetik']
products = pd.DataFrame({
    'Product_ID': range(N_PRODUCTS),
    'Product_Name': [f'Product_{i:03d}' for i in range(N_PRODUCTS)],
    'Category': np.random.choice(product_categories, N_PRODUCTS),
    'Base_Price': np.random.choice([
        np.random.uniform(5_000, 50_000),      # Obat murah
        np.random.uniform(50_000, 200_000),    # Obat sedang
        np.random.uniform(200_000, 1_000_000), # Obat mahal
        np.random.uniform(1_000_000, 5_000_000) # Alat kesehatan
    ], N_PRODUCTS)
})

# Generate transactions
transactions = []

for i in range(N_TRANSACTIONS):
    customer_id = np.random.choice(customers['Customer_ID'])
    customer_type = customers.loc[customers['Customer_ID'] == customer_id, 'Customer_Type'].values[0]
    
    product_id = np.random.choice(products['Product_ID'])
    base_price = products.loc[products['Product_ID'] == product_id, 'Base_Price'].values[0]
    
    # Quantity depends on customer type
    if customer_type == 'Corporate':
        quantity = np.random.randint(10, 200)
    elif customer_type == 'VIP':
        quantity = np.random.randint(5, 50)
    else:
        quantity = np.random.randint(1, 20)
    
    # Add some price variation (discount, markup, etc)
    price_variation = np.random.normal(1.0, 0.1)
    unit_price = base_price * max(0.5, price_variation)
    
    total_bayar = unit_price * quantity
    
    # Transaction date
    days_offset = np.random.randint(0, 365)
    transaction_date = START_DATE + timedelta(days=days_offset)
    
    transactions.append({
        'Transaction_ID': i,
        'Transaction_Date': transaction_date,
        'Customer_ID': customer_id,
        'Customer_Type': customer_type,
        'Product_ID': product_id,
        'Quantity': quantity,
        'Unit_Price': unit_price,
        'Total_Bayar': total_bayar
    })

df_transactions = pd.DataFrame(transactions)

# Add ERRORS (Kesalahan Input) - 10 transactions
print("Adding errors (kesalahan input)...")
error_indices = np.random.choice(N_TRANSACTIONS, 10, replace=False)

for idx in error_indices:
    error_type = np.random.choice(['extra_zero', 'wrong_quantity', 'decimal_error'])
    
    if error_type == 'extra_zero':
        # Accidentally added extra zero(s)
        df_transactions.loc[idx, 'Total_Bayar'] *= np.random.choice([10, 100])
        df_transactions.loc[idx, 'Error_Type'] = 'Extra Zero'
        
    elif error_type == 'wrong_quantity':
        # Wrong quantity entered
        df_transactions.loc[idx, 'Quantity'] *= np.random.randint(50, 200)
        df_transactions.loc[idx, 'Total_Bayar'] = df_transactions.loc[idx, 'Unit_Price'] * df_transactions.loc[idx, 'Quantity']
        df_transactions.loc[idx, 'Error_Type'] = 'Wrong Quantity'
        
    elif error_type == 'decimal_error':
        # Decimal point error
        df_transactions.loc[idx, 'Total_Bayar'] *= 1000
        df_transactions.loc[idx, 'Error_Type'] = 'Decimal Error'

# Add LEGITIMATE HIGH-VALUE transactions - 5 transactions
print("Adding legitimate high-value transactions...")
legit_indices = np.random.choice(
    [i for i in range(N_TRANSACTIONS) if i not in error_indices],
    5,
    replace=False
)

for idx in legit_indices:
    # Corporate customer buying expensive products in bulk
    corporate_customers = customers[customers['Customer_Type'] == 'Corporate']['Customer_ID'].values
    expensive_products = products.nlargest(10, 'Base_Price')['Product_ID'].values
    
    df_transactions.loc[idx, 'Customer_ID'] = np.random.choice(corporate_customers)
    df_transactions.loc[idx, 'Customer_Type'] = 'Corporate'
    df_transactions.loc[idx, 'Product_ID'] = np.random.choice(expensive_products)
    
    product_id = df_transactions.loc[idx, 'Product_ID']
    base_price = products.loc[products['Product_ID'] == product_id, 'Base_Price'].values[0]
    
    quantity = np.random.randint(50, 300)
    df_transactions.loc[idx, 'Quantity'] = quantity
    df_transactions.loc[idx, 'Unit_Price'] = base_price
    df_transactions.loc[idx, 'Total_Bayar'] = base_price * quantity
    df_transactions.loc[idx, 'Legitimate_High_Value'] = True

# Fill NaN in marker columns
df_transactions['Error_Type'] = df_transactions['Error_Type'].fillna('None')
df_transactions['Legitimate_High_Value'] = df_transactions['Legitimate_High_Value'].fillna(False)

# Add ground truth label
df_transactions['Ground_Truth'] = 'Normal'
df_transactions.loc[df_transactions['Error_Type'] != 'None', 'Ground_Truth'] = 'Error'
df_transactions.loc[df_transactions['Legitimate_High_Value'] == True, 'Ground_Truth'] = 'Legitimate High-Value'

# Save to CSV
print("\nSaving data...")
df_transactions.to_csv('sample_transactions.csv', index=False)
customers.to_csv('customers.csv', index=False)
products.to_csv('products.csv', index=False)

# Print summary
print("\n" + "=" * 80)
print("DATA GENERATION SUMMARY")
print("=" * 80)
print(f"Total transactions: {len(df_transactions)}")
print(f"Date range: {df_transactions['Transaction_Date'].min()} to {df_transactions['Transaction_Date'].max()}")
print(f"\nTotal Bayar statistics:")
print(f"  Mean: Rp {df_transactions['Total_Bayar'].mean():,.0f}")
print(f"  Median: Rp {df_transactions['Total_Bayar'].median():,.0f}")
print(f"  Min: Rp {df_transactions['Total_Bayar'].min():,.0f}")
print(f"  Max: Rp {df_transactions['Total_Bayar'].max():,.0f}")
print(f"\nGround Truth distribution:")
print(df_transactions['Ground_Truth'].value_counts())
print(f"\nTransactions > Rp 10 juta: {(df_transactions['Total_Bayar'] > 10_000_000).sum()}")
print("=" * 80)

print("\nFiles saved:")
print("  - sample_transactions.csv")
print("  - customers.csv")
print("  - products.csv")
