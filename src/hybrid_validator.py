"""
Hybrid Anomaly Validator
Kombinasi Statistical + ML + Rule-Based untuk membedakan:
- Kesalahan Input (Error)
- Transaksi Valid Besar (Legitimate High-Value Transaction)
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from .statistical_methods import detect_anomalies_iqr, detect_anomalies_zscore
from .ml_methods import detect_anomalies_iforest, detect_anomalies_lof


class HybridAnomalyValidator:
    """
    Validator hybrid yang menggabungkan:
    1. Deteksi anomali (Statistical + ML)
    2. Validasi kontekstual (Rule-Based)
    3. Confidence scoring
    """
    
    def __init__(self):
        self.anomaly_methods = []
        self.validation_rules = []
        
    def detect_statistical_anomalies(
        self,
        df: pd.DataFrame,
        column: str
    ) -> pd.Series:
        """
        Deteksi anomali menggunakan metode statistik
        Returns: Boolean series (True = anomaly detected)
        """
        # IQR
        anomalies_iqr, _ = detect_anomalies_iqr(df, column)
        
        # Z-Score
        anomalies_zscore, _ = detect_anomalies_zscore(df, column)
        
        # Combine: anomaly if detected by either method
        is_anomaly = pd.Series(False, index=df.index)
        is_anomaly.loc[anomalies_iqr.index] = True
        is_anomaly.loc[anomalies_zscore.index] = True
        
        return is_anomaly
    
    def detect_ml_anomalies(
        self,
        df: pd.DataFrame,
        features: List[str],
        contamination: float = 0.01
    ) -> pd.Series:
        """
        Deteksi anomali menggunakan ML
        Returns: Boolean series (True = anomaly detected)
        """
        # Isolation Forest
        anomalies_if, _ = detect_anomalies_iforest(
            df, features, contamination=contamination
        )
        
        # LOF
        anomalies_lof, _ = detect_anomalies_lof(
            df, features, contamination=contamination
        )
        
        # Combine: anomaly if detected by either method
        is_anomaly = pd.Series(False, index=df.index)
        is_anomaly.loc[anomalies_if.index] = True
        is_anomaly.loc[anomalies_lof.index] = True
        
        return is_anomaly
    
    def validate_customer_history(
        self,
        df: pd.DataFrame,
        customer_col: str = 'Customer_ID',
        amount_col: str = 'Total_Bayar'
    ) -> pd.Series:
        """
        Validasi berdasarkan history customer
        Returns: Boolean series (True = valid/normal untuk customer ini)
        """
        if customer_col not in df.columns:
            return pd.Series(True, index=df.index)
        
        # Calculate customer statistics
        customer_stats = df.groupby(customer_col)[amount_col].agg(['mean', 'std', 'max'])
        
        # Check if current transaction is within customer's normal range
        is_valid = pd.Series(False, index=df.index)
        
        for idx, row in df.iterrows():
            customer_id = row[customer_col]
            amount = row[amount_col]
            
            if customer_id in customer_stats.index:
                cust_mean = customer_stats.loc[customer_id, 'mean']
                cust_std = customer_stats.loc[customer_id, 'std']
                cust_max = customer_stats.loc[customer_id, 'max']
                
                # Valid if within 3 std or not exceeding previous max by more than 2x
                if cust_std > 0:
                    z_score = abs((amount - cust_mean) / cust_std)
                    is_valid.loc[idx] = (z_score <= 3) or (amount <= cust_max * 2)
                else:
                    is_valid.loc[idx] = amount <= cust_max * 2
        
        return is_valid
    
    def validate_product_price(
        self,
        df: pd.DataFrame,
        product_col: str = 'Product_ID',
        quantity_col: str = 'Quantity',
        amount_col: str = 'Total_Bayar'
    ) -> pd.Series:
        """
        Validasi berdasarkan harga produk
        Returns: Boolean series (True = harga wajar)
        """
        if product_col not in df.columns or quantity_col not in df.columns:
            return pd.Series(True, index=df.index)
        
        # Calculate unit price
        df_copy = df.copy()
        df_copy['unit_price'] = df_copy[amount_col] / df_copy[quantity_col].replace(0, 1)
        
        # Calculate product price statistics
        product_stats = df_copy.groupby(product_col)['unit_price'].agg(['mean', 'std'])
        
        # Check if unit price is within normal range
        is_valid = pd.Series(False, index=df.index)
        
        for idx, row in df_copy.iterrows():
            product_id = row[product_col]
            unit_price = row['unit_price']
            
            if product_id in product_stats.index:
                prod_mean = product_stats.loc[product_id, 'mean']
                prod_std = product_stats.loc[product_id, 'std']
                
                # Valid if within 3 std
                if prod_std > 0:
                    z_score = abs((unit_price - prod_mean) / prod_std)
                    is_valid.loc[idx] = z_score <= 3
                else:
                    is_valid.loc[idx] = abs(unit_price - prod_mean) < prod_mean * 0.5
        
        return is_valid
    
    def validate_business_rules(
        self,
        df: pd.DataFrame,
        amount_col: str = 'Total_Bayar',
        quantity_col: str = 'Quantity',
        max_single_item_price: float = 50_000_000,
        max_quantity: int = 10000
    ) -> pd.Series:
        """
        Validasi berdasarkan business rules
        Returns: Boolean series (True = memenuhi business rules)
        """
        is_valid = pd.Series(True, index=df.index)
        
        # Rule 1: Total bayar tidak boleh negatif
        is_valid &= df[amount_col] >= 0
        
        # Rule 2: Quantity harus positif
        if quantity_col in df.columns:
            is_valid &= df[quantity_col] > 0
            is_valid &= df[quantity_col] <= max_quantity
            
            # Rule 3: Unit price tidak boleh terlalu tinggi
            unit_price = df[amount_col] / df[quantity_col].replace(0, 1)
            is_valid &= unit_price <= max_single_item_price
        
        return is_valid
    
    def calculate_confidence_score(
        self,
        is_anomaly: pd.Series,
        validation_results: Dict[str, pd.Series]
    ) -> pd.Series:
        """
        Calculate confidence score (0-1)
        0 = definitely error
        1 = definitely legitimate
        
        Args:
            is_anomaly: Boolean series dari deteksi anomali
            validation_results: Dict of validation results
        
        Returns:
            Series of confidence scores
        """
        confidence = pd.Series(0.5, index=is_anomaly.index)
        
        # If not detected as anomaly, high confidence it's legitimate
        confidence[~is_anomaly] = 0.9
        
        # If detected as anomaly, check validations
        for idx in is_anomaly[is_anomaly].index:
            passed_validations = sum([
                validation_results[key].loc[idx]
                for key in validation_results
            ])
            total_validations = len(validation_results)
            
            if total_validations > 0:
                # More validations passed = higher confidence it's legitimate
                validation_score = passed_validations / total_validations
                
                # If anomaly but passes validations = likely legitimate high-value
                # If anomaly and fails validations = likely error
                confidence.loc[idx] = validation_score
        
        return confidence
    
    def classify_anomaly_type(
        self,
        is_anomaly: pd.Series,
        confidence: pd.Series,
        threshold_error: float = 0.3,
        threshold_legitimate: float = 0.7
    ) -> pd.Series:
        """
        Classify anomaly type based on confidence score
        
        Returns:
            Series with values: 'Normal', 'Error', 'Legitimate High-Value', 'Uncertain'
        """
        classification = pd.Series('Normal', index=is_anomaly.index)
        
        # Detected as anomaly
        anomaly_indices = is_anomaly[is_anomaly].index
        
        for idx in anomaly_indices:
            conf = confidence.loc[idx]
            
            if conf <= threshold_error:
                classification.loc[idx] = 'Kesalahan Input'
            elif conf >= threshold_legitimate:
                classification.loc[idx] = 'Transaksi Valid Besar'
            else:
                classification.loc[idx] = 'Perlu Review Manual'
        
        return classification
    
    def validate(
        self,
        df: pd.DataFrame,
        amount_col: str = 'Total_Bayar',
        features: List[str] = None,
        customer_col: str = None,
        product_col: str = None,
        quantity_col: str = None,
        threshold_amount: float = 10_000_000,
        contamination: float = 0.01
    ) -> pd.DataFrame:
        """
        Full validation pipeline
        
        Args:
            df: DataFrame input
            amount_col: Nama kolom total pembayaran
            features: List features untuk ML (default: [amount_col])
            customer_col: Nama kolom customer ID (optional)
            product_col: Nama kolom product ID (optional)
            quantity_col: Nama kolom quantity (optional)
            threshold_amount: Threshold untuk high-value transaction
            contamination: Contamination rate untuk ML
        
        Returns:
            DataFrame dengan kolom tambahan:
            - is_anomaly: Boolean
            - confidence_score: 0-1
            - classification: 'Normal', 'Error', 'Legitimate High-Value', 'Uncertain'
            - methods_flagged: List metode yang mendeteksi anomali
        """
        if features is None:
            features = [amount_col]
        
        df_result = df.copy()
        
        # 1. Detect anomalies
        print("Step 1: Detecting anomalies...")
        is_anomaly_stat = self.detect_statistical_anomalies(df, amount_col)
        is_anomaly_ml = self.detect_ml_anomalies(df, features, contamination)
        
        # Combine: anomaly if detected by any method
        is_anomaly = is_anomaly_stat | is_anomaly_ml
        
        # Track which methods flagged it
        methods_flagged = []
        for idx in df.index:
            methods = []
            if is_anomaly_stat.loc[idx]:
                methods.append('Statistical')
            if is_anomaly_ml.loc[idx]:
                methods.append('ML')
            methods_flagged.append(methods)
        
        df_result['methods_flagged'] = methods_flagged
        
        # 2. Run validations
        print("Step 2: Running validations...")
        validation_results = {}
        
        # Business rules (always run)
        validation_results['business_rules'] = self.validate_business_rules(
            df, amount_col, quantity_col
        )
        
        # Customer history (if available)
        if customer_col and customer_col in df.columns:
            validation_results['customer_history'] = self.validate_customer_history(
                df, customer_col, amount_col
            )
        
        # Product price (if available)
        if product_col and product_col in df.columns and quantity_col and quantity_col in df.columns:
            validation_results['product_price'] = self.validate_product_price(
                df, product_col, quantity_col, amount_col
            )
        
        # 3. Calculate confidence score
        print("Step 3: Calculating confidence scores...")
        confidence = self.calculate_confidence_score(is_anomaly, validation_results)
        
        # 4. Classify anomaly type
        print("Step 4: Classifying anomalies...")
        classification = self.classify_anomaly_type(is_anomaly, confidence)
        
        # Add results to dataframe
        df_result['is_anomaly'] = is_anomaly
        df_result['confidence_score'] = confidence
        df_result['classification'] = classification
        df_result['above_threshold'] = df[amount_col] > threshold_amount
        
        # Summary
        print("\n" + "=" * 80)
        print("VALIDATION SUMMARY")
        print("=" * 80)
        print(f"Total records: {len(df)}")
        print(f"Anomalies detected: {is_anomaly.sum()} ({is_anomaly.sum()/len(df)*100:.2f}%)")
        print(f"Above threshold (Rp {threshold_amount:,}): {df_result['above_threshold'].sum()}")
        print("\nClassification breakdown:")
        print(df_result['classification'].value_counts())
        print("=" * 80)
        
        return df_result


