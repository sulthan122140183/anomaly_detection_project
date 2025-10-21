"""
Statistical Methods for Anomaly Detection
- IQR (Interquartile Range)
- Z-Score
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict


def detect_anomalies_iqr(
    df: pd.DataFrame,
    column: str,
    multiplier: float = 1.5
) -> Tuple[pd.DataFrame, Dict]:
    """
    Deteksi anomali menggunakan metode IQR (Interquartile Range)
    
    Args:
        df: DataFrame input
        column: Nama kolom yang akan dianalisis
        multiplier: Multiplier untuk IQR (default 1.5 untuk outlier, 3.0 untuk extreme outlier)
    
    Returns:
        Tuple of (anomalies_df, statistics_dict)
    """
    # Hitung Q1, Q3, dan IQR
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    # Hitung batas bawah dan atas
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR
    
    # Identifikasi anomali
    anomalies = df[(df[column] < lower_bound) | (df[column] > upper_bound)].copy()
    anomalies['anomaly_type'] = anomalies[column].apply(
        lambda x: 'below_lower_bound' if x < lower_bound else 'above_upper_bound'
    )
    anomalies['method'] = 'IQR'
    
    # Statistik
    stats = {
        'Q1': Q1,
        'Q3': Q3,
        'IQR': IQR,
        'lower_bound': lower_bound,
        'upper_bound': upper_bound,
        'total_anomalies': len(anomalies),
        'percentage': (len(anomalies) / len(df)) * 100
    }
    
    return anomalies, stats


def detect_anomalies_zscore(
    df: pd.DataFrame,
    column: str,
    threshold: float = 3.0
) -> Tuple[pd.DataFrame, Dict]:
    """
    Deteksi anomali menggunakan Z-Score
    
    Args:
        df: DataFrame input
        column: Nama kolom yang akan dianalisis
        threshold: Threshold Z-score (default 3.0 = 99.7% confidence)
    
    Returns:
        Tuple of (anomalies_df, statistics_dict)
    """
    # Hitung mean dan std
    mean = df[column].mean()
    std = df[column].std()
    
    # Hitung Z-score
    df_copy = df.copy()
    df_copy['zscore'] = (df_copy[column] - mean) / std
    
    # Identifikasi anomali
    anomalies = df_copy[df_copy['zscore'].abs() > threshold].copy()
    anomalies['anomaly_type'] = anomalies['zscore'].apply(
        lambda x: 'negative_outlier' if x < -threshold else 'positive_outlier'
    )
    anomalies['method'] = 'Z-Score'
    
    # Statistik
    stats = {
        'mean': mean,
        'std': std,
        'threshold': threshold,
        'total_anomalies': len(anomalies),
        'percentage': (len(anomalies) / len(df)) * 100,
        'max_zscore': anomalies['zscore'].abs().max() if len(anomalies) > 0 else 0
    }
    
    return anomalies, stats


def detect_anomalies_modified_zscore(
    df: pd.DataFrame,
    column: str,
    threshold: float = 3.5
) -> Tuple[pd.DataFrame, Dict]:
    """
    Deteksi anomali menggunakan Modified Z-Score (lebih robust terhadap outlier)
    Menggunakan median dan MAD (Median Absolute Deviation)
    
    Args:
        df: DataFrame input
        column: Nama kolom yang akan dianalisis
        threshold: Threshold modified Z-score (default 3.5)
    
    Returns:
        Tuple of (anomalies_df, statistics_dict)
    """
    # Hitung median dan MAD
    median = df[column].median()
    mad = np.median(np.abs(df[column] - median))
    
    # Hitung Modified Z-score
    df_copy = df.copy()
    df_copy['modified_zscore'] = 0.6745 * (df_copy[column] - median) / mad if mad != 0 else 0
    
    # Identifikasi anomali
    anomalies = df_copy[df_copy['modified_zscore'].abs() > threshold].copy()
    anomalies['anomaly_type'] = anomalies['modified_zscore'].apply(
        lambda x: 'negative_outlier' if x < -threshold else 'positive_outlier'
    )
    anomalies['method'] = 'Modified Z-Score'
    
    # Statistik
    stats = {
        'median': median,
        'mad': mad,
        'threshold': threshold,
        'total_anomalies': len(anomalies),
        'percentage': (len(anomalies) / len(df)) * 100,
        'max_modified_zscore': anomalies['modified_zscore'].abs().max() if len(anomalies) > 0 else 0
    }
    
    return anomalies, stats


def compare_statistical_methods(
    df: pd.DataFrame,
    column: str
) -> Dict:
    """
    Bandingkan hasil dari berbagai metode statistik
    
    Args:
        df: DataFrame input
        column: Nama kolom yang akan dianalisis
    
    Returns:
        Dictionary berisi hasil dari semua metode
    """
    results = {}
    
    # IQR
    anomalies_iqr, stats_iqr = detect_anomalies_iqr(df, column)
    results['IQR'] = {
        'anomalies': anomalies_iqr,
        'stats': stats_iqr
    }
    
    # Z-Score
    anomalies_zscore, stats_zscore = detect_anomalies_zscore(df, column)
    results['Z-Score'] = {
        'anomalies': anomalies_zscore,
        'stats': stats_zscore
    }
    
    # Modified Z-Score
    anomalies_mod_zscore, stats_mod_zscore = detect_anomalies_modified_zscore(df, column)
    results['Modified Z-Score'] = {
        'anomalies': anomalies_mod_zscore,
        'stats': stats_mod_zscore
    }
    
    # Summary
    results['summary'] = {
        'IQR_count': len(anomalies_iqr),
        'ZScore_count': len(anomalies_zscore),
        'ModifiedZScore_count': len(anomalies_mod_zscore),
        'total_records': len(df)
    }
    
    return results


if __name__ == "__main__":
    # Test dengan data sample
    np.random.seed(42)
    
    # Generate data normal + beberapa outlier
    normal_data = np.random.normal(5_000_000, 1_000_000, 1000)
    outliers = [15_000_000, 20_000_000, 25_000_000, 100_000, 50_000]
    data = np.concatenate([normal_data, outliers])
    
    df_test = pd.DataFrame({
        'Transaction_ID': range(len(data)),
        'Total_Bayar': data
    })
    
    print("=" * 80)
    print("TESTING STATISTICAL METHODS")
    print("=" * 80)
    
    # Test IQR
    print("\n1. IQR Method:")
    anomalies_iqr, stats_iqr = detect_anomalies_iqr(df_test, 'Total_Bayar')
    print(f"   Detected {len(anomalies_iqr)} anomalies ({stats_iqr['percentage']:.2f}%)")
    print(f"   Lower bound: Rp {stats_iqr['lower_bound']:,.0f}")
    print(f"   Upper bound: Rp {stats_iqr['upper_bound']:,.0f}")
    
    # Test Z-Score
    print("\n2. Z-Score Method:")
    anomalies_zscore, stats_zscore = detect_anomalies_zscore(df_test, 'Total_Bayar')
    print(f"   Detected {len(anomalies_zscore)} anomalies ({stats_zscore['percentage']:.2f}%)")
    print(f"   Mean: Rp {stats_zscore['mean']:,.0f}")
    print(f"   Std: Rp {stats_zscore['std']:,.0f}")
    
    # Test Modified Z-Score
    print("\n3. Modified Z-Score Method:")
    anomalies_mod, stats_mod = detect_anomalies_modified_zscore(df_test, 'Total_Bayar')
    print(f"   Detected {len(anomalies_mod)} anomalies ({stats_mod['percentage']:.2f}%)")
    print(f"   Median: Rp {stats_mod['median']:,.0f}")
    print(f"   MAD: Rp {stats_mod['mad']:,.0f}")
    
    print("\n" + "=" * 80)
