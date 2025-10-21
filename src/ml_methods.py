"""
Machine Learning Methods for Anomaly Detection
- Isolation Forest
- Local Outlier Factor (LOF)
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Dict
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler


def detect_anomalies_iforest(
    df: pd.DataFrame,
    features: List[str],
    contamination: float = 0.01,
    random_state: int = 42,
    n_estimators: int = 200
) -> Tuple[pd.DataFrame, Dict]:
    """
    Deteksi anomali menggunakan Isolation Forest
    
    Args:
        df: DataFrame input
        features: List nama kolom yang akan digunakan sebagai features
        contamination: Proporsi anomali yang diharapkan (default 0.01 = 1%)
        random_state: Random seed untuk reproducibility
        n_estimators: Jumlah trees dalam forest
    
    Returns:
        Tuple of (anomalies_df, statistics_dict)
    """
    # Prepare data
    X = df[features].fillna(0).values
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train Isolation Forest
    model = IsolationForest(
        contamination=contamination,
        random_state=random_state,
        n_estimators=n_estimators
    )
    
    # Predict
    predictions = model.fit_predict(X_scaled)
    scores = model.score_samples(X_scaled)
    
    # Add results to dataframe
    df_copy = df.copy()
    df_copy['iforest_prediction'] = predictions
    df_copy['iforest_score'] = scores
    df_copy['is_anomaly'] = predictions == -1
    
    # Extract anomalies
    anomalies = df_copy[df_copy['is_anomaly']].copy()
    anomalies['method'] = 'Isolation Forest'
    
    # Statistics
    stats = {
        'total_anomalies': len(anomalies),
        'percentage': (len(anomalies) / len(df)) * 100,
        'contamination': contamination,
        'n_estimators': n_estimators,
        'mean_score': scores.mean(),
        'min_score': scores.min(),
        'max_score': scores.max(),
        'anomaly_score_threshold': scores[predictions == -1].max() if len(anomalies) > 0 else None
    }
    
    return anomalies, stats


def detect_anomalies_lof(
    df: pd.DataFrame,
    features: List[str],
    n_neighbors: int = 20,
    contamination: float = 0.01
) -> Tuple[pd.DataFrame, Dict]:
    """
    Deteksi anomali menggunakan Local Outlier Factor (LOF)
    
    Args:
        df: DataFrame input
        features: List nama kolom yang akan digunakan sebagai features
        n_neighbors: Jumlah neighbors untuk density estimation
        contamination: Proporsi anomali yang diharapkan
    
    Returns:
        Tuple of (anomalies_df, statistics_dict)
    """
    # Prepare data
    X = df[features].fillna(0).values
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train LOF
    model = LocalOutlierFactor(
        n_neighbors=n_neighbors,
        contamination=contamination
    )
    
    # Predict
    predictions = model.fit_predict(X_scaled)
    scores = model.negative_outlier_factor_
    
    # Add results to dataframe
    df_copy = df.copy()
    df_copy['lof_prediction'] = predictions
    df_copy['lof_score'] = scores
    df_copy['is_anomaly'] = predictions == -1
    
    # Extract anomalies
    anomalies = df_copy[df_copy['is_anomaly']].copy()
    anomalies['method'] = 'LOF'
    
    # Statistics
    stats = {
        'total_anomalies': len(anomalies),
        'percentage': (len(anomalies) / len(df)) * 100,
        'n_neighbors': n_neighbors,
        'contamination': contamination,
        'mean_score': scores.mean(),
        'min_score': scores.min(),
        'max_score': scores.max(),
        'anomaly_score_threshold': scores[predictions == -1].max() if len(anomalies) > 0 else None
    }
    
    return anomalies, stats


def detect_anomalies_multivariate(
    df: pd.DataFrame,
    features: List[str],
    method: str = 'iforest',
    **kwargs
) -> Tuple[pd.DataFrame, Dict]:
    """
    Deteksi anomali multivariate (menggunakan multiple features)
    
    Args:
        df: DataFrame input
        features: List nama kolom yang akan digunakan
        method: 'iforest' atau 'lof'
        **kwargs: Additional parameters untuk method yang dipilih
    
    Returns:
        Tuple of (anomalies_df, statistics_dict)
    """
    if method == 'iforest':
        return detect_anomalies_iforest(df, features, **kwargs)
    elif method == 'lof':
        return detect_anomalies_lof(df, features, **kwargs)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'iforest' or 'lof'")


def compare_ml_methods(
    df: pd.DataFrame,
    features: List[str],
    contamination: float = 0.01
) -> Dict:
    """
    Bandingkan hasil dari berbagai metode ML
    
    Args:
        df: DataFrame input
        features: List nama kolom yang akan digunakan
        contamination: Proporsi anomali yang diharapkan
    
    Returns:
        Dictionary berisi hasil dari semua metode
    """
    results = {}
    
    # Isolation Forest
    anomalies_if, stats_if = detect_anomalies_iforest(
        df, features, contamination=contamination
    )
    results['Isolation Forest'] = {
        'anomalies': anomalies_if,
        'stats': stats_if
    }
    
    # LOF
    anomalies_lof, stats_lof = detect_anomalies_lof(
        df, features, contamination=contamination
    )
    results['LOF'] = {
        'anomalies': anomalies_lof,
        'stats': stats_lof
    }
    
    # Find common anomalies (detected by both methods)
    if_indices = set(anomalies_if.index)
    lof_indices = set(anomalies_lof.index)
    common_indices = if_indices.intersection(lof_indices)
    
    results['summary'] = {
        'IsolationForest_count': len(anomalies_if),
        'LOF_count': len(anomalies_lof),
        'common_anomalies': len(common_indices),
        'total_records': len(df),
        'agreement_rate': len(common_indices) / max(len(if_indices), len(lof_indices)) if max(len(if_indices), len(lof_indices)) > 0 else 0
    }
    
    return results


def ensemble_ml_detection(
    df: pd.DataFrame,
    features: List[str],
    contamination: float = 0.01,
    min_votes: int = 1
) -> Tuple[pd.DataFrame, Dict]:
    """
    Ensemble detection: kombinasi Isolation Forest dan LOF
    Anomali terdeteksi jika minimal min_votes metode mendeteksinya
    
    Args:
        df: DataFrame input
        features: List nama kolom yang akan digunakan
        contamination: Proporsi anomali yang diharapkan
        min_votes: Minimum jumlah metode yang harus mendeteksi (1 atau 2)
    
    Returns:
        Tuple of (anomalies_df, statistics_dict)
    """
    # Run both methods
    anomalies_if, _ = detect_anomalies_iforest(df, features, contamination=contamination)
    anomalies_lof, _ = detect_anomalies_lof(df, features, contamination=contamination)
    
    # Count votes
    df_copy = df.copy()
    df_copy['votes'] = 0
    df_copy.loc[anomalies_if.index, 'votes'] += 1
    df_copy.loc[anomalies_lof.index, 'votes'] += 1
    
    # Extract anomalies based on min_votes
    anomalies = df_copy[df_copy['votes'] >= min_votes].copy()
    anomalies['method'] = f'Ensemble (min_votes={min_votes})'
    
    # Add which methods detected it
    anomalies['detected_by'] = anomalies['votes'].apply(
        lambda v: 'Both' if v == 2 else 'One method'
    )
    
    # Statistics
    stats = {
        'total_anomalies': len(anomalies),
        'percentage': (len(anomalies) / len(df)) * 100,
        'min_votes': min_votes,
        'detected_by_both': len(anomalies[anomalies['votes'] == 2]),
        'detected_by_one': len(anomalies[anomalies['votes'] == 1]),
        'if_only': len(set(anomalies_if.index) - set(anomalies_lof.index)),
        'lof_only': len(set(anomalies_lof.index) - set(anomalies_if.index))
    }
    
    return anomalies, stats


if __name__ == "__main__":
    # Test dengan data sample
    np.random.seed(42)
    
    # Generate data normal + beberapa outlier
    normal_data = np.random.normal(5_000_000, 1_000_000, 1000)
    outliers = [15_000_000, 20_000_000, 25_000_000, 100_000, 50_000]
    data = np.concatenate([normal_data, outliers])
    
    # Generate additional features
    quantities = np.random.randint(1, 100, len(data))
    
    df_test = pd.DataFrame({
        'Transaction_ID': range(len(data)),
        'Total_Bayar': data,
        'Quantity': quantities
    })
    
    print("=" * 80)
    print("TESTING MACHINE LEARNING METHODS")
    print("=" * 80)
    
    # Test Isolation Forest
    print("\n1. Isolation Forest:")
    anomalies_if, stats_if = detect_anomalies_iforest(
        df_test, ['Total_Bayar'], contamination=0.01
    )
    print(f"   Detected {len(anomalies_if)} anomalies ({stats_if['percentage']:.2f}%)")
    print(f"   Score range: [{stats_if['min_score']:.4f}, {stats_if['max_score']:.4f}]")
    
    # Test LOF
    print("\n2. Local Outlier Factor:")
    anomalies_lof, stats_lof = detect_anomalies_lof(
        df_test, ['Total_Bayar'], contamination=0.01
    )
    print(f"   Detected {len(anomalies_lof)} anomalies ({stats_lof['percentage']:.2f}%)")
    print(f"   Score range: [{stats_lof['min_score']:.4f}, {stats_lof['max_score']:.4f}]")
    
    # Test Ensemble
    print("\n3. Ensemble (Both methods):")
    anomalies_ensemble, stats_ensemble = ensemble_ml_detection(
        df_test, ['Total_Bayar'], contamination=0.01, min_votes=2
    )
    print(f"   Detected {len(anomalies_ensemble)} anomalies ({stats_ensemble['percentage']:.2f}%)")
    print(f"   Detected by both: {stats_ensemble['detected_by_both']}")
    print(f"   IF only: {stats_ensemble['if_only']}")
    print(f"   LOF only: {stats_ensemble['lof_only']}")
    
    print("\n" + "=" * 80)
