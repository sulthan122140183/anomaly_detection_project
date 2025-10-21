"""
Deep Learning Methods for Anomaly Detection
- Autoencoder
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, Optional
import warnings
warnings.filterwarnings('ignore')

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, Model
    from tensorflow.keras.callbacks import EarlyStopping
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("Warning: TensorFlow not installed. Deep learning methods will not be available.")

from sklearn.preprocessing import StandardScaler


class AnomalyAutoencoder:
    """
    Autoencoder untuk deteksi anomali
    Model belajar merekonstruksi data normal
    Data anomali akan memiliki reconstruction error tinggi
    """
    
    def __init__(
        self,
        input_dim: int,
        encoding_dim: int = None,
        hidden_layers: list = None
    ):
        """
        Args:
            input_dim: Dimensi input (jumlah features)
            encoding_dim: Dimensi encoding layer (default: input_dim // 2)
            hidden_layers: List dimensi hidden layers (default: [input_dim*2, input_dim])
        """
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow is required for Autoencoder. Install with: pip install tensorflow")
        
        self.input_dim = input_dim
        self.encoding_dim = encoding_dim or max(1, input_dim // 2)
        self.hidden_layers = hidden_layers or [input_dim * 2, input_dim]
        
        self.model = None
        self.scaler = StandardScaler()
        self.threshold = None
        self.history = None
        
    def build_model(self):
        """Build autoencoder architecture"""
        # Input layer
        input_layer = layers.Input(shape=(self.input_dim,))
        
        # Encoder
        encoded = input_layer
        for units in self.hidden_layers:
            encoded = layers.Dense(units, activation='relu')(encoded)
        
        # Bottleneck (encoding layer)
        encoded = layers.Dense(self.encoding_dim, activation='relu', name='encoding')(encoded)
        
        # Decoder
        decoded = encoded
        for units in reversed(self.hidden_layers):
            decoded = layers.Dense(units, activation='relu')(decoded)
        
        # Output layer
        decoded = layers.Dense(self.input_dim, activation='linear')(decoded)
        
        # Create model
        self.model = Model(inputs=input_layer, outputs=decoded)
        self.model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae']
        )
        
        return self.model
    
    def train(
        self,
        X_train: np.ndarray,
        epochs: int = 100,
        batch_size: int = 32,
        validation_split: float = 0.2,
        verbose: int = 1
    ):
        """
        Train autoencoder
        
        Args:
            X_train: Training data (hanya data normal)
            epochs: Jumlah epochs
            batch_size: Batch size
            validation_split: Proporsi data untuk validation
            verbose: Verbosity level
        """
        # Scale data
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Build model if not exists
        if self.model is None:
            self.build_model()
        
        # Early stopping
        early_stop = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        # Train
        self.history = self.model.fit(
            X_train_scaled,
            X_train_scaled,  # Autoencoder: input = output
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=[early_stop],
            verbose=verbose
        )
        
        # Calculate threshold (95th percentile of reconstruction error on training data)
        reconstructions = self.model.predict(X_train_scaled, verbose=0)
        mse = np.mean(np.power(X_train_scaled - reconstructions, 2), axis=1)
        self.threshold = np.percentile(mse, 95)
        
        return self.history
    
    def predict_reconstruction_error(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate reconstruction error for input data
        
        Args:
            X: Input data
        
        Returns:
            Array of reconstruction errors (MSE)
        """
        X_scaled = self.scaler.transform(X)
        reconstructions = self.model.predict(X_scaled, verbose=0)
        mse = np.mean(np.power(X_scaled - reconstructions, 2), axis=1)
        return mse
    
    def predict_anomalies(
        self,
        X: np.ndarray,
        threshold: Optional[float] = None
    ) -> np.ndarray:
        """
        Predict anomalies based on reconstruction error
        
        Args:
            X: Input data
            threshold: Custom threshold (default: use calculated threshold)
        
        Returns:
            Boolean array (True = anomaly)
        """
        mse = self.predict_reconstruction_error(X)
        threshold = threshold or self.threshold
        return mse > threshold
    
    def get_anomaly_scores(self, X: np.ndarray) -> np.ndarray:
        """
        Get anomaly scores (normalized reconstruction error)
        
        Args:
            X: Input data
        
        Returns:
            Array of anomaly scores (0-1, higher = more anomalous)
        """
        mse = self.predict_reconstruction_error(X)
        # Normalize to 0-1 range
        scores = (mse - mse.min()) / (mse.max() - mse.min() + 1e-8)
        return scores


def detect_anomalies_autoencoder(
    df: pd.DataFrame,
    features: list,
    encoding_dim: int = None,
    epochs: int = 100,
    contamination: float = 0.05,
    verbose: int = 0
) -> Tuple[pd.DataFrame, Dict]:
    """
    Deteksi anomali menggunakan Autoencoder
    
    Args:
        df: DataFrame input
        features: List nama kolom yang akan digunakan
        encoding_dim: Dimensi encoding layer
        epochs: Jumlah epochs untuk training
        contamination: Proporsi anomali yang diharapkan (untuk threshold)
        verbose: Verbosity level
    
    Returns:
        Tuple of (anomalies_df, statistics_dict)
    """
    if not TENSORFLOW_AVAILABLE:
        raise ImportError("TensorFlow is required. Install with: pip install tensorflow")
    
    # Prepare data
    X = df[features].fillna(0).values
    
    # Create and train autoencoder
    autoencoder = AnomalyAutoencoder(
        input_dim=len(features),
        encoding_dim=encoding_dim
    )
    
    autoencoder.train(X, epochs=epochs, verbose=verbose)
    
    # Get reconstruction errors
    reconstruction_errors = autoencoder.predict_reconstruction_error(X)
    
    # Set threshold based on contamination
    threshold = np.percentile(reconstruction_errors, (1 - contamination) * 100)
    
    # Predict anomalies
    is_anomaly = reconstruction_errors > threshold
    
    # Add results to dataframe
    df_copy = df.copy()
    df_copy['reconstruction_error'] = reconstruction_errors
    df_copy['is_anomaly'] = is_anomaly
    df_copy['anomaly_score'] = autoencoder.get_anomaly_scores(X)
    
    # Extract anomalies
    anomalies = df_copy[df_copy['is_anomaly']].copy()
    anomalies['method'] = 'Autoencoder'
    
    # Statistics
    stats = {
        'total_anomalies': len(anomalies),
        'percentage': (len(anomalies) / len(df)) * 100,
        'threshold': threshold,
        'mean_reconstruction_error': reconstruction_errors.mean(),
        'std_reconstruction_error': reconstruction_errors.std(),
        'max_reconstruction_error': reconstruction_errors.max(),
        'min_reconstruction_error': reconstruction_errors.min(),
        'encoding_dim': autoencoder.encoding_dim,
        'input_dim': autoencoder.input_dim
    }
    
    return anomalies, stats


if __name__ == "__main__":
    if not TENSORFLOW_AVAILABLE:
        print("TensorFlow not available. Skipping tests.")
    else:
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
        print("TESTING DEEP LEARNING METHODS")
        print("=" * 80)
        
        # Test Autoencoder
        print("\n1. Autoencoder:")
        print("   Training model...")
        anomalies_ae, stats_ae = detect_anomalies_autoencoder(
            df_test,
            features=['Total_Bayar', 'Quantity'],
            epochs=50,
            contamination=0.01,
            verbose=0
        )
        print(f"   Detected {len(anomalies_ae)} anomalies ({stats_ae['percentage']:.2f}%)")
        print(f"   Threshold: {stats_ae['threshold']:.6f}")
        print(f"   Mean reconstruction error: {stats_ae['mean_reconstruction_error']:.6f}")
        print(f"   Max reconstruction error: {stats_ae['max_reconstruction_error']:.6f}")
        
        # Show top anomalies
        print("\n   Top 5 anomalies by reconstruction error:")
        top_anomalies = anomalies_ae.nlargest(5, 'reconstruction_error')
        for idx, row in top_anomalies.iterrows():
            print(f"   - ID {row['Transaction_ID']}: Rp {row['Total_Bayar']:,.0f} (error: {row['reconstruction_error']:.6f})")
        
        print("\n" + "=" * 80)
