import pandas as pd
import numpy as np
import os
from scipy import signal
from scipy.stats import zscore
from sklearn.decomposition import FastICA
import matplotlib.pyplot as plt
import warnings

class EEGPreprocessor:
    def __init__(self, sampling_rate=250):
        """
        EEG Preprocessing Pipeline with ICA for artifact removal
        
        Parameters:
        sampling_rate: int, sampling frequency of EEG data (default: 250 Hz)
        """
        self.sampling_rate = sampling_rate
        self.eeg_channels = ['FZ', 'C3', 'CZ', 'C4', 'PZ', 'PO7', 'OZ', 'PO8']
        self.ica = None
        self.fitted = False
        
    def bandpass_filter(self, data, low_freq=1.0, high_freq=40.0, order=4):
        """
        Apply bandpass filter to remove artifacts and noise
        
        Parameters:
        data: numpy array, EEG data (samples x channels)
        low_freq: float, low cutoff frequency (Hz)
        high_freq: float, high cutoff frequency (Hz)
        order: int, filter order
        
        Returns:
        filtered_data: numpy array, filtered EEG data
        """
        nyquist = self.sampling_rate / 2
        low = low_freq / nyquist
        high = high_freq / nyquist
        
        # Design Butterworth bandpass filter
        b, a = signal.butter(order, [low, high], btype='band')
        
        # Apply filter to each channel
        filtered_data = np.zeros_like(data)
        for i in range(data.shape[1]):
            filtered_data[:, i] = signal.filtfilt(b, a, data[:, i])
            
        return filtered_data
    
    def notch_filter(self, data, notch_freq=50.0, quality_factor=30):
        """
        Apply notch filter to remove power line interference
        
        Parameters:
        data: numpy array, EEG data (samples x channels)
        notch_freq: float, notch frequency (Hz) - 50Hz for most countries, 60Hz for US
        quality_factor: float, quality factor of the notch filter
        
        Returns:
        filtered_data: numpy array, notch filtered EEG data
        """
        # Design notch filter
        b, a = signal.iirnotch(notch_freq, quality_factor, fs=self.sampling_rate)
        
        # Apply filter to each channel
        filtered_data = np.zeros_like(data)
        for i in range(data.shape[1]):
            filtered_data[:, i] = signal.filtfilt(b, a, data[:, i])
            
        return filtered_data
    
    def remove_baseline(self, data, baseline_window=None):
        """
        Remove baseline from EEG data
        
        Parameters:
        data: numpy array, EEG data (samples x channels)
        baseline_window: tuple, (start_sample, end_sample) for baseline period
                        If None, uses entire trial for baseline removal
        
        Returns:
        corrected_data: numpy array, baseline corrected EEG data
        """
        if baseline_window is None:
            # Remove mean of entire trial
            baseline_mean = np.mean(data, axis=0)
        else:
            # Remove mean of specified baseline period
            start_idx, end_idx = baseline_window
            baseline_mean = np.mean(data[start_idx:end_idx, :], axis=0)
        
        corrected_data = data - baseline_mean
        return corrected_data
    
    def detect_bad_channels(self, data, threshold=3.0):
        """
        Detect bad channels based on statistical outliers
        
        Parameters:
        data: numpy array, EEG data (samples x channels)
        threshold: float, z-score threshold for outlier detection
        
        Returns:
        bad_channels: list, indices of bad channels
        """
        # Calculate variance and kurtosis for each channel
        variances = np.var(data, axis=0)
        kurtosis_vals = []
        
        for i in range(data.shape[1]):
            # Calculate kurtosis (measure of tail heaviness)
            kurtosis_vals.append(self._kurtosis(data[:, i]))
        
        kurtosis_vals = np.array(kurtosis_vals)
        
        # Z-score normalization
        var_z = np.abs(zscore(variances))
        kurt_z = np.abs(zscore(kurtosis_vals))
        
        # Identify bad channels
        bad_channels = []
        for i in range(len(var_z)):
            if var_z[i] > threshold or kurt_z[i] > threshold:
                bad_channels.append(i)
        
        return bad_channels
    
    def _kurtosis(self, x):
        """Calculate kurtosis of a signal"""
        n = len(x)
        mean = np.mean(x)
        var = np.var(x)
        
        if var == 0:
            return 0
        
        kurtosis = (np.sum((x - mean) ** 4) / n) / (var ** 2) - 3
        return kurtosis
    
    def interpolate_bad_channels(self, data, bad_channels):
        """
        Interpolate bad channels using spherical spline interpolation
        For simplicity, we'll use linear interpolation of neighboring channels
        
        Parameters:
        data: numpy array, EEG data (samples x channels)
        bad_channels: list, indices of bad channels to interpolate
        
        Returns:
        interpolated_data: numpy array, data with interpolated channels
        """
        interpolated_data = data.copy()
        
        # Channel positions (simplified for 8-channel setup)
        channel_positions = {
            0: 'FZ',   # Frontal center
            1: 'C3',   # Left central
            2: 'CZ',   # Central
            3: 'C4',   # Right central
            4: 'PZ',   # Parietal center
            5: 'PO7',  # Left parieto-occipital
            6: 'OZ',   # Occipital center
            7: 'PO8'   # Right parieto-occipital
        }
        
        # Define neighboring channels for interpolation
        neighbors = {
            0: [2, 4],        # FZ -> CZ, PZ
            1: [0, 2, 5],     # C3 -> FZ, CZ, PO7
            2: [0, 1, 3, 4],  # CZ -> FZ, C3, C4, PZ
            3: [0, 2, 7],     # C4 -> FZ, CZ, PO8
            4: [0, 2, 6],     # PZ -> FZ, CZ, OZ
            5: [1, 6],        # PO7 -> C3, OZ
            6: [4, 5, 7],     # OZ -> PZ, PO7, PO8
            7: [3, 6]         # PO8 -> C4, OZ
        }
        
        for bad_ch in bad_channels:
            if bad_ch in neighbors:
                # Get neighboring channels that are not bad
                good_neighbors = [ch for ch in neighbors[bad_ch] if ch not in bad_channels]
                
                if good_neighbors:
                    # Average the neighboring channels
                    interpolated_data[:, bad_ch] = np.mean(data[:, good_neighbors], axis=1)
                    print(f"Interpolated channel {self.eeg_channels[bad_ch]} using neighbors: {[self.eeg_channels[ch] for ch in good_neighbors]}")
        
        return interpolated_data
    
    def apply_ica(self, data, n_components=None, max_iter=200, random_state=42):
        """
        Apply Independent Component Analysis for artifact removal
        
        Parameters:
        data: numpy array, EEG data (samples x channels)
        n_components: int, number of ICA components (default: same as channels)
        max_iter: int, maximum iterations for ICA convergence
        random_state: int, random seed for reproducibility
        
        Returns:
        ica_data: numpy array, ICA transformed data
        components: numpy array, ICA mixing matrix
        """
        if n_components is None:
            n_components = data.shape[1]
        
        # Initialize ICA
        self.ica = FastICA(n_components=n_components, 
                          max_iter=max_iter, 
                          random_state=random_state,
                          whiten='unit-variance')
        
        # Fit ICA and transform data
        ica_data = self.ica.fit_transform(data)
        self.fitted = True
        
        return ica_data, self.ica.components_
    
    def remove_ica_components(self, data, components_to_remove):
        """
        Remove specific ICA components and reconstruct the signal
        
        Parameters:
        data: numpy array, original EEG data (samples x channels)
        components_to_remove: list, indices of ICA components to remove
        
        Returns:
        reconstructed_data: numpy array, reconstructed EEG data
        """
        if not self.fitted:
            raise ValueError("ICA must be fitted first using apply_ica()")
        
        # Transform data to ICA space
        ica_data = self.ica.transform(data)
        
        # Remove specified components by setting them to zero
        ica_cleaned = ica_data.copy()
        ica_cleaned[:, components_to_remove] = 0
        
        # Reconstruct the signal
        reconstructed_data = self.ica.inverse_transform(ica_cleaned)
        
        return reconstructed_data
    def validate_data(self, data):
        """Check for NaN, Inf, and constant channels before processing"""
        # Check for NaN/Inf
        if np.any(np.isnan(data)) or np.any(np.isinf(data)):
            print(f"WARNING: Input data contains NaN/Inf values!")
            return False
        
        # Check for constant channels
        channel_stds = np.std(data, axis=0)
        constant_channels = np.where(channel_stds < 1e-10)[0]
        if len(constant_channels) > 0:
            print(f"WARNING: Constant channels detected: {[self.eeg_channels[i] for i in constant_channels]}")
            return False
    
        return True
    def check_for_nans(self, data, step_name):
        """Check for NaN values and report"""
        nan_count = np.sum(np.isnan(data))
        if nan_count > 0:
            nan_channels = np.any(np.isnan(data), axis=0)
            bad_channels = [self.eeg_channels[i] for i, is_nan in enumerate(nan_channels) if is_nan]
            print(f"ERROR at {step_name}: Found {nan_count} NaN values in channels: {bad_channels}")
            return True
        return False
    def standardize_data(self, data):
        """Standardize EEG data with protection against zero variance"""
        standardized_data = np.zeros_like(data)
        for i in range(data.shape[1]):
            channel_std = np.std(data[:, i])
            if channel_std < 1e-10:  # Essentially zero variance
                print(f"Warning: Channel {self.eeg_channels[i]} has near-zero variance, skipping standardization")
                standardized_data[:, i] = data[:, i] - np.mean(data[:, i])
            else:
                standardized_data[:, i] = zscore(data[:, i])
        
        return standardized_data
    
    def preprocess_trial(self, trial_data, 
                        apply_bandpass=True, 
                        apply_notch=True,
                        remove_baseline_drift=True,
                        detect_bad_channels_flag=True,
                        apply_ica_flag=True,
                        standardize=True,
                        ica_components_to_remove=None):
        """
        Complete preprocessing pipeline for a single trial
        
        Parameters:
        trial_data: pandas DataFrame, raw trial data
        apply_bandpass: bool, whether to apply bandpass filter
        apply_notch: bool, whether to apply notch filter
        remove_baseline_drift: bool, whether to remove baseline
        detect_bad_channels_flag: bool, whether to detect and interpolate bad channels
        apply_ica_flag: bool, whether to apply ICA
        standardize: bool, whether to standardize the data
        ica_components_to_remove: list, specific ICA components to remove (auto-detect if None)
        
        Returns:
        preprocessed_data: numpy array, preprocessed EEG data
        preprocessing_info: dict, information about preprocessing steps
        """
        eeg_data = trial_data[self.eeg_channels].values
    
        # Initial validation
        if not self.validate_data(eeg_data):
            print("Initial data validation failed!")
        
        preprocessing_info = {
            'original_shape': eeg_data.shape,
            'bad_channels': [],
            'ica_components_removed': [],
            'preprocessing_steps': []
        }
        
        print(f"Starting preprocessing for trial with shape: {eeg_data.shape}")
        
        # Check after each step
        self.check_for_nans(eeg_data, "Initial data")
        
        # Step 1: Bandpass filtering
        if apply_bandpass:
            eeg_data = self.bandpass_filter(eeg_data, low_freq=1.0, high_freq=40.0)
            self.check_for_nans(eeg_data, "After bandpass filter")
            preprocessing_info['preprocessing_steps'].append('Bandpass filter (1-40 Hz)')
            print("✓ Applied bandpass filter (1-40 Hz)")
            # Step 2: Notch filtering
        if apply_notch:
            eeg_data = self.notch_filter(eeg_data, notch_freq=50.0)
            preprocessing_info['preprocessing_steps'].append('Notch filter (50 Hz)')
            print("✓ Applied notch filter (50 Hz)")
        
        # Step 3: Baseline removal
        if remove_baseline_drift:
            eeg_data = self.remove_baseline(eeg_data)
            preprocessing_info['preprocessing_steps'].append('Baseline removal')
            print("✓ Removed baseline drift")
        
        # Step 4: Bad channel detection and interpolation
        if detect_bad_channels_flag:
            bad_channels = self.detect_bad_channels(eeg_data, threshold=3.0)
            if bad_channels:
                print(f"⚠ Detected bad channels: {[self.eeg_channels[ch] for ch in bad_channels]}")
                eeg_data = self.interpolate_bad_channels(eeg_data, bad_channels)
                preprocessing_info['bad_channels'] = bad_channels
                preprocessing_info['preprocessing_steps'].append(f'Bad channel interpolation ({len(bad_channels)} channels)')
            else:
                print("✓ No bad channels detected")
        if apply_ica_flag:
            # Handle NaN/Inf values
            if np.any(np.isnan(eeg_data)) or np.any(np.isinf(eeg_data)):
                print("⚠ NaN or Inf detected in data before ICA. Applying np.nan_to_num.")
                eeg_data = np.nan_to_num(eeg_data, nan=0.0, posinf=np.finfo(eeg_data.dtype).max, neginf=np.finfo(eeg_data.dtype).min)

            # Handle constant channels (zero variance)
            stds = np.std(eeg_data, axis=0)
            if np.any(stds < 1e-9): # Using a small threshold for near-zero variance
                constant_channel_indices = np.where(stds < 1e-9)[0]
                print(f"⚠ Constant channel(s) detected at indices: {constant_channel_indices} before ICA. Adding small noise.")
                for ch_idx in constant_channel_indices:
                    eeg_data[:, ch_idx] += np.random.normal(0, 1e-7, size=eeg_data.shape[0])
                
                # Re-check if ICA is still viable, or skip if too many channels are bad
                if np.sum(np.std(eeg_data, axis=0) < 1e-9) >= eeg_data.shape[1] -1 : # If almost all channels are still constant
                    print("‼ Too many constant channels even after adding noise. Skipping ICA for this trial.")
                    apply_ica_flag = False
        
        # Step 5: ICA for artifact removal
        if apply_ica_flag:
            ica_data, components = self.apply_ica(eeg_data)
            
            # Auto-detect artifacts if not specified
            if ica_components_to_remove is None:
                ica_components_to_remove = self._auto_detect_artifacts(ica_data, components)
            
            if ica_components_to_remove:
                eeg_data = self.remove_ica_components(eeg_data, ica_components_to_remove)
                preprocessing_info['ica_components_removed'] = ica_components_to_remove
                preprocessing_info['preprocessing_steps'].append(f'ICA artifact removal ({len(ica_components_to_remove)} components)')
                print(f"✓ Removed ICA components: {ica_components_to_remove}")
            else:
                print("✓ No artifact components detected by ICA")
        
        # Step 6: Standardization
        if standardize:
            eeg_data = self.standardize_data(eeg_data)
            preprocessing_info['preprocessing_steps'].append('Z-score standardization')
            print("✓ Applied standardization")
        
        preprocessing_info['final_shape'] = eeg_data.shape
        print(f"✓ Preprocessing complete. Final shape: {eeg_data.shape}")
        
        return eeg_data, preprocessing_info
    
    def _auto_detect_artifacts(self, ica_data, components, threshold_percentile=95):
        """
        Automatically detect artifact components based on statistical properties
        
        Parameters:
        ica_data: numpy array, ICA transformed data
        components: numpy array, ICA mixing matrix
        threshold_percentile: float, percentile threshold for artifact detection
        
        Returns:
        artifact_components: list, indices of detected artifact components
        """
        artifact_components = []
        
        # Calculate metrics for each component
        for i in range(ica_data.shape[1]):
            component_data = ica_data[:, i]
            
            # Metric 1: High variance (muscle artifacts, eye movements)
            variance = np.var(component_data)
            
            # Metric 2: High kurtosis (blink artifacts, spikes)
            kurtosis_val = self._kurtosis(component_data)
            
            # Metric 3: Frequency content analysis
            freqs, psd = signal.welch(component_data, fs=self.sampling_rate, nperseg=min(256, len(component_data)//2))
            
            # Check for high power in artifact frequency ranges
            # Eye blinks/movements: < 4 Hz
            # Muscle artifacts: > 30 Hz
            low_freq_power = np.sum(psd[freqs < 4])
            high_freq_power = np.sum(psd[freqs > 30])
            total_power = np.sum(psd)
            
            low_freq_ratio = low_freq_power / total_power if total_power > 0 else 0
            high_freq_ratio = high_freq_power / total_power if total_power > 0 else 0
            
            # Heuristic thresholds for artifact detection
            variance_threshold = np.percentile([np.var(ica_data[:, j]) for j in range(ica_data.shape[1])], threshold_percentile)
            kurtosis_threshold = 5.0  # High kurtosis indicates non-Gaussian artifacts
            
            # Detect artifacts
            is_artifact = (variance > variance_threshold or 
                          abs(kurtosis_val) > kurtosis_threshold or
                          low_freq_ratio > 0.6 or  # Dominated by low frequencies (eye artifacts)
                          high_freq_ratio > 0.3)   # High muscle artifact content
            
            if is_artifact:
                artifact_components.append(i)
        
        return artifact_components
    
    def plot_preprocessing_comparison(self, original_data, preprocessed_data, trial_info):
        """
        Plot comparison between original and preprocessed data
        
        Parameters:
        original_data: numpy array, original EEG data
        preprocessed_data: numpy array, preprocessed EEG data
        trial_info: dict, trial information
        """
        fig, axes = plt.subplots(4, 2, figsize=(15, 12))
        fig.suptitle(f'EEG Preprocessing Comparison - {trial_info.get("task", "Unknown")} Task', fontsize=16)
        
        # Time vector
        time = np.arange(original_data.shape[0]) / self.sampling_rate
        
        # Plot first 4 channels
        channels_to_plot = min(4, len(self.eeg_channels))
        
        for i in range(channels_to_plot):
            # Original data
            axes[i, 0].plot(time, original_data[:, i], 'b-', alpha=0.7)
            axes[i, 0].set_title(f'Original - {self.eeg_channels[i]}')
            axes[i, 0].set_ylabel('Amplitude (µV)')
            axes[i, 0].grid(True, alpha=0.3)
            
            # Preprocessed data
            axes[i, 1].plot(time, preprocessed_data[:, i], 'r-', alpha=0.7)
            axes[i, 1].set_title(f'Preprocessed - {self.eeg_channels[i]}')
            axes[i, 1].set_ylabel('Amplitude (Standardized)')
            axes[i, 1].grid(True, alpha=0.3)
            
            if i == channels_to_plot - 1:
                axes[i, 0].set_xlabel('Time (s)')
                axes[i, 1].set_xlabel('Time (s)')
        
        plt.tight_layout()
        plt.show()

# Usage example with your existing code
