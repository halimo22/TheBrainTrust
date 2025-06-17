import numpy as np
import torch
import torch.nn as nn
from scipy.linalg import eigh, inv


class EEGNetv4(nn.Module):
    def __init__(self, nb_classes=2, Chans=8, Samples=2250, 
                 dropoutRate=0.5, kernLength=64, F1=8, D=2, F2=16,
                 norm_rate=0.25, dropoutType='Dropout'):
        """
        EEGNetv4 implementation
        
        Parameters:
        nb_classes: int, number of classes (2 for MI: Left/Right)
        Chans: int, number of channels (8 for your dataset)
        Samples: int, number of time samples (2250 for MI)
        dropoutRate: float, dropout rate
        kernLength: int, temporal kernel length
        F1: int, number of temporal filters
        D: int, depth multiplier for spatial filters  
        F2: int, number of separable filters
        norm_rate: float, max norm constraint
        dropoutType: str, type of dropout
        """
        super(EEGNetv4, self).__init__()
        
        self.nb_classes = nb_classes
        self.Chans = Chans
        self.Samples = Samples
        self.dropoutRate = dropoutRate
        self.kernLength = kernLength
        self.F1 = F1
        self.D = D
        self.F2 = F2
        self.norm_rate = norm_rate
        
        # Block 1: Temporal Convolution
        self.conv1 = nn.Conv2d(1, F1, (1, kernLength), padding=(0, kernLength // 2), bias=False)
        self.batchnorm1 = nn.BatchNorm2d(F1)
        
        # Block 2: Spatial Convolution (Depthwise)
        self.conv2 = nn.Conv2d(F1, F1 * D, (Chans, 1), groups=F1, bias=False)
        self.batchnorm2 = nn.BatchNorm2d(F1 * D)
        self.activation1 = nn.ELU()
        self.pooling1 = nn.AvgPool2d((1, 4))
        
        if dropoutType == 'Dropout':
            self.dropout1 = nn.Dropout(dropoutRate)
        else:
            self.dropout1 = nn.Dropout2d(dropoutRate)
        
        # Block 3: Separable Convolution
        self.conv3 = nn.Conv2d(F1 * D, F2, (1, 16), padding=(0, 8), bias=False)
        self.batchnorm3 = nn.BatchNorm2d(F2)
        self.activation2 = nn.ELU()
        self.pooling2 = nn.AvgPool2d((1, 8))
        
        if dropoutType == 'Dropout':
            self.dropout2 = nn.Dropout(dropoutRate)
        else:
            self.dropout2 = nn.Dropout2d(dropoutRate)
        
        # Classification head
        self.flatten = nn.Flatten()
        
        # Calculate the size of flattened features
        self._calculate_flatten_size()
        
        self.classifier = nn.Linear(self.flatten_size, nb_classes)
        
        # Apply constraints
        self._apply_constraints()
    
    def _calculate_flatten_size(self):
        """Calculate the size after all conv and pooling layers"""
        # Simulate forward pass to get the size
        with torch.no_grad():
            x = torch.zeros(1, 1, self.Chans, self.Samples)
            x = self.conv1(x)
            x = self.conv2(x)
            x = self.pooling1(x)
            x = self.conv3(x)
            x = self.pooling2(x)
            self.flatten_size = x.numel()
    
    def _apply_constraints(self):
        """Apply max norm constraints as in original EEGNet"""
        def apply_constraint(module):
            if hasattr(module, 'weight'):
                with torch.no_grad():
                    norm = module.weight.norm(dim=0, keepdim=True)
                    desired = torch.clamp(norm, 0, self.norm_rate)
                    module.weight *= (desired / (norm + 1e-8))
        
        # Apply to spatial conv layer
        apply_constraint(self.conv2)
    
    def forward(self, x):
        # Block 1
        x = self.conv1(x)
        x = self.batchnorm1(x)
        
        # Block 2
        x = self.conv2(x)
        x = self.batchnorm2(x)
        x = self.activation1(x)
        x = self.pooling1(x)
        x = self.dropout1(x)
        
        # Block 3
        x = self.conv3(x)
        x = self.batchnorm3(x)
        x = self.activation2(x)
        x = self.pooling2(x)
        x = self.dropout2(x)
        
        # Classification
        x = self.flatten(x)
        x = self.classifier(x)
        
        return x



# ==================== CSP-Net Implementation ====================

class CSPLayer(nn.Module):
    def __init__(self, n_channels, n_components=8):
        """
        CSP Layer that can be integrated into neural networks
        
        Parameters:
        n_channels: int, number of EEG channels
        n_components: int, number of CSP components
        """
        super(CSPLayer, self).__init__()
        self.n_channels = n_channels
        self.n_components = n_components
        
        # Learnable spatial filters
        self.spatial_filters = nn.Parameter(
            torch.randn(n_components, n_channels) / n_channels**0.5
        )
        
    def forward(self, x):
        """
        Apply CSP transformation
        Input shape: (batch_size, n_features, n_samples) or (batch_size, n_channels, height, width)
        Output shape: (batch_size, n_components, n_samples) or (batch_size, n_components, height, width)
        """
        # Handle both 3D and 4D inputs
        if x.dim() == 4:
            # Conv2d output: (batch_size, n_features, height, width)
            batch_size, n_features, height, width = x.shape
            
            # For EEGNet, after first conv layer, n_features might be F1 (not n_channels)
            # We need to handle this case differently
            
            if n_features == self.n_channels:
                # Standard case: features match channels
                x_reshaped = x.view(batch_size, n_features, -1)
                x_filtered = torch.matmul(self.spatial_filters, x_reshaped)
                x_filtered = x_filtered.view(batch_size, self.n_components, height, width)
            else:
                # Case where conv has changed the feature dimension
                # Apply CSP across the spatial dimension instead
                x_reshaped = x.permute(0, 2, 1, 3).contiguous()  # (batch, height, features, width)
                x_reshaped = x_reshaped.view(batch_size * height, n_features, width)
                
                # Create adapted filters if needed
                if n_features != self.n_channels:
                    # Use a projection to match dimensions
                    projection = nn.Linear(n_features, self.n_channels, bias=False).to(x.device)
                    x_projected = projection(x_reshaped.transpose(1, 2)).transpose(1, 2)
                    x_filtered = torch.matmul(self.spatial_filters, x_projected)
                else:
                    x_filtered = torch.matmul(self.spatial_filters, x_reshaped)
                
                x_filtered = x_filtered.view(batch_size, height, self.n_components, width)
                x_filtered = x_filtered.permute(0, 2, 1, 3).contiguous()
                
        elif x.dim() == 3:
            # Standard 3D input: (batch_size, n_channels, n_samples)
            batch_size, n_channels, n_samples = x.shape
            assert n_channels == self.n_channels, f"Channel mismatch: expected {self.n_channels}, got {n_channels}"
            
            # Apply spatial filtering
            x_filtered = torch.matmul(self.spatial_filters, x)
        else:
            raise ValueError(f"Expected 3D or 4D input, got {x.dim()}D")
        
        return x_filtered
    def initialize_filters(self, csp_filters):
        """Initialize spatial filters with pre-computed CSP filters"""
        with torch.no_grad():
            self.spatial_filters.data = torch.FloatTensor(csp_filters)

class AdaptiveCSPLayer(nn.Module):
    """
    Adaptive CSP layer that handles dimension mismatches gracefully
    """
    def __init__(self, n_channels, n_components):
        super(AdaptiveCSPLayer, self).__init__()
        self.n_channels = n_channels
        self.n_components = n_components
        
        # Learnable spatial filters
        self.spatial_filters = nn.Parameter(
            torch.randn(n_components, n_channels) / n_channels**0.5
        )
        
        # Adaptive projection layer (will be created dynamically if needed)
        self.projection = None
    def forward(self, x):
        """
        Apply CSP transformation with adaptive dimension handling
        """
        if x.dim() == 4:
            batch_size, n_features, height, width = x.shape
            
            # Create projection layer if needed
            if n_features != self.n_channels and self.projection is None:
                self.projection = nn.Conv1d(n_features, self.n_channels, 1, bias=False).to(x.device)
                nn.init.xavier_uniform_(self.projection.weight)
            
            # Reshape to (batch_size, n_features, height*width)
            x_reshaped = x.view(batch_size, n_features, -1)
            
            # Apply projection if dimensions don't match
            if n_features != self.n_channels:
                x_reshaped = self.projection(x_reshaped)
            
            # Apply CSP
            x_filtered = torch.matmul(self.spatial_filters, x_reshaped)
            
            # Reshape back
            x_filtered = x_filtered.view(batch_size, self.n_components, height, width)
            
        elif x.dim() == 3:
            batch_size, n_features, n_samples = x.shape
            
            # Create projection layer if needed
            if n_features != self.n_channels and self.projection is None:
                self.projection = nn.Linear(n_features, self.n_channels, bias=False).to(x.device)
                nn.init.xavier_uniform_(self.projection.weight)
            
            if n_features != self.n_channels:
                # Project features to match expected channels
                x_proj = x.transpose(1, 2)  # (batch, samples, features)
                x_proj = self.projection(x_proj)  # (batch, samples, channels)
                x_proj = x_proj.transpose(1, 2)  # (batch, channels, samples)
                x_filtered = torch.matmul(self.spatial_filters, x_proj)
            else:
                x_filtered = torch.matmul(self.spatial_filters, x)
        else:
            raise ValueError(f"Expected 3D or 4D input, got {x.dim()}D")
        
        return x_filtered
    
    def initialize_filters(self, csp_filters):
        """Initialize spatial filters with pre-computed CSP filters"""
        with torch.no_grad():
            self.spatial_filters.data = torch.FloatTensor(csp_filters)

class CSPNet1(nn.Module):
    def __init__(self, backbone_model, n_channels=8, n_components=8):
        """
        CSP-Net-1: Add CSP layer before CNN backbone
        
        Parameters:
        backbone_model: nn.Module, the CNN backbone (EEGNetv4)
        n_channels: int, number of EEG channels
        n_components: int, number of CSP components
        """
        super(CSPNet1, self).__init__()
        self.n_channels = n_channels
        self.n_components = n_components
        self.csp_layer = CSPLayer(n_channels, n_components)
        self.backbone = backbone_model
        
        # Create a 1x1 convolution to adapt CSP output to EEGNet input format
        # CSP outputs (batch, n_components, samples) but EEGNet expects (batch, 1, channels, samples)
        self.adapt_layer = nn.Conv2d(n_components, 1, kernel_size=1, bias=False)
        nn.init.xavier_uniform_(self.adapt_layer.weight)
        
        # Update backbone's spatial convolution to work with n_components instead of n_channels
        if hasattr(self.backbone, 'conv2'):
            original_conv2 = self.backbone.conv2
            # The spatial conv expects (batch, F1, n_components, time) after conv1
            # So we need to change its kernel size from (Chans, 1) to (n_components, 1)
            self.backbone.conv2 = nn.Conv2d(
                original_conv2.in_channels,  # F1
                original_conv2.out_channels,  # F1 * D
                (n_components, 1),  # Changed from (Chans, 1)
                stride=original_conv2.stride,
                padding=original_conv2.padding,
                groups=original_conv2.groups,
                bias=False
            )
            nn.init.xavier_uniform_(self.backbone.conv2.weight)
    
    def initialize_csp_filters(self, csp_filters):
        """Initialize CSP layer with pre-computed CSP filters"""
        with torch.no_grad():
            self.csp_layer.spatial_filters.data = torch.tensor(csp_filters, dtype=torch.float32)
    
    def forward(self, x):
        # Input shape: (batch, 1, channels, samples)
        batch_size = x.shape[0]
        
        # Remove the singleton channel dimension for CSP
        x = x.squeeze(1)  # (batch, channels, samples)
        
        # Apply CSP transformation
        x = self.csp_layer(x)  # (batch, n_components, samples)
        
        # Reshape for CNN: add spatial dimension
        x = x.unsqueeze(2)  # (batch, n_components, 1, samples)
        
        # Use 1x1 conv to create the expected input format for EEGNet
        x = self.adapt_layer(x)  # (batch, 1, 1, samples)
        
        # Expand to have n_components as the spatial dimension
        x = x.squeeze(1).unsqueeze(1)  # (batch, 1, 1, samples)
        x = x.expand(-1, 1, self.n_components, -1)  # (batch, 1, n_components, samples)
        
        # Pass through backbone
        return self.backbone(x)


class CSPNet2(nn.Module):
    """
    CSP-Net-2: CSP layer after first convolution + batch norm
    """
    def __init__(self, backbone_model, n_channels=8, n_components=8):
        super(CSPNet2, self).__init__()
        self.backbone = backbone_model
        self.n_channels = n_channels
        self.n_components = n_components
        
        # Extract layers from EEGNetv4 (not block1)
        self.conv1 = self.backbone.conv1
        self.batchnorm1 = self.backbone.batchnorm1
        
        # CSP layer - note that conv1 outputs F1 channels, not n_channels
        self.csp_layer = CSPLayer(self.backbone.F1, n_components)
        
        # Modify the spatial convolution (conv2) to work with CSP output
        original_conv2 = self.backbone.conv2
        self.backbone.conv2 = nn.Conv2d(
            n_components,  # Changed from F1 to n_components
            original_conv2.out_channels,
            original_conv2.kernel_size,
            stride=original_conv2.stride,
            padding=original_conv2.padding,
            groups=n_components if n_components <= original_conv2.out_channels else 1,
            bias=False
        )
        
        # Initialize the new conv2 weights
        nn.init.xavier_uniform_(self.backbone.conv2.weight)
        
        # Recalculate classifier input size
        self._recalculate_classifier()
    
    def _recalculate_classifier(self):
        """Recalculate and update classifier input dimensions"""
        # Create dummy input
        dummy_input = torch.randn(1, 1, self.n_channels, 2250)
        
        # Pass through the network to get final feature size
        with torch.no_grad():
            # First conv + batchnorm
            x = self.conv1(dummy_input)
            x = self.batchnorm1(x)
            
            # CSP layer
            x = self.csp_layer(x)
            
            # Continue through the rest of the backbone
            x = self.backbone.conv2(x)
            x = self.backbone.batchnorm2(x)
            x = self.backbone.activation1(x)
            x = self.backbone.pooling1(x)
            x = self.backbone.dropout1(x)
            
            x = self.backbone.conv3(x)
            x = self.backbone.batchnorm3(x)
            x = self.backbone.activation2(x)
            x = self.backbone.pooling2(x)
            x = self.backbone.dropout2(x)
            
            # Flatten
            x = self.backbone.flatten(x)
            
            # Update classifier input size
            feature_size = x.shape[1]
            old_out_features = self.backbone.classifier.out_features
            self.backbone.classifier = nn.Linear(feature_size, old_out_features)
    
    def forward(self, x):
        # First conv + batchnorm
        x = self.conv1(x)
        x = self.batchnorm1(x)
        
        # Apply CSP transformation
        x = self.csp_layer(x)
        
        # Continue through the rest of EEGNet
        x = self.backbone.conv2(x)
        x = self.backbone.batchnorm2(x)
        x = self.backbone.activation1(x)
        x = self.backbone.pooling1(x)
        x = self.backbone.dropout1(x)
        
        x = self.backbone.conv3(x)
        x = self.backbone.batchnorm3(x)
        x = self.backbone.activation2(x)
        x = self.backbone.pooling2(x)
        x = self.backbone.dropout2(x)
        
        # Classification
        x = self.backbone.flatten(x)
        x = self.backbone.classifier(x)
        
        return x
    def initialize_csp_filters(self, csp_filters):
        """Initialize CSP layer with pre-computed filters"""
        print(f"Warning: CSP filters shape {csp_filters.shape} may not match conv1 output channels {self.backbone.F1}")
        
        self.csp_layer.initialize_filters(csp_filters)
class CSPFilter:
    def __init__(self, n_components=8, reg=1e-6):
        """
        Common Spatial Pattern filter implementation with robust error handling
        
        Parameters:
        n_components: int, number of CSP components
        reg: float, regularization parameter
        """
        self.n_components = n_components
        self.reg = reg
        self.filters_ = None
        self.patterns_ = None
        self.eigenvalues_ = None
        
    def fit(self, X1, X2):
        """
        Fit CSP filters for binary classification
        
        Parameters:
        X1: array-like, shape (n_trials, n_channels, n_samples) - Class 1 (Left)
        X2: array-like, shape (n_trials, n_channels, n_samples) - Class 2 (Right)
        """
        print("🔧 Starting CSP computation...")
        
        # Convert to numpy arrays with proper dtype
        X1 = np.asarray(X1, dtype=np.float64)
        X2 = np.asarray(X2, dtype=np.float64)
        
        print(f"Input shapes - Class 1: {X1.shape}, Class 2: {X2.shape}")
        
        # Clean data
        X1_clean = self._clean_data(X1, "Class 1 (Left)")
        X2_clean = self._clean_data(X2, "Class 2 (Right)")
        
        # Compute covariance matrices
        print("Computing covariance matrices...")
        C1 = self._compute_covariance_matrix(X1_clean, "Class 1")
        C2 = self._compute_covariance_matrix(X2_clean, "Class 2")
        
        # Validate covariance matrices
        self._validate_covariance_matrix(C1, "C1")
        self._validate_covariance_matrix(C2, "C2")
        
        # Add regularization
        eye_matrix = np.eye(C1.shape[0])
        C1_reg = C1 + self.reg * eye_matrix
        C2_reg = C2 + self.reg * eye_matrix
        
        print(f"Added regularization: {self.reg}")
        
        # Solve generalized eigenvalue problem
        print("Solving generalized eigenvalue problem...")
        eigenvalues, eigenvectors = self._solve_gevd_robust(C1_reg, C2_reg)
        
        # Sort eigenvalues and eigenvectors
        sort_indices = np.argsort(eigenvalues)[::-1]  # Descending order
        eigenvalues = eigenvalues[sort_indices]
        eigenvectors = eigenvectors[:, sort_indices]
        
        # Select CSP filters: first and last n_components/2
        n_select = self.n_components // 2
        selected_indices = list(range(n_select)) + list(range(-n_select, 0))
        selected_filters = eigenvectors[:, selected_indices]
        
        self.filters_ = selected_filters
        self.eigenvalues_ = eigenvalues
        self.patterns_ = np.linalg.pinv(selected_filters.T)
        
        print(f"✅ CSP computation successful!")
        print(f"Selected filters shape: {self.filters_.shape}")
        print(f"Eigenvalue range: {eigenvalues.min():.6f} to {eigenvalues.max():.6f}")
        
        return self
    
    def _clean_data(self, X, class_name):
        """Clean data by handling NaN/inf values and removing bad trials"""
        print(f"Cleaning {class_name} data...")
        
        original_shape = X.shape
        
        # Replace NaN/inf with finite values
        X_clean = np.where(np.isfinite(X), X, 0.0)
        
        # Remove trials with no variation (all same values)
        valid_trials = []
        for i in range(X_clean.shape[0]):
            trial_data = X_clean[i]
            
            # Check if trial has sufficient variation
            if np.std(trial_data) > 1e-10:  # Some minimum variation
                valid_trials.append(i)
        
        if len(valid_trials) == 0:
            raise ValueError(f"No valid trials found in {class_name}")
        
        X_clean = X_clean[valid_trials]
        
        print(f"{class_name}: {len(valid_trials)}/{original_shape[0]} trials kept")
        
        return X_clean
    
    def _compute_covariance_matrix(self, X, class_name):
        """Compute normalized covariance matrix"""
        n_trials, n_channels, n_samples = X.shape
        
        print(f"Computing covariance for {class_name}: {n_trials} trials, {n_channels} channels, {n_samples} samples")
        
        # Method 1: Average of normalized trial covariances (more stable)
        cov_sum = np.zeros((n_channels, n_channels))
        valid_count = 0
        
        for trial_idx in range(n_trials):
            trial_data = X[trial_idx]  # Shape: (n_channels, n_samples)
            
            try:
                # Compute covariance for this trial
                trial_cov = np.cov(trial_data)
                
                # Check if covariance is valid
                if np.isfinite(trial_cov).all():
                    trace_val = np.trace(trial_cov)
                    if trace_val > 1e-10:  # Avoid division by very small numbers
                        # Normalize by trace
                        trial_cov_norm = trial_cov / trace_val
                        cov_sum += trial_cov_norm
                        valid_count += 1
                        
            except Exception as e:
                print(f"Warning: Failed to compute covariance for trial {trial_idx}: {e}")
                continue
        
        if valid_count == 0:
            raise ValueError(f"No valid covariance matrices computed for {class_name}")
        
        # Average the normalized covariances
        cov_avg = cov_sum / valid_count
        
        print(f"{class_name}: Used {valid_count}/{n_trials} trials for covariance")
        
        return cov_avg
    
    def _validate_covariance_matrix(self, C, name):
        """Validate covariance matrix"""
        if not np.isfinite(C).all():
            raise ValueError(f"Covariance matrix {name} contains non-finite values")
        
        if not np.allclose(C, C.T):
            print(f"Warning: Covariance matrix {name} is not symmetric, symmetrizing...")
            C = (C + C.T) / 2
        
        eigenvals = np.linalg.eigvals(C)
        if np.any(eigenvals <= 0):
            print(f"Warning: Covariance matrix {name} is not positive definite")
            print(f"Minimum eigenvalue: {eigenvals.min()}")
        
        print(f"Covariance {name} validation passed")
    
    def _solve_gevd_robust(self, C1, C2):
        """Solve generalized eigenvalue problem with multiple fallback methods"""
        
        # Method 1: Standard scipy generalized eigenvalue decomposition
        try:
            print("Trying scipy.linalg.eigh for generalized eigenvalue problem...")
            eigenvalues, eigenvectors = eigh(C1, C2)
            
            if np.isfinite(eigenvalues).all() and np.isfinite(eigenvectors).all():
                print("✅ Method 1 (scipy.eigh) successful")
                return eigenvalues, eigenvectors
            else:
                raise ValueError("Non-finite eigenvalues/eigenvectors")
                
        except Exception as e:
            print(f"❌ Method 1 failed: {e}")
        
        # Method 2: Manual generalized eigenvalue using inverse
        try:
            print("Trying manual method: inv(C2) @ C1...")
            
            # Add more regularization to C2 for stability
            C2_reg = C2 + 1e-3 * np.eye(C2.shape[0])
            
            # Compute C2^-1 @ C1
            C2_inv = inv(C2_reg)
            M = C2_inv @ C1
            
            # Standard eigenvalue decomposition
            eigenvalues, eigenvectors = np.linalg.eig(M)
            
            if np.isfinite(eigenvalues).all() and np.isfinite(eigenvectors).all():
                print("✅ Method 2 (manual inverse) successful")
                return eigenvalues.real, eigenvectors.real
            else:
                raise ValueError("Non-finite eigenvalues/eigenvectors")
                
        except Exception as e:
            print(f"❌ Method 2 failed: {e}")
        
        # Method 3: Cholesky decomposition approach
        try:
            print("Trying Cholesky decomposition method...")
            
            # Ensure C2 is positive definite
            C2_reg = C2 + 1e-2 * np.eye(C2.shape[0])
            
            # Cholesky decomposition of C2
            L = np.linalg.cholesky(C2_reg)
            L_inv = inv(L)
            
            # Transform the problem: L_inv @ C1 @ L_inv.T
            M = L_inv @ C1 @ L_inv.T
            
            # Standard eigenvalue decomposition
            eigenvalues, eigenvectors_transformed = np.linalg.eig(M)
            
            # Transform eigenvectors back
            eigenvectors = L_inv.T @ eigenvectors_transformed
            
            if np.isfinite(eigenvalues).all() and np.isfinite(eigenvectors).all():
                print("✅ Method 3 (Cholesky) successful")
                return eigenvalues.real, eigenvectors.real
            else:
                raise ValueError("Non-finite eigenvalues/eigenvectors")
                
        except Exception as e:
            print(f"❌ Method 3 failed: {e}")
        
        # Method 4: Regularized simple approach
        try:
            print("Trying heavily regularized approach...")
            
            # Heavy regularization
            reg_heavy = 1e-1
            C1_heavy = C1 + reg_heavy * np.eye(C1.shape[0])
            C2_heavy = C2 + reg_heavy * np.eye(C2.shape[0])
            
            eigenvalues, eigenvectors = eigh(C1_heavy, C2_heavy)
            
            if np.isfinite(eigenvalues).all() and np.isfinite(eigenvectors).all():
                print("✅ Method 4 (heavy regularization) successful")
                return eigenvalues, eigenvectors
            else:
                raise ValueError("Non-finite eigenvalues/eigenvectors")
                
        except Exception as e:
            print(f"❌ Method 4 failed: {e}")
        
        # If all methods fail, raise an error
        raise ValueError("All CSP computation methods failed. Data may be too noisy or insufficient.")
    
    def transform(self, X):
        """Transform data using fitted CSP filters"""
        if self.filters_ is None:
            raise ValueError("CSP filters not fitted yet. Call fit() first.")
        
        if X.ndim == 2:
            # Single trial: (channels, samples)
            return self.filters_.T @ X
        else:
            # Multiple trials: (n_trials, channels, samples)
            X_transformed = np.zeros((X.shape[0], self.filters_.shape[1], X.shape[2]))
            for i in range(X.shape[0]):
                X_transformed[i] = self.filters_.T @ X[i]
            return X_transformed

