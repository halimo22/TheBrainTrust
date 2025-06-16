import numpy as np
import pandas as pd
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
# 
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

# ==================== CSP Implementation ====================

class CSPFilter:
    def __init__(self, n_components=8, reg=None):
        """
        Common Spatial Pattern filter implementation
        
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
        X1: array-like, shape (n_trials, n_channels, n_samples) - Class 1
        X2: array-like, shape (n_trials, n_channels, n_samples) - Class 2
        """
        # Convert to numpy arrays
        X1 = np.array(X1)
        X2 = np.array(X2)
        
        # Calculate covariance matrices
        C1 = self._compute_covariance_matrix(X1)
        C2 = self._compute_covariance_matrix(X2)
        
        # Add regularization if specified
        if self.reg is not None:
            C1 += self.reg * np.eye(C1.shape[0])
            C2 += self.reg * np.eye(C2.shape[0])
        
        # Solve generalized eigenvalue problem: C1 * w = lambda * C2 * w
        eigenvalues, eigenvectors = self._solve_gevd(C1, C2)
        
        # Sort by eigenvalues (descending order)
        sort_indices = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[sort_indices]
        eigenvectors = eigenvectors[:, sort_indices]
        
        # Select components: first n_components/2 and last n_components/2
        if self.n_components is None:
            self.n_components = eigenvectors.shape[1]
        
        n_select = self.n_components // 2
        selected_filters = np.column_stack([
            eigenvectors[:, :n_select],
            eigenvectors[:, -n_select:]
        ])
        
        self.filters_ = selected_filters
        self.eigenvalues_ = eigenvalues
        self.patterns_ = np.linalg.pinv(selected_filters.T)
        
        return self
    
    def _compute_covariance_matrix(self, X):
        """Compute normalized covariance matrix"""
        n_trials, n_channels, n_samples = X.shape
        cov_matrix = np.zeros((n_channels, n_channels))
        
        for trial in range(n_trials):
            trial_data = X[trial]
            trial_cov = np.cov(trial_data)
            cov_matrix += trial_cov / np.trace(trial_cov)
        
        return cov_matrix / n_trials
    
    def _solve_gevd(self, C1, C2):
        """Solve generalized eigenvalue decomposition"""
        # Regularize C2 to ensure it's invertible
        C2_reg = C2 + 1e-10 * np.eye(C2.shape[0])
        
        # Compute C2^-1 * C1
        try:
            C2_inv = np.linalg.inv(C2_reg)
            eigenvalues, eigenvectors = np.linalg.eig(C2_inv @ C1)
        except np.linalg.LinAlgError:
            # Fallback to generalized eigenvalue problem
            eigenvalues, eigenvectors = np.linalg.eigh(C1, C2_reg)
        
        return eigenvalues.real, eigenvectors.real
    
    def transform(self, X):
        """Transform data using fitted CSP filters"""
        if self.filters_ is None:
            raise ValueError("CSP filters not fitted yet. Call fit() first.")
        
        # X shape: (n_trials, n_channels, n_samples) or (n_channels, n_samples)
        if X.ndim == 2:
            # Single trial
            return self.filters_.T @ X
        else:
            # Multiple trials
            X_transformed = np.zeros((X.shape[0], self.filters_.shape[1], X.shape[2]))
            for i in range(X.shape[0]):
                X_transformed[i] = self.filters_.T @ X[i]
            return X_transformed
    
    def fit_transform(self, X1, X2, X):
        """Fit CSP filters and transform data"""
        self.fit(X1, X2)
        return self.transform(X)

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
        
        # Initialize CSP filters as learnable parameters
        self.spatial_filters = nn.Parameter(torch.randn(n_components, n_channels))
        
    def forward(self, x):
        """
        Apply CSP spatial filtering
        
        Parameters:
        x: tensor, shape (batch_size, 1, n_channels, n_samples)
        
        Returns:
        tensor, shape (batch_size, n_components, 1, n_samples)
        """
        batch_size, _, n_channels, n_samples = x.shape
        
        # Reshape for matrix multiplication
        x_reshaped = x.view(batch_size, n_channels, n_samples)
        
        # Apply spatial filtering: (batch_size, n_components, n_samples)
        x_filtered = torch.matmul(self.spatial_filters, x_reshaped)
        
        # Reshape back to 4D: (batch_size, n_components, 1, n_samples)
        x_filtered = x_filtered.unsqueeze(2)
        
        return x_filtered

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
        self.csp_layer = CSPLayer(n_channels, n_components)
        self.backbone = backbone_model
        
        # Update backbone to expect n_components channels instead of n_channels
        if hasattr(self.backbone, 'conv2'):
            # Modify the spatial convolution to match CSP output
            original_conv2 = self.backbone.conv2
            self.backbone.conv2 = nn.Conv2d(
                original_conv2.in_channels, 
                original_conv2.out_channels,
                (1, original_conv2.kernel_size[1]),  # Change from (Chans, 1) to (1, 1)
                groups=original_conv2.groups,
                bias=original_conv2.bias
            )
    
    def initialize_csp_filters(self, csp_filters):
        """Initialize CSP layer with pre-computed CSP filters"""
        with torch.no_grad():
            self.csp_layer.spatial_filters.data = torch.tensor(csp_filters, dtype=torch.float32)
    
    def forward(self, x):
        x = self.csp_layer(x)
        return self.backbone(x)

class CSPNet2(nn.Module):
    def __init__(self, backbone_model, n_channels=8, n_components=8):
        """
        CSP-Net-2: Replace spatial conv layer in CNN backbone with CSP layer
        
        Parameters:
        backbone_model: nn.Module, the CNN backbone (EEGNetv4)
        n_channels: int, number of EEG channels  
        n_components: int, number of CSP components
        """
        super(CSPNet2, self).__init__()
        self.backbone = backbone_model
        
        # Replace the spatial convolution (conv2) with CSP layer
        self.csp_layer = CSPLayer(n_channels, n_components)
        
        # Remove the original spatial conv layer
        self.backbone.conv2 = nn.Identity()
        
        # Store other components
        self.conv1 = self.backbone.conv1
        self.batchnorm1 = self.backbone.batchnorm1
        self.batchnorm2 = self.backbone.batchnorm2
        self.activation1 = self.backbone.activation1
        self.pooling1 = self.backbone.pooling1
        self.dropout1 = self.backbone.dropout1
        self.conv3 = self.backbone.conv3
        self.batchnorm3 = self.backbone.batchnorm3
        self.activation2 = self.backbone.activation2
        self.pooling2 = self.backbone.pooling2
        self.dropout2 = self.backbone.dropout2
        self.flatten = self.backbone.flatten
        self.classifier = self.backbone.classifier
        
        # Recalculate classifier input size
        self._recalculate_classifier()
    
    def _recalculate_classifier(self):
        """Recalculate classifier input size after CSP layer replacement"""
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, 8, 2250)  # Adjust based on your data
            x = self.conv1(dummy_input)
            x = self.batchnorm1(x)
            x = self.csp_layer(x)
            x = self.batchnorm2(x)
            x = self.activation1(x)
            x = self.pooling1(x)
            x = self.conv3(x)
            x = self.batchnorm3(x)
            x = self.activation2(x)
            x = self.pooling2(x)
            x = self.flatten(x)
            
            # Update classifier
            self.classifier = nn.Linear(x.shape[1], self.backbone.nb_classes)
    
    def initialize_csp_filters(self, csp_filters):
        """Initialize CSP layer with pre-computed CSP filters"""
        with torch.no_grad():
            self.csp_layer.spatial_filters.data = torch.tensor(csp_filters, dtype=torch.float32)
    
    def forward(self, x):
        # Block 1: Temporal convolution
        x = self.conv1(x)
        x = self.batchnorm1(x)
        
        # Block 2: CSP spatial filtering (replacing conv2)
        x = self.csp_layer(x)
        x = self.batchnorm2(x)
        x = self.activation1(x)
        x = self.pooling1(x)
        x = self.dropout1(x)
        
        # Block 3: Separable convolution
        x = self.conv3(x)
        x = self.batchnorm3(x)
        x = self.activation2(x)
        x = self.pooling2(x)
        x = self.dropout2(x)
        
        # Classification
        x = self.flatten(x)
        x = self.classifier(x)
        
        return x