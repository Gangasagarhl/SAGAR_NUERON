from torch import nn
import torch


class SpectralFilter(nn.Module):
    def __init__(self, retain_ratio=0.5):
        super().__init__()
        self.r = retain_ratio

    def dct(self, x, norm='ortho'):
        """
        Discrete Cosine Transform (Type-II) along dimension 1
        x: (B, L, D)
        """
        B, L, D = x.size()
        
        # Reshape to (B*D, L) for processing
        x = x.transpose(1, 2).reshape(B * D, L)
        
        # Create DCT basis matrix
        n = torch.arange(L, dtype=x.dtype, device=x.device)
        k = n.unsqueeze(0)
        n = n.unsqueeze(1)
        
        # DCT-II formula: X[k] = sum(x[n] * cos(pi*k*(n+0.5)/L))
        dct_matrix = torch.cos(torch.pi * k * (n + 0.5) / L)
        
        # Apply normalization
        if norm == 'ortho':
            dct_matrix[0] *= (1.0 / torch.sqrt(torch.tensor(L, dtype=x.dtype)))
            dct_matrix[1:] *= (torch.sqrt(torch.tensor(2.0 / L, dtype=x.dtype)))
        
        # Compute DCT
        X = torch.mm(x, dct_matrix.t())
        
        # Reshape back to (B, D, L) then transpose to (B, L, D)
        X = X.reshape(B, D, L).transpose(1, 2)
        return X

    def idct(self, X, norm='ortho'):
        """
        Inverse Discrete Cosine Transform (Type-III) along dimension 1
        X: (B, L_new, D)
        """
        B, L_new, D = X.size()
        
        # Reshape to (B*D, L_new) for processing
        X = X.transpose(1, 2).reshape(B * D, L_new)
        
        # Create IDCT basis matrix
        n = torch.arange(L_new, dtype=X.dtype, device=X.device)
        k = n.unsqueeze(0)
        n = n.unsqueeze(1)
        
        # IDCT-III formula (inverse of DCT-II)
        idct_matrix = torch.cos(torch.pi * n * (k + 0.5) / L_new)
        
        # Apply normalization
        if norm == 'ortho':
            idct_matrix[:, 0] *= (1.0 / torch.sqrt(torch.tensor(L_new, dtype=X.dtype)))
            idct_matrix[:, 1:] *= (torch.sqrt(torch.tensor(2.0 / L_new, dtype=X.dtype)))
        
        # Compute IDCT
        x = torch.mm(X, idct_matrix.t())
        
        # Reshape back to (B, D, L_new) then transpose to (B, L_new, D)
        x = x.reshape(B, D, L_new).transpose(1, 2)
        return x

    def forward(self, x):  # x: (B, L, D)
        B, L, D = x.size()

        # DCT along sequence length
        X = self.dct(x, norm="ortho")

        # Truncate frequency dimension → shorter sequence (low freq only)
        L_new = max(1, int(self.r * L))
        X = X[:, :L_new, :]

        # IDCT back to time domain
        x = self.idct(X, norm="ortho")
        return x  # shape: (B, L_new, D)