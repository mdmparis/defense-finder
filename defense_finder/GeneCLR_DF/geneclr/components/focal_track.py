import torch
import torch.nn as nn
import torch.nn.functional as F

class SwiGLULayer(nn.Module):
    """
    A single SwiGLU (Swish Gated Linear Unit) layer.
    Includes optional layer normalization, residual connections, and dropout.
    """
    def __init__(self, 
                 input_dim: int, 
                 output_dim: int, 
                 glu_dim: int = None, 
                 use_residual: bool = True,
                 use_layer_norm: bool = True,
                 dropout: float = 0.0):
        """
        Args:
            input_dim (int): Dimension of the input embeddings.
            output_dim (int): Dimension of the output embeddings.
            glu_dim (int, optional): Intermediate dimension for the SwiGLU's linear projection.
                                     The projection will be to 2 * glu_dim.
                                     If None, defaults to output_dim.
            use_residual (bool): If True, adds a residual connection from input to output.
                                 Defaults to True.
            use_layer_norm (bool): If True, applies layer normalization before SwiGLU.
                                   Defaults to True.
            dropout (float): Dropout probability applied after SwiGLU. Defaults to 0.0.
        """
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.glu_intermediate_dim = glu_dim if glu_dim is not None else output_dim
        self.use_residual = use_residual
        self.use_layer_norm = use_layer_norm
        
        # Layer normalization (applied before SwiGLU)
        if use_layer_norm:
            self.layer_norm = nn.LayerNorm(input_dim)
        
        # Linear layer for SwiGLU, projecting to twice the glu_intermediate_dim
        # One part for the main signal (value), one for the gate
        self.glu_linear = nn.Linear(input_dim, 2 * self.glu_intermediate_dim)

        # If the SwiGLU's intermediate dimension is different from the final output_dim,
        # an additional linear layer is needed to project to output_dim.
        if self.glu_intermediate_dim != output_dim:
            self.output_projection = nn.Linear(self.glu_intermediate_dim, output_dim)
        else:
            self.output_projection = nn.Identity()

        # Residual connection projection
        if use_residual:
            if input_dim != output_dim:
                self.residual_projection = nn.Linear(input_dim, output_dim)
            else:
                self.residual_projection = nn.Identity()
        
        # Dropout
        if dropout > 0:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the SwiGLU layer.
        Args:
            x (torch.Tensor): Input embeddings.
                              Shape: (batch_size, num_proteins, input_dim)
        Returns:
            torch.Tensor: Processed embeddings.
                          Shape: (batch_size, num_proteins, output_dim)
        """
        identity = x
        
        # Apply layer normalization if enabled
        if self.use_layer_norm:
            x = self.layer_norm(x)

        # Project to 2 * glu_intermediate_dim
        projected_x = self.glu_linear(x)  # Shape: (batch_size, num_proteins, 2 * glu_intermediate_dim)

        # Split into two halves for SwiGLU
        value, gate = torch.chunk(projected_x, 2, dim=-1)
        # value shape: (batch_size, num_proteins, glu_intermediate_dim)
        # gate shape: (batch_size, num_proteins, glu_intermediate_dim)

        # Apply SwiGLU: value * SiLU(gate)
        # SiLU(x) = x * sigmoid(x), implemented as F.silu(x) in PyTorch
        swiglu_output = value * F.silu(gate)  # Shape: (batch_size, num_proteins, glu_intermediate_dim)

        # Project to final output_dim if necessary
        output = self.output_projection(swiglu_output)  # Shape: (batch_size, num_proteins, output_dim)
        
        # Apply dropout
        output = self.dropout(output)

        # Add residual connection if enabled
        if self.use_residual:
            projected_identity = self.residual_projection(identity)
            output = output + projected_identity
        
        return output


class FocalTrack(nn.Module):
    """
    Focal Track of the GeneCLR model.
    Processes protein embeddings using one or more SwiGLU (Swish Gated Linear Unit) layers.
    It does not use inter-protein attention.
    Includes optional residual connections and layer normalization.
    """
    def __init__(self, 
                 input_dim: int, 
                 output_dim: int, 
                 num_layers: int = 1,
                 intermediate_dim: int = 480,
                 glu_dim: int = None, 
                 use_residual: bool = True,
                 use_layer_norm: bool = True,
                 dropout: float = 0.0):
        """
        Args:
            input_dim (int): Dimension of the input protein embeddings.
            output_dim (int): Dimension of the output embeddings.
            num_layers (int): Number of SwiGLU layers to stack. Defaults to 1.
            intermediate_dim (int): Hidden dimension between layers (for num_layers > 1). Defaults to 480.
            glu_dim (int, optional): Intermediate dimension for each SwiGLU's linear projection.
                                     If None, defaults to intermediate_dim for multi-layer or output_dim for single layer.
            use_residual (bool): If True, adds residual connections around each layer.
                                 Defaults to True.
            use_layer_norm (bool): If True, applies layer normalization before each SwiGLU.
                                   Defaults to True for multi-layer, maintains backward compatibility for single layer.
            dropout (float): Dropout probability applied after each SwiGLU layer.
                            Defaults to 0.0.
        """
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.intermediate_dim = intermediate_dim
        
        # For backward compatibility: single layer doesn't use layer norm by default
        if num_layers == 1 and use_layer_norm is True:
            use_layer_norm = False
        
        # Create layers
        self.layers = nn.ModuleList()
        
        for i in range(num_layers):
            # Determine input and output dimensions for this layer
            if i == 0:
                layer_input_dim = input_dim
            else:
                layer_input_dim = intermediate_dim
                
            if i == num_layers - 1:
                layer_output_dim = output_dim
            else:
                layer_output_dim = intermediate_dim
            
            # Create the SwiGLU layer
            layer = SwiGLULayer(
                input_dim=layer_input_dim,
                output_dim=layer_output_dim,
                glu_dim=glu_dim,
                use_residual=use_residual,
                use_layer_norm=use_layer_norm,
                dropout=dropout
            )
            
            self.layers.append(layer)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the FocalTrack.
        Args:
            x (torch.Tensor): Input protein embeddings.
                              Shape: (batch_size, num_proteins, input_dim)
        Returns:
            torch.Tensor: Processed embeddings.
                          Shape: (batch_size, num_proteins, output_dim)
        """
        for layer in self.layers:
            x = layer(x)
        
        return x

if __name__ == '__main__':
    # Example Usage
    batch_size = 4
    num_proteins = 10
    embedding_dim = 480  # ESM2 embedding dimension
    
    print("--- Test Case 1: Single layer (backward compatibility) ---")
    focal_track_1 = FocalTrack(input_dim=embedding_dim, output_dim=embedding_dim, use_residual=True)
    dummy_embeddings_1 = torch.randn(batch_size, num_proteins, embedding_dim)
    output_1 = focal_track_1(dummy_embeddings_1)
    print(f"Input shape: {dummy_embeddings_1.shape}")
    print(f"Output shape: {output_1.shape}")
    assert output_1.shape == (batch_size, num_proteins, embedding_dim)
    print("Passed.\n")

    print("--- Test Case 2: Multi-layer FocalTrack (3 layers) ---")
    focal_track_2 = FocalTrack(
        input_dim=embedding_dim, 
        output_dim=embedding_dim, 
        num_layers=3,
        intermediate_dim=480,
        use_residual=True,
        use_layer_norm=True
    )
    dummy_embeddings_2 = torch.randn(batch_size, num_proteins, embedding_dim)
    output_2 = focal_track_2(dummy_embeddings_2)
    print(f"Input shape: {dummy_embeddings_2.shape}")
    print(f"Output shape: {output_2.shape}")
    assert output_2.shape == (batch_size, num_proteins, embedding_dim)
    print("Passed.\n")

    print("--- Test Case 3: Multi-layer with different output dim ---")
    output_dim_3 = 256
    focal_track_3 = FocalTrack(
        input_dim=embedding_dim, 
        output_dim=output_dim_3, 
        num_layers=5,
        intermediate_dim=480,
        use_residual=True
    )
    dummy_embeddings_3 = torch.randn(batch_size, num_proteins, embedding_dim)
    output_3 = focal_track_3(dummy_embeddings_3)
    print(f"Input shape: {dummy_embeddings_3.shape}")
    print(f"Output shape: {output_3.shape}")
    assert output_3.shape == (batch_size, num_proteins, output_dim_3)
    print("Passed.\n")

    print("--- Test Case 4: Custom glu_dim with dropout ---")
    focal_track_4 = FocalTrack(
        input_dim=embedding_dim, 
        output_dim=embedding_dim, 
        num_layers=2,
        intermediate_dim=480,
        glu_dim=1024,  # Custom GLU intermediate dimension
        use_residual=True,
        dropout=0.1
    )
    dummy_embeddings_4 = torch.randn(batch_size, num_proteins, embedding_dim)
    output_4 = focal_track_4(dummy_embeddings_4)
    print(f"Input shape: {dummy_embeddings_4.shape}")
    print(f"Output shape: {output_4.shape}")
    assert output_4.shape == (batch_size, num_proteins, embedding_dim)
    print("Passed.\n")

    print("--- Test Case 5: No residual connections ---")
    focal_track_5 = FocalTrack(
        input_dim=embedding_dim, 
        output_dim=embedding_dim, 
        num_layers=3,
        intermediate_dim=480,
        use_residual=False,
        use_layer_norm=True
    )
    dummy_embeddings_5 = torch.randn(batch_size, num_proteins, embedding_dim)
    output_5 = focal_track_5(dummy_embeddings_5)
    print(f"Input shape: {dummy_embeddings_5.shape}")
    print(f"Output shape: {output_5.shape}")
    assert output_5.shape == (batch_size, num_proteins, embedding_dim)
    print("Passed.\n")

    print("FocalTrack with SwiGLULayer components works correctly with dummy data.")
    
    # Parameter count comparison
    print(f"\n--- Parameter Count Comparison ---")
    
    # Single layer version
    single_layer_params = sum(p.numel() for p in focal_track_1.parameters())
    
    # Multi-layer version
    multi_layer_params = sum(p.numel() for p in focal_track_2.parameters())
    
    print(f"Single-layer FocalTrack parameters: {single_layer_params:,}")
    print(f"3-layer FocalTrack parameters: {multi_layer_params:,}")
    print(f"Parameter ratio: {multi_layer_params / single_layer_params:.2f}x") 