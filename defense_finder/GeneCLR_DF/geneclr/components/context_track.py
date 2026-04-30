import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union
from einops import rearrange

# Imports from sibling modules in geneclr.components
from .modeling_fastem import (
    EsmSelfOutput,
    EsmIntermediate,
    EsmOutput,
    RotaryEmbedding,  # Import the existing RoPE implementation
    FastEsmConfig # For testing and type hinting
)

# --- Double Stranded Attention Components --- #

class DSAttentionBiasModule(nn.Module):
    """
    Computes an attention bias term (B_ij) based on double stranded distances
    between protein pairs (e.g., d1_ij, d2_ij, ..., d6_ij).
    Architecture: [Distance Transform] -> Linear(input_features, 2*H) -> SwiGLU -> (+Noise during training) -> Linear(H, output_features).
    """
    def __init__(self, input_features: int = 6, hidden_features: int = 64, output_features: int = 1, noise_std: float = 1.0, distance_transformation: str = None, scale_factor: float = 1000.0):
        """
        Args:
            input_features (int): Number of input distance features (e.g., 6 for d1-d6).
            hidden_features (int): Number of features in the intermediate SwiGLU representation.
            output_features (int): Number of output features for the bias term, typically the number of attention heads.
            noise_std (float): Standard deviation of Gaussian noise to add to SwiGLU hidden features
                               during training. Defaults to 0.0 (no noise).
            distance_transformation (str): Type of distance transformation to apply. Options:
                                         - None: No transformation
                                         - "asinh": Inverse hyperbolic sine (recommended for signed values)
                                         - "log": Log transformation (for positive values only)
                                         - "tanh": Tanh transformation
            scale_factor (float): Scale factor applied to distances before transformation to keep 
                                asinh in linear range for small distances. Defaults to 1000.0.
        """
        super().__init__()
        self.input_features = input_features
        self.hidden_features = hidden_features
        self.output_features = output_features
        self.noise_std = noise_std
        self.distance_transformation = distance_transformation
        self.scale_factor = scale_factor

        self.linear_in = nn.Linear(input_features, 2 * hidden_features)
        self.linear_out = nn.Linear(hidden_features, output_features)

    def _transform_distances(self, distances: torch.Tensor, scale_factor: Optional[float] = None) -> torch.Tensor:
        """Apply distance transformation if specified."""
        if scale_factor is None:
            scaled_distances = distances * self.scale_factor
        else:
            scaled_distances = distances * scale_factor

        if self.distance_transformation is None:
            return scaled_distances
        elif self.distance_transformation.lower() == "asinh":
            # Asinh transformation - handles positive and negative values naturally
            transformed = torch.asinh(scaled_distances)
            # Normalize to standard range
            normalized = (transformed - transformed.mean()) / (transformed.std() + 1e-8)
            return normalized
        elif self.distance_transformation.lower() == "log":
            # Log transformation - requires positive values
            epsilon = 1e-8
            sign = torch.sign(scaled_distances)
            log_abs = torch.log(torch.abs(scaled_distances) + epsilon)
            log_abs = (log_abs - log_abs.mean()) / (log_abs.std() + epsilon)
            return sign * log_abs
        elif self.distance_transformation.lower() == "tanh":
            # Tanh transformation with automatic scaling
            tanh_scale_factor = 1.0 / (torch.abs(scaled_distances).mean() + 1e-8) * 0.1
            return torch.tanh(scaled_distances * tanh_scale_factor)
        else:
            raise ValueError(f"Unsupported distance_transformation: {self.distance_transformation}. "
                           f"Supported options: None, 'asinh', 'log', 'tanh'")

    def forward(self, distances: torch.Tensor, scale_factor: Optional[float] = None) -> torch.Tensor:
        """
        Args:
            distances (torch.Tensor): (batch_size, num_proteins_i, num_proteins_j, num_distance_features)
            scale_factor (Optional[float]): Optional scale factor to use instead of the initialized one.
                                          If None, uses self.scale_factor.
        Returns:
            torch.Tensor: (batch_size, num_proteins_i, num_proteins_j) if output_features=1,
                          else (batch_size, num_proteins_i, num_proteins_j, output_features)
        """
        # Apply distance transformation if specified
        distances = self._transform_distances(distances, scale_factor)
        
        projected_x = self.linear_in(distances)
        if self.training and self.noise_std > 0.0:
            noise = torch.randn_like(projected_x) * self.noise_std
            projected_x = projected_x + noise
        
        value, gate = torch.chunk(projected_x, 2, dim=-1)
        swiglu_hidden_repr = value * F.silu(gate)
        bias_matrix_elements = self.linear_out(swiglu_hidden_repr)

        if self.output_features == 1:
            return bias_matrix_elements.squeeze(-1)
        return bias_matrix_elements

class DSSelfAttention(nn.Module):
    """
    Self-attention mechanism that incorporates an additive double stranded attention bias.
    """
    def __init__(self, config):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        self.scale = self.attention_head_size**-0.5

        self.dsattn_bias_module = DSAttentionBiasModule(
            input_features=config.dsattn_bias_input_dim, 
            hidden_features=config.dsattn_bias_hidden_size, 
            output_features=config.num_attention_heads,
            noise_std=config.dsattn_bias_noise_std,
            distance_transformation=getattr(config, 'distance_transformation', None),
            scale_factor=getattr(config, 'distance_scale_factor', 1000.0)
        )
        self.dropout_prob = config.attention_probs_dropout_prob

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, sequence_length, all_head_size)
        # all_head_size = num_attention_heads * attention_head_size
        # output shape: (batch_size, num_attention_heads, sequence_length, attention_head_size)
        return rearrange(x, 'b s (h d) -> b h s d', h=self.num_attention_heads)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        distance_tensor: Optional[torch.Tensor] = None,
        distance_scale_factor: Optional[float] = None,  # Add learnable scale factor parameter
        output_attentions: Optional[bool] = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        query_layer = self.transpose_for_scores(self.query(hidden_states))
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))

        combined_mask = attention_mask

        if distance_tensor is not None:
            dsattn_bias = self.dsattn_bias_module(distance_tensor, distance_scale_factor)  # Pass learnable scale factor
            dsattn_bias_permuted = dsattn_bias.permute(0, 3, 1, 2)
            
            if combined_mask is None:
                combined_mask = dsattn_bias_permuted
            else:
                combined_mask = combined_mask + dsattn_bias_permuted
        
        context_layer_tuple = () 
        if output_attentions:
            attention_scores = torch.matmul(query_layer * self.scale, key_layer.transpose(-1, -2))
            if combined_mask is not None:
                attention_scores = attention_scores + combined_mask
            attention_probs = F.softmax(attention_scores, dim=-1)
            if self.dropout_prob > 0.0:
                attention_probs = F.dropout(attention_probs, p=self.dropout_prob, training=self.training)
            context_layer = torch.matmul(attention_probs, value_layer)
            context_layer_tuple = (context_layer, attention_probs)
        else:
            context_layer = F.scaled_dot_product_attention(
                query_layer * self.scale,
                key_layer,
                value_layer,
                attn_mask=combined_mask,
                dropout_p=self.dropout_prob if self.training else 0.0,
                scale=1.0 
            )
            context_layer_tuple = (context_layer,)
        
        # context_layer_tuple[0] has shape (batch_size, num_attention_heads, sequence_length, attention_head_size)
        # We want to transform it to (batch_size, sequence_length, all_head_size)
        context_layer_output = rearrange(context_layer_tuple[0], 'b h s d -> b s (h d)')

        if output_attentions:
            return context_layer_output, context_layer_tuple[1]
        return context_layer_output

class RoPESelfAttention(nn.Module):
    """
    Self-attention mechanism using Rotary Position Embeddings (RoPE).
    This is a conventional attention mechanism that ignores distance tensors.
    """
    def __init__(self, config):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        self.scale = self.attention_head_size**-0.5

        # RoPE embeddings
        self.rotary_embeddings = RotaryEmbedding(dim=self.attention_head_size)
        
        self.dropout_prob = config.attention_probs_dropout_prob

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, sequence_length, all_head_size)
        # all_head_size = num_attention_heads * attention_head_size
        # output shape: (batch_size, num_attention_heads, sequence_length, attention_head_size)
        return rearrange(x, 'b s (h d) -> b h s d', h=self.num_attention_heads)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        distance_tensor: Optional[torch.Tensor] = None,  # Ignored for RoPE attention
        distance_scale_factor: Optional[float] = None,  # Ignored for RoPE attention
        output_attentions: Optional[bool] = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        query_layer = self.transpose_for_scores(self.query(hidden_states)) * self.scale
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))

        # Apply RoPE to query and key
        query_layer, key_layer = self.rotary_embeddings(query_layer, key_layer)

        # Note: distance_tensor and distance_scale_factor are ignored for RoPE attention
        
        if output_attentions:
            # Manual attention computation to get attention weights
            attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
            if attention_mask is not None:
                attention_scores = attention_scores + attention_mask
            attention_probs = F.softmax(attention_scores, dim=-1)
            if self.dropout_prob > 0.0:
                attention_probs = F.dropout(attention_probs, p=self.dropout_prob, training=self.training)
            context_layer = torch.matmul(attention_probs, value_layer)
            context_layer_output = rearrange(context_layer, 'b h s d -> b s (h d)')
            return context_layer_output, attention_probs
        else:
            context_layer = F.scaled_dot_product_attention(
                query_layer,
                key_layer,
                value_layer,
                attn_mask=attention_mask,
                dropout_p=self.dropout_prob if self.training else 0.0,
                scale=1.0
            )
            context_layer_output = rearrange(context_layer, 'b h s d -> b s (h d)')
            return context_layer_output

# --- Context Track Transformer Components --- #

class ContextAttention(nn.Module):
    """
    Attention block for the Context Track, supporting both DS and RoPE attention.
    """
    def __init__(self, config):
        super().__init__()
        # Determine which attention mechanism to use
        attention_type = getattr(config, 'attention_type', 'ds')  # Default to DS attention
        
        if attention_type.lower() == 'rope':
            self.self_attention = RoPESelfAttention(config)
            self.uses_distance_tensor = False
        elif attention_type.lower() == 'ds':
            self.self_attention = DSSelfAttention(config)
            self.uses_distance_tensor = True
        else:
            raise ValueError(f"Unsupported attention_type: {attention_type}. Supported types: 'ds', 'rope'")
        
        self.attention_type = attention_type.lower()
        self.output_projection = EsmSelfOutput(config) 
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=getattr(config, "layer_norm_eps", 1e-12))

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        distance_tensor: Optional[torch.Tensor] = None,
        distance_scale_factor: Optional[float] = None,  # Add learnable scale factor parameter
        output_attentions: Optional[bool] = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        
        hidden_states_ln = self.LayerNorm(hidden_states)
        
        # For RoPE attention, we don't need to pass distance_tensor or scale_factor
        if self.uses_distance_tensor:
            self_outputs = self.self_attention(
                hidden_states_ln,
                attention_mask,
                distance_tensor,
                distance_scale_factor,  # Pass learnable scale factor
                output_attentions,
            )
        else:
            # RoPE attention ignores distance_tensor and scale_factor
            self_outputs = self.self_attention(
                hidden_states_ln,
                attention_mask,
                None,  # Explicitly pass None for distance_tensor
                None,  # Explicitly pass None for distance_scale_factor
                output_attentions,
            )
        
        attention_output_val = self_outputs[0] if output_attentions else self_outputs
        attention_output = self.output_projection(attention_output_val, hidden_states) 

        if output_attentions:
            return attention_output, self_outputs[1]
        return attention_output

class ContextTrackLayer(nn.Module):
    """
    A single layer of the Context Track Transformer.
    """
    def __init__(self, config):
        super().__init__()
        self.attention = ContextAttention(config)
        self.intermediate = EsmIntermediate(config)
        self.output_projection = EsmOutput(config) 
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=getattr(config, "layer_norm_eps", 1e-12)) 

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        distance_tensor: Optional[torch.Tensor] = None,
        distance_scale_factor: Optional[float] = None,  # Add learnable scale factor parameter
        output_attentions: Optional[bool] = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        
        attention_outputs_tuple = self.attention(
            hidden_states,
            attention_mask,
            distance_tensor,
            distance_scale_factor,  # Pass learnable scale factor
            output_attentions,
        )
        
        attention_output_val = attention_outputs_tuple[0] if output_attentions else attention_outputs_tuple
        
        ffn_input = self.LayerNorm(attention_output_val)
        intermediate_output = self.intermediate(ffn_input) 
        layer_output = self.output_projection(intermediate_output, attention_output_val) 

        if output_attentions:
            return layer_output, attention_outputs_tuple[1]
        return layer_output

class ContextTrackEncoder(nn.Module):
    """
    Encoder for the Context Track, consisting of multiple ContextTrackLayers.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([ContextTrackLayer(config) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = getattr(config, "gradient_checkpointing", False)
        
        # Check if we're using distance tensors
        attention_type = getattr(config, 'attention_type', 'ds')
        self.uses_distance_tensor = attention_type.lower() == 'ds'

    def forward(
        self,
        hidden_states: torch.Tensor, 
        attention_mask: Optional[torch.Tensor] = None, 
        distance_tensor: Optional[torch.Tensor] = None, 
        distance_scale_factor: Optional[float] = None,  # Add learnable scale factor parameter
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        # Warn if distance_tensor is provided but not used
        if distance_tensor is not None and not self.uses_distance_tensor:
            print(f"Warning: distance_tensor provided but attention_type is '{getattr(self.config, 'attention_type', 'ds')}'. Distance tensor will be ignored.")

        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        current_hidden_states = hidden_states

        for i, layer_module in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (current_hidden_states,)

            if self.gradient_checkpointing and self.training:
                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # Ensure all necessary arguments for ContextTrackLayer.forward are passed
                        # Inputs will be (current_hidden_states, attention_mask, distance_tensor, distance_scale_factor)
                        return module(inputs[0], attention_mask=inputs[1], distance_tensor=inputs[2], distance_scale_factor=inputs[3], output_attentions=output_attentions) 
                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer_module),
                    current_hidden_states, # Pass current_hidden_states
                    attention_mask,
                    distance_tensor,
                    distance_scale_factor,  # Pass learnable scale factor
                    use_reentrant=False 
                )
            else:
                layer_outputs = layer_module(
                    current_hidden_states, # Pass current_hidden_states
                    attention_mask,
                    distance_tensor,
                    distance_scale_factor,  # Pass learnable scale factor
                    output_attentions=output_attentions,
                )
            
            current_hidden_states = layer_outputs[0] if isinstance(layer_outputs, tuple) else layer_outputs # Update current_hidden_states for the next iteration
            if output_attentions:
                if not isinstance(layer_outputs, tuple) or len(layer_outputs) < 2:
                    raise ValueError(f"Layer {i} was expected to return a tuple with attentions but did not.")
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (current_hidden_states,) # Add the final hidden state

        from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions 
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=current_hidden_states, # Use the final current_hidden_states
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )

# Example usage and tests
if __name__ == '__main__':
    batch_size_ds = 2
    num_proteins_ds = 5 
    num_dist_features_ds = 6 # Updated for 6 distances
    hidden_features_dsattn_bias = 8
    
    print("--- Testing DSAttentionBiasModule (SwiGLU with Noise and Distance Transform) ---")
    # Create dummy distances with large values to test transformation
    dummy_distances_ds = torch.randn(batch_size_ds, num_proteins_ds, num_proteins_ds, num_dist_features_ds) * 10000  # Large values

    dsattn_bias_module_no_noise = DSAttentionBiasModule(input_features=num_dist_features_ds, hidden_features=hidden_features_dsattn_bias, noise_std=0.0)
    dsattn_bias_module_no_noise.eval()
    output_bias_no_noise = dsattn_bias_module_no_noise(dummy_distances_ds)
    print(f"Output shape (no transform, no noise): {output_bias_no_noise.shape}")
    assert output_bias_no_noise.shape == (batch_size_ds, num_proteins_ds, num_proteins_ds)

    dsattn_bias_module_with_noise = DSAttentionBiasModule(input_features=num_dist_features_ds, hidden_features=hidden_features_dsattn_bias, noise_std=0.1)
    dsattn_bias_module_with_noise.train()
    output_bias_with_noise = dsattn_bias_module_with_noise(dummy_distances_ds)
    print(f"Output shape (no transform, with noise): {output_bias_with_noise.shape}")
    assert output_bias_with_noise.shape == (batch_size_ds, num_proteins_ds, num_proteins_ds)
    if num_dist_features_ds > 0:
        assert not torch.allclose(output_bias_no_noise, output_bias_with_noise), "Noise did not change output."

    # Test with asinh transformation
    dsattn_bias_module_asinh = DSAttentionBiasModule(
        input_features=num_dist_features_ds, 
        hidden_features=hidden_features_dsattn_bias, 
        noise_std=0.0,
        distance_transformation="asinh",
        scale_factor=1000.0
    )
    dsattn_bias_module_asinh.eval()
    output_bias_asinh = dsattn_bias_module_asinh(dummy_distances_ds)
    print(f"Output shape (asinh transform): {output_bias_asinh.shape}")
    assert output_bias_asinh.shape == (batch_size_ds, num_proteins_ds, num_proteins_ds)
    assert not torch.allclose(output_bias_no_noise, output_bias_asinh), "Asinh transformation did not change output."
    
    print("DSAttentionBiasModule tests passed.")

    print("\n--- Testing DSSelfAttention and RoPESelfAttention ---")
    test_config_realistic = FastEsmConfig(
        hidden_size=12, 
        num_attention_heads=4,
        attention_probs_dropout_prob=0.0,
        dsattn_bias_input_dim=num_dist_features_ds, # Should be 6
        dsattn_bias_hidden_size=16, 
        dsattn_bias_noise_std=0.0,
        output_attentions=False, # Default for FastEsmConfig
        output_hidden_states=False, # Default for FastEsmConfig
        attention_type='ds'  # Test DS attention
    )

    batch_size_ctx = 2
    seq_len_ctx = 5
    hidden_dim_ctx = test_config_realistic.hidden_size

    dummy_hidden_states_ctx = torch.randn(batch_size_ctx, seq_len_ctx, hidden_dim_ctx)
    dummy_distances_ctx = torch.randn(batch_size_ctx, seq_len_ctx, seq_len_ctx, num_dist_features_ds)
    
    dummy_attention_mask_ctx = torch.ones(batch_size_ctx, seq_len_ctx, dtype=torch.bool)
    additive_attention_mask = dummy_attention_mask_ctx.unsqueeze(1).unsqueeze(2)
    additive_attention_mask = additive_attention_mask.to(dtype=dummy_hidden_states_ctx.dtype)
    additive_attention_mask = (1.0 - additive_attention_mask) * torch.finfo(dummy_hidden_states_ctx.dtype).min

    print("Testing DSSelfAttention directly...")
    ds_self_attn = DSSelfAttention(test_config_realistic)
    ds_self_attn.eval()
    attn_output_direct_tuple = ds_self_attn(dummy_hidden_states_ctx, additive_attention_mask, dummy_distances_ctx, output_attentions=True)
    assert isinstance(attn_output_direct_tuple, tuple) and len(attn_output_direct_tuple) == 2
    attn_output_direct, attn_probs_direct = attn_output_direct_tuple
    print(f"DSSelfAttention output shape: {attn_output_direct.shape}")
    assert attn_output_direct.shape == (batch_size_ctx, seq_len_ctx, hidden_dim_ctx)
    print("DSSelfAttention direct test passed.")

    print("Testing RoPESelfAttention directly...")
    rope_self_attn = RoPESelfAttention(test_config_realistic)
    rope_self_attn.eval()
    rope_output_direct_tuple = rope_self_attn(dummy_hidden_states_ctx, additive_attention_mask, dummy_distances_ctx, output_attentions=True)
    assert isinstance(rope_output_direct_tuple, tuple) and len(rope_output_direct_tuple) == 2
    rope_output_direct, rope_attn_probs_direct = rope_output_direct_tuple
    print(f"RoPESelfAttention output shape: {rope_output_direct.shape}")
    assert rope_output_direct.shape == (batch_size_ctx, seq_len_ctx, hidden_dim_ctx)
    print("RoPESelfAttention direct test passed.")

    print("Testing ContextAttention with DS attention...")
    test_config_realistic.attention_type = 'ds'
    context_attention_ds = ContextAttention(test_config_realistic)
    context_attention_ds.eval()
    context_attn_tuple = context_attention_ds(dummy_hidden_states_ctx, additive_attention_mask, dummy_distances_ctx, output_attentions=True)
    assert isinstance(context_attn_tuple, tuple) and len(context_attn_tuple) == 2
    context_attn_output, context_attn_probs = context_attn_tuple
    print(f"ContextAttention (DS) output shape: {context_attn_output.shape}")
    assert context_attn_output.shape == (batch_size_ctx, seq_len_ctx, hidden_dim_ctx)
    print("ContextAttention (DS) test passed.")

    print("Testing ContextAttention with RoPE attention...")
    test_config_realistic.attention_type = 'rope'
    context_attention_rope = ContextAttention(test_config_realistic)
    context_attention_rope.eval()
    context_attn_rope_tuple = context_attention_rope(dummy_hidden_states_ctx, additive_attention_mask, dummy_distances_ctx, output_attentions=True)
    assert isinstance(context_attn_rope_tuple, tuple) and len(context_attn_rope_tuple) == 2
    context_attn_rope_output, context_attn_rope_probs = context_attn_rope_tuple
    print(f"ContextAttention (RoPE) output shape: {context_attn_rope_output.shape}")
    assert context_attn_rope_output.shape == (batch_size_ctx, seq_len_ctx, hidden_dim_ctx)
    print("ContextAttention (RoPE) test passed.")

    print("Testing ContextTrackEncoder with both attention types...")
    test_config_realistic.num_hidden_layers = 2 
    
    # Test DS attention
    test_config_realistic.attention_type = 'ds'
    context_encoder_ds = ContextTrackEncoder(test_config_realistic)
    context_encoder_ds.eval()
    encoder_output_ds = context_encoder_ds(dummy_hidden_states_ctx, additive_attention_mask, dummy_distances_ctx)
    print(f"ContextTrackEncoder (DS) last_hidden_state shape: {encoder_output_ds.last_hidden_state.shape}")
    assert encoder_output_ds.last_hidden_state.shape == (batch_size_ctx, seq_len_ctx, hidden_dim_ctx)
    
    # Test RoPE attention
    test_config_realistic.attention_type = 'rope'
    context_encoder_rope = ContextTrackEncoder(test_config_realistic)
    context_encoder_rope.eval()
    encoder_output_rope = context_encoder_rope(dummy_hidden_states_ctx, additive_attention_mask, dummy_distances_ctx)
    print(f"ContextTrackEncoder (RoPE) last_hidden_state shape: {encoder_output_rope.last_hidden_state.shape}")
    assert encoder_output_rope.last_hidden_state.shape == (batch_size_ctx, seq_len_ctx, hidden_dim_ctx)
    
    print("ContextTrackEncoder tests passed for both attention types.")

    print("\nAll context_track.py component tests passed!") 