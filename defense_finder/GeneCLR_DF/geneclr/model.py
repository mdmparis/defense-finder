import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import Callback
import numpy as np

# Removed numpy and sklearn.metrics as they are now in macro_avg_metric.py

from .components.focal_track import FocalTrack
from .components.context_track import ContextTrackEncoder
from .components.modeling_fastem import FastEsmConfig # For configuring ContextTrackEncoder
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions
#from metrics.macro_avg_metric import macro_average_precision_pos_vs_single_neg, macro_auroc_pos_vs_single_neg # Changed to absolute import


class LoraLinear(nn.Module):
    """
    Manual LoRA implementation for Linear layers.
    
    This adds low-rank adaptation matrices to existing linear layers:
    output = base_layer(x) + lora_B(lora_A(x)) * scaling
    
    Args:
        base_layer: The original nn.Linear layer to adapt
        r: Rank of the adaptation (lower = more efficient, higher = more expressive)
        alpha: Scaling factor (typically set to r or higher)
        dropout: Dropout probability for LoRA layers
    """
    def __init__(self, base_layer: nn.Linear, r: int = 8, alpha: int = 16, dropout: float = 0.1):
        super().__init__()
        self.base_layer = base_layer
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        
        # LoRA matrices
        self.lora_A = nn.Linear(base_layer.in_features, r, bias=False)
        self.lora_B = nn.Linear(r, base_layer.out_features, bias=False)
        self.dropout = nn.Dropout(dropout)
        
        # Initialize LoRA weights
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)
        
        # Freeze base layer
        for param in self.base_layer.parameters():
            param.requires_grad = False
            
    def forward(self, x):
        base_output = self.base_layer(x)
        lora_output = self.lora_B(self.dropout(self.lora_A(x))) * self.scaling
        return base_output + lora_output
        

            
    def get_merged_linear(self):
        """
        Create a new standard nn.Linear layer with merged LoRA weights.
        This is used for converting LoRA layers to standard layers permanently.
        """
        # Create merged weights
        if self.r > 0:
            delta_weight = (self.lora_B.weight @ self.lora_A.weight) * self.scaling
            merged_weight = self.base_layer.weight.data + delta_weight
        else:
            merged_weight = self.base_layer.weight.data.clone()
        
        # Create new standard linear layer
        merged_linear = nn.Linear(
            self.base_layer.in_features,
            self.base_layer.out_features,
            bias=self.base_layer.bias is not None
        )
        
        # Copy merged weights
        merged_linear.weight.data = merged_weight
        if self.base_layer.bias is not None:
            merged_linear.bias.data = self.base_layer.bias.data.clone()
            
        return merged_linear


def apply_lora_to_model(model, target_modules, r=8, alpha=16, dropout=0.1):
    """
    Apply LoRA to specific modules in a model.
    
    Args:
        model: The model to modify
        target_modules: List of module name patterns to target
        r: LoRA rank
        alpha: LoRA alpha
        dropout: LoRA dropout
    """
    lora_modules = {}
    
    def apply_lora_recursive(module, prefix=""):
        for name, child in module.named_children():
            full_name = f"{prefix}.{name}" if prefix else name
            
            # Check if this module should get LoRA
            if isinstance(child, nn.Linear) and any(pattern in full_name for pattern in target_modules):
                lora_layer = LoraLinear(child, r=r, alpha=alpha, dropout=dropout)
                setattr(module, name, lora_layer)
                lora_modules[full_name] = lora_layer
                print(f"Applied LoRA to: {full_name} (in_features={child.in_features}, out_features={child.out_features})")
            else:
                # Recursively apply to children
                apply_lora_recursive(child, full_name)
    
    apply_lora_recursive(model)
    return lora_modules


class LoRAMergeCallback(Callback):
    """
    Callback to merge LoRA weights before saving checkpoints and unmerge them after saving.
    This ensures saved checkpoints contain merged weights for easier deployment while
    maintaining separate LoRA adapters during training.
    """
    
    def __init__(self, merge_before_save: bool = True):
        """
        Args:
            merge_before_save: Whether to merge LoRA weights before saving checkpoints
        """
        self.merge_before_save = merge_before_save
        
    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        """Called when a checkpoint is about to be saved."""
        if self.merge_before_save and hasattr(pl_module, 'lora_modules'):
            # Create a clean state_dict using the LoraLinear.get_merged_linear() approach
            self._create_merged_state_dict(checkpoint, pl_module)
            
    def _create_merged_state_dict(self, checkpoint, pl_module):
        """
        Create a clean state_dict where LoRA layers are replaced with standard linear layers.
        This doesn't modify the actual model, just the saved state_dict.
        """
        original_state_dict = checkpoint.get('state_dict', {})
        clean_state_dict = {}
        
        for key, value in original_state_dict.items():
            # Skip LoRA-specific parameters
            if 'lora_A' in key or 'lora_B' in key:
                continue
            # Convert base_layer paths to standard paths and get merged weights
            elif '.base_layer.weight' in key:
                # Extract the module path without .base_layer.weight
                module_path = key.replace('.base_layer.weight', '')
                
                # Get the LoRA module if it exists
                if module_path in pl_module.lora_modules:
                    lora_module = pl_module.lora_modules[module_path]
                    merged_linear = lora_module.get_merged_linear()
                    clean_state_dict[module_path + '.weight'] = merged_linear.weight
                    if merged_linear.bias is not None:
                        clean_state_dict[module_path + '.bias'] = merged_linear.bias
                else:
                    # Not a LoRA module, keep as-is but fix path
                    clean_state_dict[module_path + '.weight'] = value
            elif '.base_layer.bias' in key:
                # Handle bias - if weight was already processed, skip (already handled above)
                module_path = key.replace('.base_layer.bias', '')
                if module_path not in pl_module.lora_modules:
                    clean_state_dict[module_path + '.bias'] = value
            else:
                # Keep all other parameters as-is
                clean_state_dict[key] = value
        
        checkpoint['state_dict'] = clean_state_dict
            
    def on_load_checkpoint(self, trainer, pl_module, checkpoint):
        """Called when loading a checkpoint."""
        # Check if the checkpoint was saved with merged weights
        # We can detect this by checking if the state_dict contains LoRA parameters
        state_dict = checkpoint.get('state_dict', {})
        has_lora_params = any('lora_A' in key or 'lora_B' in key for key in state_dict.keys())
        
        if self.merge_before_save and not has_lora_params:
            pass  # Loading merged checkpoint
        else:
            pass  # Loading LoRA checkpoint





class OptimizersMixin:
    def configure_optimizers(self):
        opt_config = self.model_config.get('optimizer', {})
        lr = float(opt_config.get('lr', 1e-4))  # Ensure lr is float
        weight_decay = float(opt_config.get('weight_decay', 0.01))  # Ensure weight_decay is float
        
        optimizer = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=weight_decay)
        
        scheduler_config = self.model_config.get('scheduler', None)
        if scheduler_config:
            if scheduler_config.get('name') == 'cosine_annealing':
                t_max = scheduler_config.get('T_max')
                
                # If t_max not provided, estimate from training config
                if t_max is None:
                    if self.trainer and hasattr(self.trainer, 'estimated_stepping_batches'):
                        t_max = self.trainer.estimated_stepping_batches
                    else:
                        # Estimate t_max from training configuration
                        max_epochs = self.model_config.get('max_epochs', 200)
                        # This is approximate - actual value depends on dataset size and batch size
                        # You may want to set T_max explicitly in config for precision
                        estimated_steps_per_epoch = scheduler_config.get('estimated_steps_per_epoch', 100)
                        t_max = max_epochs * estimated_steps_per_epoch
                        print(f"Warning: T_max estimated as {t_max} steps. Consider setting T_max explicitly in scheduler config.")
                
                if t_max is None:
                    raise ValueError("Could not determine T_max. Please set T_max in scheduler config or ensure trainer is available.")

                # Calculate warmup steps (half an epoch)
                warmup_epochs = float(scheduler_config.get('warmup_epochs', 0.5))
                max_epochs = int(self.model_config.get('max_epochs', 200))
                steps_per_epoch = t_max / max_epochs
                warmup_steps = int(steps_per_epoch * warmup_epochs)
                
                # Warmup start LR (typically 10% of base LR)
                base_lr = float(self.model_config.get('optimizer', {}).get('lr', 1e-4))
                warmup_start_lr = float(scheduler_config.get('warmup_start_lr', base_lr * 0.1))
                eta_min = float(scheduler_config.get('eta_min', base_lr * 0.01))
                
                # Combined warmup + cosine annealing scheduler
                def lr_lambda(step):
                    if step < warmup_steps:
                        # Linear warmup from warmup_start_lr to base_lr
                        return warmup_start_lr / base_lr + (1.0 - warmup_start_lr / base_lr) * step / warmup_steps
                    else:
                        # Cosine annealing from base_lr to eta_min
                        progress = (step - warmup_steps) / (t_max - warmup_steps)
                        import math
                        return eta_min / base_lr + (1.0 - eta_min / base_lr) * 0.5 * (1 + math.cos(math.pi * progress))
                
                scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
                return [optimizer], [{'scheduler': scheduler, 'interval': 'step'}]
            elif scheduler_config.get('name') == 'reduce_on_plateau':
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer,
                    mode='min',
                    patience=int(scheduler_config.get('patience', 10)),
                    factor=float(scheduler_config.get('factor', 0.1)),
                    min_lr=float(scheduler_config.get('min_lr', 1e-6)),
                    verbose=True
                )
                return [optimizer], [{
                    'scheduler': scheduler,
                    'monitor': 'val_loss',
                    'interval': 'epoch',
                    'frequency': 1
                }]
        return optimizer


class GeneCLR(OptimizersMixin, pl.LightningModule):
    def __init__(self, model_config: dict):
        super().__init__()
        self.save_hyperparameters(model_config) # Saves model_config to self.hparams
        self.model_config = model_config # Also keep a direct reference if needed

        _hidden_dim = self.model_config.get('hidden_dim', 480)

        # --- Focal Track ---xr
        focal_config = self.model_config.get('focal_track', {})
        self.focal_track = FocalTrack(
            input_dim=focal_config.get('input_dim', _hidden_dim),
            output_dim=focal_config.get('output_dim', _hidden_dim),
            glu_dim=focal_config.get('glu_dim', _hidden_dim * 4),
            use_residual=focal_config.get('use_residual', True)
        )

        # --- Context Track ---
        context_track_config = self.model_config.get('context_track', {})
        
        # --- Distance Scale Factor Configuration ---
        # Must be set up before FastEsmConfig creation since get_distance_scale_factor() is called there
        self.distance_scale_factor = self.model_config.get('distance_scale_factor', 1000.0)
        self.use_learnable_distance_scale = self.model_config.get('use_learnable_distance_scale', False)
        if self.use_learnable_distance_scale:
            # Initialize with log of the default value (log(1000) ≈ 6.91)
            self.log_distance_scale_factor = nn.Parameter(torch.log(torch.tensor(self.distance_scale_factor)))
            print(f"Using learnable distance scale factor. Initial value: {torch.exp(self.log_distance_scale_factor).item():.2f}")
        else:
            self.log_distance_scale_factor = None
        
        # Create FastEsmConfig for the ContextTrackEncoder
        ct_esm_config = FastEsmConfig(
            hidden_size=_hidden_dim, # Global hidden_dim
            num_hidden_layers=context_track_config.get('num_layers', 6),
            num_attention_heads=context_track_config.get('num_heads', 8),
            intermediate_size=context_track_config.get('intermediate_size', _hidden_dim * 4),
            hidden_dropout_prob=context_track_config.get('hidden_dropout', 0.1),
            attention_probs_dropout_prob=context_track_config.get('attention_dropout', 0.1),
            layer_norm_eps=context_track_config.get('layer_norm_eps', 1e-12),
            # Add Double Stranded Attention specific parameters to this FastEsmConfig object
            # The ContextTrackEncoder and its sub-modules will expect these.
            dsattn_bias_input_dim=context_track_config.get('dsattn_bias_input_dim', 6), 
            dsattn_bias_hidden_size=context_track_config.get('dsattn_bias_hidden_size', 16),
            dsattn_bias_noise_std=context_track_config.get('dsattn_bias_noise_std', 1.0),
            distance_transformation=context_track_config.get('distance_transformation', None),
            distance_scale_factor=self.distance_scale_factor,  # Always use the base value for initialization
            attention_type=context_track_config.get('attention_type', 'ds'),
            emb_layer_norm_before=context_track_config.get('emb_layer_norm_before', False),
            gradient_checkpointing=context_track_config.get('gradient_checkpointing', False)
        )

        self.context_track_encoder = ContextTrackEncoder(
            config=ct_esm_config # Pass the populated FastEsmConfig (gradient_checkpointing is now inside)
        )

        # --- Projection Heads (Optional) ---
        self.projection_dim = self.model_config.get('projection_dim', None)
        focal_out_dim = focal_config.get('output_dim', _hidden_dim)
        context_out_dim = ct_esm_config.hidden_size # Derived from FastEsmConfig

        if self.projection_dim:
            self.focal_projection = nn.Linear(focal_out_dim, self.projection_dim)
            self.context_projection = nn.Linear(context_out_dim, self.projection_dim)
        else:
            if focal_out_dim != context_out_dim:
                print(f"Warning: No projection_dim set, but FocalTrack output_dim ({focal_out_dim}) "
                      f"and ContextTrack output_dim ({context_out_dim}) differ. "
                      "Ensure they are compatible for InfoNCE loss.")

        # --- InfoNCE Loss Configuration ---
        self.use_temperature_free_loss = self.model_config.get('use_temperature_free_loss', False)  # New flag
        self.use_learnable_temperature = self.model_config.get('use_learnable_temperature', False)  # New flag for learnable temperature

        # New loss function parameters
        self.use_soft_target_loss = self.model_config.get('use_soft_target_loss', False)

        # Initialize self.temperature based on projection_dim or default to 1/sqrt(projection_dim) or 1/sqrt(64) = 1/8 = 0.125
        # This update ensures a principled default for tau
        projection_dim = self.model_config.get('projection_dim', None)
        tau_init = 1/torch.sqrt(torch.tensor(projection_dim, dtype=torch.float32)) if projection_dim is not None else torch.tensor(1/8.0)
        temperature_value = self.model_config.get('temperature')
        if temperature_value is None:
            temperature_value = tau_init.item()
        self.temperature = nn.Parameter(torch.tensor(temperature_value))

        if self.use_learnable_temperature:
            self.log_temperature = nn.Parameter(torch.log(torch.tensor(temperature_value)))
            print(f"Using learnable temperature. Initial temperature: {temperature_value}")
        else:
            self.log_temperature = None

        # Loss function setup
        if self.use_temperature_free_loss:
            print("Using temperature-free loss (e.g., ArcTanh-based).")
            self.loss_fn = self._temperature_free_loss
        elif self.use_soft_target_loss:
            print(f"Using Symmetric Soft-Target InfoNCE loss with confusion temperature: {self.confusion_temperature}")
            self.loss_fn = self._st_infonce_loss
        else:
            print(f"Using standard InfoNCE loss with temperature: {self.temperature.item()}")
            self.loss_fn = self._infonce_loss

        # Metric computation - now handled by macro_avg_metric.py
        # No change needed here for loss function logic
        self.metric_return_metrics = self.model_config.get('return_metrics', True)

        # --- Memory Efficiency Configuration ---
        self.use_bf16_similarity = self.model_config.get('use_bf16_similarity', True)
        
        # --- Learnable Mask Token ---
        self.mask_token = nn.Parameter(torch.randn(1, 1, _hidden_dim))
        
        # --- Logging Configuration ---
        self.log_every_n_steps = self.model_config.get('log_every_n_steps', 10)

    def forward(self, embeddings, pairwise_distances, attention_mask=None, random_context_mask: Optional[torch.Tensor] = None, return_context_pre_projection: bool = False):
        """
        Args:
            embeddings (torch.Tensor): Precomputed ESM2 embeddings (B, S, D_esm=480).
            pairwise_distances (torch.Tensor): Pairwise distances for double stranded attention (B, S, S, N_dist=6).
            attention_mask (torch.Tensor, optional): Boolean PADDING mask (B, S) where True indicates valid tokens, 
                                                     False indicates padding. Defaults to None.
            random_context_mask (torch.Tensor, optional): Boolean MASKING mask from dataloader (B,S) 
                                                                  where True indicates a token to be masked for context track input.
            return_context_pre_projection (bool): If True, return 480-dim context hidden states instead of projected.
        Returns:
            Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]: 
                Projected representations from focal track,
                Context representations (projected or 480-dim if return_context_pre_projection),
                Applied INPUT mask for context track (B,S) or None if no masking was applied.
        """
        # Debug GPU usage
        if hasattr(self, '_debug_gpu_printed') and not self._debug_gpu_printed:
            print(f"Model forward - embeddings device: {embeddings.device}")
            print(f"Model forward - current CUDA device: {torch.cuda.current_device()}")
            print(f"Model forward - local rank: {getattr(self, 'local_rank', 'N/A')}")
            self._debug_gpu_printed = True
        
        # Focal Track always sees original embeddings, shape (B, S, D)
        focal_output = self.focal_track(embeddings)

        # Context Track - potentially with masked input embeddings
        context_input_embeddings = embeddings.clone()
        applied_input_mask = None 

        if random_context_mask is not None:
            applied_input_mask = random_context_mask
            if embeddings.shape[1] > 0: 
                all_masked_in_sequence = applied_input_mask.all(dim=1)
                if all_masked_in_sequence.any():
                    for i in range(embeddings.shape[0]): 
                        if all_masked_in_sequence[i]:
                            applied_input_mask[i, 0] = False 
            
            # Replace masked positions with learnable mask token
            # Invert the mask: True in random_context_mask means visible, but torch.where expects True for mask token
            mask_indices = ~applied_input_mask  # Invert: True for masked positions
            mask_token_expanded = self.mask_token.expand(embeddings.shape[0], embeddings.shape[1], -1)
            context_input_embeddings = torch.where(
                mask_indices.unsqueeze(-1).expand_as(embeddings),
                mask_token_expanded,
                context_input_embeddings
            )
        
        context_transformer_attention_mask = None
        if attention_mask is not None:
            context_transformer_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2) 
            context_transformer_attention_mask = context_transformer_attention_mask.to(dtype=embeddings.dtype) 
            context_transformer_attention_mask = (1.0 - context_transformer_attention_mask) * torch.finfo(embeddings.dtype).min
        
        context_output_pack = self.context_track_encoder(
            hidden_states=context_input_embeddings, 
            distance_tensor=pairwise_distances,
            distance_scale_factor=self.get_distance_scale_factor() if self.use_learnable_distance_scale else None,  # Pass learnable scale factor
            attention_mask=context_transformer_attention_mask 
        )
        context_output_hidden_states = context_output_pack.last_hidden_state

        if self.projection_dim and not return_context_pre_projection:
            focal_projected = self.focal_projection(focal_output)
            context_projected = self.context_projection(context_output_hidden_states)
        else:
            focal_projected = self.focal_projection(focal_output) if self.projection_dim else focal_output
            context_projected = context_output_hidden_states if return_context_pre_projection else (
                self.context_projection(context_output_hidden_states) if self.projection_dim else context_output_hidden_states
            )

        return focal_projected, context_projected, applied_input_mask

    def _compute_similarity_matrix_masked_only(self, z_i, z_j, attention_mask=None, input_mask=None):
        """
        Memory-efficient method to compute similarity matrix only for masked tokens.
        This avoids computing the full similarity matrix for all tokens.
        Uses bfloat16 for memory efficiency during similarity computation.
        
        Args:
            z_i: First set of representations (focal track), shape (B, S, D)
            z_j: Second set of representations (context track), shape (B, S, D)
            attention_mask: Boolean mask (B, S) to exclude padded tokens from loss. True for valid tokens.
            input_mask: Boolean mask (B, S) indicating which tokens were masked in input. True for visible tokens, False for masked tokens.
        Returns:
            Tuple containing:
            - similarity_matrix: The computed similarity matrix for masked tokens only
            - labels: The labels for the loss computation
            - num_masked_tokens: Number of masked tokens
            - device: The device of the tensors
        """
        batch_size, seq_len, dim = z_i.shape
        device = z_i.device
        original_dtype = z_i.dtype
        temperature = self.get_temperature()
        # Reshape to (B*S, D)
        z_i_flat = z_i.reshape(batch_size * seq_len, dim)
        z_j_flat = z_j.reshape(batch_size * seq_len, dim)
        # Normalize the representations
        z_i_norm = F.normalize(z_i_flat, p=2, dim=1)
        z_j_norm = F.normalize(z_j_flat, p=2, dim=1)
        # Apply attention mask first if provided
        if attention_mask is not None:
            mask_flat = attention_mask.reshape(-1)
            z_i_norm = z_i_norm[mask_flat]
            z_j_norm = z_j_norm[mask_flat]
            # Also apply to input_mask
            if input_mask is not None:
                input_mask = input_mask.reshape(-1)[mask_flat]
        # Get masked indices - input_mask is True for visible, False for masked
        if input_mask is not None:
            # input_mask is True for visible tokens, so masked tokens are where it's False
            # But we want to compute loss over masked tokens, so we need the positions where input_mask is False
            masked_indices = ~input_mask  # True for masked tokens (where input_mask is False)
        else:
            return None, None, 0, device
        
        num_masked_tokens = masked_indices.sum().item()
        if num_masked_tokens == 0:
            return None, None, 0, device
        
        # Filter to only masked positions
        z_i_masked = z_i_norm[masked_indices]
        z_j_masked = z_j_norm[masked_indices]
        
        # Compute similarity matrix with optional bfloat16 optimization
        if hasattr(self, 'use_bf16_similarity') and self.use_bf16_similarity:
            z_i_masked_bf16 = z_i_masked.to(torch.bfloat16)
            z_j_masked_bf16 = z_j_masked.to(torch.bfloat16)
            similarity_matrix_bf16 = torch.matmul(z_i_masked_bf16, z_j_masked_bf16.t()) / temperature
            similarity_matrix = similarity_matrix_bf16.to(original_dtype)
        else:
            # Standard float32 computation
            similarity_matrix = torch.matmul(z_i_masked, z_j_masked.t()) / temperature
        
        # Create labels (diagonal matrix)
        labels = torch.arange(num_masked_tokens, device=device)
        return similarity_matrix, labels, num_masked_tokens, device, z_i_masked, z_j_masked

    def _compute_similarity_matrix_masked_only_temperature_free(self, z_i, z_j, attention_mask=None, input_mask=None):
        """
        Memory-efficient method to compute temperature-free similarity matrix only for masked tokens.
        Uses inverse hyperbolic tangent function instead of temperature scaling.
        
        Args:
            z_i: First set of representations (focal track), shape (B, S, D)
            z_j: Second set of representations (context track), shape (B, S, D)
            attention_mask: Boolean mask (B, S) to exclude padded tokens from loss. True for valid tokens.
            input_mask: Boolean mask (B, S) indicating which tokens were masked in input. True for visible tokens, False for masked tokens.
        Returns:
            Tuple containing:
            - similarity_matrix: The computed similarity matrix for masked tokens only with arctanh transformation
            - labels: The labels for the loss computation
            - num_masked_tokens: Number of masked tokens
            - device: The device of the tensors
        """
        batch_size, seq_len, dim = z_i.shape
        device = z_i.device
        original_dtype = z_i.dtype
        # Reshape to (B*S, D)
        z_i_flat = z_i.reshape(batch_size * seq_len, dim)
        z_j_flat = z_j.reshape(batch_size * seq_len, dim)
        # Apply attention mask first if provided
        if attention_mask is not None:
            mask_flat = attention_mask.reshape(-1)
            z_i_flat = z_i_flat[mask_flat]
            z_j_flat = z_j_flat[mask_flat]
            # Also apply to input_mask
            if input_mask is not None:
                input_mask = input_mask.reshape(-1)[mask_flat]

        # Normalize the representations
        z_i_norm = F.normalize(z_i_flat, p=2, dim=1)
        z_j_norm = F.normalize(z_j_flat, p=2, dim=1)
        
        # Get masked indices - input_mask is True for visible, False for masked
        if input_mask is not None:
            # input_mask is True for visible tokens, so masked tokens are where it's False
            # But we want to compute loss over masked tokens, so we need the positions where input_mask is False
            masked_indices = ~input_mask  # True for masked tokens (where input_mask is False)
        else:
            return None, None, 0, device, None, None
        
        num_masked_tokens = masked_indices.sum().item()
        if num_masked_tokens == 0:
            return None, None, 0, device, None, None
        
        # Filter to only masked positions
        z_i_masked = z_i_norm[masked_indices]
        z_j_masked = z_j_norm[masked_indices]

        # Compute temperature-free similarity matrix with optional bfloat16 optimization
        if hasattr(self, 'use_bf16_similarity') and self.use_bf16_similarity:
            z_i_masked_bf16 = z_i_masked.to(torch.bfloat16)
            z_j_masked_bf16 = z_j_masked.to(torch.bfloat16)
            cosine_similarity_bf16 = torch.matmul(z_i_masked_bf16, z_j_masked_bf16.t())
            
            # Clamp to avoid numerical issues near ±1
            similarity_matrix_bf16 = torch.atanh(torch.clamp(cosine_similarity_bf16, -0.99, 0.99))
            similarity_matrix = similarity_matrix_bf16.to(original_dtype)
        else:
            # Standard float32 computation
            cosine_similarity = torch.matmul(z_i_masked, z_j_masked.t())
            similarity_matrix = torch.atanh(torch.clamp(cosine_similarity, -0.99, 0.99))
        
        # Create labels (diagonal matrix)
        labels = torch.arange(num_masked_tokens, device=device)
        return similarity_matrix, labels, num_masked_tokens, device, z_i_masked, z_j_masked

    def get_temperature(self):
        """Get the current temperature value, either fixed or learnable."""
        if self.use_learnable_temperature:
            return torch.exp(self.log_temperature)
        else:
            return self.temperature
        
    def get_distance_scale_factor(self):
        """Get the current distance scale factor, either fixed or learnable."""
        if self.use_learnable_distance_scale:
            return torch.exp(self.log_distance_scale_factor)
        else:
            return self.distance_scale_factor
        
    def get_confusion_temperature(self):
        """
        Returns the confusion temperature. For now, this is linked to the regular temperature.
        Later, we might introduce a scaling factor.
        """
        return self.get_temperature()

    def _infonce_loss(self, z_i, z_j, attention_mask=None, input_mask=None, return_metrics=False):
        """
        Compute the InfoNCE loss only over masked positions.
        Uses memory-efficient similarity matrix computation.
        
        Args:
            z_i: First set of representations (focal track), shape (B, S, D)
            z_j: Second set of representations (context track), shape (B, S, D)
            attention_mask: Boolean mask (B, S) to exclude padded tokens from loss. True for valid tokens.
            input_mask: Boolean mask (B, S) indicating which tokens were masked in input. True for visible tokens, False for masked tokens.
            return_metrics: Whether to return additional metrics
        """
        current_temperature = None

        if self.use_temperature_free_loss:
            similarity_matrix, labels, num_masked_tokens, device, z_i_masked, z_j_masked = self._compute_similarity_matrix_masked_only_temperature_free(
                z_i, z_j, attention_mask, input_mask
            )
        else:
            similarity_matrix, labels, num_masked_tokens, device, z_i_masked, z_j_masked = self._compute_similarity_matrix_masked_only(
                z_i, z_j, attention_mask, input_mask
            )

        if num_masked_tokens == 0:
            if return_metrics:
                return torch.tensor(0.0, device=device), {}
            return torch.tensor(0.0, device=device)
        
        # Compute predictions for both directions
        pred_focal_to_context = torch.argmax(similarity_matrix, dim=1)
        pred_context_to_focal = torch.argmax(similarity_matrix.t(), dim=1)

        metrics = {}
        metrics['accuracy_focal_to_context_masked'] = (pred_focal_to_context == labels).float().mean()
        metrics['accuracy_context_to_focal_masked'] = (pred_context_to_focal == labels).float().mean()
        
        if self.use_learnable_temperature:
            current_temperature = self.get_temperature()
            metrics['temperature'] = current_temperature
        
        if self.use_learnable_distance_scale:
            current_distance_scale = self.get_distance_scale_factor()
            metrics['distance_scale_factor'] = current_distance_scale
        
        # Compute InfoNCE loss in both directions
        focal_to_context_loss = F.cross_entropy(similarity_matrix, labels)
        context_to_focal_loss = F.cross_entropy(similarity_matrix.t(), labels)
        
        metrics['focal_to_context_loss_masked'] = focal_to_context_loss
        metrics['context_to_focal_loss_masked'] = context_to_focal_loss
        total_loss = (focal_to_context_loss + context_to_focal_loss) / 2.0
        
        if return_metrics:
            return total_loss, metrics
        return total_loss

    def _st_infonce_loss(self, z_i, z_j, attention_mask=None, input_mask=None, return_metrics: bool = False):
        """
        Implements Symmetric Soft-Target InfoNCE Loss.
        
        Args:
            z_i: Focal track representations (B, S, D)
            z_j: Context track representations (B, S, D)
            attention_mask: Padding mask (B, S), True for valid tokens.
            input_mask: Mask indicating original masked tokens (B, S), True for visible, False for masked.
            return_metrics: Whether to return additional metrics (e.g., accuracy, AUC).
        Returns:
            Tuple: (loss, metrics_dict)
        """
        # 1. Extract necessary components using the refactored helper
        if self.use_temperature_free_loss:
            sim_matrix_raw, labels, num_masked_tokens, device, z_i_masked, z_j_masked = \
                self._compute_similarity_matrix_masked_only_temperature_free(z_i, z_j, attention_mask, input_mask)
        else:
            sim_matrix_raw, labels, num_masked_tokens, device, z_i_masked, z_j_masked = \
                self._compute_similarity_matrix_masked_only(z_i, z_j, attention_mask, input_mask)

        if sim_matrix_raw is None or num_masked_tokens == 0:
            return torch.tensor(0.0, device=device, requires_grad=True), {}

        # Ensure original_dtype is consistent for bfloat16 conversions
        original_dtype = z_i.dtype
        temperature = self.get_temperature()
        confusion_temperature = self.get_confusion_temperature()

        # 2. Confusion Matrix M_SS (for H_S): Z_S . Z_S^T
        # Use bfloat16 for dot product if enabled
        if hasattr(self, 'use_bf16_similarity') and self.use_bf16_similarity:
            z_i_masked_bf16 = z_i_masked.to(torch.bfloat16)
            M_SS_bf16 = torch.matmul(z_i_masked_bf16, z_i_masked_bf16.t())
            M_SS = M_SS_bf16.to(original_dtype)
        else:
            M_SS = torch.matmul(z_i_masked, z_i_masked.t())
        
        # Calculate H_S = Softmax(M_SS / tau_c)
        H_S = F.softmax(M_SS / confusion_temperature, dim=1)

        # 3. Confusion Matrix M_CC (for H_C): Z_C . Z_C^T
        # Use bfloat16 for dot product if enabled
        if hasattr(self, 'use_bf16_similarity') and self.use_bf16_similarity:
            z_j_masked_bf16 = z_j_masked.to(torch.bfloat16)
            M_CC_bf16 = torch.matmul(z_j_masked_bf16, z_j_masked_bf16.t())
            M_CC = M_CC_bf16.to(original_dtype)
        else:
            M_CC = torch.matmul(z_j_masked, z_j_masked.t())

        # Calculate H_C = Softmax(M_CC / tau_c)
        H_C = F.softmax(M_CC / confusion_temperature, dim=1)

        # 4. Calculate Prediction Probabilities
        if self.use_temperature_free_loss:
            P_CS_log_softmax = F.log_softmax(sim_matrix_raw / temperature, dim=1)
            P_SC_log_softmax = F.log_softmax(sim_matrix_raw.t() / temperature, dim=1)
        else:
            P_CS_log_softmax = F.log_softmax(sim_matrix_raw, dim=1) # sim_matrix_raw already has temperature applied
            P_SC_log_softmax = F.log_softmax(sim_matrix_raw.t(), dim=1) # sim_matrix_raw already has temperature applied

        # 5. Calculate Loss Terms
        # L_C_to_S = - (H_S * P_CS).sum(dim=1).mean()
        # The prompt specifies the sum over j and then sum over i / N.
        # This is equivalent to (H_S * P_CS_log_softmax).sum() / N
        L_C_to_S = - (H_S * P_CS_log_softmax).sum(dim=1).mean()
        L_S_to_C = - (H_C * P_SC_log_softmax).sum(dim=1).mean()

        # 6. Final Symmetric Loss
        total_loss = (L_C_to_S + L_S_to_C) / 2.0

        metrics = {}
        if return_metrics:
            # For evaluation, we can still compute standard InfoNCE-like metrics
            # to see how the model performs on hard examples, even if it's not optimized directly.
            # This would require adapting the InfoNCE metric to this new loss, or just using the similarity_matrix_raw.
            # For now, we will return an empty metrics dictionary, or a simple placeholder.
            # A proper implementation would involve calculating macro_average_precision_pos_vs_single_neg
            # based on sim_matrix_raw and labels, similar to the _infonce_loss method.
            pass # Placeholder for metrics in soft-target loss

        return total_loss, metrics

    def training_step(self, batch, batch_idx):
        embeddings = batch['embeddings']
        pairwise_distances = batch['pairwise_distances']
        padding_attention_mask = batch.get('attention_mask', None) 
        random_context_mask = batch.get('random_context_mask', None)

        focal_representations, context_representations, applied_input_mask = self(
            embeddings,
            pairwise_distances,
            attention_mask=padding_attention_mask,
            random_context_mask=random_context_mask
        )
        
        if batch_idx % self.log_every_n_steps == 0:
            loss, metrics = self.loss_fn(
                focal_representations, 
                context_representations, 
                attention_mask=padding_attention_mask,
                input_mask=applied_input_mask,
                return_metrics=True
            )
            self.log('train_loss', loss, on_step=True, on_epoch=True, logger=True, sync_dist=True)
            for key, value in metrics.items():
                self.log(f'train_{key}', value, on_step=True, on_epoch=True, logger=True, sync_dist=True)
        else:
            loss = self.loss_fn(
                focal_representations, 
                context_representations, 
                attention_mask=padding_attention_mask,
                input_mask=applied_input_mask,
                return_metrics=False
            )
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        embeddings = batch['embeddings']
        pairwise_distances = batch['pairwise_distances']
        padding_attention_mask = batch.get('attention_mask', None) 
        random_context_mask = batch.get('random_context_mask', None)

        focal_representations, context_representations, applied_input_mask = self(
            embeddings,
            pairwise_distances,
            attention_mask=padding_attention_mask,
            random_context_mask=random_context_mask
        )
        
        loss, metrics = self.loss_fn(
            focal_representations,  
            context_representations, 
            attention_mask=padding_attention_mask,
            input_mask=applied_input_mask,
            return_metrics=True
        )
        
        # Log validation metrics (use val_ prefix) - log on epoch for early stopping
        self.log('val_loss', loss, on_step=False, on_epoch=True, logger=True, sync_dist=True)
        for key, value in metrics.items():
            self.log(f'val_{key}', value, on_step=False, on_epoch=True, logger=True, sync_dist=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)


class GeneClrForTokenClassification(OptimizersMixin, pl.LightningModule):
    def __init__(self, model_config: dict, pretrained_geneclr_path: Optional[str] = None):
        super().__init__()
        self.save_hyperparameters(model_config)
        self.model_config = model_config

        _hidden_dim = self.model_config.get('hidden_dim', 480) # This should match the GeneCLR's hidden_dim for context track input

        # --- Context Track (Trunk) ---
        context_track_config = self.model_config.get('context_track', {})
        
        # --- Distance Scale Factor Configuration ---
        self.distance_scale_factor = self.model_config.get('distance_scale_factor', 1000.0)
        self.use_learnable_distance_scale = self.model_config.get('use_learnable_distance_scale', False)
        if self.use_learnable_distance_scale:
            # Initialize with log of the default value (log(1000) ≈ 6.91)
            self.log_distance_scale_factor = nn.Parameter(torch.log(torch.tensor(self.distance_scale_factor)))
            print(f"Using learnable distance scale factor. Initial value: {torch.exp(self.log_distance_scale_factor).item():.2f}")
        else:
            self.log_distance_scale_factor = None
            
        ct_esm_config = FastEsmConfig(
            hidden_size=_hidden_dim,
            num_hidden_layers=context_track_config.get('num_layers', 6),
            num_attention_heads=context_track_config.get('num_heads', 8),
            intermediate_size=context_track_config.get('intermediate_size', _hidden_dim * 4),
            hidden_dropout_prob=context_track_config.get('hidden_dropout', 0.1),
            attention_probs_dropout_prob=context_track_config.get('attention_dropout', 0.1),
            layer_norm_eps=context_track_config.get('layer_norm_eps', 1e-12),
            dsattn_bias_input_dim=context_track_config.get('dsattn_bias_input_dim', 6), 
            dsattn_bias_hidden_size=context_track_config.get('dsattn_bias_hidden_size', 16),
            dsattn_bias_noise_std=context_track_config.get('dsattn_bias_noise_std', 1.0),
            distance_transformation=context_track_config.get('distance_transformation', None),
            distance_scale_factor=self.distance_scale_factor,  # Always use the base value for initialization
            attention_type=context_track_config.get('attention_type', 'ds'),
            emb_layer_norm_before=context_track_config.get('emb_layer_norm_before', False),
            gradient_checkpointing=context_track_config.get('gradient_checkpointing', False)
        )
        self.context_track_encoder = ContextTrackEncoder(config=ct_esm_config)

        # --- Context Track Projection Head ---
        self.projection_dim = self.model_config.get('projection_dim', 64) # User specified 64
        context_out_dim = ct_esm_config.hidden_size # This is _hidden_dim (480)
        self.context_projection = nn.Linear(context_out_dim, self.projection_dim)

        # --- Classification Head ---
        num_classes = self.model_config.get('num_classes', 1) # Changed to 1 for binary classification with BCEWithLogits
        self.classification_head = nn.Linear(self.projection_dim, num_classes)

        # --- Load pre-trained weights if path is provided ---
        if pretrained_geneclr_path:
            print(f"Loading pre-trained GeneCLR weights from {pretrained_geneclr_path}...")
            # Load the state_dict from the checkpoint
            # For PyTorch Lightning checkpoints, the model's state_dict is usually under 'state_dict' key
            checkpoint = torch.load(pretrained_geneclr_path, map_location='cpu')
            pretrained_state_dict = checkpoint.get('state_dict', checkpoint) # Handle cases where state_dict is top-level

            # Create a new state_dict for the current model, mapping only relevant keys
            new_state_dict = {}
            for k, v in pretrained_state_dict.items():
                # Keys for context_track_encoder start with 'context_track_encoder.'
                if k.startswith('context_track_encoder.'):
                    new_key = k # Key name is already correct
                    new_state_dict[new_key] = v
                # Keys for context_projection start with 'context_projection.'
                elif k.startswith('context_projection.'):
                    new_key = k # Key name is already correct
                    new_state_dict[new_key] = v
                # Also include classification_head if it exists (for loading fine-tuned models)
                elif k.startswith('classification_head.'):
                    new_key = k # Key name is already correct
                    new_state_dict[new_key] = v
            
            # Load the filtered state_dict. strict=False allows missing keys (like focal_track)
            missing_keys, unexpected_keys = self.load_state_dict(new_state_dict, strict=False)
            
            # Print what was loaded
            loaded_modules = set([k.split('.')[0] for k in new_state_dict.keys()])
            print(f"Successfully loaded pre-trained weights for: {', '.join(sorted(loaded_modules))}")
            if missing_keys:
                print(f"Missing keys (will be randomly initialized): {len(missing_keys)} keys")
            if unexpected_keys:
                print(f"Unexpected keys (ignored): {len(unexpected_keys)} keys")

            # Optional: Freeze the loaded layers if you don't want to fine-tune them
            # for param in self.context_track_encoder.parameters():
            #     param.requires_grad = False
            # for param in self.context_projection.parameters():
            #     param.requires_grad = False
            # print("Pre-trained context track and projection layers are frozen.")

    def apply_lora(self, target_modules=None, r=8, alpha=16, dropout=0.1):
        """
        Apply LoRA adaptation to the model.
        
        Args:
            target_modules: List of module patterns to target. If None, uses default targets.
            r: LoRA rank (lower = more efficient)
            alpha: LoRA alpha (scaling factor)
            dropout: LoRA dropout rate
            
        Returns:
            dict: Mapping of module names to LoRA layers
        """
        if target_modules is None:
            # Default targets: attention layers, feed-forward layers, and head layers
            target_modules = [
                'query', 'key', 'value',                    # Attention layers
                'output_projection.dense',                   # Attention output projection
                'intermediate.dense',                        # Feed-forward layers
                'context_projection',
                'dsattn_bias_module.linear_in',
                'dsattn_bias_module.linear_out'                    # Context projection head
            ]
        
        print(f"Applying LoRA with r={r}, alpha={alpha}, dropout={dropout}")
        print(f"Target modules: {target_modules}")
        
        lora_modules = apply_lora_to_model(self, target_modules, r=r, alpha=alpha, dropout=dropout)
        self.lora_modules = lora_modules
        
        # Count trainable parameters
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,} ({trainable_params/total_params*100:.2f}%)")
        
        return lora_modules
        

            
    def convert_to_standard_linear(self):
        """Permanently convert LoRA layers to standard linear layers with merged weights."""
        if hasattr(self, 'lora_modules'):
            for name, lora_layer in self.lora_modules.items():
                # Get the merged linear layer
                merged_linear = lora_layer.get_merged_linear()
                
                # Replace the LoRA layer with the standard linear layer
                # Navigate to the parent module and replace the child
                module_path = name.split('.')
                parent_module = self
                for part in module_path[:-1]:
                    parent_module = getattr(parent_module, part)
                
                setattr(parent_module, module_path[-1], merged_linear)
            
            # Clear the lora_modules reference since they're no longer LoRA layers
            del self.lora_modules

    def get_distance_scale_factor(self):
        """Get the current distance scale factor, either fixed or learnable."""
        if self.use_learnable_distance_scale:
            return torch.exp(self.log_distance_scale_factor)
        else:
            return self.distance_scale_factor

    def forward(self, embeddings: torch.Tensor, pairwise_distances: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        """
        Forward pass for token classification.
        
        Args:
            embeddings: Precomputed embeddings (B, S, D)
            pairwise_distances: Distance features for attention (B, S, S, N_dist)
            attention_mask: Attention mask (B, S)
        """
        # Context Track as the trunk
        context_transformer_attention_mask = None
        if attention_mask is not None:
            # Prepare attention mask for the transformer (B, 1, 1, S)
            context_transformer_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2) 
            # Convert to appropriate dtype and invert for attention bias (0.0 for visible, -inf for masked)
            context_transformer_attention_mask = context_transformer_attention_mask.to(dtype=embeddings.dtype) 
            context_transformer_attention_mask = (1.0 - context_transformer_attention_mask) * torch.finfo(embeddings.dtype).min
        
        context_output_pack = self.context_track_encoder(
            hidden_states=embeddings, # Use original embeddings as input for the trunk
            distance_tensor=pairwise_distances,
            distance_scale_factor=self.get_distance_scale_factor() if self.use_learnable_distance_scale else None,  # Pass learnable scale factor
            attention_mask=context_transformer_attention_mask 
        )
        context_output_hidden_states = context_output_pack.last_hidden_state

        # Apply context projection head
        context_projected = self.context_projection(context_output_hidden_states)

        # Apply classification head
        logits = self.classification_head(context_projected)

        return logits

    def training_step(self, batch, batch_idx):
        embeddings = batch['embeddings']
        pairwise_distances = batch['pairwise_distances']
        attention_mask = batch.get('attention_mask', None)
        labels = batch['labels']  # Shape: (B, S)
        weights = batch['weights']  # Shape: (B, S)

        # Forward pass
        logits = self(embeddings, pairwise_distances, attention_mask=attention_mask) # Shape: (B, S, num_classes)

        # Ensure labels and weights are tensors
        if not isinstance(labels, torch.Tensor):
            labels = torch.tensor(labels, device=logits.device)
        if not isinstance(weights, torch.Tensor):
            weights = torch.tensor(weights, device=logits.device)

        # Flatten logits, labels, and weights
        logits_flat = logits.view(-1)                  # (B*S) - squeeze to remove the last dim of size 1
        labels_flat = labels.view(-1)                  # (B*S)
        weights_flat = weights.view(-1)                # (B*S)

        # Filter out positions where labels are -100
        valid_positions_mask = (labels_flat != -100)
        
        if not valid_positions_mask.any():
            # No valid labels in this batch, return 0 loss
            return torch.tensor(0.0, device=self.device, requires_grad=True)

        valid_logits = logits_flat[valid_positions_mask]
        valid_labels = labels_flat[valid_positions_mask].float() # BCEWithLogitsLoss expects float targets
        valid_weights = weights_flat[valid_positions_mask]

        # Compute BCEWithLogitsLoss
        loss = F.binary_cross_entropy_with_logits(valid_logits, valid_labels, reduction='none')

        # Apply weights and compute weighted average
        weighted_loss = (loss * valid_weights).sum() / valid_weights.sum()

        self.log('train_loss', weighted_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return weighted_loss

    def on_validation_epoch_start(self):
        # Initialize lists to store outputs for epoch-level metric calculation
        self.all_val_labels = []
        self.all_val_probabilities = []
        self.all_val_group_strings = []

    def validation_step(self, batch, batch_idx):
        embeddings = batch['embeddings']
        pairwise_distances = batch['pairwise_distances']
        attention_mask = batch.get('attention_mask', None)
        labels = batch['labels']
        weights = batch['weights']
        group_strings = batch['group_strings'] # Assuming this comes from your DataLoader

        logits = self(embeddings, pairwise_distances, attention_mask=attention_mask)

        # Ensure labels and weights are tensors
        if not isinstance(labels, torch.Tensor):
            labels = torch.tensor(labels, device=logits.device)
        if not isinstance(weights, torch.Tensor):
            weights = torch.tensor(weights, device=logits.device)

        logits_flat = logits.view(-1)
        labels_flat = labels.view(-1)
        weights_flat = weights.view(-1)
        group_strings_flat = np.asarray(group_strings).flatten() # Flatten group_strings too

        valid_positions_mask = (labels_flat != -100)

        if not valid_positions_mask.any():
            self.log('val_loss', torch.tensor(0.0, device=self.device), on_step=False, on_epoch=True, prog_bar=True, logger=True)
            # No valid samples, so skip logging macro_ap for this step, it will be NaN at epoch end if no data
            return torch.tensor(0.0, device=self.device)

        valid_logits = logits_flat[valid_positions_mask]
        valid_labels = labels_flat[valid_positions_mask].float()
        valid_weights = weights_flat[valid_positions_mask]
        valid_group_strings = group_strings_flat[valid_positions_mask.cpu().numpy()] # Use filtered group_strings
        # Set group_strings for negative labels to "non-defensive"
        valid_group_strings[valid_labels.cpu() == 0] = "non-defensive"

        loss = F.binary_cross_entropy_with_logits(valid_logits, valid_labels, reduction='none')

        weighted_loss = (loss * valid_weights).sum() / valid_weights.sum()

        self.log('val_loss', weighted_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        # Collect data for epoch-level macro AP
        probabilities = torch.sigmoid(valid_logits).detach().cpu().numpy()
        self.all_val_labels.append(valid_labels.cpu().numpy())
        self.all_val_probabilities.append(probabilities)
        self.all_val_group_strings.append(valid_group_strings)

        return weighted_loss
    
    def on_validation_epoch_end(self, verbose: bool = True):
        # Concatenate all collected data from validation steps
        if not self.all_val_labels:
            # No valid samples were processed in this epoch
            self.log('val_macro_ap', torch.tensor(np.nan, device=self.device), on_step=False, on_epoch=True, prog_bar=True, logger=True)
            self.log('val_macro_auroc', torch.tensor(np.nan, device=self.device), on_step=False, on_epoch=True, prog_bar=True, logger=True)
            return

        all_labels_np = np.concatenate(self.all_val_labels)
        all_probabilities_np = np.concatenate(self.all_val_probabilities)
        all_group_strings_np = np.concatenate(self.all_val_group_strings)

        # Compute the macro AP metric once on the aggregated data
        macro_ap = macro_average_precision_pos_vs_single_neg(
            all_labels_np, 
            all_probabilities_np, 
            all_group_strings_np
        )
        self.log('val_macro_ap', torch.tensor(macro_ap, device=self.device), on_step=False, on_epoch=True, prog_bar=True, logger=True)
        if verbose:
            print(f"Validation macro AP: {macro_ap}")

        # Compute the macro AUROC metric once on the aggregated data
        macro_auroc = macro_auroc_pos_vs_single_neg(
            all_labels_np, 
            all_probabilities_np, 
            all_group_strings_np
        )
        self.log('val_macro_auroc', torch.tensor(macro_auroc, device=self.device), on_step=False, on_epoch=True, prog_bar=True, logger=True)
        if verbose:
            print(f"Validation macro AUROC: {macro_auroc}")
        
        # Clear lists for the next epoch
        self.all_val_labels = []
        self.all_val_probabilities = []
        self.all_val_group_strings = []

    def test_step(self, batch, batch_idx):
        embeddings = batch['embeddings']
        pairwise_distances = batch['pairwise_distances']
        attention_mask = batch.get('attention_mask', None)
        labels = batch['labels']
        weights = batch['weights']
        group_strings = batch['group_strings'] # Assuming this comes from your DataLoader

        logits = self(embeddings, pairwise_distances, attention_mask=attention_mask)

        # Ensure labels and weights are tensors
        if not isinstance(labels, torch.Tensor):
            labels = torch.tensor(labels, device=logits.device)
        if not isinstance(weights, torch.Tensor):
            weights = torch.tensor(weights, device=logits.device)

        logits_flat = logits.view(-1)
        labels_flat = labels.view(-1)
        weights_flat = weights.view(-1)
        group_strings_flat = np.asarray(group_strings).flatten()

        valid_positions_mask = (labels_flat != -100)

        if not valid_positions_mask.any():
            self.log('test_loss', torch.tensor(0.0, device=self.device), on_step=False, on_epoch=True, prog_bar=True, logger=True)
            return torch.tensor(0.0, device=self.device)

        valid_logits = logits_flat[valid_positions_mask]
        valid_labels = labels_flat[valid_positions_mask].float()
        valid_weights = weights_flat[valid_positions_mask]
        valid_group_strings = group_strings_flat[valid_positions_mask.cpu().numpy()]
        # valid group strings associated to a label == 0 should be "non-defensive"
        valid_group_strings[valid_labels.cpu() == 0] = "non-defensive"

        loss = F.binary_cross_entropy_with_logits(valid_logits, valid_labels, reduction='none')

        weighted_loss = (loss * valid_weights).sum() / valid_weights.sum()

        self.log('test_loss', weighted_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        # Collect data for epoch-level macro AP
        probabilities = torch.sigmoid(valid_logits).detach().cpu().numpy()
        self.all_test_labels.append(valid_labels.cpu().numpy())
        self.all_test_probabilities.append(probabilities)
        self.all_test_group_strings.append(valid_group_strings)

        return weighted_loss

    def on_test_epoch_start(self):
        # Initialize lists to store outputs for epoch-level metric calculation for test set
        self.all_test_labels = []
        self.all_test_probabilities = []
        self.all_test_group_strings = []

    def on_test_epoch_end(self):
        # Concatenate all collected data from test steps
        if not self.all_test_labels:
            # No valid samples were processed in this epoch
            self.log('test_macro_ap', torch.tensor(np.nan, device=self.device), on_step=False, on_epoch=True, prog_bar=True, logger=True)
            self.log('test_macro_auroc', torch.tensor(np.nan, device=self.device), on_step=False, on_epoch=True, prog_bar=True, logger=True)
            return

        all_labels_np = np.concatenate(self.all_test_labels)
        all_probabilities_np = np.concatenate(self.all_test_probabilities)
        all_group_strings_np = np.concatenate(self.all_test_group_strings)

        # Compute the macro AP metric once on the aggregated data
        macro_ap = macro_average_precision_pos_vs_single_neg(
            all_labels_np, 
            all_probabilities_np, 
            all_group_strings_np
        )
        self.log('test_macro_ap', torch.tensor(macro_ap, device=self.device), on_step=False, on_epoch=True, prog_bar=True, logger=True)

        # Compute the macro AUROC metric once on the aggregated data
        macro_auroc = macro_auroc_pos_vs_single_neg(
            all_labels_np, 
            all_probabilities_np, 
            all_group_strings_np
        )
        self.log('test_macro_auroc', torch.tensor(macro_auroc, device=self.device), on_step=False, on_epoch=True, prog_bar=True, logger=True)
        
        # Clear lists for the next epoch
        self.all_test_labels = []
        self.all_test_probabilities = []
        self.all_test_group_strings = []


if __name__ == '__main__':
    print("GeneCLR model placeholder for direct execution.")

    logger = TensorBoardLogger(
        save_dir="logs/",
        name="geneclr_experiment"
    )

    trainer = pl.Trainer(
        max_epochs=10,
        logger=logger,
    ) 
