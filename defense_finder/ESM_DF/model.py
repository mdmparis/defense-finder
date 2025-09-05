import pytorch_lightning as pl
import torch
import torch.nn as nn
import numpy as np
from transformers import EsmModel
from peft import get_peft_model, LoraConfig, TaskType
from torchmetrics import AUROC
import torch.nn.functional as F
# from metrics.macro_avg_metric import macro_average_precision_pos_vs_single_neg, macro_auroc_pos_vs_single_neg
from typing import Optional


id2label = {
    0: "non defense gene",
    1: "defense gene"
}

label2id = {
    "non defense gene": 0,
    "defense gene": 1,
}

label_list = [
    "non defense gene",
    "defense gene"
]


class MeanPoolingClassificationHead(nn.Module):
    """Classification head with mean pooling instead of CLS pooling"""


    def __init__(self, hidden_size: int, num_labels: int, dropout_prob: float = 0.1, intermediate_size: int = None):
        super().__init__()
        if intermediate_size is None:
            intermediate_size = hidden_size * 4  # Common expansion factor

        self.dropout = nn.Dropout(dropout_prob)
        # SwiGLU requires two linear layers for gating
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, num_labels)

    def mean_pooling(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        """Apply mean pooling over sequence dimension, respecting attention mask"""
        if attention_mask is None:
            pooled = hidden_states.mean(dim=1)
        else:
            # Expand attention mask to match hidden_states dimensions and dtype
            attention_mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).to(hidden_states.dtype)

            # Apply mask and compute mean
            sum_hidden = (hidden_states * attention_mask_expanded).sum(dim=1)
            sum_mask = attention_mask_expanded.sum(dim=1)

            # Avoid division by zero
            pooled = sum_hidden / torch.clamp(sum_mask, min=1e-9)

        # L2 normalize the pooled vector to make it length-invariant
        # pooled = F.normalize(pooled, p=2, dim=1)
        return pooled

    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        # Apply mean pooling
        pooled_output = self.mean_pooling(hidden_states, attention_mask)

        # Apply classification layers
        pooled_output = self.dropout(pooled_output)

        # SwiGLU with residual connection
        residual = pooled_output
        gate = torch.nn.functional.silu(self.gate_proj(pooled_output))
        up = self.up_proj(pooled_output)
        pooled_output = gate * up

        # Project back down to hidden_size and add residual
        pooled_output = self.down_proj(pooled_output)
        pooled_output = pooled_output + residual  # Residual connection
        pooled_output = self.dropout(pooled_output)
        logits = self.out_proj(pooled_output)

        return logits

class EsmForSequenceClassificationLightning(pl.LightningModule):
    def __init__(
        self,
        model_checkpoint: str,
        num_labels: int = 2,
        id2label: dict = id2label,
        label2id: dict = label2id,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.1,
        classifier_dropout: float = 0.1,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.0,
        scheduler: dict = None,
        max_epochs: int = 10,
        ):

        super().__init__()
        self.save_hyperparameters()

        # Load the base ESM model (without classification head)
        self.esm = EsmModel.from_pretrained(model_checkpoint)

        # Create custom mean pooling classification head
        self.classifier = MeanPoolingClassificationHead(
            hidden_size=self.esm.config.hidden_size,
            num_labels=num_labels,
            dropout_prob=classifier_dropout
        )

        # Store config info for compatibility
        self.config = self.esm.config
        self.config.num_labels = num_labels
        self.config.id2label = id2label
        self.config.label2id = label2id
        self.config.problem_type = 'single_label_classification'
        print('Model config:', self.config)
        print(f'Using mean pooling instead of CLS pooling for RoPE compatibility')

        # Apply LoRA to the ESM backbone
        peft_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,  # Changed from SEQ_CLS since we're using base model
            inference_mode=False,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias="all",
            target_modules=['key', 'query', 'value', 'output_projection.dense'],  # Remove dense layers to save memory
            use_rslora=True,
        )
        self.esm = get_peft_model(self.esm, peft_config)
        
        # Enable gradient checkpointing to save memory
        if hasattr(self.esm.base_model.model, 'gradient_checkpointing_enable'):
            self.esm.base_model.model.gradient_checkpointing_enable()
        
        # Ensure classifier is trainable
        self.classifier.requires_grad_(True)
        
        # Print trainable parameters
        self.esm.print_trainable_parameters()
        classifier_params = sum(p.numel() for p in self.classifier.parameters() if p.requires_grad)
        print(f'Classifier trainable parameters: {classifier_params:,}')

        # Loss - will be used with sample weights
        self.loss_fn = torch.nn.CrossEntropyLoss(reduction='none')  # No reduction, we'll apply weights manually

        # Metrics
        self.val_auroc = AUROC(task="binary")
        self.test_auroc = AUROC(task="binary")

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        """Override to handle metadata fields that shouldn't be moved to GPU"""
        # Separate metadata fields from tensor fields
        metadata_fields = {}
        tensor_fields = {}
        
        for key, value in batch.items():
            if key in ['defense_finder_subtypes', 'defense_finder_profiles']:
                # Keep metadata on CPU
                metadata_fields[key] = value
            else:
                # Move tensors to device
                tensor_fields[key] = value
        
        # Move only tensor fields to device
        moved_batch = super().transfer_batch_to_device(tensor_fields, device, dataloader_idx)
        
        # Add back metadata fields (create new dict to avoid reference issues)
        result_batch = dict(moved_batch)
        result_batch.update(metadata_fields)
        del tensor_fields, metadata_fields
        return result_batch

    def forward(self, input_ids, attention_mask, labels=None):
        # Get hidden states from ESM backbone
        esm_outputs = self.esm(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = esm_outputs.last_hidden_state
        
        # Ensure hidden states match the dtype of the classifier weights
        # This is necessary when the model is loaded with mixed precision
        if hasattr(self.classifier.gate_proj, 'weight'):
            target_dtype = self.classifier.gate_proj.weight.dtype
            if hidden_states.dtype != target_dtype:
                hidden_states = hidden_states.to(dtype=target_dtype)
        
        # Apply mean pooling classification head
        logits = self.classifier(hidden_states, attention_mask=attention_mask)
        
        # Return in the same format as the original model
        class SimpleOutput:
            def __init__(self, logits):
                self.logits = logits
        
        return SimpleOutput(logits)

    def compute_weighted_loss(self, logits, labels, sample_weights, return_mean_of_weights=False):
        """Compute weighted cross-entropy loss"""
        # Compute per-sample losses
        losses = self.loss_fn(logits, labels)  # Shape: (batch_size,)
        
        # Ensure sample_weights have the same dtype and device as losses for mixed precision
        sample_weights = sample_weights.to(dtype=losses.dtype, device=losses.device)
        
        # Apply sample weights
        weighted_losses = losses * sample_weights
        
        # Return mean weighted loss
        if return_mean_of_weights:
            return weighted_losses.mean(), sample_weights.mean()
        else:
            return weighted_losses.mean()

    def training_step(self, batch, batch_idx):
        outputs = self(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        )
        logits = outputs.logits
        
        # Compute weighted loss
        loss, mean_of_weights = self.compute_weighted_loss(
            logits=logits,
            labels=batch["labels"],
            sample_weights=batch["sample_weight"],
            return_mean_of_weights=True
        )
        
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("normalized_loss", loss / mean_of_weights, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    # def _shared_eval_step(self, batch, batch_idx, stage):
    #     """Shared evaluation logic for validation and test steps"""
    #     outputs = self(
    #         input_ids=batch["input_ids"],
    #         attention_mask=batch["attention_mask"],
    #     )
    #     logits = outputs.logits
        
    #     # Compute weighted loss
    #     loss = self.compute_weighted_loss(
    #         logits=logits,
    #         labels=batch["labels"],
    #         sample_weights=batch["sample_weight"]
    #     )
        
    #     probs = F.softmax(logits, dim=-1)[:, 1]
        
    #     # Update appropriate AUROC metric
    #     if stage == "val":
    #         self.val_auroc.update(probs, batch["labels"])
    #         auroc_metric = self.val_auroc
    #         outputs_attr = '_val_outputs'
    #     else:  # test
    #         self.test_auroc.update(probs, batch["labels"])
    #         auroc_metric = self.test_auroc
    #         outputs_attr = '_test_outputs'
        
    #     # Log metrics
    #     self.log(f"{stage}_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
    #     self.log(f"{stage}_auroc", auroc_metric, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        
    #     # Accumulate for grouped metrics
    #     if not hasattr(self, outputs_attr):
    #         setattr(self, outputs_attr, [])

    #     # Explicitly move to CPU and clear CUDA cache
    #     probs_cpu = probs.detach().cpu()
    #     labels_cpu = batch['labels'].detach().cpu()
    #     getattr(self, outputs_attr).append({
    #         'probs': probs_cpu,
    #         'labels': labels_cpu,
    #         'subtypes': batch.get('defense_finder_subtypes', None)
    #     })
    #     del probs_cpu, labels_cpu
    #     torch.cuda.empty_cache()

    #     return loss

    # def _shared_epoch_end(self, stage):
    #     """Shared epoch end logic for validation and test"""
    #     outputs_attr = f'_{stage}_outputs'
    #     dataset_attr = 'eval_dataset' if stage == 'val' else 'test_dataset'
    #     stage_upper = stage.upper()
        
    #     if not hasattr(self, outputs_attr) or not getattr(self, outputs_attr):
    #         return
            
    #     outputs = getattr(self, outputs_attr)
    #     all_probs = torch.cat([x['probs'] for x in outputs]).numpy()
    #     all_labels = torch.cat([x['labels'] for x in outputs]).numpy()
    #     all_subtypes = []
        
    #     for x in outputs:
    #         if x['subtypes'] is not None:
    #             if isinstance(x['subtypes'], list):
    #                 all_subtypes.extend(x['subtypes'])
    #             elif hasattr(x['subtypes'], 'tolist'):
    #                 all_subtypes.extend(x['subtypes'].tolist())
    #             else:
    #                 all_subtypes.extend([''] * len(x['labels']))
    #         else:
    #             all_subtypes.extend([''] * len(x['labels']))
                
    #     if not all_subtypes or len(all_subtypes) != len(all_labels):
    #         if hasattr(self.trainer.datamodule, dataset_attr):
    #             all_subtypes = getattr(self.trainer.datamodule, dataset_attr)['defense_finder_subtypes']
                
    #     # Debug: Print what we have for subtype analysis
    #     n_pos = np.sum(all_labels == 1)
    #     n_neg = np.sum(all_labels == 0)
    #     pos_subtypes = [s for s, l in zip(all_subtypes, all_labels) if l == 1 and s != 'non_defense' and s != '']
    #     unique_pos_subtypes = set(pos_subtypes)
    #     print(f'[{stage_upper}] Debug: {len(all_labels)} total, {n_pos} pos, {n_neg} neg, {len(unique_pos_subtypes)} unique pos subtypes')
    #     print(f'[{stage_upper}] Debug: Unique pos subtypes: {sorted(unique_pos_subtypes)[:10]}...')  # Show first 10
        
    #     macro_ap = macro_average_precision_pos_vs_single_neg(all_labels, all_probs, all_subtypes)
    #     macro_auroc = macro_auroc_pos_vs_single_neg(all_labels, all_probs, all_subtypes)
    #     print(f'[{stage_upper}] Macro-averaged AP (per subtype): {macro_ap}')
    #     print(f'[{stage_upper}] Macro-averaged AUROC (per subtype): {macro_auroc}')
    #     self.log(f'{stage}_macro_ap', macro_ap, prog_bar=True, logger=True)
    #     self.log(f'{stage}_macro_auroc', macro_auroc, prog_bar=True, logger=True)
    #     getattr(self, outputs_attr).clear()
        
    #     # Clear CUDA cache to free memory
    #     if torch.cuda.is_available():
    #         torch.cuda.empty_cache()


    def validation_step(self, batch, batch_idx):
        return self._shared_eval_step(batch, batch_idx, "val")

    def on_validation_epoch_end(self):
        self._shared_epoch_end("val")

    def test_step(self, batch, batch_idx):
        return self._shared_eval_step(batch, batch_idx, "test")

    def on_test_epoch_end(self):
        self._shared_epoch_end("test")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )
        
        # Add scheduler support similar to GeneCLR
        scheduler_config = getattr(self.hparams, 'scheduler', None)
        if scheduler_config:
            if scheduler_config.get('name') == 'cosine_annealing':
                t_max = scheduler_config.get('T_max')
                
                # If t_max not provided, estimate from training config
                if t_max is None:
                    if self.trainer and hasattr(self.trainer, 'estimated_stepping_batches'):
                        t_max = self.trainer.estimated_stepping_batches
                    else:
                        # Estimate t_max from training configuration
                        max_epochs = getattr(self.hparams, 'max_epochs', 10)
                        # This is approximate - actual value depends on dataset size and batch size
                        # You may want to set T_max explicitly in config for precision
                        estimated_steps_per_epoch = scheduler_config.get('estimated_steps_per_epoch', 100)
                        t_max = max_epochs * estimated_steps_per_epoch
                        print(f"Warning: T_max estimated as {t_max} steps. Consider setting T_max explicitly in scheduler config.")
                
                if t_max is None:
                    raise ValueError("Could not determine T_max. Please set T_max in scheduler config or ensure trainer is available.")

                # Calculate warmup steps
                warmup_epochs = scheduler_config.get('warmup_epochs', 0.5)
                max_epochs = getattr(self.hparams, 'max_epochs', 10)
                steps_per_epoch = t_max / max_epochs
                warmup_steps = int(steps_per_epoch * warmup_epochs)
                
                # Warmup start LR (typically 10% of base LR)
                base_lr = self.hparams.learning_rate
                warmup_start_lr = scheduler_config.get('warmup_start_lr', base_lr * 0.1)
                eta_min = scheduler_config.get('eta_min', base_lr * 0.01)
                
                print(f"Scheduler config: warmup_steps={warmup_steps}, t_max={t_max}, base_lr={base_lr:.2e}, warmup_start_lr={warmup_start_lr:.2e}, eta_min={eta_min:.2e}")
                
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
                    patience=scheduler_config.get('patience', 10),
                    factor=scheduler_config.get('factor', 0.1),
                    min_lr=scheduler_config.get('min_lr', 1e-6),
                    verbose=True
                )
                return [optimizer], [{
                    'scheduler': scheduler,
                    'monitor': 'val_loss',
                    'interval': 'epoch',
                    'frequency': 1
                }]
        
        return optimizer