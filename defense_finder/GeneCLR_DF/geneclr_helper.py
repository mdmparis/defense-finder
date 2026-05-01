#!/usr/bin/env python3
import copy
import os
from typing import Optional, Union

import pandas as pd
import numpy as np
import torch
import yaml

from .geneclr.model import GeneCLR, GeneClrForTokenClassification
from .geneclr.prodigal_io import pyrodigal_annotation_dict_to_dataframe

from pyhmmer.easel import SequenceFile, TextSequence, Alphabet
import colorlog

logger = colorlog.getLogger("Defense_Finder")


def create_overlapping_windows(genes_df: pd.DataFrame, window_size: int = 64, stride: int = 32) -> tuple:
    """
    Create overlapping windows of genes.
    
    Returns:
        windows: List of fragments (each = list of gene dicts, may have < window_size genes)
        window_to_gene_indices: List of length-64 lists; each position maps to global gene index (-1 = padding)
    """
    genes = genes_df.to_dict("records")
    n_genes = len(genes)
    
    if n_genes == 0:
        return [], []
    
    windows = []
    window_to_gene_indices = []
    
    start = 0
    while start < n_genes:
        end = min(start + window_size, n_genes)
        window_genes = []
        gene_indices = []
        
        for i in range(window_size):
            global_idx = start + i
            if global_idx < n_genes:
                g = genes[global_idx]
                window_genes.append({
                    "sequence": str(g["sequence"]),
                    "start": int(g["start"]),
                    "end": int(g["end"]),
                })
                gene_indices.append(global_idx)
            else:
                gene_indices.append(-1)
        
        # Don't pad window_genes - collate will pad. But gene_indices must match output positions (64)
        windows.append(window_genes)
        window_to_gene_indices.append(gene_indices)
        start += stride
        
        if end >= n_genes:
            break
    
    return windows, window_to_gene_indices


def run_classifier_inference_cli(
    checkpoint_path: str,
    finetuning_config_path: str,
    device: str,
    dataloader,
    window_to_gene_indices: list,
    n_genes: int,
    hit_ids: np.ndarray,
    output_path: str,
    output_format: str,
) -> bool:
    """Classifier path: load model, run windows, aggregate logits, write table. Returns False if no logits."""
    clf_model = load_geneclr_classification_for_inference(
        checkpoint_path, finetuning_config_path, device=device
    )
    window_logits = run_classification_inference(clf_model, dataloader, device)
    if window_logits.numel() == 0:
        print("No logits produced.")
        return False
    gene_logits = average_overlapping_scores(window_logits, window_to_gene_indices, n_genes)
    if gene_logits.ndim == 1:
        out_df = pd.DataFrame({"hit_id": hit_ids, "logit_Def": gene_logits})
    else:
        cols = {"hit_id": hit_ids}
        for j in range(gene_logits.shape[1]):
            cols[f"logit_{j}"] = gene_logits[:, j]
        out_df = pd.DataFrame(cols)
    #write_dataframe_output(output_path, output_format, out_df, index=False)
    #print(f"Saved {n_genes} gene classifier logits {gene_logits.shape} to {output_path} ({output_format.lower()})")
    return out_df


def load_geneclr_classification_for_inference(
    checkpoint_path: str,
    finetuning_config_path: Optional[str] = None,
    device: str = "cuda",
) -> GeneClrForTokenClassification:
    """
    Load GeneClrForTokenClassification (fine-tuned defense / binary classifier) for inference.

    **Initialization:** if the checkpoint contains ``hyper_parameters`` (PyTorch Lightning default),
    ``model_config`` is rebuilt from that dict so it matches ``scripts/finetune_geneclr.py`` at train time.
    Otherwise ``finetuning_config_path`` is required and :func:`build_model_config_for_classification` is used.

    Supports merged LoRA weights or unmerged checkpoints; for the latter, pass ``finetuning_config_path``
    so the ``lora`` section can be read (defaults ``r=4, alpha=4, dropout=0.1`` if section missing).

    When ``finetuning_config_path`` is set, ``model.distance_scale_factor`` or
    ``context_track.distance_scale_factor`` from that YAML overrides the checkpoint for inference
    (then :func:`finalize_classification_inference_model_config` fixes a non-learnable scale).

    Args:
        checkpoint_path: Fine-tuned ``.ckpt``.
        finetuning_config_path: Fine-tuning YAML (LoRA settings; optional distance scale override;
            required if the checkpoint has no usable ``hyper_parameters``).
        device: Target device.
    """
    try:
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(checkpoint_path, map_location="cpu")

    cfg: dict = {}
    if finetuning_config_path:
        with open(finetuning_config_path) as f:
            cfg = yaml.safe_load(f)

    hp = ckpt.get("hyper_parameters") or {}
    if isinstance(hp, dict) and "hidden_dim" in hp and "context_track" in hp:
        model_config = model_config_from_checkpoint_hyperparameters(hp)
        logger.debug(
            "[load_geneclr_classification] Using hyper_parameters from checkpoint "
            "(matches training-time GeneClrForTokenClassification init)"
        )
    else:
        if not cfg:
            raise ValueError(
                "Checkpoint has no usable hyper_parameters; pass finetuning_config_path "
                "so the model can be built from YAML."
            )
        model_config = build_model_config_for_classification(cfg)
        logger.debug("[load_geneclr_classification] Using finetuning YAML (no hyper_parameters in checkpoint)")

    # Prefer explicit distance scale from fine-tuning YAML when provided (over checkpoint hp)
    if cfg:
        ymod = cfg.get("model", {})
        yct = cfg.get("context_track", {})
        if "distance_scale_factor" in ymod:
            model_config["distance_scale_factor"] = float(ymod["distance_scale_factor"])
        elif "distance_scale_factor" in yct:
            model_config["distance_scale_factor"] = float(yct["distance_scale_factor"])

    finalize_classification_inference_model_config(model_config)
    logger.debug(
        f"[load_geneclr_classification] distance_scale_factor={model_config['distance_scale_factor']} "
        "(fixed, non-learnable)"
    )

    state = _normalize_pl_state_dict(ckpt.get("state_dict", ckpt))
    has_lora = any("lora_A" in k or "lora_B" in k for k in state)

    model = GeneClrForTokenClassification(model_config, pretrained_geneclr_path=None)

    if has_lora:
        lcfg = cfg.get("lora", {}) if cfg else {}
        for p in model.parameters():
            p.requires_grad = False
        model.apply_lora(
            r=lcfg.get("r", 4),
            alpha=lcfg.get("alpha", 4),
            dropout=lcfg.get("dropout", 0.1),
        )

    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        logger.error(f"[load_geneclr_classification] Missing keys ({len(missing)}): first few: {missing[:5]}")
    if unexpected:
        logger.error(f"[load_geneclr_classification] Unexpected keys ({len(unexpected)}): first few: {unexpected[:5]}")

    model.eval()
    return model.to(device)


def run_classification_inference(
    model: GeneClrForTokenClassification,
    dataloader,
    device: str,
) -> torch.Tensor:
    """
    Run token-level logits for each window.

    Returns:
        ``(num_windows, seq_len)`` if ``num_classes == 1``, else ``(num_windows, seq_len, num_classes)``.
    """
    all_logits: list[torch.Tensor] = []
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            if batch is None:
                continue
            embeddings = batch["embeddings"].to(device)
            pairwise_distances = batch["pairwise_distances"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            logits = model(embeddings, pairwise_distances, attention_mask=attention_mask)
            if logits.dim() == 3 and logits.shape[-1] == 1:
                logits = logits.squeeze(-1)
            all_logits.append(logits.cpu())
    if not all_logits:
        return torch.empty(0)
    return torch.cat(all_logits, dim=0)


def build_model_config_for_classification(yaml_config: dict) -> dict:
    """
    Build ``model_config`` for ``GeneClrForTokenClassification`` from a fine-tuning YAML.

    Same keys as ``scripts/finetune_geneclr.py`` ``create_model_config``. Optional top-level
    ``model.distance_scale_factor`` / ``model.use_learnable_distance_scale`` are passed through if set.
    """
    model = yaml_config["model"]
    training = yaml_config["training"]
    model_config = {
        "hidden_dim": model["hidden_dim"],
        "projection_dim": model.get("projection_dim", 64),
        "num_classes": model.get("num_classes", 1),
        "context_track": copy.deepcopy(yaml_config["context_track"]),
        "optimizer": yaml_config.get("optimizer", {"lr": 1e-4, "weight_decay": 0.01}),
        "scheduler": yaml_config.get("scheduler"),
        "max_epochs": training.get("max_epochs", 1),
    }
    if "distance_scale_factor" in model:
        model_config["distance_scale_factor"] = model["distance_scale_factor"]
    if "use_learnable_distance_scale" in model:
        model_config["use_learnable_distance_scale"] = model["use_learnable_distance_scale"]
    return model_config


def average_overlapping_scores(
    window_logits: torch.Tensor,
    window_to_gene_indices: list,
    n_genes: int,
) -> np.ndarray:
    """
    Mean logits per gene across overlapping windows (same indexing as average_overlapping_representations).

    Args:
        window_logits: (num_windows, seq_len) or (num_windows, seq_len, num_classes)
        window_to_gene_indices: list of length num_windows, each entry length seq_len with global gene index or -1
        n_genes: number of genes in the replicon

    Returns:
        ``(n_genes,)`` float32 if 2D logits; ``(n_genes, num_classes)`` if 3D.
    """
    arr = window_logits.detach().cpu().numpy()
    if arr.ndim == 2:
        sums = np.zeros(n_genes, dtype=np.float64)
        counts = np.zeros(n_genes, dtype=np.int64)
        for w_idx, gene_indices in enumerate(window_to_gene_indices):
            for pos, g_idx in enumerate(gene_indices):
                if g_idx >= 0:
                    sums[g_idx] += arr[w_idx, pos]
                    counts[g_idx] += 1
        counts = np.maximum(counts, 1)
        return (sums / counts).astype(np.float32)
    if arr.ndim == 3:
        _, _, n_cls = arr.shape
        sums = np.zeros((n_genes, n_cls), dtype=np.float64)
        counts = np.zeros(n_genes, dtype=np.int64)
        for w_idx, gene_indices in enumerate(window_to_gene_indices):
            for pos, g_idx in enumerate(gene_indices):
                if g_idx >= 0:
                    sums[g_idx] += arr[w_idx, pos]
                    counts[g_idx] += 1
        counts = np.maximum(counts, 1)
        return (sums / counts[:, np.newaxis]).astype(np.float32)
    raise ValueError(f"window_logits must be 2D or 3D, got shape {arr.shape}")







def finalize_classification_inference_model_config(model_config: dict) -> None:
    """
    Fix distance handling for classifier inference: non-learnable scale from config.

    Uses top-level ``distance_scale_factor`` if already set, otherwise
    ``context_track['distance_scale_factor']`` (typical fine-tuning YAMLs: ``0.03`` with ``asinh``).
    Sets ``use_learnable_distance_scale`` to ``False`` so the forward path uses a fixed scale baked
    into ``FastEsmConfig`` / ``DSAttentionBiasModule`` at construction time.
    """
    ct = model_config.get("context_track") or {}
    scale = model_config.get("distance_scale_factor", ct.get("distance_scale_factor", 0.03))
    model_config["distance_scale_factor"] = float(scale)
    model_config["use_learnable_distance_scale"] = False


def model_config_from_checkpoint_hyperparameters(hyper_parameters: dict) -> dict:
    """
    Rebuild ``model_config`` from Lightning ``hyper_parameters`` (same keys as training).
    Prefer this over YAML-only reconstruction so architecture matches the saved run.
    """
    return {
        "hidden_dim": hyper_parameters["hidden_dim"],
        "projection_dim": hyper_parameters["projection_dim"],
        "num_classes": hyper_parameters.get("num_classes", 1),
        "context_track": copy.deepcopy(hyper_parameters["context_track"]),
        "optimizer": copy.deepcopy(hyper_parameters.get("optimizer", {"lr": 1e-4, "weight_decay": 0.01})),
        "scheduler": copy.deepcopy(hyper_parameters["scheduler"])
        if hyper_parameters.get("scheduler") is not None
        else None,
        "max_epochs": hyper_parameters.get("max_epochs", 1),
    }

def _normalize_pl_state_dict(state: dict) -> dict:
    """Strip common Lightning prefixes from checkpoint state_dict keys."""
    if not state:
        return state
    if any(k.startswith("model.") for k in state):
        return {k[len("model.") :]: v for k, v in state.items() if k.startswith("model.")}
    return state