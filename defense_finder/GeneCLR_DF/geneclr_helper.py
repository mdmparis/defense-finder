#!/usr/bin/env python3
import copy
import os
import sys
import argparse
from typing import Optional, Union

import pandas as pd
import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .geneclr.model import GeneCLR, GeneClrForTokenClassification
from .geneclr.datamodule_inference import InferenceGeneCLRDataModule
from .geneclr.prodigal_io import pyrodigal_annotation_dict_to_dataframe

from pyhmmer.easel import SequenceFile, TextSequence, Alphabet
import colorlog

logger = colorlog.getLogger("Defense_Finder")

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


def load_genes_from_csv(path: str) -> pd.DataFrame:
    """Load genes from CSV with columns: hit_id, start, end, sequence."""
    df = pd.read_csv(path)
    required = {"hit_id", "start", "end", "sequence"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV must have columns {required}. Missing: {missing}")
    return df


def load_genes_from_fasta_pyrodigal(path: str, meta_threshold: int = 100_000) -> pd.DataFrame:
    """
    Run Pyrodigal on each nucleotide record in a FASTA file and return a gene DataFrame.

    Uses the same meta vs single-genome threshold as typical Prodigal usage:
    sequences shorter than ``meta_threshold`` use ``GeneFinder(meta=True)``;
    longer sequences are trained per record then annotated.

    Requires: ``pip install pyrodigal``

    Args:
        path: Path to nucleotide FASTA (.fna / .fa / .fasta).
        meta_threshold: Length below which metagenomic mode is used (default 100_000).

    Returns:
        DataFrame with columns hit_id, start, end, sequence (GeneCLR / CSV contract).
    """
    try:
        import pyrodigal
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "load_genes_from_fasta_pyrodigal requires pyrodigal. Install with: pip install pyrodigal"
        ) from e
    # try:
    #     from Bio import SeqIO
    # except ImportError as e:  # pragma: no cover
    #     raise ImportError(
    #         "load_genes_from_fasta_pyrodigal requires biopython. Install with: pip install biopython"
    #     ) from e

    # dic_genes = {}
    # for record in SeqIO.parse(path, "fasta"):
    #     sname = record.id
    #     sseq = bytes(str(record.seq).upper(), encoding="utf-8")
    #     if len(sseq) < meta_threshold:
    #         orf_finder = pyrodigal.GeneFinder(meta=True)
    #         dic_genes[sname] = orf_finder.find_genes(sseq)
    #     else:
    #         orf_finder = pyrodigal.GeneFinder()
    #         orf_finder.train(sseq)
    #         dic_genes[sname] = orf_finder.find_genes(sseq)

    with SequenceFile(path) as sf:
            seq = TextSequence()
            dic_genes = {}
            if sf.guess_alphabet() == Alphabet.dna():
                #logger.info(f"{filename} is a nucleotide fasta file. Prodigal will annotate the CDS")
                while sf.readinto(seq) is not None: # iterate over sequences in case multifasta
                    sseq = bytes(seq.sequence, encoding="utf-8")
                    sname = seq.name.decode()
                    if len(sseq) < 100000: # it is recommended to use the mode meta when seq is less than 100kb
                        orf_finder = pyrodigal.GeneFinder(meta=True)
                        dic_genes[sname] = orf_finder.find_genes(sseq)
                    else:
                        orf_finder = pyrodigal.GeneFinder()
                        orf_finder.train(sseq)
                        dic_genes[sname] = orf_finder.find_genes(sseq)
                    seq.clear()


    return pyrodigal_annotation_dict_to_dataframe(dic_genes)


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


def run_inference(
    model,
    dataloader,
    device: str,
    return_context_pre_projection: bool = True,
) -> torch.Tensor:
    """Run inference and return context outputs only."""
    all_context = []
    with torch.no_grad():
        for batch in dataloader:
            if batch is None:
                continue
            embeddings = batch["embeddings"].to(device)
            pairwise_distances = batch["pairwise_distances"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            random_context_mask = batch.get("random_context_mask")
            if random_context_mask is not None:
                random_context_mask = random_context_mask.to(device)
            
            _, context_out, _ = model(
                embeddings,
                pairwise_distances,
                attention_mask=attention_mask,
                random_context_mask=random_context_mask,
                return_context_pre_projection=return_context_pre_projection,
            )
            all_context.append(context_out.cpu())
    
    return torch.cat(all_context, dim=0) if all_context else torch.empty(0)


def average_overlapping_representations(
    context_outputs: torch.Tensor,
    window_to_gene_indices: list,
    n_genes: int,
    embed_dim: int,
) -> np.ndarray:
    """
    Average representations for genes that appear in multiple windows.
    
    context_outputs: (num_windows, 64, embed_dim)
    window_to_gene_indices: list of lists, each of length 64, values are global gene indices (-1 for padding)
    """
    # Accumulate sum and count per gene
    sums = np.zeros((n_genes, embed_dim), dtype=np.float64)
    counts = np.zeros(n_genes, dtype=np.int64)
    
    for w_idx, gene_indices in enumerate(window_to_gene_indices):
        for pos, g_idx in enumerate(gene_indices):
            if g_idx >= 0:
                vec = context_outputs[w_idx, pos].numpy()
                sums[g_idx] += vec
                counts[g_idx] += 1
    
    # Average (avoid div by zero for genes never seen - shouldn't happen)
    counts = np.maximum(counts, 1)
    return (sums.T / counts).T.astype(np.float32)


def create_model_config(config_path: str) -> dict:
    """Create model config from YAML."""
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return {
        "hidden_dim": config["model"]["hidden_dim"],
        "temperature": config["model"].get("temperature"),
        "projection_dim": config["model"].get("projection_dim"),
        "focal_track": config["focal_track"],
        "context_track": config["context_track"],
        "use_bf16_similarity": config["model"].get("use_bf16_similarity", True),
        "use_temperature_free_loss": config["model"].get("use_temperature_free_loss", False),
        "use_learnable_temperature": config["model"].get("use_learnable_temperature", False),
        "distance_scale_factor": config["model"].get("distance_scale_factor", 1000.0),
        "use_learnable_distance_scale": config["model"].get("use_learnable_distance_scale", False),
    }


def write_dataframe_output(
    path: str,
    output_format: str,
    df: pd.DataFrame,
    *,
    index: bool = False,
    index_label: Optional[str] = None,
) -> None:
    """Create parent dirs if needed; write Parquet or CSV from ``output_format`` (``parquet`` / ``csv``)."""
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    fmt = output_format.lower()
    if fmt == "csv":
        if index:
            df.to_csv(path, index=True, index_label=index_label or "hit_id")
        else:
            df.to_csv(path, index=False)
    else:
        df.to_parquet(path, index=index)


def save_genes_df_for_debug(genes_df: pd.DataFrame, path: str, output_format: str) -> None:
    """Write ``genes_df`` (post-load, same columns as CSV / Pyrodigal contract) for debugging."""
    write_dataframe_output(path, output_format, genes_df, index=False)
    logger.debug(f"[debug] Saved genes table ({len(genes_df)} rows) to {path} ({output_format.lower()})")


def post_output_path(primary_path: str, output_format: str) -> str:
    """Path for the post-projection file when ``--output-mode both`` (same format as primary)."""
    root, ext = os.path.splitext(primary_path)
    fmt = output_format.lower()
    if fmt == "csv":
        return f"{root}_post.csv"
    if ext.lower() in (".parquet", ".pqt"):
        return f"{root}_post{ext}"
    return primary_path + "_post.parquet"


def project_pre_to_post(model: GeneCLR, pre_embeddings: np.ndarray, device: str) -> np.ndarray:
    """Apply ``model.context_projection`` to pre-projection embeddings (full gene matrix)."""
    with torch.no_grad():
        pre_t = torch.from_numpy(pre_embeddings).to(device)
        post_t = model.context_projection(pre_t)
        return post_t.cpu().numpy().astype(np.float32)


def save_embeddings_matrix(
    hit_ids: Union[np.ndarray, pd.Series],
    embeddings: np.ndarray,
    path: str,
    output_format: str,
    desc: str,
) -> None:
    """Save ``(n_genes, embed_dim)`` matrix with ``hit_id`` as row index."""
    embed_dim = embeddings.shape[-1]
    columns = list(range(embed_dim))
    result_df = pd.DataFrame(embeddings, index=hit_ids, columns=columns)
    write_dataframe_output(path, output_format, result_df, index=True, index_label="hit_id")
    print(f"Saved {len(hit_ids)} x {embed_dim} {desc} embeddings to {path} ({output_format.lower()})")


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


def run_embeddings_inference_cli(
    checkpoint_path: str,
    pretrain_config_path: str,
    device: str,
    dataloader,
    window_to_gene_indices: list,
    n_genes: int,
    hit_ids: np.ndarray,
    output_path: str,
    output_format: str,
    output_mode: str,
) -> None:
    """GeneCLR embedding path: load checkpoint, window inference, optional post projection, write file(s)."""
    model_config = create_model_config(pretrain_config_path)
    model = GeneCLR.load_from_checkpoint(
        checkpoint_path,
        model_config=model_config,
        strict=False,
    )
    model = model.to(device).eval()
    context_outputs = run_inference(model, dataloader, device, return_context_pre_projection=True)
    pre_embeddings = average_overlapping_representations(
        context_outputs, window_to_gene_indices, n_genes, context_outputs.shape[-1]
    )
    if output_mode == "pre":
        save_embeddings_matrix(hit_ids, pre_embeddings, output_path, output_format, "pre-projection")
    elif output_mode == "post":
        post_embeddings = project_pre_to_post(model, pre_embeddings, device)
        save_embeddings_matrix(hit_ids, post_embeddings, output_path, output_format, "post-projection")
    else:
        save_embeddings_matrix(hit_ids, pre_embeddings, output_path, output_format, "pre-projection")
        post_embeddings = project_pre_to_post(model, pre_embeddings, device)
        save_embeddings_matrix(
            hit_ids,
            post_embeddings,
            post_output_path(output_path, output_format),
            output_format,
            "post-projection",
        )


def main():
    parser = argparse.ArgumentParser(
        description="Full-genome GeneCLR inference with overlapping windows"
    )
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="YAML: pretraining config for --inference-mode embeddings; fine-tuning config for classifier",
    )
    parser.add_argument(
        "--inference-mode",
        type=str,
        default="embeddings",
        choices=["embeddings", "classifier"],
        help="embeddings: GeneCLR + pretrain config; classifier: GeneClrForTokenClassification logits + finetuning config",
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--input",
        type=str,
        default=None,
        help="Path to CSV with columns hit_id, start, end, sequence",
    )
    input_group.add_argument(
        "--input-fasta",
        type=str,
        default=None,
        dest="input_fasta",
        help="Path to nucleotide FASTA; annotate with Pyrodigal (requires: pip install pyrodigal)",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to output file (Parquet or CSV; see --output-format)",
    )
    parser.add_argument(
        "--output-format",
        type=str,
        default="parquet",
        choices=["parquet", "csv"],
        help="File format for --output (default: parquet). For embeddings + output-mode=both, post file is <stem>_post.csv or <stem>_post.parquet",
    )
    parser.add_argument(
        "--meta-threshold",
        type=int,
        default=100_000,
        help="For --input-fasta: use metagenomic Prodigal mode when sequence length is below this (default: 100000)",
    )
    parser.add_argument("--stride", type=int, default=DEFAULT_STRIDE, help="Window stride (default: 32)")
    parser.add_argument(
        "--output-mode",
        type=str,
        default="pre",
        choices=["pre", "post", "both"],
        help="Only for --inference-mode embeddings: pre/post projection (default: pre)",
    )
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/cpu)")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for inference")
    parser.add_argument(
        "--save-genes-debug",
        type=str,
        default=None,
        metavar="PATH",
        help="If set, save the loaded genes DataFrame (hit_id, start, end, sequence) to PATH; "
        "format follows --output-format (default parquet)",
    )

    args = parser.parse_args()
    
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load genes
    if args.input_fasta:
        genes_df = load_genes_from_fasta_pyrodigal(args.input_fasta, meta_threshold=args.meta_threshold)
        input_label = args.input_fasta
    else:
        genes_df = load_genes_from_csv(args.input)
        input_label = args.input
    n_genes = len(genes_df)
    print(f"Loaded {n_genes} genes from {input_label}")
    
    if n_genes == 0:
        print("No genes to process.")
        return

    if args.save_genes_debug:
        save_genes_df_for_debug(genes_df, args.save_genes_debug, args.output_format)

    # Create overlapping windows
    windows, window_to_gene_indices = create_overlapping_windows(
        genes_df, window_size=WINDOW_SIZE, stride=args.stride
    )
    print(f"Created {len(windows)} windows (stride={args.stride})")
    
    hit_ids = genes_df["hit_id"].values

    datamodule = InferenceGeneCLRDataModule(
        fragments=windows,
        batch_size=args.batch_size,
        num_workers=0,
        device=device,
        fragment_length=WINDOW_SIZE,
    )
    datamodule.setup()
    dataloader = datamodule.val_dataloader()

    if args.inference_mode == "classifier":
        run_classifier_inference_cli(
            args.checkpoint,
            args.config,
            device,
            dataloader,
            window_to_gene_indices,
            n_genes,
            hit_ids,
            args.output,
            args.output_format,
        )
        return

    run_embeddings_inference_cli(
        args.checkpoint,
        args.config,
        device,
        dataloader,
        window_to_gene_indices,
        n_genes,
        hit_ids,
        args.output,
        args.output_format,
        args.output_mode,
    )


if __name__ == "__main__":
    main()
