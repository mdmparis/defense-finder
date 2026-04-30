"""
Inference data module for GeneCLR with on-the-fly ESM2-35M embedding.

Enables the context track to work with arbitrary genomes by embedding protein
sequences on-the-fly during inference, without requiring pre-computed embeddings.
"""

import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
import torch
from typing import Optional, List, Dict, Any, Union
from transformers import AutoModel, AutoTokenizer

# Gene structure: dict with keys: sequence, start, end, strand (optional)
# - sequence: raw protein sequence (required for on-the-fly embedding)
# - start, end: genomic coordinates (start > end implies reverse strand)
# - strand: optional; can be derived from start/end if omitted
# Fragment: list of 64 genes
GeneDict = Dict[str, Any]
Fragment = List[GeneDict]

# Synthyra/ESM2-35M requires entrypoint_setup; use Facebook ESM2 as fallback (same 480-dim output)
ESM2_MODEL_PATH = "facebook/esm2_t12_35M_UR50D"
ESM2_TOKENIZER_PATH = "facebook/esm2_t12_35M_UR50D"
ESM2_MAX_LENGTH = 1024
ESM2_TRUNCATE_LENGTH = 1022  # ESM2 tokenizer adds special tokens


class GeneFragmentSequenceDataset(Dataset):
    """Dataset that holds a list of gene fragments for inference.
    
    Each fragment is a list of 64 genes. Each gene is a dict with:
        - sequence: str (raw protein sequence)
        - start: int (start position)
        - end: int (end position)
        - strand: optional; if omitted, strand is implicit (start > end = reverse)
    """
    def __init__(self, fragments: List[Fragment]):
        self.fragments = fragments

    def __len__(self):
        return len(self.fragments)

    def __getitem__(self, idx):
        return self.fragments[idx]


class InferenceGeneFragmentCollate:
    """
    Collate function for inference that embeds protein sequences on-the-fly using ESM2-35M.
    
    Accepts batches of fragments (each fragment = list of 64 genes with sequence, start, end).
    Returns the same format as GeneFragmentCollate for compatibility with the model.
    """
    
    def __init__(
        self,
        esm_model_path: str = ESM2_MODEL_PATH,
        tokenizer_path: str = ESM2_TOKENIZER_PATH,
        device: str = "cpu",
        seq_len: int = 64,
    ):
        self.esm_model_path = esm_model_path
        self.tokenizer_path = tokenizer_path
        self.device = device
        self.seq_len = seq_len
        
        self._esm_model = None
        self._tokenizer = None

    def _load_esm(self):
        """Lazy load ESM2 model and tokenizer."""
        if self._esm_model is not None:
            return

        dtype = (
            torch.bfloat16
            if self.device != "cpu" and torch.cuda.is_bf16_supported()
            else torch.float32
        )
        self._tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path)
        self._esm_model = AutoModel.from_pretrained(
            self.esm_model_path,
            trust_remote_code=True,
            dtype=dtype,
        ).eval().to(self.device)

    def _truncate_sequence(self, seq: str) -> str:
        """Truncate sequence to fit ESM2 max length."""
        if len(seq) > ESM2_TRUNCATE_LENGTH:
            return seq[:ESM2_TRUNCATE_LENGTH]
        return seq

    def _embed_sequences(self, sequences: List[str]) -> torch.Tensor:
        """
        Embed a batch of protein sequences using ESM2-35M with mean pooling.
        
        Args:
            sequences: List of protein sequences (amino acid strings)
            
        Returns:
            Tensor of shape (batch_size, 480) - mean-pooled embeddings
        """
        self._load_esm()
        
        truncated = [self._truncate_sequence(s) for s in sequences]
        tokenized = self._tokenizer(
            truncated,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=ESM2_MAX_LENGTH,
        )
        
        input_ids = tokenized["input_ids"].to(self.device)
        attention_mask = tokenized["attention_mask"].to(self.device)
        
        with torch.no_grad():
            outputs = self._esm_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
        
        # Mean pool over sequence length (excluding padding)
        # last_hidden_state: (B, seq_len, 480)
        hidden = outputs.last_hidden_state
        mask_expanded = attention_mask.unsqueeze(-1).expand(hidden.size()).float()
        sum_hidden = (hidden * mask_expanded).sum(dim=1)
        sum_mask = attention_mask.sum(dim=1, keepdim=True).float().clamp(min=1e-9)
        pooled = sum_hidden / sum_mask
        
        return pooled.float()

    def _compute_distance_tensor(
        self,
        starts: torch.Tensor,
        ends: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute pairwise distance tensor from start/end positions.
        
        Same 6 features as GeneFragmentCollate:
        - start_diff, end_diff, start_diff_abs, end_diff_abs
        - starts (broadcasted), ends (broadcasted)
        """
        start_diff = starts.unsqueeze(2) - starts.unsqueeze(1)
        end_diff = ends.unsqueeze(2) - ends.unsqueeze(1)
        start_diff_abs = torch.abs(start_diff)
        end_diff_abs = torch.abs(end_diff)
        
        distance_tensor = torch.stack([
            start_diff,
            end_diff,
            start_diff_abs,
            end_diff_abs,
            starts.unsqueeze(2).expand(-1, -1, self.seq_len),
            ends.unsqueeze(2).expand(-1, -1, self.seq_len),
        ], dim=-1)
        
        return distance_tensor

    def __call__(self, batch_fragments: List[Fragment]) -> Dict[str, torch.Tensor]:
        """
        Process a batch of fragments and return model-ready tensors.
        
        Args:
            batch_fragments: List of fragments, each a list of 64 gene dicts
            
        Returns:
            Dict with keys: embeddings, pairwise_distances, attention_mask,
            missing_embeddings_mask, random_context_mask
        """
        batch_size = len(batch_fragments)
        
        # Flatten: collect all sequences, positions, and padding flags (batch_size * seq_len total)
        all_sequences = []
        all_starts = []
        all_ends = []
        all_is_padding = []
        
        for fragment in batch_fragments:
            n_genes = len(fragment)
            for i in range(self.seq_len):
                if i < n_genes:
                    gene = fragment[i]
                    seq = gene.get("sequence", "")
                    all_sequences.append(seq)  # Assume genes have valid sequences
                    all_starts.append(gene.get("start", 0))
                    all_ends.append(gene.get("end", 0))
                    all_is_padding.append(False)
                else:
                    all_sequences.append(None)  # Padding: will use zeros
                    all_starts.append(0)
                    all_ends.append(0)
                    all_is_padding.append(True)
        
        # Embed only real sequences (padding positions get zero vectors)
        real_sequences = [s for s in all_sequences if s is not None]
        real_indices = [i for i, s in enumerate(all_sequences) if s is not None]

        if real_sequences:
            sub_batch_size = 128
            all_embeddings = []
            for i in range(0, len(real_sequences), sub_batch_size):
                chunk = real_sequences[i : i + sub_batch_size]
                chunk_emb = self._embed_sequences(chunk)
                all_embeddings.append(chunk_emb)
            real_embeddings = torch.cat(all_embeddings, dim=0)
        else:
            real_embeddings = torch.empty(0, 480)

        # Build full embedding tensor: real positions from ESM2, padding = zeros
        embed_dim = 480
        embeddings = torch.zeros(batch_size * self.seq_len, embed_dim, dtype=torch.float32)
        if len(real_indices) > 0:
            embeddings[real_indices] = real_embeddings.cpu()

        # Reshape to (batch_size, seq_len, 480)
        embeddings = embeddings.view(batch_size, self.seq_len, -1)
        
        # Build starts/ends tensors
        starts = torch.tensor(all_starts, dtype=torch.float32).view(batch_size, self.seq_len)
        ends = torch.tensor(all_ends, dtype=torch.float32).view(batch_size, self.seq_len)
        
        # Missing mask: True where position was padding (no real gene)
        missing_embeddings_mask = torch.tensor(
            all_is_padding,
            dtype=torch.bool
        ).view(batch_size, self.seq_len)
        
        # Attention mask: True for valid genes, False for padding
        attention_mask = ~missing_embeddings_mask
        
        # Distance tensor
        distance_tensor = self._compute_distance_tensor(starts, ends)
        
        # For inference: no random masking
        random_context_mask = torch.ones(batch_size, self.seq_len, dtype=torch.bool)
        
        return {
            "embeddings": embeddings,
            "pairwise_distances": distance_tensor,
            "attention_mask": attention_mask,
            "missing_embeddings_mask": missing_embeddings_mask,
            "random_context_mask": random_context_mask,
            "weights": None,
            "labels": None,
            "group_strings": None,
        }


class InferenceGeneCLRDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for inference with on-the-fly ESM2 embedding.
    
    Accepts a list of gene fragments (each fragment = 64 genes with sequence, start, end)
    and provides a DataLoader that embeds sequences on-the-fly.
    """
    
    def __init__(
        self,
        fragments: List[Fragment],
        esm_model_path: str = ESM2_MODEL_PATH,
        tokenizer_path: str = ESM2_TOKENIZER_PATH,
        batch_size: int = 1,
        num_workers: int = 0,
        device: str = "cpu",
        fragment_length: int = 64,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.fragments = fragments

    def setup(self, stage: Optional[str] = None):
        pass

    def _filter_none_collate(self, collate_fn):
        def wrapper(batch):
            batch = [item for item in batch if item is not None]
            if not batch:
                return None
            return collate_fn(batch)
        return wrapper

    def _make_dataloader(self, fragments: List[Fragment], shuffle: bool = False):
        dataset = GeneFragmentSequenceDataset(fragments)
        collate_fn = InferenceGeneFragmentCollate(
            esm_model_path=self.hparams.esm_model_path,
            tokenizer_path=self.hparams.tokenizer_path,
            device=self.hparams.device,
            seq_len=self.hparams.fragment_length,
        )
        filter_none_collate = self._filter_none_collate(collate_fn)
        
        return DataLoader(
            dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=shuffle,
            pin_memory=True,
            collate_fn=filter_none_collate,
            persistent_workers=self.hparams.num_workers > 0,
        )

    def train_dataloader(self):
        return self._make_dataloader(self.fragments, shuffle=False)

    def val_dataloader(self):
        return self._make_dataloader(self.fragments, shuffle=False)

    def test_dataloader(self):
        return self._make_dataloader(self.fragments, shuffle=False)
