# Only present as a template for the GeneCLR's context track model.

import torch
import torch.nn as nn
import os
from torch.nn import functional as F
from torch.utils.data import Dataset as TorchDataset
from torch.utils.data import DataLoader as DataLoader
from typing import Optional, Tuple, Union, Callable, List, Dict, Any
from einops import rearrange
from dataclasses import dataclass
from transformers import PreTrainedModel, PretrainedConfig, EsmTokenizer, PreTrainedTokenizerBase
from transformers.modeling_outputs import (
    ModelOutput,
    BaseModelOutputWithPastAndCrossAttentions,
    BaseModelOutputWithPoolingAndCrossAttentions,
    SequenceClassifierOutput,
    TokenClassifierOutput
)
from transformers.models.esm.modeling_esm import (
    EsmIntermediate,
    EsmOutput,
    EsmPooler,
    EsmLMHead,
    EsmSelfOutput,
    EsmClassificationHead,
)
from tqdm.auto import tqdm


@dataclass
class EsmMaskedLMOutput(ModelOutput):
    loss: Optional[torch.Tensor] = None
    logits: Optional[torch.Tensor] = None
    last_hidden_state: Optional[torch.Tensor] = None
    hidden_states: Optional[Tuple[torch.Tensor, ...]] = None
    attentions: Optional[Tuple[torch.Tensor, ...]] = None


class FastEsmConfig(PretrainedConfig):
    model_type = "fast_esm"
    def __init__(
        self,
        vocab_size: int = None,
        mask_token_id: int = None,
        pad_token_id: int = None,
        hidden_size: int = 768,
        num_hidden_layers: int = 12,
        num_attention_heads: int = 12,
        intermediate_size: int = 3072,
        hidden_dropout_prob: float = 0.1,
        attention_probs_dropout_prob: float = 0.1,
        max_position_embeddings: int = 1026,
        initializer_range: float = 0.02,
        layer_norm_eps: float = 1e-12,
        position_embedding_type: str = "absolute",
        emb_layer_norm_before: bool = None,
        **kwargs,
    ):
        super().__init__(
            pad_token_id=pad_token_id,
            mask_token_id=mask_token_id,
            **kwargs,
        )

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.position_embedding_type = position_embedding_type
        self.emb_layer_norm_before = emb_layer_norm_before
        self.tie_word_embeddings = False

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes this instance to a Python dictionary. Override the default [`~PretrainedConfig.to_dict`].

        Returns:
            `Dict[str, any]`: Dictionar y of all the attributes that make up this configuration instance,
        """
        output = super().to_dict()
        return output


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    cos = cos[:, :, : x.shape[-2], :]
    sin = sin[:, :, : x.shape[-2], :]

    return (x * cos) + (rotate_half(x) * sin)


def symmetrize(x: torch.Tensor) -> torch.Tensor:
    "Make layer symmetric in final two dimensions, used for contact prediction."
    return x + x.transpose(-1, -2)


def average_product_correct(x: torch.Tensor) -> torch.Tensor:
    "Perform average product correct, used for contact prediction."
    a1 = x.sum(-1, keepdims=True)
    a2 = x.sum(-2, keepdims=True)
    a12 = x.sum((-1, -2), keepdims=True)

    avg = a1 * a2
    avg.div_(a12)  # in-place to reduce memory
    normalized = x - avg
    return normalized


class EsmContactPredictionHead(nn.Module):
    """Performs symmetrization, apc, and computes a logistic regression on the output features"""

    def __init__(
        self,
        in_features: int,
        bias: bool = True,
        eos_idx: int = 2,
    ):
        super().__init__()
        self.in_features = in_features
        self.eos_idx = eos_idx
        self.regression = nn.Linear(in_features, 1, bias=bias)
        self.activation = nn.Sigmoid()

    def forward(self, input_ids: torch.Tensor, attentions: torch.Tensor) -> torch.Tensor:
        # remove eos token attentions
        eos_mask = input_ids.ne(self.eos_idx).to(attentions)
        eos_mask = eos_mask.unsqueeze(1) * eos_mask.unsqueeze(2)
        attentions = attentions * eos_mask[:, None, None, :, :]
        attentions = attentions[..., :-1, :-1]
        # remove cls token attentions
        attentions = attentions[..., 1:, 1:]
        batch_size, layers, heads, seqlen, _ = attentions.size()
        attentions = attentions.view(batch_size, layers * heads, seqlen, seqlen)

        # features: batch x channels x tokens x tokens (symmetric)
        attentions = attentions.to(
            self.regression.weight.device
        )  # attentions always float32, may need to convert to float16
        attentions = average_product_correct(symmetrize(attentions))
        attentions = attentions.permute(0, 2, 3, 1)
        return self.activation(self.regression(attentions).squeeze(3))


class RotaryEmbedding(torch.nn.Module):
    """
    Rotary position embeddings based on those in
    [RoFormer](https://huggingface.co/docs/transformers/model_doc/roformer). Query and keys are transformed by rotation
    matrices which depend on their relative positions.
    """

    def __init__(self, dim: int):
        super().__init__()
        # Generate and save the inverse frequency buffer (non trainable)
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2, dtype=torch.int64).float() / dim))
        inv_freq = inv_freq
        self.register_buffer("inv_freq", inv_freq)

        self._seq_len_cached = None
        self._cos_cached = None
        self._sin_cached = None

    def _update_cos_sin_tables(self, x: torch.Tensor, seq_dimension: int = 2) -> Tuple[torch.Tensor, torch.Tensor]:
        seq_len = x.shape[seq_dimension]

        # Reset the tables if the sequence length has changed,
        # or if we're on a new device (possibly due to tracing for instance)
        if seq_len != self._seq_len_cached or self._cos_cached.device != x.device:
            self._seq_len_cached = seq_len
            t = torch.arange(x.shape[seq_dimension], device=x.device).type_as(self.inv_freq)
            freqs = torch.outer(t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)

            self._cos_cached = emb.cos()[None, None, :, :].to(x.dtype)
            self._sin_cached = emb.sin()[None, None, :, :].to(x.dtype)

        return self._cos_cached, self._sin_cached

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        self._cos_cached, self._sin_cached = self._update_cos_sin_tables(k, seq_dimension=-2)

        return (
            apply_rotary_pos_emb(q, self._cos_cached, self._sin_cached),
            apply_rotary_pos_emb(k, self._cos_cached, self._sin_cached),
        )


class EsmEmbeddings(nn.Module):
    """
    Same as BertEmbeddings with a tiny tweak for positional embeddings indexing.
    """

    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        if config.emb_layer_norm_before:
            self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        else:
            self.layer_norm = None
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
        self.register_buffer(
            "position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)), persistent=False
        )

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        past_key_values_length: Optional[int] = 0,
    ):
        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)

        embeddings = inputs_embeds

        if self.layer_norm is not None:
            embeddings = self.layer_norm(embeddings)
        if attention_mask is not None:
            embeddings = (embeddings * attention_mask.unsqueeze(-1)).to(embeddings.dtype)
        return embeddings

    def create_position_ids_from_inputs_embeds(self, inputs_embeds):
        """
        We are provided embeddings directly. We cannot infer which are padded so just generate sequential position ids.

        Args:
            inputs_embeds: torch.Tensor

        Returns: torch.Tensor
        """
        input_shape = inputs_embeds.size()[:-1]
        sequence_length = input_shape[1]

        position_ids = torch.arange(
            self.padding_idx + 1, sequence_length + self.padding_idx + 1, dtype=torch.long, device=inputs_embeds.device
        )
        return position_ids.unsqueeze(0).expand(input_shape)


class EsmSelfAttention(nn.Module):
    def __init__(self, config, position_embedding_type: Optional[str] = None):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
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

        self.dropout_prob = config.attention_probs_dropout_prob
        self.position_embedding_type = position_embedding_type or getattr(
            config, "position_embedding_type", "absolute"
        )
        self.rotary_embeddings = None
        if self.position_embedding_type == "rotary":
            self.rotary_embeddings = RotaryEmbedding(dim=self.attention_head_size)

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        return rearrange(x, 'b s (h d) -> b h s d', h=self.num_attention_heads)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass for self attention.
        
        Args:
            hidden_states: Input tensor
            attention_mask: Optional attention mask
            output_attentions: Whether to return attention weights
            
        Returns:
            Output tensor and optionally attention weights
        """
        query_layer = self.transpose_for_scores(self.query(hidden_states)) * self.scale
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))

        if self.position_embedding_type == "rotary":
            query_layer, key_layer = self.rotary_embeddings(query_layer, key_layer)

        if output_attentions:
            # Manual attention computation to get attention weights
            attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
            if attention_mask is not None:
                attention_scores = attention_scores + attention_mask
            attention_probs = F.softmax(attention_scores, dim=-1)
            if self.dropout_prob > 0:
                attention_probs = F.dropout(attention_probs, p=self.dropout_prob, training=self.training)
            context_layer = torch.matmul(attention_probs, value_layer)
            context_layer = rearrange(context_layer, 'b h s d -> b s (h d)')
            return context_layer, attention_probs
        else:
            context_layer = F.scaled_dot_product_attention(
                query_layer,
                key_layer,
                value_layer,
                attn_mask=attention_mask,
                dropout_p=self.dropout_prob,
                scale=1.0
            )
            context_layer = rearrange(context_layer, 'b h s d -> b s (h d)')
            return context_layer
        

class EsmAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self = EsmSelfAttention(config)
        self.output = EsmSelfOutput(config)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass for attention layer.
        
        Args:
            hidden_states: Input tensor
            attention_mask: Optional attention mask
            output_attentions: Whether to return attention weights
            
        Returns:
            Output tensor and optionally attention weights
        """
        hidden_states_ln = self.LayerNorm(hidden_states)
        self_outputs = self.self(
            hidden_states_ln,
            attention_mask,
            output_attentions,
        )
        if output_attentions:
            attention_output, attention_weights = self_outputs
            attention_output = self.output(attention_output, hidden_states)
            return attention_output, attention_weights
        else:
            attention_output = self_outputs
            return self.output(attention_output, hidden_states)


class EsmLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = EsmAttention(config)
        self.intermediate = EsmIntermediate(config)
        self.output = EsmOutput(config)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass for transformer layer.
        
        Args:
            hidden_states: Input tensor
            attention_mask: Optional attention mask
            output_attentions: Whether to return attention weights
            
        Returns:
            Output tensor and optionally attention weights
        """
        attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            output_attentions,
        )
        if output_attentions:
            attention_output, attention_weights = attention_outputs
        else:
            attention_output = attention_outputs
            attention_weights = None

        layer_output = self.feed_forward_chunk(attention_output)
        
        if output_attentions:
            return layer_output, attention_weights
        return layer_output

    def feed_forward_chunk(self, attention_output):
        attention_output_ln = self.LayerNorm(attention_output)
        intermediate_output = self.intermediate(attention_output_ln)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class EsmEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([EsmLayer(config) for _ in range(config.num_hidden_layers)])
        self.emb_layer_norm_after = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_hidden_states: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
    ) -> BaseModelOutputWithPastAndCrossAttentions:
        """Forward pass for transformer encoder.
        
        Args:
            hidden_states: Input tensor
            attention_mask: Optional attention mask
            output_hidden_states: Whether to return all hidden states
            output_attentions: Whether to return attention weights
            
        Returns:
            BaseModelOutputWithPastAndCrossAttentions containing model outputs
        """
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        for layer_module in self.layer:
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    layer_module.__call__,
                    hidden_states,
                    attention_mask,
                    output_attentions,
                )
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask,
                    output_attentions,
                )

            if output_attentions:
                hidden_states, attention_weights = layer_outputs
                all_attentions = all_attentions + (attention_weights,)
            else:
                hidden_states = layer_outputs

        if self.emb_layer_norm_after:
            hidden_states = self.emb_layer_norm_after(hidden_states)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
        )


### Support for embedding datasets with low code
class Pooler:
    def __init__(self, pooling_types: List[str]):
        self.pooling_types = pooling_types
        self.pooling_options = {
            'mean': self.mean_pooling,
            'max': self.max_pooling,
            'min': self.min_pooling,
            'norm': self.norm_pooling,
            'prod': self.prod_pooling,
            'median': self.median_pooling,
            'std': self.std_pooling,
            'var': self.var_pooling,
            'cls': self.cls_pooling,
        }

    def mean_pooling(self, emb: torch.Tensor, attention_mask: Optional[torch.Tensor] = None): # (b, L, d) -> (b, d)
        if attention_mask is None:
            return emb.mean(dim=1)
        else:
            attention_mask = attention_mask.unsqueeze(-1)
            return (emb * attention_mask).sum(dim=1) / attention_mask.sum(dim=1)

    def max_pooling(self, emb: torch.Tensor, attention_mask: Optional[torch.Tensor] = None): # (b, L, d) -> (b, d)
        if attention_mask is None:
            return emb.max(dim=1).values
        else:
            attention_mask = attention_mask.unsqueeze(-1)
            return (emb * attention_mask).max(dim=1).values
    
    def min_pooling(self, emb: torch.Tensor, attention_mask: Optional[torch.Tensor] = None): # (b, L, d) -> (b, d)
        if attention_mask is None:
            return emb.min(dim=1).values
        else:
            attention_mask = attention_mask.unsqueeze(-1)
            return (emb * attention_mask).min(dim=1).values

    def norm_pooling(self, emb: torch.Tensor, attention_mask: Optional[torch.Tensor] = None): # (b, L, d) -> (b, d)
        if attention_mask is None:
            return emb.norm(dim=1, p=2)
        else:
            attention_mask = attention_mask.unsqueeze(-1)
            return (emb * attention_mask).norm(dim=1, p=2)

    def prod_pooling(self, emb: torch.Tensor, attention_mask: Optional[torch.Tensor] = None): # (b, L, d) -> (b, d)
        length = emb.shape[1]
        if attention_mask is None:
            return emb.prod(dim=1) / length
        else:
            attention_mask = attention_mask.unsqueeze(-1)
            return ((emb * attention_mask).prod(dim=1) / attention_mask.sum(dim=1)) / length

    def median_pooling(self, emb: torch.Tensor, attention_mask: Optional[torch.Tensor] = None): # (b, L, d) -> (b, d)
        if attention_mask is None:
            return emb.median(dim=1).values
        else:
            attention_mask = attention_mask.unsqueeze(-1)
            return (emb * attention_mask).median(dim=1).values
    
    def std_pooling(self, emb: torch.Tensor, attention_mask: Optional[torch.Tensor] = None): # (b, L, d) -> (b, d)
        if attention_mask is None:
            return emb.std(dim=1)
        else:
            attention_mask = attention_mask.unsqueeze(-1)
            return (emb * attention_mask).std(dim=1)
    
    def var_pooling(self, emb: torch.Tensor, attention_mask: Optional[torch.Tensor] = None): # (b, L, d) -> (b, d)
        if attention_mask is None:
            return emb.var(dim=1)
        else:
            attention_mask = attention_mask.unsqueeze(-1)
            return (emb * attention_mask).var(dim=1)

    def cls_pooling(self, emb: torch.Tensor, attention_mask: Optional[torch.Tensor] = None): # (b, L, d) -> (b, d)
        return emb[:, 0, :]

    def __call__(self, emb: torch.Tensor, attention_mask: Optional[torch.Tensor] = None): # [mean, max]
        final_emb = []
        for pooling_type in self.pooling_types:
            final_emb.append(self.pooling_options[pooling_type](emb, attention_mask)) # (b, d)
        return torch.cat(final_emb, dim=-1) # (b, n_pooling_types * d)


class ProteinDataset(TorchDataset):
    """Simple dataset for protein sequences."""
    def __init__(self, sequences: list[str]):
        self.sequences = sequences

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> str:
        return self.sequences[idx]


def build_collator(tokenizer) -> Callable[[list[str]], tuple[torch.Tensor, torch.Tensor]]:
    def _collate_fn(sequences: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        """Collate function for batching sequences."""
        return tokenizer(sequences, return_tensors="pt", padding='longest', pad_to_multiple_of=8)
    return _collate_fn


class EmbeddingMixin:
    def _embed(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        raise NotImplementedError

    @property
    def device(self) -> torch.device:
        """Get the device of the model."""
        return next(self.parameters()).device

    def _read_sequences_from_db(self, db_path: str) -> set[str]:
        """Read sequences from SQLite database."""
        import sqlite3
        sequences = []
        with sqlite3.connect(db_path) as conn:
            c = conn.cursor()
            c.execute("SELECT sequence FROM embeddings")
            while True:
                row = c.fetchone()
                if row is None:
                    break
                sequences.append(row[0])
        return set(sequences)

    def embed_dataset(
        self,
        sequences: List[str],
        tokenizer: PreTrainedTokenizerBase,
        batch_size: int = 2,
        max_len: int = 512,
        truncate: bool = True,
        full_embeddings: bool = False,
        embed_dtype: torch.dtype = torch.float32,
        pooling_types: List[str] = ['mean'],
        num_workers: int = 0,
        sql: bool = False,
        save: bool = True,
        sql_db_path: str = 'embeddings.db',
        save_path: str = 'embeddings.pth',
    ) -> Optional[dict[str, torch.Tensor]]:
        """Embed a dataset of protein sequences.
        
        Args:
            sequences: List of protein sequences
            batch_size: Batch size for processing
            max_len: Maximum sequence length
            full_embeddings: Whether to return full residue-wise (True) embeddings or pooled (False)
            pooling_type: Type of pooling ('mean' or 'cls')
            num_workers: Number of workers for data loading, 0 for the main process
            sql: Whether to store embeddings in SQLite database - will be stored in float32
            sql_db_path: Path to SQLite database
            
        Returns:
            Dictionary mapping sequences to embeddings, or None if sql=True

        Note:
            - If sql=True, embeddings can only be stored in float32
            - sql is ideal if you need to stream a very large dataset for training in real-time
            - save=True is ideal if you can store the entire embedding dictionary in RAM
            - sql will be used if it is True and save is True or False
            - If your sql database or .pth file is already present, they will be scanned first for already embedded sequences
            - Sequences will be truncated to max_len and sorted by length in descending order for faster processing

        Example:
            >>> embedder = EmbeddingMixin()
            >>> embedding_dict = embedder.embed_dataset(
                sequences=[
                    'MALWMRLLPLLALLALWGPDPAAA', ... # list of protein sequences
                ],
                batch_size=2, # adjust for your GPU memory
                max_len=512, # adjust for your needs
                full_embeddings=False, # if True, no pooling is performed
                embed_dtype=torch.float32, # cast to what dtype you want
                pooling_type=['mean', 'cls'], # more than one pooling type will be concatenated together
                num_workers=0, # if you have many cpu cores, we find that num_workers = 4 is fast for large datasets
                sql=False, # if True, embeddings will be stored in SQLite database
                sql_db_path='embeddings.db',
                save=True, # if True, embeddings will be saved as a .pth file
                save_path='embeddings.pth',
            )
            >>> # embedding_dict is a dictionary mapping sequences to their embeddings as tensors for .pth or numpy arrays for sql
        """
        sequences = list(set([seq[:max_len] if truncate else seq for seq in sequences]))
        sequences = sorted(sequences, key=len, reverse=True)
        hidden_size = self.config.hidden_size
        collate_fn = build_collator(tokenizer)
        device = self.device
        pooler = Pooler(pooling_types) if not full_embeddings else None

        def get_embeddings(residue_embeddings: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
            if full_embeddings or residue_embeddings.ndim == 2: # if already pooled or want residue-wise embeddings
                return residue_embeddings
            else:
                return pooler(residue_embeddings, attention_mask)

        if sql:
            import sqlite3
            conn = sqlite3.connect(sql_db_path)
            c = conn.cursor()
            c.execute('CREATE TABLE IF NOT EXISTS embeddings (sequence text PRIMARY KEY, embedding blob)')
            already_embedded = self._read_sequences_from_db(sql_db_path)
            to_embed = [seq for seq in sequences if seq not in already_embedded]
            print(f"Found {len(already_embedded)} already embedded sequences in {sql_db_path}")
            print(f"Embedding {len(to_embed)} new sequences")
            if len(to_embed) > 0:
                dataset = ProteinDataset(to_embed)
                dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, collate_fn=collate_fn, shuffle=False)
                with torch.no_grad():
                    for i, batch in tqdm(enumerate(dataloader), total=len(dataloader), desc='Embedding batches'):
                        seqs = to_embed[i * batch_size:(i + 1) * batch_size]
                        input_ids, attention_mask = batch['input_ids'].to(device), batch['attention_mask'].to(device)
                        residue_embeddings = self._embed(input_ids, attention_mask).float() # sql requires float32
                        embeddings = get_embeddings(residue_embeddings, attention_mask).cpu()
                        for seq, emb, mask in zip(seqs, embeddings, attention_mask):
                            if full_embeddings:
                                emb = emb[mask.bool()].reshape(-1, hidden_size)
                            c.execute("INSERT OR REPLACE INTO embeddings VALUES (?, ?)", 
                                    (seq, emb.cpu().numpy().tobytes()))
                        
                        if (i + 1) % 100 == 0:
                            conn.commit()
            
                conn.commit()
            conn.close()
            return None

        embeddings_dict = {}
        if os.path.exists(save_path):
            embeddings_dict = torch.load(save_path, map_location='cpu', weights_only=True)
            to_embed = [seq for seq in sequences if seq not in embeddings_dict]
            print(f"Found {len(embeddings_dict)} already embedded sequences in {save_path}")
            print(f"Embedding {len(to_embed)} new sequences")
        else:
            to_embed = sequences
            print(f"Embedding {len(to_embed)} new sequences")

        if len(to_embed) > 0:
            dataset = ProteinDataset(to_embed)
            dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, collate_fn=collate_fn, shuffle=False)
            with torch.no_grad():
                for i, batch in tqdm(enumerate(dataloader), total=len(dataloader), desc='Embedding batches'):
                    seqs = to_embed[i * batch_size:(i + 1) * batch_size]
                    input_ids, attention_mask = batch['input_ids'].to(device), batch['attention_mask'].to(device)
                    residue_embeddings = self._embed(input_ids, attention_mask)
                    embeddings = get_embeddings(residue_embeddings, attention_mask).to(embed_dtype)
                    for seq, emb, mask in zip(seqs, embeddings, attention_mask):
                        if full_embeddings:
                            emb = emb[mask.bool()].reshape(-1, hidden_size)
                        embeddings_dict[seq] = emb.cpu()

        if save:
            torch.save(embeddings_dict, save_path)

        return embeddings_dict


class FastEsmPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    config_class = FastEsmConfig
    base_model_prefix = "fastesm"
    supports_gradient_checkpointing = True
    tokenizer = EsmTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")
    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            if module.bias is not None:
                module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def get_input_embeddings(self) -> nn.Module:
        try:
            return self.embeddings.word_embeddings
        except AttributeError:
            return self.esm.embeddings.word_embeddings


class FAST_ESM_ENCODER(FastEsmPreTrainedModel, EmbeddingMixin):
    def __init__(self, config, add_pooling_layer: Optional[bool] = True, **kwargs):
        FastEsmPreTrainedModel.__init__(self, config, **kwargs)
        self.config = config
        self.embeddings = EsmEmbeddings(config)
        self.encoder = EsmEncoder(config)
        self.contact_head = EsmContactPredictionHead(
            in_features=config.num_hidden_layers * config.num_attention_heads, bias=True
        )
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _embed(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        token_embedding_output = self.embeddings(input_ids, attention_mask=attention_mask)
        batch_size, seq_length = input_ids.shape
        if attention_mask is not None:
            extended_attention_mask = attention_mask[:, None, None, :].expand(
                batch_size, 1, seq_length, seq_length
            ).bool()
        else:
            extended_attention_mask = None
        encoder_outputs = self.encoder(
            token_embedding_output,
            attention_mask=extended_attention_mask,
            output_hidden_states=False,
            output_attentions=False,
        )
        return encoder_outputs.last_hidden_state

    def predict_contacts(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        attns = self(input_ids, attention_mask=attention_mask, output_attentions=True).attentions
        attns = torch.stack(attns, dim=1)
        attns *= attention_mask.unsqueeze(1).unsqueeze(2).unsqueeze(3)
        attns *= attention_mask.unsqueeze(1).unsqueeze(2).unsqueeze(4)
        return self.contact_head(input_ids, attns)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None, # to play nice with HF adjacent packages
    ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPoolingAndCrossAttentions]:
        """Forward pass for base model.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Optional attention mask
            position_ids: Optional position IDs
            inputs_embeds: Optional input embeddings
            output_hidden_states: Whether to return all hidden states
            output_attentions: Whether to return attention weights
            
        Returns:
            Model outputs including hidden states and optionally attention weights
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        batch_size, seq_length = input_shape
        token_embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
        )

        if attention_mask is not None:
            extended_attention_mask = attention_mask[:, None, None, :].expand(
                batch_size, 1, seq_length, seq_length
            ).bool()
        else:
            extended_attention_mask = None

        encoder_outputs = self.encoder(
            token_embedding_output,
            attention_mask=extended_attention_mask,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
        )
        sequence_output = encoder_outputs.last_hidden_state

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


class FastEsmModel(FastEsmPreTrainedModel, EmbeddingMixin):
    def __init__(self, config, add_pooling_layer: Optional[bool] = True, **kwargs):
        FastEsmPreTrainedModel.__init__(self, config, **kwargs)
        self.config = config
        self.esm = FAST_ESM_ENCODER(config)
        self.pooler = EsmPooler(config) if add_pooling_layer else None
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _embed(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.esm._embed(input_ids, attention_mask)

    def predict_contacts(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return self.esm.predict_contacts(input_ids, attention_mask=attention_mask)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None, # to play nice with HF adjacent packages
        **kwargs,
    ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPoolingAndCrossAttentions]:
        """Forward pass for base model.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Optional attention mask
            position_ids: Optional position IDs
            inputs_embeds: Optional input embeddings
            output_hidden_states: Whether to return all hidden states
            output_attentions: Whether to return attention weights
            
        Returns:
            Model outputs including hidden states and optionally attention weights
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        outputs = self.esm(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
        )
        sequence_output = outputs.last_hidden_state
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class FastEsmForMaskedLM(FastEsmPreTrainedModel, EmbeddingMixin):
    def __init__(self, config, **kwargs):
        FastEsmPreTrainedModel.__init__(self, config, **kwargs)
        self.esm = FAST_ESM_ENCODER(config, add_pooling_layer=False)
        self.lm_head = EsmLMHead(config)
        self.loss_fct = nn.CrossEntropyLoss()
        self.init_weights()

    def get_output_embeddings(self):
        return self.lm_head.decoder

    def set_output_embeddings(self, new_embeddings):
        self.lm_head.decoder = new_embeddings

    def _embed(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.esm._embed(input_ids, attention_mask)

    def predict_contacts(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return self.esm.predict_contacts(input_ids, attention_mask=attention_mask)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None, # to play nice with HF adjacent packages
        **kwargs,
    ) -> Union[Tuple, EsmMaskedLMOutput]:
        outputs = self.esm(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
        )
        sequence_output = outputs.last_hidden_state
        prediction_scores = self.lm_head(sequence_output)

        loss = None
        if labels is not None:
            labels = labels.to(prediction_scores.device)
            loss = self.loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))

        return EsmMaskedLMOutput(
            loss=loss,
            logits=prediction_scores,
            last_hidden_state=sequence_output,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class FastEsmForSequenceClassification(FastEsmPreTrainedModel, EmbeddingMixin):
    def __init__(self, config, **kwargs):
        FastEsmPreTrainedModel.__init__(self, config, **kwargs)
        self.num_labels = config.num_labels
        self.config = config
        self.esm = FAST_ESM_ENCODER(config, add_pooling_layer=False)
        self.classifier = EsmClassificationHead(config)
        self.mse = nn.MSELoss()
        self.ce = nn.CrossEntropyLoss()
        self.bce = nn.BCEWithLogitsLoss()
        self.init_weights()

    def _embed(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.esm._embed(input_ids, attention_mask)

    def predict_contacts(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return self.esm.predict_contacts(input_ids, attention_mask=attention_mask)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[Tuple, SequenceClassifierOutput]:
        outputs = self.esm(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        sequence_output = outputs.last_hidden_state
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            labels = labels.to(logits.device)
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                if self.num_labels == 1:
                    loss = self.mse(logits.squeeze(), labels.squeeze())
                else:
                    loss = self.mse(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss = self.ce(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss = self.bce(logits, labels)

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class FastEsmForTokenClassification(FastEsmPreTrainedModel, EmbeddingMixin):
    def __init__(self, config, **kwargs):
        FastEsmPreTrainedModel.__init__(self, config, **kwargs)
        self.num_labels = config.num_labels
        self.esm = FAST_ESM_ENCODER(config, add_pooling_layer=False)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.loss_fct = nn.CrossEntropyLoss()
        self.init_weights()

    def _embed(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.esm._embed(input_ids, attention_mask)

    def predict_contacts(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return self.esm.predict_contacts(input_ids, attention_mask=attention_mask)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[Tuple, TokenClassifierOutput]:
        outputs = self.esm(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        sequence_output = outputs.last_hidden_state
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            labels = labels.to(logits.device)
            loss = self.loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


if __name__ == "__main__":
    """
    Test the hidden state differences between the FastEsmModel and the HF EsmModel.
    In full precision, the differences are very very small, but nonzero due to floating point issues with F.scaled_dot_product_attention.
    In Pytorch 2.5+ (and linux kernel), this implementation is very fast and uses less memory than the HF implementation.
    """
    import random
    from transformers import EsmForMaskedLM as TransformersEsmModel, EsmTokenizer

    model_paths = [
        "facebook/esm2_t6_8M_UR50D",
        "facebook/esm2_t12_35M_UR50D",
        #"facebook/esm2_t30_150M_UR50D",
        #"facebook/esm2_t33_650M_UR50D",
    ]
    canonical_amino_acids = "ACDEFGHIKLMNPQRSTVWY"
    length = 64
    seq_count = 100
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tolerances = [1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8]

    def generate_random_sequence(length: int) -> str:
        return 'M' + "".join(random.choices(canonical_amino_acids, k=length))

    print("Percentage of hidden states that are within the tolerance:")
    for model_path in model_paths:
        print(f"Testing {model_path}...")
        tokenizer = EsmTokenizer.from_pretrained(model_path)
        config = FastEsmConfig.from_pretrained(model_path)
        fast_model = FastEsmForMaskedLM(config).from_pretrained(model_path).to(device)
        print('fast model')
        print(fast_model)
        model = TransformersEsmModel.from_pretrained(model_path, token_dropout=False).to(device)
        print('transformers model')
        print(model)

        counts = [0] * len(tolerances)
        for _ in range(seq_count):
            example_seq = generate_random_sequence(length)
            fast_tokens = tokenizer(example_seq, return_tensors="pt").input_ids.to(device)
            fast_output = fast_model(fast_tokens, output_hidden_states=True).hidden_states[-1].detach().cpu()

            model_tokens = tokenizer(example_seq, return_tensors="pt").input_ids.to(device)
            model_output = model(model_tokens, output_hidden_states=True).hidden_states[-1].detach().cpu()

            for i, atol in enumerate(tolerances):
                if torch.allclose(fast_output, model_output, atol=atol):
                    counts[i] += 1

        print(f"{model_path}:")
        for i, atol in enumerate(tolerances):
            print(f"    tolerance={atol}: {counts[i] / seq_count * 100}%")
    
        model.cpu()
        fast_model.cpu()
        del model
        del fast_model
        torch.cuda.empty_cache()
