"""
Convert Pyrodigal / Prodigal gene predictions to GeneCLR inference tables.

Requires optional dependency::

    pip install pyrodigal

Example (after running Prodigal as in your pipeline)::

    dic_genes = {}  # seq_name -> pyrodigal.Genes
    # ... fill dic_genes with orf_finder.find_genes(sseq) per contig ...
    from geneclr.prodigal_io import pyrodigal_annotation_dict_to_dataframe
    genes_df = pyrodigal_annotation_dict_to_dataframe(dic_genes)
    # genes_df has columns hit_id, start, end, sequence — pass to
    # create_overlapping_windows + InferenceGeneCLRDataModule as for CSV input.

Coordinate convention: Pyrodigal uses Prodigal's 1-based, end-inclusive ``begin`` / ``end``.
GeneCLR expects ``start > end`` on the reverse strand (same as Gembase FASTA parsing).
"""

from __future__ import annotations

from typing import Any, Dict, List, TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    pass


def _require_pyrodigal():
    try:
        import pyrodigal  # noqa: F401
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "geneclr.prodigal_io requires pyrodigal. Install with: pip install pyrodigal"
        ) from e


def pyrodigal_genes_to_dataframe(
    genes,
    sequence_id: str,
    *,
    include_stop_in_translation: bool = False,
) -> pd.DataFrame:
    """
    Build a GeneCLR-compatible gene table from one Pyrodigal ``Genes`` object.

    Args:
        genes: ``pyrodigal.Genes`` from ``GeneFinder.find_genes(...)``.
        sequence_id: Contig / record name (used in ``hit_id``).
        include_stop_in_translation: If False (default), omit stop codon as ``*``
            for ESM compatibility.

    Returns:
        DataFrame with columns ``hit_id``, ``start``, ``end``, ``sequence``.
    """
    _require_pyrodigal()

    rows: List[dict] = []
    gene_list = list(genes)
    gene_list.sort(key=lambda g: (g.begin, g.end))

    for i, gene in enumerate(gene_list, start=1):
        try:
            seq = gene.translate(include_stop=include_stop_in_translation)
        except Exception:
            continue
        if not seq or not str(seq).strip():
            continue
        sequence = str(seq).strip()
        if sequence.endswith("*"):
            sequence = sequence.rstrip("*")

        b, e = int(gene.begin), int(gene.end)
        if gene.strand == -1:
            start, end = e, b
        else:
            start, end = b, e

        rows.append(
            {
                "hit_id": f"{sequence_id}_{i}",
                "start": start,
                "end": end,
                "sequence": sequence,
            }
        )

    return pd.DataFrame(rows, columns=["hit_id", "start", "end", "sequence"])


def pyrodigal_annotation_dict_to_dataframe(
    dic_genes: Dict[str, Any],
    *,
    include_stop_in_translation: bool = False,
) -> pd.DataFrame:
    """
    Concatenate tables from a ``{sequence_name: pyrodigal.Genes}`` mapping.

    ``hit_id`` values are prefixed with ``sequence_id`` so IDs are unique
    across a multifasta / multi-replicon run.

    Args:
        dic_genes: Mapping from sequence/contig name to ``pyrodigal.Genes``.
        include_stop_in_translation: Passed to :func:`pyrodigal_genes_to_dataframe`.

    Returns:
        Single DataFrame with columns ``hit_id``, ``start``, ``end``, ``sequence``.
    """
    parts: List[pd.DataFrame] = []
    for seq_id in sorted(dic_genes.keys()):
        df = pyrodigal_genes_to_dataframe(
            dic_genes[seq_id],
            seq_id,
            include_stop_in_translation=include_stop_in_translation,
        )
        if len(df) > 0:
            parts.append(df)
    if not parts:
        return pd.DataFrame(columns=["hit_id", "start", "end", "sequence"])
    return pd.concat(parts, ignore_index=True)
