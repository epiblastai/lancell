import lancell.builtins  # noqa: F401  # register built-in specs
import lancell.codecs.bitpacking  # noqa: F401  # register bitpacking codec

__all__ = [
    # Atlas
    "RaggedAtlas",
    # Query
    "AtlasQuery",
    # Dataloader
    "CellDataset",
    "MultimodalCellDataset",
    "SparseBatch",
    "DenseBatch",
    "MultimodalBatch",
    "make_loader",
    "sparse_to_dense_collate",
    # Samplers
    "CellSampler",
    # Ingestion
    "add_from_anndata",
    "add_anndata_batch",
    "add_csc",
    # Schema
    "LancellBaseSchema",
    "FeatureBaseSchema",
    "DatasetRecord",
]


def __getattr__(name: str):
    """Lazy imports for public API to avoid heavy import costs at init."""
    _import_map = {
        "RaggedAtlas": "lancell.atlas",
        "AtlasQuery": "lancell.query",
        "CellDataset": "lancell.dataloader",
        "MultimodalCellDataset": "lancell.dataloader",
        "SparseBatch": "lancell.dataloader",
        "DenseBatch": "lancell.dataloader",
        "MultimodalBatch": "lancell.dataloader",
        "make_loader": "lancell.dataloader",
        "sparse_to_dense_collate": "lancell.dataloader",
        "CellSampler": "lancell.sampler",
        "add_from_anndata": "lancell.ingestion",
        "add_anndata_batch": "lancell.ingestion",
        "add_csc": "lancell.ingestion",
        "LancellBaseSchema": "lancell.schema",
        "FeatureBaseSchema": "lancell.schema",
        "DatasetRecord": "lancell.schema",
    }
    if name in _import_map:
        import importlib

        module = importlib.import_module(_import_map[name])
        return getattr(module, name)
    raise AttributeError(f"module 'lancell' has no attribute {name!r}")
