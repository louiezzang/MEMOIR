from .amazon import AmazonReviewDataset
from .mind import MINDDataset
from .movielens import MovieLensDataset
from .temporal import TemporalBehaviorSegmenter
from .collator import MEMOIRCollator

DATASET_REGISTRY = {
    "amazon": AmazonReviewDataset,
    "amazon_100k": AmazonReviewDataset,
    "amazon_100k_sampled": AmazonReviewDataset,
    "mind": MINDDataset,
    "movielens": MovieLensDataset,
}


def build_dataset(name: str, **kwargs):
    if name not in DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset: {name}. Choose from {list(DATASET_REGISTRY)}")
    return DATASET_REGISTRY[name](**kwargs)
