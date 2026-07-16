"""Baseline models for MEMOIR comparison.

Uses lazy imports to avoid requiring heavy dependencies (transformers, peft)
when only using lightweight baselines like SASRec or CL4SRec.
"""


def _lazy_import(module_path: str, class_name: str):
    import importlib
    mod = importlib.import_module(module_path, package="baselines")
    return getattr(mod, class_name)


def get_baseline(name: str):
    """Get a baseline model class by name."""
    registry = {
        "gru4rec": (".gru4rec", "GRU4Rec"),
        "sasrec": (".sasrec", "SASRec"),
        "bert4rec": (".bert4rec", "BERT4Rec"),
        "cl4srec": (".cl4srec", "CL4SRec"),
        "duorec": (".duorec", "DuoRec"),
        "cllmrec": (".cllmrec", "CLLMRec"),
        "tallrec": (".tallrec", "TALLRec"),
        "llm_ranker": (".llm_ranker", "LLMRanker"),
        "sracl": (".sracl", "SRACL"),
        "sasrec_text": (".sasrec_text", "SASRecText"),
        "unisrec": (".unisrec", "UniSRec"),
    }
    if name not in registry:
        raise ValueError(f"Unknown baseline: {name}. Choose from {list(registry)}")
    module_path, class_name = registry[name]
    return _lazy_import(module_path, class_name)


BASELINE_NAMES = ["gru4rec", "sasrec", "bert4rec", "cl4srec", "duorec", "cllmrec", "sracl", "sasrec_text", "unisrec", "tallrec", "llm_ranker"]
SEQUENTIAL_BASELINES = ["gru4rec", "sasrec", "bert4rec", "cl4srec", "duorec", "cllmrec", "sracl", "sasrec_text", "unisrec"]
LLM_BASELINES = ["tallrec", "llm_ranker"]
