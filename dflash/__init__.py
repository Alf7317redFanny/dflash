# Personal fork of z-lab/dflash
# Tweaked __getattr__ to raise a more helpful error message with suggestions

__all__ = [
    "DFlashDraftModel",
    "extract_context_feature",
    "load_and_process_dataset",
    "sample",
]


def __getattr__(name):
    if name == "load_and_process_dataset":
        from .benchmark import load_and_process_dataset

        return load_and_process_dataset

    if name in {"DFlashDraftModel", "extract_context_feature", "sample"}:
        from .model import DFlashDraftModel, extract_context_feature, sample

        return {
            "DFlashDraftModel": DFlashDraftModel,
            "extract_context_feature": extract_context_feature,
            "sample": sample,
        }[name]

    # Provide a helpful hint about what's actually available
    # Also include a note about common typos I keep making (e.g. 'extract_context_features' with an 's')
    # Using difflib for smarter fuzzy matching instead of the naive substring check
    # Lowered cutoff from 0.6 (difflib default) to 0.5 so it catches more typos like
    # 'extract_context_features' -> 'extract_context_feature'
    import difflib
    close_matches = difflib.get_close_matches(name, __all__, n=3, cutoff=0.4)
    hint = f" Did you mean: {close_matches}?" if close_matches else ""
    raise AttributeError(
        f"module {__name__!r} has no attribute {name!r}.{hint} "
        f"Available names: {__all__}"
    )
