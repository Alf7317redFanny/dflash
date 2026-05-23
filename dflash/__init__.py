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
    raise AttributeError(
        f"module {__name__!r} has no attribute {name!r}. "
        f"Available names: {__all__}"
    )
