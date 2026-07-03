"""
OSNet Model Loader
==================

Loads the pretrained OSNet-x1.0 architecture from TorchReID.

The model is intended to be loaded **once** and shared across all
camera pipelines to avoid redundant memory and download time.
"""

import torch


def load_osnet(
    model_name: str = "osnet_x1_0",
    pretrained: bool = True,
    device: str = "",
) -> torch.nn.Module:
    """Load a pretrained OSNet model via TorchReID.

    Args:
        model_name: TorchReID model name.  Default is ``osnet_x1_0``
            (1.0× width, ~2.2 M parameters, 512-d features).
        pretrained: Whether to download and load pretrained weights.
        device: Target device (``"cpu"``, ``"cuda"``, or ``""`` for
            automatic selection).

    Returns:
        The loaded OSNet model in eval mode on the target device.

    Raises:
        ImportError: If ``torchreid`` is not installed.
        RuntimeError: If model loading fails.
    """
    try:
        import torchreid
    except ImportError:
        raise ImportError(
            "torchreid is required for Person Re-ID.  Install with:\n"
            "  pip install torchreid\n"
            "Or from source:\n"
            "  pip install git+https://github.com/"
            "KaiyangZhou/deep-person-reid.git"
        )

    if not device:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    try:
        model = torchreid.models.build_model(
            name=model_name,
            num_classes=1000,
            pretrained=pretrained,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Failed to build OSNet model '{model_name}': {exc}"
        ) from exc

    model = model.to(device)
    model.eval()

    param_count = sum(p.numel() for p in model.parameters()) / 1e6
    print(
        f"[ReID] Loaded {model_name} on {device} "
        f"({param_count:.1f}M parameters)"
    )

    return model
