from .rotation_classifier import ECGRotationClassifier


def ECGClassifier() -> ECGRotationClassifier:
    return ECGRotationClassifier(r"rotation_classifier\mobilenet_small-0-21985.pt")


__all__ = ["ECGRotationClassifier"]
