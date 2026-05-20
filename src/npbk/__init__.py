"""NPBK — Neural Network Biometry-Code Converter (GOST R 52633.5-2011)."""
from .augmentation import augment_own, augmentation_info
from .authenticator import NPBKAuthenticator
from .container import NBKContainer
from .metrics import EvaluationReport, evaluate
from .trainer import NPBKTrainer, TrainerConfig
from .types import AuthResult, LayerReport, QualityReport, TrainingResult, TrainingSet

__version__ = "0.1.0"
__all__ = [
    "NPBKTrainer",
    "NPBKAuthenticator",
    "NBKContainer",
    "TrainingSet",
    "TrainingResult",
    "AuthResult",
    "QualityReport",
    "LayerReport",
    "EvaluationReport",
    "TrainerConfig",
    "augment_own",
    "augmentation_info",
    "evaluate",
]
