"""NPBK — Neural Network Biometry-Code Converter (GOST R 52633.5-2011)."""
from .authenticator import NPBKAuthenticator
from .container import NBKContainer
from .metrics import EvaluationReport, evaluate
from .trainer import NPBKTrainer, TrainerConfig
from .types import AuthResult, QualityReport, TrainingResult, TrainingSet

__version__ = "0.1.0"
__all__ = [
    "NPBKTrainer",
    "NPBKAuthenticator",
    "NBKContainer",
    "TrainingSet",
    "TrainingResult",
    "AuthResult",
    "QualityReport",
    "EvaluationReport",
    "TrainerConfig",
    "evaluate",
]
