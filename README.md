# npbk — Neural Network Biometry-Code Converter

Implementation of **GOST R 52633.5-2011** — the Russian national standard for automatic training of neural network biometry-to-code converters.

## What it does

Converts a high-dimensional biometric feature vector (e.g. facial expression features) into a stable cryptographic binary code:
- **Own** biometrics → deterministic binary code (same person, same emotion → same code)
- **Other** biometrics → random code (different person → unpredictable)

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Usage

```python
from npbk import NPBKTrainer, NPBKAuthenticator, NBKContainer, TrainingSet
import numpy as np

# Training
own = np.random.randn(15, 256)    # ≥11 own-class samples
other = np.random.randn(100, 256) # ≥64 diverse other-class samples
result = NPBKTrainer().train(TrainingSet(own, other))
container = NBKContainer.from_result(result, feature_dim=256)
container.save("user.nbk")

# Authentication
container = NBKContainer.load("user.nbk")
auth = NPBKAuthenticator(container).authenticate(test_vector)
print("Accepted:", auth.accepted, "| Hamming:", auth.hamming_distance)
```

## Tests

```bash
pytest tests/ -v
```

## GOST compliance

See [GOST_COMPLIANCE.md](GOST_COMPLIANCE.md) for mapping of each class/method to GOST R 52633.5-2011 sections.
