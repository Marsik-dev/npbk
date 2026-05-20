# GOST R 52633.5-2011 mandated thresholds
MIN_OWN_SAMPLES: int = 11
MIN_OTHER_SAMPLES: int = 64

# Minimum stability index ω_i for a neuron to be included in the key
MIN_STABILITY_INDEX: float = 0.6

# Minimum fraction of neurons that must pass the stability threshold
MIN_NEURON_PASS_RATE: float = 0.8

# Default design targets for FAR/FRR
TARGET_FAR: float = 1e-6
TARGET_FRR: float = 0.1

# NBK container format version
CONTAINER_VERSION: str = "1.0"
