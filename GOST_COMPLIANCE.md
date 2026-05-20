# GOST R 52633.5-2011 Compliance Map

| GOST Section | Implementation |
|---|---|
| 6.2 — First layer weight computation: `μᵢ = E_own[vᵢ] / σ_own[vᵢ]` | `layer1.py:Layer1.fit()` |
| 6.2 — Sign assignment: `sign_i = sign(E_own[vᵢ] - E_other[vᵢ])` | `layer1.py:Layer1.fit()` |
| 6.2 — Threshold: `tᵢ = μᵢ · E_own[vᵢ]` | `layer1.py:Layer1.fit()` |
| 6.3 — Second layer error correction | `layer2.py:Layer2` |
| 6.4 — Stability index: `ωᵢ = 2·\|0.5 − P(bᵢ=1\|Own)\|` | `metrics.py:stability_index()` |
| 6.5 — Minimum Own samples ≥ 11 | `constants.py:MIN_OWN_SAMPLES`, `trainer.py:validate()` |
| 6.5 — Minimum Other samples ≥ 64 | `constants.py:MIN_OTHER_SAMPLES`, `trainer.py:validate()` |
| 6.6 — FAR/FRR prediction | `metrics.py:predict_far()`, `predict_frr()` |
| 6.7 — Optimal Hamming threshold | `metrics.py:optimal_hamming_threshold()` |
| 7.1 — Neural Biometric Container (НБК) | `container.py:NBKContainer` |
| 7.2 — No raw biometrics in container | `container.py` — only weights stored |
