# Trained model checkpoints

Produced by `python scripts/export_models.py`. Each `.pt` file is a
self-describing PyTorch checkpoint containing:

| key | meaning |
|---|---|
| `state_dict` | the trained weights (best-validation checkpoint) |
| `arch` | `baseline` or `multiscale` |
| `in_channels`, `n_classes`, `dropout`, `use_rr` | everything needed to rebuild the model |
| `classes` | `["N","S","V","F","Q"]` |
| `config` | the full experiment configuration that produced it |
| `best_val_macro_f1` | validation score of the selected epoch |

`manifest.json` lists every checkpoint with its test metrics.

## Loading one

```bash
python scripts/load_model.py models/E4_inter_proposed_seed0.pt
```

or in your own code:

```python
import torch
from src.models import build_model

ckpt  = torch.load("models/E4_inter_proposed_seed0.pt", map_location="cpu", weights_only=False)
model = build_model(ckpt["arch"], ckpt["in_channels"], ckpt["n_classes"],
                    ckpt["dropout"], use_rr=ckpt["use_rr"])
model.load_state_dict(ckpt["state_dict"])
model.eval()

# forward(beat, context, rr):
#   beat    (B, C, 360)  1 s window centred on the R peak
#   context (B, C, 360)  4 s window, decimated x4
#   rr      (B, 4)       pre-RR, post-RR, local average, prematurity ratio
```

The weights were trained with the same seeds and code path as the numbers in
`RESULTS.md`, so they correspond to the reported results.
