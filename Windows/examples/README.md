# Sterling — Examples

Standalone reference implementations for planned Sterling features.
These are working, runnable scripts — not part of the main Sterling runtime.
Think of them as the prototype before the feature gets properly integrated.

Each example has its own README with setup and usage instructions.

---

## Contents

| Folder | What it is |
|---|---|
| `webcam_vision/` | USB webcam + YOLO + face recognition — replacement for HuskyLens2 |

---

## Running Examples

Each example has its own `requirements.txt`. Install into the Sterling venv:

```bash
source ster/bin/activate
pip install -r examples/<folder>/requirements.txt
python examples/<folder>/<script>.py
```

Examples are written to be self-contained — they don't depend on Sterling being running.
