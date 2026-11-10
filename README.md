# aeot — helper functions for the AEOT practicals

Helper code for the practical sessions of *Advanced Earth Observation Techniques* (VUB).
Students do not need to read or modify it; the notebooks import it.

| Module | Practical |
| --- | --- |
| `aeot.unmixing` | Pixel unmixing (Landsat, Leuven) |

## Use in a notebook (Colab or local)

```python
!pip install -q git+https://github.com/OWNER/aeot.git@2026.1
from aeot.unmixing import *
img, X, ref = load_data(DATA_DIR, LANDSAT, REFERENCE)
```

`@2026.1` is a git tag: the notebooks of an academic year always install the same version,
even if the repository changes later.

## Releasing a version for a new academic year

```bash
git tag 2027.1
git push origin 2027.1
```

Then update the tag in the `pip install` line of the notebooks.
