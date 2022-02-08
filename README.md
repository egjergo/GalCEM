# GalCEM

Eda Gjergo (Wuhan University) <GalacticCEM@gmail.com>

![GalCEM flowchart](/docs/figs/GalCEMdiagram.jpg "GalCEM flowchart")

## Setup

```
git clone git@github.com:egjergo/GalCEM.git
cd GalCEM
conda env create -f environment.yml
conda activate gce
conda develop .
make sphinxdocs
python examples/mwe.py
```

## TODO

- delete/merge code from `_scratch/`
- update docs using `docs/index.rst` and `docs/components.rst` and in-code docstrings
- modify/delete `examples/GalCEM_notebook.ipynb`
- update `yield_interpolation/` files

