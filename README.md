# GalCEM

Eda Gjergo (Wuhan University) <GalacticCEM@gmail.com>

![GalCEM flowchart](/docs/figs/GalCEMdiagram.jpg "GalCEM flowchart")

## TODO

- build interpolation models based on `galcem/input/yields/lims/c15/...dat` and the corresponding .txt files
- delete/merge code from `_scratch/`
- update docs using `docs/index.rst` and `docs/components.rst` and in-code docstrings
- update `yield_interpolation/` files
- compile pkl files when init is first called instead of distributing with MANIFEST.in
- use `'%d'%my_num` format instead of `f'...'` format
- remove `pandas` req
- refine `MANIFEST.IN`
- make `verbose` option for printing output


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

## Upload to TestPyPI

upload to test repo

```
python setup.py sdist
twine check dist/*
twine upload -r testpypi dist/*
```

test package from testpypi

```
conda create --name tmp python=3.8
conda activate tmp
pip install numpy scipy pandas
pip install pandas
pip install -i https://test.pypi.org/simple/ galcem==...
python
```

then in the python console

```python
import galcem as gc
inputs = gc.Inputs()
inputs.nTimeStep = .25
oz = gc.OneZone(inputs,outdir='runs/ags1/')
oz.main()
```

back in the terminal install matplotlib

```
pip install matplotlib
python
```

now in the new python console 

```python
import galcem as gc
inputs = gc.Inputs()
inputs.nTimeStep = .25
oz = gc.OneZone(inputs,outdir='runs/ags1/')
oz.main()
oz.plots()
```

## Upload to PyPI

```
python setup.py sdist
twine check dist/*
twine upload -r testpypi dist/*
twine upload dist/*
```

