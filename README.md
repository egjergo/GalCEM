# GalCEM 
Modern IPA pronunciation: gálkɛ́m.

Eda Gjergo (Nanjing University) <GalacticCEM@gmail.com>

![GalCEM flowchart](/docs/figs/GalCEMdiagram.jpg "GalCEM flowchart")



## Setup

```
git clone git@github.com:egjergo/GalCEM.git
cd GalCEM
conda env create -f environment.yml
conda activate gce
```

## Pre-process the SNCC and LIMs yields 
(e.g., Limongi & Chieffi, 2018, and Cristallo et al., 2015)
```
python yield_interpolation/lc18/main.py
python yield_interpolation/c15/main.py
```

Under "fit_names" (for models) and "plot_names" (for figures) of the yield folder, you can choose to preprocess all ('all'), none ([]) or invididual elements (e.g., ['lc18_z8.a16.irv0.O16'])


## Run the minimum working example
```
python examples/mwe.py
```

## Run a minimum working example from a Python console:

```python
import galcem as gc
inputs = gc.Inputs()
inputs.nTimeStep = .25
oz = gc.OneZone(inputs,outdir='runs/MYDIR/')
oz.main()
```
