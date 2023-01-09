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
