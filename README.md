# GalCEM
## Author
Eda Gjergo (Wuhan University) <GalacticCEM@gmail.com>

![GalCEM flowchart](/docs/GalCEM_flowchart.png "GalCEM flowchart")

## Setup

```
git clone git@github.com:egjergo/GalCEM.git
cd GalCEM
conda env create -f environment.yml
conda activate gce
conda develop .
make sphinxdocs
python uses/example_use.py
```