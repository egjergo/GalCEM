# Comparing abundance histories at equal metallicity

`Plots.elem_abundance()` plots the baseline and lifetime variants on their own
`[Fe/H]` coordinates. For each IMF or SFR pair, it reports whether the sampled
metallicities are identical. Bands compare `[X/Fe]` at equal `[Fe/H]`, never at
equal time indices or on the baseline's grid.

For each element and model pair, the band uses piecewise-linear interpolation
in the plotted dex coordinates, on the union of both models' metallicity samples
inside their common finite range. Both overlap boundaries and curve crossings
are included. There is no extrapolation. The shaded region spans the two supplied
models; it is neither a confidence interval nor a guarantee that intermediate
parameter choices fall within that range. The baseline need not lie inside it.

## Generate a figure from existing runs

Use the GalCEM environment described in the repository README, with Matplotlib
installed and the preprocessing required by `Plots` completed. This method reads
existing outputs and does not run chemical evolution. `Plots` initialization
still uses the existing setup and its preprocessed yield/lifetime files.

Each directory must contain `Mass_i.dat`. The baseline also needs `inputs.pkl`
for the usual `Plots` constructor. All files must have the same isotope row
layout (`Z, A`, then masses in time order), as checked against the baseline
setup. The baseline solar abundance normalization is applied to every run.
Only compare runs with compatible conventions and intended parameter changes.

```python
from pathlib import Path
import galcem as glc
import matplotlib.pyplot as plt

root = Path('/path/to/existing/runs')
runs = {
    'baseline': root / 'baseline',
    'lowIMF': root / 'IMF1pt2',
    'highIMF': root / 'IMF1pt7',
    'lowSFR': root / 'k_SFR_0.5',
    'highSFR': root / 'k_SFR_2',
    'lowZ': root / 'lifetimeZ0003',
    'highZ': root / 'lifetimeZ06',
}
plots = glc.Plots(outdir=str(runs['baseline']) + '/')
with plt.rc_context({'font.size': 12, 'pdf.fonttype': 42, 'ps.fonttype': 42}):
    fig, axes = plots.elem_abundance(
        run_dirs=runs,
        elements=[6, 8, 12, 14, 20, 28],  # C, O, Mg, Si, Ca, Ni
        figsiz=(11, 6.5),
        setxlim=(-4, 0.5),
        setylim=(-1.5, 1.5),
        output_path='figures/galactic_abundance_patterns.pdf',
        show=False,
    )
    fig.savefig('figures/galactic_abundance_patterns.png', dpi=300,
                bbox_inches='tight')
plt.close(fig)
```

Choose the elements and axis limits for the science being shown; inspect the
full histories before restricting the view. The PDF is vector output, and the
PNG can be placed in presentation slides. The legend uses generic variant
labels: specify the parameter values in the caption or adjust the returned
figure's legend to match the supplied runs. Exporting the corrected figure may
require the history handling below before a band is scientifically defined.

With an explicit `run_dirs` mapping, omit either complete IMF/SFR pair and either
lifetime variant if those outputs are unavailable. An omitted baseline uses
the run supplied to `Plots`. With `run_dirs=None`, the historical seven relative
paths remain the defaults for compatibility. A missing file raises a file error.
`elements=None` selects all tracked elements; choose a subset if an element has
no finite solar-normalized abundance. The legacy `c` convention is unchanged:
`c=3` skips five time samples, and `c=0` skips two, in addition to the two isotope
identifier columns. This preserves the original time selection.

## Dilution loops, plateaus, and missing values

Band construction checks each retained history before interpolation:

- A wholly increasing or decreasing metallicity track is allowed. A decreasing
  track is reversed as a whole, preserving the adjacency of its samples.
- A turning point is rejected. Sorting a dilution loop would mix distinct
  evolutionary phases at the same metallicity.
- Identical consecutive `(FeH, XFe)` points are coalesced. Repeated metallicity
  with different abundances is rejected, without a numerical tolerance that
  could erase a physical plateau.
- Non-finite endpoints are excluded from the band's domain. An interior gap,
  fewer than two valid distinct metallicities, or no finite-width overlap raises
  an error. Missing values are never bridged or extrapolated.

For a multi-valued history, use `comparison='tracks'`. This draws every supplied
model on its native coordinates in time order, preserves gaps, and makes no
bands. It is also the appropriate view for disjoint metallicity ranges:

```python
fig, axes = plots.elem_abundance(
    run_dirs=runs, elements=[8, 12], figsiz=(9, 4),
    comparison='tracks', output_path='figures/abundance_tracks.pdf', show=False,
)
plt.close(fig)
```

If a particular physical phase is the intended comparison, pass explicit
`history_slices={'lowIMF': slice(start, stop), 'highIMF': slice(start2, stop2)}`
alongside `comparison='bands'`. Indices refer to the extracted histories after
the legacy initial-time exclusion. Identify corresponding phases from each
run's time history (`phys.dat`); matching numerical indices across runs is not
sufficient. Slices must be contiguous with step 1. Select every relevant run
explicitly if the baseline/lifetime curves should show the same phases.

## Numerical regression

From the repository root in the GalCEM environment:

```sh
python -m unittest discover -s tests -p 'test_abundance_comparison.py' -v
```

The tests construct synthetic histories and actual temporary `Mass_i.dat` files.
They check both rendered bands, native baseline/lifetime coordinates, shifted
and unequal grids, curve crossings, decreasing histories, explicit branches,
and rejection of ambiguous or undefined interpolation. The equal-length shifted
grid example has a true separation of 0.3 dex; the old index pairing gives
0.5 dex. This is a synthetic regression, not a measurement of the error in any
GalCEM simulation. No simulation outputs are bundled with these tests.
