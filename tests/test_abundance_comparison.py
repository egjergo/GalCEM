"""Numerical regressions for comparison plots; no evolution or run data needed."""

from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from galcem.plottingtool import Plots


class AbundanceBandTests(unittest.TestCase):
    def test_shifted_equal_length_grids_compare_at_metallicity(self):
        a = np.array([-3., -2., -1., 0.])
        b = a + 0.5
        grid, low, high = Plots._abundance_band(a, .4*a + .7, b, .4*b + 1.)
        assert_allclose(grid, [-2.5, -2., -1.5, -1., -.5, 0.])
        assert_allclose(low, .4*grid + .7, atol=1e-14)
        assert_allclose(high, .4*grid + 1., atol=1e-14)
        # Index pairing would give a spurious width of 0.5 dex, not 0.3.
        assert_allclose(high - low, .3)

    def test_unequal_sampling_and_decreasing_track(self):
        grid, low, high = Plots._abundance_band(
            [0., -2., -3.], [0., -2., -3.],
            [-2.5, -2., -1., -.5], [-1.5, -1., 0., .5])
        assert_allclose(grid, [-2.5, -2., -1., -.5])
        assert_allclose(low, grid)
        assert_allclose(high, grid + 1.)

    def test_common_grid_is_retained(self):
        x = np.array([-3., -1., 0.])
        grid, low, high = Plots._abundance_band(x, [0., 1., 2.], x, [2., 3., 4.])
        assert_array_equal(grid, x)
        assert_array_equal(low, [0., 1., 2.])
        assert_array_equal(high, [2., 3., 4.])

    def test_crossing_is_included_before_enveloping(self):
        grid, low, high = Plots._abundance_band([0., 2.], [0., 2.],
                                               [0., 2.], [2., 0.])
        assert_allclose(grid, [0., 1., 2.])
        assert_allclose(low, [0., 1., 0.])
        assert_allclose(high, [2., 1., 2.])

    def test_undefined_endpoints_and_identical_duplicates(self):
        grid, low, high = Plots._abundance_band(
            [-np.inf, -3., -3., -2., -1., np.nan],
            [np.nan, 0., 0., 1., 2., np.nan], [-3., -1.], [1., 3.])
        assert_allclose(grid, [-3., -2., -1.])
        assert_allclose(low, [0., 1., 2.])
        assert_allclose(high, [1., 2., 3.])

    def test_ambiguous_or_invalid_histories_are_rejected(self):
        cases = [
            ([-3., -1., -2., 0.], [0., 1., 2., 3.], 'nonmonotonic'),
            ([-3., -2., -2., 0.], [0., 1., 2., 3.], 'repeated'),
            ([-3., -2., -1., 0.], [0., np.nan, 2., 3.], 'interior'),
            ([-3., np.nan, -1., 0.], [0., 1., 2., 3.], 'interior'),
            ([-3., -2.], [np.nan, np.nan], 'two finite'),
            ([-3., -3.], [0., 0.], 'two distinct'),
            ([-3., -2.], [0.], 'equal-length'),
        ]
        for x, y, message in cases:
            with self.subTest(message=message, x=x):
                with self.assertRaisesRegex(ValueError, message):
                    Plots._abundance_band(x, y, [-3., 0.], [0., 0.])
        # Even identical metallicity arrays cannot license an enrichment loop.
        with self.assertRaisesRegex(ValueError, 'nonmonotonic'):
            Plots._abundance_band([-3., -1., -2.], [0., 1., 2.],
                                  [-3., -1., -2.], [0., 1., 2.])

    def test_disjoint_or_point_overlap_is_rejected(self):
        for x in ([1., 2.], [0., 1.]):
            with self.subTest(x=x):
                with self.assertRaisesRegex(ValueError, 'overlapping'):
                    Plots._abundance_band([-2., 0.], [0., 0.], x, [1., 1.])


class ComparisonPlotTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.root = Path(self.temp.name)
        # Supply plotting metadata only. Never initialize or run the solver.
        self.plotter = Plots.__new__(Plots)
        self.plotter.ZA_sorted = np.array([[1, 1], [8, 16], [26, 56]])
        self.plotter.c_class = SimpleNamespace(
            solarA09_vs_H_bymass={1: 0., 8: 0., 26: 0.},
            solarA09_vs_Fe_bymass={1: 0., 8: 0., 26: 0.})
        self.plotter.IN = SimpleNamespace(
            periodic={'elemSymb': {1: 'H', 8: 'O', 26: 'Fe'}})
        self.plotter._dir_out = str(self.root / 'baseline')
        self.plotter._dir_out_figs = str(self.root / 'figs')
        self.runs = {}
        self.write_run('baseline', [-4., -3., -2., -1., 0., .3], [0.]*6)
        a = np.array([-3., -2., -1., 0.])
        b = np.array([-2.5, -1.5, -.5])
        self.write_run('lowIMF', a, .4*a + .7)
        self.write_run('highIMF', b, .4*b + 1.)
        a = np.array([-2., -1., 0., 1.])
        b = np.array([-1.5, -.5, .5])
        self.write_run('lowSFR', a, -.2*a + .1)
        self.write_run('highSFR', b, -.2*b + .5)
        self.write_run('lowZ', [-2.7, -1.7, -.7], [0.1, 0.2, 0.3])
        self.write_run('highZ', [-2.9, -1.4, -.2, .1], [0.4, 0.3, 0.2, 0.1])

    def tearDown(self):
        plt.close('all')
        self.temp.cleanup()

    def write_run(self, name, x, y):
        # Two initial time columns test the legacy c=0 -> skip 2 convention.
        x, y = np.r_[-5., -4.5, x], np.r_[0., 0., y]
        fe = 10.**x
        masses = np.array([np.ones(len(x)), fe * 10.**y, fe])
        folder = self.root / name
        folder.mkdir(exist_ok=True)
        np.savetxt(folder / 'Mass_i.dat',
                   np.column_stack((self.plotter.ZA_sorted, masses)))
        self.runs[name] = folder

    def plot(self, **kwargs):
        options = dict(run_dirs=self.runs, elements=[8], c=0, show=False,
                       figsiz=(6, 4), setxlim=(-4, 1), setylim=(-1, 2))
        options.update(kwargs)
        return self.plotter.elem_abundance(**options)

    def test_both_rendered_bands_use_pairwise_metallicities(self):
        fig, axes = self.plot()
        ax = axes[0, 0]
        self.assertEqual(len(ax.collections), 2)
        for collection, (left, right, slope, lower, upper) in zip(
                ax.collections, [(-2.5, -.5, .4, .7, 1.),
                                 (-1.5, .5, -.2, .1, .5)]):
            vertices = collection.get_paths()[0].vertices
            x, y = vertices[:, 0], vertices[:, 1]
            assert_allclose([min(x), max(x)], [left, right])
            self.assertTrue(np.all(np.isclose(y, slope*x + lower)
                                   | np.isclose(y, slope*x + upper)))
        tracks = {line.get_label(): line for line in ax.lines}
        assert_allclose(tracks['Baseline'].get_xdata(), [-4, -3, -2, -1, 0, .3])
        assert_allclose(tracks['Lifetime variant 1'].get_xdata(), [-2.7, -1.7, -.7])
        assert_allclose(tracks['Lifetime variant 2'].get_xdata(), [-2.9, -1.4, -.2, .1])
        output = self.root / 'figs' / 'elem_abundance.pdf'
        self.assertTrue(output.read_bytes().startswith(b'%PDF'))

    def test_loop_raises_before_saving_and_tracks_preserve_time_order(self):
        x = np.array([-3., -1., -2., 0.])
        self.write_run('lowIMF', x, [0., .2, .5, .7])
        with self.assertRaisesRegex(ValueError, 'IMF band, Z=8.*nonmonotonic'):
            self.plot()
        self.assertFalse((self.root / 'figs' / 'elem_abundance.pdf').exists())
        fig, axes = self.plot(comparison='tracks')
        self.assertEqual(len(axes[0, 0].collections), 0)
        track = next(line for line in axes[0, 0].lines
                     if line.get_label() == 'IMF variant 1')
        assert_allclose(track.get_xdata(), x)
        assert_allclose(track.get_ydata(), [0., .2, .5, .7])

    def test_explicit_branch_and_optional_comparisons(self):
        self.write_run('lowIMF', [-3., -1., -2., 0.], [0., .2, .5, .7])
        selected = {key: self.runs[key] for key in ('lowIMF', 'highIMF')}
        fig, axes = self.plot(run_dirs=selected, history_slices={'lowIMF': slice(0, 2)},
                              output_path=self.root / 'custom' / 'comparison.svg')
        self.assertEqual(len(axes[0, 0].collections), 1)
        self.assertTrue((self.root / 'custom' / 'comparison.svg').is_file())

    def test_configuration_and_isotope_layout_errors(self):
        for options, message in [
                ({'run_dirs': {'lowIMF': self.runs['lowIMF']}}, 'requires both'),
                ({'run_dirs': {'typo': self.root}}, 'Unknown'),
                ({'history_slices': {'baseline': slice(None, None, -1)}}, 'contiguous'),
                ({'elements': [8.5]}, 'atomic numbers')]:
            with self.subTest(options=options):
                with self.assertRaisesRegex(ValueError, message):
                    self.plot(**options)
        path = self.runs['lowIMF'] / 'Mass_i.dat'
        masses = np.loadtxt(path)
        np.savetxt(path, masses[::-1])
        with self.assertRaisesRegex(ValueError, 'isotope rows'):
            self.plot()


if __name__ == '__main__':
    unittest.main()
