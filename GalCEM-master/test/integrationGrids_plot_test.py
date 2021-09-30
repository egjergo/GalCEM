import matplotlib.pylab as plt
import numpy as np
import input_parameters as IN
from onezone import *
plt.rcParams['xtick.major.size'] = 15
plt.rcParams['xtick.major.width'] = 2
plt.rcParams['xtick.minor.size'] = 10
plt.rcParams['xtick.minor.width'] = 1
plt.rcParams['ytick.major.size'] = 15
plt.rcParams['ytick.major.width'] = 2
plt.rcParams['ytick.minor.size'] = 10
plt.rcParams['ytick.minor.width'] = 1
plt.rcParams['xtick.labelsize'] = 20
plt.rcParams['ytick.labelsize'] = 20
plt.rcParams['axes.linewidth'] = 2

age_idx = 1000
metallicity = 0.002 # e.g. subsolar
Wi_class = Wi(metallicity, age_idx)

fig = plt.figure(figsize=(10,7))
ax = fig.add_subplot(111)
ax2 = ax.twinx()
ax.plot(np.arange(IN.num_MassGrid), Wi_class.LIMs_birthtime_grid, color = 'red', label = r'birthtime')
ax.plot(np.arange(IN.num_MassGrid), Wi_class.LIMs_lifetime_grid, color = 'orange', label = r'$\tau_M$')
ax2.semilogy(np.arange(IN.num_MassGrid), Wi_class.LIMs_mass_grid, color='black', label = 'mass')
ax.legend(frameon=False, loc = 'upper left', fontsize = 20)
ax2.legend(frameon=False, loc = 'upper right', fontsize = 20)
ax2.set_title(r'Integration grids at $Z= %s$ and Age = %.4g Gyr' % (str(metallicity), time_chosen[age_idx]), fontsize = 20)
ax.set_ylabel(r'time [Gyr]', fontsize = 20)
ax2.set_ylabel(r'mass [$M_{\odot}$]', fontsize = 20)
ax.set_xlabel(r'grid index', fontsize = 20)
plt.tight_layout()
plt.savefig('figures/test/integrationGrids_fit_test.pdf')
plt.show(block=False)
