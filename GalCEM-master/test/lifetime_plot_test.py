import matplotlib.pylab as plt
import numpy as np
import input_parameters as IN
from onezone import *
plt.rcParams['xtick.major.size'], plt.rcParams['ytick.major.size'] = 10, 10
plt.rcParams['xtick.minor.size'], plt.rcParams['ytick.minor.size'] = 7, 7
plt.rcParams['xtick.major.width'], plt.rcParams['ytick.major.width'] = 2, 2
plt.rcParams['xtick.minor.width'], plt.rcParams['ytick.minor.width'] = 1, 1
plt.rcParams['xtick.labelsize'], plt.rcParams['ytick.labelsize'] = 15, 15
plt.rcParams['axes.linewidth'] = 2

Zcol = ['Z0004', 'Z008', 'Z02', 'Z05']

fig = plt.figure(figsize=(8,4))
for i in range(len(Zcol)):
	plt.scatter(np.log10(IN.s_lifetimes_p98['M']), np.log10(IN.s_lifetimes_p98[Zcol[i]]/1e9), label = Zcol[i])
	plt.plot(np.log10(IN.s_lifetimes_p98['M']), np.log10(lifetime_class.stellar_lifetimes()[i](IN.s_lifetimes_p98['M'])))
plt.legend(frameon=False, loc = 'upper right', fontsize = 15)
plt.title('Portinari+98 interpolation fit crosscheck', fontsize = 15)
plt.ylabel(r'$\log_{10}(\tau(M_*))$ lifetime [Gyr]', fontsize = 15)
plt.xlabel(r'$M$ stellar mass [$M_{\odot}$]', fontsize = 15)
plt.tight_layout(rect=[0,0,1,1])
plt.show(block=False)
plt.savefig('figures/test/Portinari_lifetimes_fit_test.pdf')

fig2 = plt.figure()
plt.semilogy(lifetime_class.interp_stellar_masses(0.00001)(time_uniform), time_uniform, label = 'y: time_uniform, x: interp')
plt.semilogy(mass_uniform, lifetime_class.interp_stellar_lifetimes(0.00001)(mass_uniform), label = 'y: interp, x: mass_uniform')
plt.legend(frameon=False, loc = 'upper right', fontsize = 15)
plt.title('Reliability check of mapping mass onto lifetimes', fontsize = 15)
plt.ylabel(r'$\log_{10}(\tau(M))$ lifetime [yr]', fontsize = 15)
plt.xlabel(r'$M$ stellar mass [$M_{\odot}$]', fontsize = 15)
plt.tight_layout()
plt.show(block=False)
plt.savefig('figures/test/Portinari_lifetimes_convergence_test.pdf')
# Maybe if I fitted the lifetimes in logarithmic space, 
# the convergence would occur on a smaller iTimeStep