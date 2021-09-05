import matplotlib.pylab as plt
import numpy as np
import input_parameters as IN
from portinari98_unified import *

Zcol = ['Z0004', 'Z008', 'Z02', 'Z05']

fig = plt.figure(figsize=(10,7))
for i in range(len(Zcol)):
	plt.scatter(np.log10(IN.s_lifetimes_p98['M']), np.log10(IN.s_lifetimes_p98[Zcol[i]]), label = Zcol[i])
	plt.plot(np.log10(IN.s_lifetimes_p98['M']), np.log10(lifetime_class.stellar_lifetimes()[i](IN.s_lifetimes_p98['M'])*1e9))
plt.legend(frameon=False, loc = 'upper right', fontsize = 15)
plt.title('Portinari+98 interpolation fit crosscheck', fontsize = 15)
plt.ylabel(r'$\log_{10}(\tau(M))$ lifetime [yr]', fontsize = 15)
plt.xlabel(r'$M$ stellar mass [$M_{\odot}$]', fontsize = 15)
plt.savefig('figures/test/Portinari_lifetimes_fit_test.pdf')
plt.show(block=False)

fig2 = plt.figure()
plt.semilogy(lifetime_class.interp_stellar_masses(0.00001)(time_uniform), time_uniform, label = 'y: time_uniform, x: interp')
plt.semilogy(mass_uniform, lifetime_class.interp_stellar_lifetimes(0.00001)(mass_uniform), label = 'y: interp, x: mass_uniform')
plt.legend(frameon=False, loc = 'upper right', fontsize = 15)
plt.title('Reliability check of mapping mass onto lifetimes', fontsize = 15)
plt.ylabel(r'$\log_{10}(\tau(M))$ lifetime [yr]', fontsize = 15)
plt.xlabel(r'$M$ stellar mass [$M_{\odot}$]', fontsize = 15)
plt.savefig('figures/test/Portinari_lifetimes_convergence_test.pdf')
plt.show(block=False)
# Maybe if I fitted the lifetimes in logarithmic space, 
# the convergence would occur on a smaller iTimeStep