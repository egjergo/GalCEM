""" I only achieve simplicity with enormous effort (Clarice Lispector) """
import time
tic = []
tic.append(time.process_time())
import math as m
import numpy as np
import scipy.integrate
import scipy.interpolate as interp
from scipy.integrate import quad
from scipy.misc import derivative

import prep.inputs as IN
import classes.morphology as morph
import classes.yields as Y
from prep.setup import *

import matplotlib.pylab as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as colors
import matplotlib.ticker as ticker
plt.rcParams['xtick.major.size'], plt.rcParams['ytick.major.size'] = 10, 10
plt.rcParams['xtick.minor.size'], plt.rcParams['ytick.minor.size'] = 7, 7
plt.rcParams['xtick.major.width'], plt.rcParams['ytick.major.width'] = 2, 2
plt.rcParams['xtick.minor.width'], plt.rcParams['ytick.minor.width'] = 1, 1
plt.rcParams['xtick.labelsize'], plt.rcParams['ytick.labelsize'] = 15, 15
plt.rcParams['axes.linewidth'] = 2


def SFR(timestep_i):
	''' 
	Actual SFR employed within the integro-differential equation
	
	Feed it every timestep appropriately
	'''
	return SFR_class.SFR(Mgas=Mgas_v, Mtot=Mtot, timestep_i=timestep_i) # Function: SFR(Mgas)
		

def f_RK4(t_i, y_i, i):
	'''
	Explicit general diff eq GCE function
	'''
	return Infall_rate[i] - SFR(i)


def no_integral():
	for i in range(len(time_chosen)-1):	
		SFR_v[i+1] = SFR(i)
		Mstar_v[i+1] = Mstar_v[i] + SFR(i) * IN.iTimeStep
		Mstar_test[i+1] = Mtot[i-1] - Mgas_v[i]
		Mgas_v[i+1] = aux.RK4(f_RK4, time_chosen[i], Mgas_v[i], i, IN.iTimeStep)		


''' GCE Classes and Functions'''
def pick_yields(channel_switch, AZ_Symb, stellar_mass_idx=None, metallicity_idx=None, vel_idx=None):
	'''
	channel_switch	[str] can be 'LIMs', 'Massive', or 'SNIa'
	AZ_Symb			[str] is the element symbol, e.g. 'Na'
	
	'LIMs' requires metallicity_idx and stellar mass_idx
	'Massive' requires metallicity_idx, stellar mass_idx, and vel_idx
	'SNIa' requires None
	'''
	if channel_switch == 'LIMs':
		if stellar_mass_idx == None:
			raise Exception('You must import the mass grid')
		idx = isotopes.pick_by_Symb(yields_LIMs_class.elemZ, AZ_Symb)
		return yields_LIMs_class.yields[metallicity_idx][idx, stellar_mass_idx]
	elif channel_switch == 'Massive':
		if stellar_mass_idx == None:
			raise Exception('You must import the mass grid')
		metallicity_idx = np.digitize(self.metallicity, self.Z_bins)
		vel_idx = IN.LC18_vel_idx
		idx = isotopes.pick_by_Symb(yields_Massive_class.elemZ, AZ_Symb)
		return yields_Massive_class.yields[metallicity_idx, vel_idx, idx, stellar_mass_idx]
	elif channel_switch == 'SNIa':
		idx = isotopes.pick_by_Symb(yields_SNIa_class.elemZ, AZ_Symb)
		return yields_SNIa_class.yields[idx]


class Wi_grid:
	'''
	birthtime grid for Wi integral
	'''
	def __init__(self, metallicity, age_idx):
		self.metallicity = metallicity
		self.age_idx = age_idx
		return None

	def grids(self, Ml_lim, Mu_lim):
		'''
		Ml_lim and Mu_lim are mass limits
		They are converted to lifetimes by integr_lim() in integration_grid()
		'''
		mass_grid = np.geomspace(Ml_lim, Mu_lim, num = IN.num_MassGrid)
		lifetime_grid = lifetime_class.interp_stellar_lifetimes(self.metallicity)(mass_grid)
		birthtime_grid = time_chosen[self.age_idx] - lifetime_grid 
		positive_idx = np.where(birthtime_grid > 0.)
		return birthtime_grid[positive_idx], lifetime_grid[positive_idx], mass_grid[positive_idx]

			
class Wi:
	'''
	Solves each integration item by integrating over birthtimes.
	
	Input upper and lower mass limits (to be mapped onto birthtimes)
	
	Gyr_age	(t) 	is the Galactic age
	birthtime (t') 	is the stellar birthtime
	lifetime (tau)	is the stellar lifetime
	'''
	def __init__(self, metallicity, age_idx):
		self.metallicity = metallicity
		self.age_idx = age_idx
		#self.SFR_class = morph.Star_Formation_Rate(Mtot, age_idx, IN.SFR_option, IN.custom_SFR)
		#self.SFR = self.SFR_class.SFR() # Function: SFR()(Mtot)(Mgas)
		#self.mass_from_age = lifetime_class.interp_stellar_masses(metallicity)(time_chosen[age_idx])
		self.Wi_grid_class = Wi_grid(metallicity, age_idx)
		self.Massive_birthtime_grid, self.Massive_lifetime_grid, self.Massive_mass_grid = (
				self.Wi_grid_class.grids(IN.Ml_Massive, IN.Mu_Massive))
		self.LIMs_birthtime_grid, self.LIMs_lifetime_grid, self.LIMs_mass_grid = (
				self.Wi_grid_class.grids(IN.Ml_LIMs, IN.Mu_LIMs))
		self.SNIa_birthtime_grid, self.SNIa_lifetime_grid, self.SNIa_mass_grid = (
				self.Wi_grid_class.grids(IN.Ml_SNIa, IN.Mu_SNIa))
		return None
		
	def grid_picker(self, channel_switch, grid_type):
		'''
		Selects e.g. "self.LIMs_birthtime_grid"
		
		channel_switch:		can be 'LIMs', 'SNIa', 'Massive'
		grid_type:			can be 'birthtime', 'lifetime', 'mass'
		'''
		return self.__dict__[channel_switch+'_'+grid_type+'_grid']
	
	def SFR_component(self, birthtime_grid):
		''' Returns the interpolated SFR vector computed at the birthtime grids'''
		SFR_interp = interp.interp1d(time_chosen, SFR(Mgas_v))
		return SFR_interp(birthtime_grid)
	
	def IMF_component(self, mass_grid):
		''' Returns the IMF vector computed at the mass grids'''
		return IMF(mass_grid)
	
	def dMdtauM_component(self, lifetime_grid, derlog = False): #!!!!!!!
		''' computes the derivative of M(tauM) w.r.t. tauM '''
		if derlog == False:
			return lifetime_class.dMdtauM(self.metallicity, time_chosen)#(lifetime_grid)
		if derlog == True:
			return 0.5	
				
	def yield_component(self, channel_switch, AZ_Symb, stellar_mass_idx=None, metallicity_idx=None, vel_idx=None):
		Yield_i_birthtime = pick_yields(channel_switch, AZ_Symb, stellar_mass_idx=stellar_mass_idx, 
										metallicity_idx=metallicity_idx, vel_idx=vel_idx)
		return Yield_i_birthtime

	def mass_component(self, channel_switch, AZ_Symb, stellar_mass_idx=None, metallicity_idx=None, vel_idx=None):
		''' page 22, last eq. first column '''
		mass_grid = self.grid_picker(channel_switch, 'mass')
		lifetime_grid = self.grid_picker(channel_switch, 'lifetime')
		return self.dMdtauM_component(lifetime_grid) * self.IMF_component(mass_grid) #* self.yield_component(channel_switch, AZ_Symb) 

	def compute_simpson(self, channel_switch, AZ_Symb, stellar_mass_idx=None, metallicity_idx=None, vel_idx=None):
		'''Computes, using the Simpson rule, the integral elements of eq. (34) Portinari+98 -- for alive stars'''	
		birthtime_grid = self.grid_picker(channel_switch, 'birthtime')
		mass_comp = self.mass_component(channel_switch, AZ_Symb, 
					stellar_mass_idx=stellar_mass_idx, metallicity_idx=metallicity_idx, vel_idx=vel_idx)
		integrand = np.multiply(self.SFR_component(birthtime_grid), mass_comp)
		#if channel_switch == 'SNIa':
		#	return (1 - IN.A) * scipy.integrate.simpson(integrand) 
		#else:
		return scipy.integrate.simps(integrand, x=birthtime_grid) 
	"""
	def compute_gauss_quad_delay(self, delay_func=None):
		'''
		Computes the Gaussian Quadrature of the integral elements of
		eq. (34) Portinari+98 -- for delayed events (SNIa, NSNS)
		'''
		SFR_SNIa = lambda M1: 1
		integrand_SNIa, M1_min, M1_max = Wi_integrand_class.SNIa_FM1(M1)
		return IN.A * quad(f_nu * IMF, M1_min, M1_max)
	"""		
	def Mass_i_infall(self, n):
		Minfall_dt = infall(time_chosen[n])
		print('Infalling mass ', Minfall_dt, ' Msun at timestep idx: ', n)
		BBN_idx = c_class.R_M_i_idx(yields_BBN_class, AZ_sorted)
		for i in range(len(BBN_idx)):
			Mass_i_v[BBN_idx[i],n] = Mass_i_v[BBN_idx[i],n-1] + yields_BBN_class.yields[i] * Minfall_dt * IN.iTimeStep
			
	def compute(self, n):
		'''
		n timestep index
		'''
		total = self.Mass_i_infall(n)
		return total
	"""	
	def yield_component(self):
		'''
		Returns the sparse vector of all elements simultaneously.
		'''
		t_min_tprime = Gyr_age - time_chosen
		t_min_tprime = t_min_tprime[np.where(t_min_tprime > 0.)]
		if channel_switch == 'LIMs':
			llimit_lifetime = lifetime_class.interp_stellar_lifetimes(self.metallicity)(Ml_X)	
			ulimit_lifetime = 10 # [Msun]
		if channel_switch == 'Massive':
			llimit_lifetime = 10 # [Msun]
			ulimit_lifetime = lifetime_class.interp_stellar_lifetimes(self.metallicity)(Mu_X)
		Yield_i_birthtime = pick_yields(channel_switch, AZ_Symb, stellar_mass_idx = stellar_mass_idx, 
										metallicity_idx = metallicity_idx, vel_idx = vel_idx)	
		all_args = tuple()
		return np.sum(Yield_i_birthtime) 
	"""
	def SNIa_FM1(self, M1):
		f_nu = lambda nu: 24 * (1 - nu)**2
		M1_min = 0.5 * IN.MBl
		M1_max = IN.MBu
		nu_min = np.max(0.5, M1 / IN.MBu)
		nu_max = np.min(1, M1 / IN.MBl)
		int_SNIa = lambda nu: f_nu(nu) * IMF(M1 / nu)
		integrand_SNIa = quad(int_SNIa, nu_min, nu_max)[0]
		return integrand_SNIa, M1_min, M1_max

	def number_2(i):
		numerator = (totdens_time[i] * deriv_gasdens[i] 
							- gasdens_time[i] * deriv_totdens[i]) 
		return np.divide(numerator, np.power(totdens(time[i]),2))
		
	def number_3(i, a = 1e-5):
		return - a * SFR(time[i]) - number_2(i)
	
	def diff_Xi(Xi, rate, i, b = 1e5):
		return (totdens_time[i] * (number_3(i) * Xi + rate[i])# + b * infall_time[i])  
						/ gasdens_time[i])
		
	def find_X_r(rate):
		X_r = np.zeros(len(time))
		for i in range(len(time)-1):
			X_r[i+1] = X_r[i] + 0.002 * diff_Xi(X_r[i], rate, i)
		return X_r


class Convergence:
	'''
	Computes eq. (27)
	'''
	def __init__(self, isotope_idx, timestep):
		self.n = timestep
		self.i = isotope_idx
		#self.Gi_n = Xi_v[isotope_idx, timestep]
		return None

	#Wi_class = Wi()
	def W_i(self):
		return Wi_class.compute(self.i)
	
	def eta_SFR_func(self):
		'''
		between eq. (24) and (25)
		'''
		return np.divide(SFR_v[self.i-1], G_v[self.i-1])
	#eta_SFR = self.eta_SFR_func()
		
	def ratio_Wi_eta(self, Wi):
		return np.divide(self.W_i(), self.eta_SFR())
		return None
			
	def betai(self, t, partial_ratio=0.5): #partial_ratio = derlog (bi=) [!!!!!!!]
		'''
		Equation (30) 
		'''
		eta_deltat = eta_SFR_n * iTimeStep
		exp_eta = np.exp(-eta_deltat) 
		part1 = (exp_eta - 1) * self.ratio_Wi_eta()
		part2 = eta_deltat * exp_eta * (Gi_n - ratio_Wi_eta)
		return 0.5 * partial_ratio * (part1 - part2)
		
	def Ai(self, eta_SFR, Gi_n, ratio_Wi_eta):
		'''
		Equation (29)
		'''
		# while deltai_k
		return (np.exp(-self.eta_SFR() * IN.iTimeStep) * (Gi_n - ratio_Wi_eta) + ratio_Wi_eta)
		        
	def deltai_kplus1(self, Ai, betai, Gi_k):
		'''
		Equation (28)
		'''
		return np.divide(Gi_k - Ai, betai - Gi_k) 

	def Gi_kplus1(self, i, Ai, betai, Gi_k):
		'''
		Equation (27)
		'''
		Gi_k1 = Gi_k
		delta_i = self.deltai_kplus1(self, Ai, betai, Gi_k)
		while delta_i >= IN.delta_max:
			Gi_k1 *= (1 + delta_i)
		return Gi_k1



class Evolution:
	'''
	Main GCE one-zone class 
	'''
	def evolve():
		for n in range(1, len(time_chosen)):
			Wi_class = Wi(Z_v[n-1], t)
			Wi_class.compute(t)
		return None

""""""""""""""""""""""""""""""""""""
"                                  "
"    One Zone evolution routine    "
"                                  "
""""""""""""""""""""""""""""""""""""

def main():
	tic.append(time.process_time())
	Evolution_class = Evolution()
	Evolution_class.evolve()
	#for j in range(1, len(time_chosen)):
	#s	Wi_class.compute(j)
	tic.append(time.process_time())
	delta_computation_m = m.floor((tic[-1] - tic[-2])/60.)
	delta_computation_s = ((tic[-1] - tic[-2])%60.)
	print("Computation time = "+str(delta_computation_m)+" minutes and "+str(delta_computation_s)+" seconds.")
	print("Saving the output...")
	np.savetxt('output/phys.dat', np.column_stack((time_chosen, Mtot, Mgas_v,
			   Mstar_v, SFR_v, Z_v, G_v, S_v)), 
			   header = ' (0) time_chosen 	(1) Mtot 	(2) Mgas_v 	(3) Mstar_v 	(4) SFR_v 	(5) Z_v 	(6) G_v 	(7) S_v')
	#np.savetxt('output/Mass_i.dat', np.column_stack((AZ_sorted, Mass_i_v)), 
	#		   header = ' (0) elemZ,	(1) elemA,	(2) masses [Msun] of every isotope for every timestep')
	np.savetxt('output/X_i.dat', np.column_stack((AZ_sorted, Xi_v)), 
			   header = ' (0) elemZ,	(1) elemA,	(2) abundance mass ratios of every isotope for every timestep (normalized to solar, Asplund et al., 2009)')
	print("Your output has been saved.")
	return None

tic.append(time.process_time())
print('Package lodaded in '+str(1e0*(tic[-1]))+' seconds.')


def no_integral_plot():
	fig = plt.figure(figsize =(7, 5))
	ax = fig.add_subplot(111)
	ax2 = ax.twinx()
	ax.hlines(IN.M_inf[IN.morphology], 0, IN.age_Galaxy, label=r'$M_{gal,f}$', linewidth = 1, linestyle = '-.')
	ax.semilogy(time_chosen, Mtot, label=r'$M_{tot}$', linewidth=2)
	ax.semilogy(time_chosen, Mgas_v, label= r'$M_{gas}$', linewidth=2)
	ax.semilogy(time_chosen, Mstar_v, label= r'$M_{star}$', linewidth=2)
	ax.semilogy(time_chosen, Mstar_v + Mgas_v, label= r'$M_g + M_s$', linestyle = '--')
	ax.semilogy(time_chosen, Mstar_test, label= r'$M_{star,t}$', linewidth=2, linestyle = ':')
	ax2.semilogy(time_chosen, Infall_rate, label= r'Infall', color = 'cyan', linestyle=':', linewidth=3)
	ax2.semilogy(time_chosen, SFR_v, label= r'SFR', color = 'gray', linestyle=':', linewidth=3)
	ax.set_xlim(0,13.8)
	ax.set_ylim(1e5, 1e11)
	#ax2.set_ylim(1e5, 1e11)
	ax2.set_ylim(1e8, 1e11)
	ax.set_xlabel(r'Age [Gyr]', fontsize = 15)
	ax.set_ylabel(r'Masses [$M_{\odot}$]', fontsize = 15)
	ax2.set_ylabel(r'Rates [$M_{\odot}/yr$]', fontsize = 15)
	#ax.set_title(r'$\alpha_{SFR} = $ %.1E' % (IN.SFR_rescaling), fontsize = 15)
	ax.set_title(r'$f_{SFR} = $ %.2f' % (IN.SFR_rescaling / IN.M_inf[IN.morphology]), fontsize=15)
	ax.legend(fontsize=15, loc='lower left', frameon=False)
	ax2.legend(fontsize=15, loc='lower right', frameon=False)
	plt.tight_layout()
	plt.show(block=False)
	plt.savefig('./figures/total_physical.pdf')
	
def AZ_sorted_plot():
	fig = plt.figure(figsize =(5, 7))
	ax = fig.add_subplot(111)
	ax.grid(True, which='major', linestyle='--', linewidth=1, color='grey', alpha=0.5)
	ax.grid(True, which='minor', linestyle=':', linewidth=1, color='grey', alpha=0.5)
	ax.scatter(AZ_sorted[:,1]- AZ_sorted[:,0], AZ_sorted[:,0], marker='o', alpha=0.2, color='r', s=20)
	ax.set_ylabel(r'Proton (Atomic) Number Z', fontsize = 15)
	ax.set_xlabel(r'Neutron Number N', fontsize = 15)
	ax.set_title(r'Tracked elements', fontsize=15)
	ax.xaxis.set_major_locator(ticker.MultipleLocator(20))
	ax.yaxis.set_major_locator(ticker.MultipleLocator(20))
	ax.xaxis.set_minor_locator(ticker.MultipleLocator(5))
	ax.yaxis.set_minor_locator(ticker.MultipleLocator(5))
	ax.tick_params(width = 2, length = 10)
	ax.tick_params(width = 1, length = 5, which = 'minor')
	plt.tight_layout()
	plt.show(block=False)
	plt.savefig('./figures/test/tracked_elements.pdf')
	
def run():
	no_integral()
	no_integral_plot()
#run()