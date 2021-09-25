''' I only achieve simplicity with enormous effort (Clarice Lispector) '''
import time
tic = []
tic.append(time.process_time())
import math as m
import numpy as np
import scipy.integrate
import scipy.interpolate as interp
from scipy.integrate import quad
from scipy.misc import derivative

import input_parameters as IN
import classes.morphology as morph
import classes.yields as Y

''' Setup '''
lifetime_class = morph.Stellar_Lifetimes()
Ml = lifetime_class.s_mass[1] # Lower limit stellar masses [Msun] 
Mu = lifetime_class.s_mass[-2] # Upper limit stellar masses [Msun]
time_uniform = np.arange(IN.time_start, IN.time_end, IN.iTimeStep)
mass_uniform = np.linspace(Ml, Mu, num = IN.num_MassGrid)
# Surface density for the disk. The bulge goes as an inverse square law.
surf_density_Galaxy = IN.sd / np.exp(IN.r / IN.Reff[IN.morphology]) #sigma(t_G) before eq(7)

infall_class = morph.Infall(morphology=IN.morphology, time=time_uniform)
infall = infall_class.inf()
SFR_class = morph.Star_Formation_Rate(IN.SFR_option, IN.custom_SFR)
SFR = SFR_class.SFR() # Function: SFR(Mgas)
IMF_class = morph.Initial_Mass_Function(Ml, Mu, IN.IMF_option, IN.custom_IMF)
IMF = IMF_class.IMF() # Function @ input stellar mass

isotopes = Y.Isotopes()
yields_LIMS_class = Y.Yields_LIMS()
yields_LIMS_class.import_yields()
yields_Massive_class = Y.Yields_Massive()
yields_Massive_class.import_yields()
yields_SNIa_class = Y.Yields_SNIa()
yields_SNIa_class.import_yields()
yields_BBN_class = Y.Yields_BBN()
yields_BBN_class.import_yields()

c_class = Y.Concentrations()
AZ_LIMS = c_class.extract_AZ_pairs_LIMS(yields_LIMS_class)
AZ_SNIa = c_class.extract_AZ_pairs_SNIa(yields_SNIa_class)
AZ_Massive = c_class.extract_AZ_pairs_Massive(yields_Massive_class)
AZ_all = np.vstack((AZ_LIMS, AZ_SNIa, AZ_Massive))
AZ_sorted = c_class.AZ_sorted(AZ_all) # 321 isotopes with 'km20', 198 w/ 'i99' # will compute over 10 million integrals and recursions
AZ_Symb_list = IN.periodic['elemSymb'][c_class.AZ_Symb(AZ_sorted)]
elemZ_for_metallicity = np.where(AZ_sorted[:,0]>2)[0][0] # metallicity starting index selection

''' Initialize tracked quantities '''
Mtot = np.insert(np.cumsum((infall(time_uniform)[1:] + infall(time_uniform)[:-1]) * IN.iTimeStep / 2), 0, IN.epsilon)
Mstar_v = IN.epsilon * np.ones(len(time_uniform))	# Global
Mass_i_v = IN.epsilon * np.ones((len(AZ_sorted), len(time_uniform)))	# Global
Xi_v = IN.epsilon * np.ones((len(AZ_sorted), len(time_uniform)))	# Xi Global
SFR_v =  IN.epsilon * np.ones(len(time_uniform)) 
Z_v = IN.epsilon * np.ones(len(time_uniform)) # Metallicity Global
S_v = IN.epsilon * np.ones(len(time_uniform)) # S = 1 - G Global
G_v = IN.epsilon * np.ones(len(time_uniform)) # G Global
Mgas_v = np.multiply(G_v, Mtot)


''' GCE Classes and Functions'''
def pick_yields(yields_switch, AZ_Symb, stellar_mass_idx = None, metallicity_idx = None, vel_idx = None):
	'''
	yields_switch	[str] can be 'LIMS', 'Massive', or 'SNIa'
	AZ_Symb			[str] is the element symbol, e.g. 'Na'
	
	'LIMS' requires metallicity_idx and stellar mass_idx
	'Massive' requires metallicity_idx, stellar mass_idx, and vel_idx
	'SNIa' requires None
	'''
	if yields_switch == 'LIMS':
		idx = isotopes.pick_by_Symb(yields_LIMS_class.elemZ, AZ_Symb)
		return yields_LIMS_class.yields[metallicity_idx][idx, stellar_mass_idx]
	elif yields_switch == 'Massive':
		idx = isotopes.pick_by_Symb(yields_Massive_class.elemZ, AZ_Symb)
		return yields_Massive_class.yields[metallicity_idx, vel_idx, idx, stellar_mass_idx]
	elif yields_switch == 'SNIa':
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
		
	def integr_lim(self, mass_lim):
		''' 
		birthtime integration lower and upper limit 
		'''
		tau_mass = lifetime_class.interp_stellar_lifetimes(self.metallicity)(mass_lim)
		return time_uniform[self.age_idx] - tau_mass
		
	def integration_grid(self, l_lim, u_lim):
		''' x array in Simpson's rule for the birthtime array '''
		lower_lim = np.maximum(self.integr_lim(l_lim), IN.time_start)
		upper_lim = np.maximum(self.integr_lim(u_lim), IN.time_start)
		return np.linspace(lower_lim, upper_lim, num = IN.num_MassGrid)
		
	def grids(self, l_lim, u_lim, age_idx):
		birthtime_grid = self.integration_grid(l_lim, u_lim)
		lifetime_grid = time_uniform[age_idx] - birthtime_grid
		mass_grid = lifetime_class.interp_stellar_masses(self.metallicity)(birthtime_grid)
		return birthtime_grid, lifetime_grid, mass_grid


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
		self.Wi_grid_class = Wi_grid(metallicity, age_idx)
		self.LIMs_birthtime_grid, self.LIMs_lifetime_grid, self.LIMs_mass_grid = self.Wi_grid_class.grids(IN.Ml_LIMs, IN.Mu_LIMs, age_idx)
		self.SNIa_birthtime_grid, self.SNIa_lifetime_grid, self.SNIa_mass_grid = self.Wi_grid_class.grids(IN.Ml_SNIa, IN.Mu_SNIa, age_idx)
		self.Massive_birthtime_grid, self.Massive_lifetime_grid, self.Massive_mass_grid = self.Wi_grid_class.grids(IN.Ml_Massive, IN.Mu_Massive, age_idx)
		return None
	
	def SFR_component(self, birthtime_grid):
		''' Returns the interpolated SFR vector computed at the birthtime grids'''
		SFR_interp = interp.interp1d(time_uniform, SFR(Mgas_v))
		return SFR_interp(birthtime_grid)
	
	def IMF_component(self, mass_grid):
		''' Returns the IMF vector computed at the mass grids'''
		return IMF(mass_grid)
	
	def dMdtauM_component(self, lifetime_grid, derlog = False):
		''' computes the derivative of M(tauM) w.r.t. tauM '''
		if derlog == False:
			return lifetime_class.dMdtauM(self.metallicity)(lifetime_grid)
		if derlog == True:
			return 0.5	
				
	def yield_component(self, yields_switch):
		Yield_i_birthtime = pick_yields(yields_switch, AZ_Symb, stellar_mass_idx = stellar_mass_idx, 
										metallicity_idx = metallicity_idx, vel_idx = vel_idx)

	def compute_gauss_quad(self, Gyr_age, metallicity, yields_switch, AZ_Symb, llimit_lifetime, ulimit_lifetime, 
					stellar_mass_idx = None, metallicity_idx = None, vel_idx = None):
		'''
		Computes the Gaussian Quadrature of the integral elements of
		eq. (34) Portinari+98 -- for alive stars
		'''	
		integrand = None
		if (llimit_lifetime == IN.MBl and ulimit_lifetime == IN.MBu):
			return (1 - IN.A) * quad(integrand, llimit_lifetime, ulimit_lifetime, args=all_args)[0]
		else:
			return quad(integrand, llimit_lifetime, ulimit_lifetime, args=all_args)[0]
		
	def compute_gauss_quad_delay(self, delay_func=None):
		'''
		Computes the Gaussian Quadrature of the integral elements of
		eq. (34) Portinari+98 -- for delayed events (SNIa, NSNS)
		'''
		SFR_SNIa = lambda M1: 1
		integrand_SNIa, M1_min, M1_max = Wi_integrand_class.SNIa_FM1(M1)
		return IN.A * quad(f_nu * IMF, M1_min, M1_max)
				
	def Mass_i_infall(self, n):
		Minfall_dt = infall(time_uniform[n])
		print('Infalling mass ', Minfall_dt, ' Msun at timestep idx: ', n)
		BBN_idx = c_class.R_M_i_idx(yields_BBN_class, AZ_sorted)
		for i in range(len(BBN_idx)):
			Mass_i_v[BBN_idx[i],n] = Mass_i_v[BBN_idx[i],n-1] + yields_BBN_class.yields[i] * Minfall_dt * IN.iTimeStep
			
	def compute(self, n):
		'''
		n timestep index
		'''
		#self.Gi_infall
		total = self.Mass_i_infall(n)
		return total
			
	def yield_component(self):
		'''
		Returns the sparse vector of all elements simultaneously.
		'''
		t_min_tprime = Gyr_age - time_uniform
		t_min_tprime = t_min_tprime[np.where(t_min_tprime > 0.)]
		if yields_switch == 'LIMS':
			llimit_lifetime = lifetime_class.interp_stellar_lifetimes(self.metallicity)(Ml_X)	
			ulimit_lifetime = 10 # [Msun]
		if yields_switch == 'Massive':
			llimit_lifetime = 10 # [Msun]
			ulimit_lifetime = lifetime_class.interp_stellar_lifetimes(self.metallicity)(Mu_X)
		Yield_i_birthtime = pick_yields(yields_switch, AZ_Symb, stellar_mass_idx = stellar_mass_idx, 
										metallicity_idx = metallicity_idx, vel_idx = vel_idx)	
		all_args = tuple()
		return np.sum(Yield_i_birthtime) 

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
		for t in time_uniform:
			run_timesteps(t)
		"..."
		return None

""""""""""""""""""""""""""""""""""""
"                                  "
"    One Zone evolution routine    "
"                                  "
""""""""""""""""""""""""""""""""""""

def main():
	tic.append(time.process_time())
	#Evolution_class = Evolution()
	#Evolution_class.evolve()
	#np.savetxt()
	Wi_class = Wi()
	for j in range(1, len(time_uniform)):
		Wi_class.compute(j)
	tic.append(time.process_time())
	np.savetxt('output/phys.dat', np.column_stack((time_uniform, Mtot, Mgas_v,
			   Mstar_v, SFR_v, Z_v, G_v, S_v)), 
			   header = ' (0) time_uniform 	(1) Mtot 	(2) Mgas_v 	(3) Mstar_v 	(4) SFR_v 	(5) Z_v 	(6) G_v 	(7) S_v')
	np.savetxt('output/Mass_i.dat', np.column_stack((AZ_sorted, Mass_i_v)), 
			   header = ' (0) elemZ,	(1) elemA,	(2) masses [Msun] of every isotope for every timestep')
	np.savetxt('output/X_i.dat', np.column_stack((AZ_sorted, Xi_v)), 
			   header = ' (0) elemZ,	(1) elemA,	(2) abundance mass ratios of every isotope for every timestep (normalized to solar, Asplund et al., 2009)')
	delta_computation_m = m.floor((tic[-1] - tic[-2])/60.)
	delta_computation_s = ((tic[-1] - tic[-2])%60.)
	print("Computation time = "+str(delta_computation_m)+" minutes and "+str(delta_computation_s)+" seconds.")
	return None

tic.append(time.process_time())
print('Package lodaded in '+str(1e0*(tic[-1]))+' seconds.')