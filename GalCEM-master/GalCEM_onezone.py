'''	applies to thick disk at 8 kpc '''
import time
tic = []
tic.append(time.process_time())
import numpy as np
import scipy.integrate
import scipy.interpolate as interp
from scipy.integrate import quad
from scipy.misc import derivative

import input_parameters as IN
import _classes as pc

''' Setup '''
lifetime_class = pc.Stellar_Lifetimes()
Ml = lifetime_class.s_mass[1] # Lower limit stellar masses [Msun] 
Mu = lifetime_class.s_mass[-2] # Upper limit stellar masses [Msun]

time_uniform = np.arange(IN.time_start, IN.time_end, IN.iTimeStep)
mass_uniform = np.linspace(Ml, Mu, num = IN.num_MassGrid)
# Surface density for the disk. The bulge goes as an inverse square law.
surf_density_Galaxy = IN.sd / np.exp(IN.r / IN.Reff[IN.morphology]) #sigma(t_G) before eq(7)

#lifetime_class.interp_stellar_lifetimes(metallicity)(mass_uniform) # lifetime func @ input metallicity
#lifetime_class.interp_stellar_masses(metallicity)(time_uniform) # mass func @ input metallicity

infall_class = pc.Infall(morphology=IN.morphology, time=time_uniform)
infall = infall_class.inf()

SFR_class = pc.Star_Formation_Rate(IN.SFR_option, IN.custom_SFR)
SFR = SFR_class.SFR() # Function: SFR(Mgas)

IMF_class = pc.Initial_Mass_Function(Ml, Mu, IN.IMF_option, IN.custom_IMF)
IMF = IMF_class.IMF() # Function @ input stellar mass

isotopes = pc.Isotopes()

yields_LIMS_class = pc.Yields_LIMS()
yields_LIMS_class.import_yields()
#yields_LIMS_class.yields[0][:,0] # yield tables with shape [FeH_ini][AZ_idx, mass_i]

yields_Massive_class = pc.Yields_Massive()
yields_Massive_class.import_yields()
#yields_Massive_class.yields[0,0,:,0] # yield tables with shape [FeH_ini, vel, AZ_idx, mass_i]

yields_SNIa_class = pc.Yields_SNIa()
yields_SNIa_class.import_yields()
#yields_SNIa_class.yields[:] # yield tables with shape [AZ_idx]

yields_BBN_class = pc.Yields_BBN()
yields_BBN_class.import_yields()
#yields_BBN_class.yields[:] # yield tables with shape [AZ_idx]

c_class = pc.Concentrations()
AZ_LIMS = c_class.extract_AZ_pairs_LIMS(yields_LIMS_class)
AZ_SNIa = c_class.extract_AZ_pairs_SNIa(yields_SNIa_class)
AZ_Massive = c_class.extract_AZ_pairs_Massive(yields_Massive_class)
AZ_all = np.vstack((AZ_LIMS, AZ_SNIa, AZ_Massive))
AZ_sorted = c_class.AZ_sorted(AZ_all) # 321 isotopes with 'km20', 198 w/ 'i99' # will compute over 10 million integrals and recursions
elemZ_for_metallicity = np.where(AZ_sorted[:,0]>2)[0][0] # metallicity index selection



''' Initialize tracked quantities '''
Gi = [np.zeros(len(AZ_sorted))]	# Gi Global
Mtot = np.insert(np.cumsum((infall(time_uniform)[1:] + infall(time_uniform)[:-1]) * IN.iTimeStep / 2), 0, 0)
Z = [0.] # Metallicity Global
G = [0.] # G Global
S = [0.]	# S = 1 - G Global

class Tracked_quantities:
	'''
	Quantities to be saved in the output
	'''
	def __init__(self, Gi_t):
		self.Gi_t = Gi_t
		
	def track_Gi(self):
		return Gi.append(self.Gi_t)
		
	def track_Z(self):
		return Z.append(np.sum(self.Gi_t[elemZ_for_metallicity:]))
		
	def track_G(self):
		return G.append(np.sum(self.Gi_t))
	
	def track_S(self):
		return S.append[1 - G[-1]]
	
	def track_Mtot(self):
		return Mtot.append
	
	def run(self):
		self.track_Gi()
		self.track_Z()
		self.track_G()
		self.track_S()

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
#pick_yields('LIMS', 'Na', stellar_mass_idx=0, metallicity_idx=0)
#pick_yields('Massive', 'Na', stellar_mass_idx=0, metallicity_idx=0, vel_idx=0)
#pick_yields('SNIa', 'Na')



class Convergence:
	'''
	Computes eq. (27)
	'''
	def __init__(self):
		return None

	def mapping(self):
		return None
	
	def eta_SFR(self, SFR_n, G_n):
		'''
		Equation after (24)
		'''
		# while deltai_k
		return np.divide(SFR_n, G_n)
		
	def ratio_Wi_eta(self):
		'''
		Wi / eta
		'''
		# while deltai_k
		
		return None
		
	def Ai(self, eta_SFR, Gi_n, ratio_Wi_eta):
		'''
		Equation (29)
		'''
		# while deltai_k
		return (np.exp(-eta_SFR * iTimeStep) * (Gi_n - ratio_Wi_eta) 
		        + ratio_Wi_eta)
			
	def betai(self, t, i, partial_ratio=0.5): #partial_ratio = derlog (bi=) [!!!!!!!]
		'''
		Equation (30) 
		'''
		eta_deltat = eta_SFR_n * iTimeStep
		exp_eta = np.exp(-eta_deltat) 
		part1 = (exp_eta - 1) * ratio_Wi_eta
		part2 = eta_deltat * exp_eta * (Gi_n - ratio_Wi_eta)
		return 0.5 * partial_ratio * (part1 - part2)
		
	def deltai_kplus1(self, Ai, betai, Gi_k):
		'''
		Equation (28)
		'''
		# while deltai_kplus1 >= delta_max: Gi_kplus1
		return np.divide(Gi_k - Ai, betai - Gi) 
	
	def Gi_kplus1(self, Gi_k, deltai_kplus1):
		'''
		Equation (27)
		'''
		return np.multiply(Gi_k, 1 + deltai_kplus1)


class Wi_integrand:
	'''
	lambda t': integrand
	'''
	def __init__(self, Mgas_tot, metallicity):
		self.Mgas_tot = Mgas_tot
		self.metallicity = metallicity
		self.Gyr_age = Gyr_age
		self.Mass_i_t = np.zeros(len(AZ_sorted))

	def SFR_component(self, Mgas):
		return None
	
	def IMF_component(self):
		return None
			
	def yield_component(self):
		'''
		Returns the sparse vector of all elements simultaneously.
		'''
		t_min_tprime = Gyr_age - time_uniform
		t_min_tprime = t_min_tprime[np.where(t_min_tprime > 0.)]
		if yields_switch == 'LIMS':
			llimit_lifetime = lifetime_class.interp_stellar_lifetimes(metallicity)(Ml)	
			ulimit_lifetime = 10 # [Msun]
		if yields_switch == 'Massive':
			llimit_lifetime = 10 # [Msun]
			ulimit_lifetime = lifetime_class.interp_stellar_lifetimes(metallicity)(Mu)
		Yield_i_birthtime = pick_yields(yields_switch, AZ_Symb, stellar_mass_idx = stellar_mass_idx, 
										metallicity_idx = metallicity_idx, vel_idx = vel_idx)	
		all_args = tuple()
		return np.sum(Yield_i_birthtime) 
	
	def dM_vs_dtauM_component(self, derlog = False): # [!!!!!!!] derlog test for comparison w/ mwgK09
		'''
		computes the derivative of M(tauM) w.r.t. tauM
		'''
		if derlog == False:
			lifetime_inverse_func = lifetime_class.interp_stellar_lifetimes_inverse()
			return derivative(lifetime_inverse_func)
		if derlog == True:
			return 0.5	
		
	def SNIa_FM1(self, M1):
		f_nu = lambda nu: 24 * (1 - nu)**2
		M1_min = 0.5 * IN.MBl
		M1_max = IN.MBu
		nu_min = np.max(0.5, M1 / IN.MBu)
		nu_max = np.min(1, M1 / IN.MBl)
		int_SNIa = lambda nu: f_nu(nu) * IMF(M1 / nu)
		integrand_SNIa = quad(int_SNIa, nu_min, nu_max)[0]
		return integrand_SNIa, M1_min, M1_max
		
	def Mass_i_infall(self):
		Minfall_dt = infall(Gyr_age)
		BBN_AZ_list = np.array(list(zip(yields_BBN_class.elemZ, yields_BBN_class.elemA)))
		BBN_idx = [np.where((AZ_sorted[:,0] == BBN_AZ_list[i,0]) 
					& (AZ_sorted[:,1] == BBN_AZ_list[i,1]))[0][0] 
					for i in range(len(BBN_AZ_list))]
		for i in range(len(BBN_AZ_list)):
			self.Mass_i_t[BBN_idx[i]] += yields_BBN_class.yields[i] * Minfall_dt
			
	def compute(self):
		self.Gi_infall
		return None

class Wi:
	'''
	Solves each integration item by integrating over birthtimes.
	
	Input upper and lower mass limits (to be mapped onto birthtimes)
	
	Gyr_age	(t) 	is Galactic age
	birthtime (t') 	is stellar birthtime
	lifetime (tau)	is stellar lifetime
	'''
	def __init__(self, Gyr_age, metallicity, Mgas, Mstar, l_mass, u_mass):
		self.metallicity = metallicity
		self.lifetime = time_uniform # (1)
		self.GasMass = Mgas
		self.Mstar = Mstar
		self.StarMass = lifetime_class.interp_stellar_masses(self.metallicity)(time_uniform)
		self.l_mass = l_mass
		self.u_mass = u_mass
		self.Gyr_age = Gyr_age

	def birthtime(self): # (2)
		'''
		Page 21, last sentence before Section 8.7	
		'''
		birthtime = self.Gyr_age - lifetime_class.interp_stellar_lifetimes(self.metallicity)(mass_uniform)
		return birthtime#[np.where(birthtime > 0)]
	
	def pick_IMF(self):
		return None

	def mapping(self):
		return None

	#Wi_integrand_class = Wi_integrand(self.Mstar, self.metallicity)
	#Wi_integrand_class.compute()
	
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
		

class Evolution:
	'''
	Main GCE one-zone class 
	'''
	#Tracking_class = Tracked_quantities(Gi_t)

	def mapping(self):
		return None

	def evolve():
		"..."
		#"..."
		#Tracking_class.run(Gi_vector)
		return None

def main():
	tic.append(time.process_time())
	Evolution_class = Evolution()
	Evolution_class.evolve()
	np.savetxt()
	tic.append(time.process_time())
	print("Computation time = "+str((tic[-1] - tic[-2])/60.)+" minutes.")

tic.append(time.process_time())
print('Package lodaded in '+str(1e0*(tic[-1]))+' seconds.')