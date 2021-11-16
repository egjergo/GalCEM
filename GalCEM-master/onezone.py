""" I only achieve simplicity with enormous effort (Clarice Lispector) """
import time
#from functools import cache, lru_cache
tic = []
tic.append(time.process_time())
import numpy as np
import scipy.integrate as integr
import scipy.interpolate as interp

import prep.inputs as INp
IN = INp.Inputs()
import classes.morphology as morph
import classes.yields as Y
import plots as plts
from prep.setup import *

''' GCE Classes and Functions'''

def SFR_tn(timestep_n):
	''' 
	Actual SFR employed within the integro-differential equation
	
	Feed it every timestep appropriately
	'''
	return SFR_class.SFR(Mgas=Mgas_v, Mtot=Mtot, timestep_n=timestep_n) # Function: SFR(Mgas)

def pick_yields(channel_switch, ZA_Symb, n, stellar_mass_idx=None, metallicity_idx=None, vel_idx=IN.LC18_vel_idx):
	''' !!!!!!! this function must be edited if you import yields from other authors
	channel_switch	[str] can be 'LIMs', 'Massive', or 'SNIa'
	ZA_Symb			[str] is the element symbol, e.g. 'Na'
	
	'LIMs' requires metallicity_idx and stellar mass_idx
	'Massive' requires metallicity_idx, stellar mass_idx, and vel_idx
	'SNIa' and 'BBN' require None
	'''
	if channel_switch == 'LIMs':
		if (stellar_mass_idx == None or metallicity_idx == None):
			raise Exception('You must import the stellar mass and metallicity grids')
		idx = isotope_class.pick_by_Symb(yields_LIMs_class.elemZ, ZA_Symb)
		return yields_LIMs_class.yields[metallicity_idx][idx, stellar_mass_idx]
	elif channel_switch == 'Massive':
		if (stellar_mass_idx == None or metallicity_idx == None or vel_idx == None):
			raise Exception('You must import the stellar mass, metallicity, and velocity grids')
		metallicity_idx = np.digitize(Z_v[n], yields_Massive_class.metallicity_bins)
		vel_idx = IN.LC18_vel_idx
		idx = isotope_class.pick_by_Symb(yields_Massive_class.elemZ, ZA_Symb)
		return idx#yields_Massive_class.yields[metallicity_idx, vel_idx, idx, stellar_mass_idx]
	elif channel_switch == 'SNIa':
		idx = isotope_class.pick_by_Symb(yields_SNIa_class.elemZ, ZA_Symb)
		return yields_SNIa_class.yields[idx]


class Wi_grid:
	'''
	birthtime grid for Wi integral
	'''
	def __init__(self, metallicity, age_idx):
		self.metallicity = metallicity * np.ones(IN.num_MassGrid) # !!!!!!!
		self.age_idx = age_idx
		return None

	def grids(self, Ml_lim, Mu_lim):
		'''
		Ml_lim and Mu_lim are mass limits
		They are converted to lifetimes by integr_lim() in integration_grid()
		'''
		mass_grid = np.geomspace(Ml_lim, Mu_lim, num = IN.num_MassGrid)
		lifetime_grid = lifetime_class.interp_stellar_lifetimes(mass_grid, self.metallicity)
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
	def __init__(self, age_idx):
		self.metallicity = Z_v[age_idx]
		self.age_idx = age_idx
		self.Wi_grid_class = Wi_grid(self.metallicity, age_idx)
		self.Massive_birthtime_grid, self.Massive_lifetime_grid, self.Massive_mass_grid = (
				self.Wi_grid_class.grids(IN.Ml_Massive, IN.Mu_Massive))
		self.LIMs_birthtime_grid, self.LIMs_lifetime_grid, self.LIMs_mass_grid = (
				self.Wi_grid_class.grids(IN.Ml_LIMs, IN.Mu_LIMs)) # !!!!!!! you should subtract SNIa fraction
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
		SFR_interp = interp.interp1d(time_chosen[:self.age_idx+1], SFR_v[:self.age_idx+1], fill_value='extrapolate')
		return SFR_interp(birthtime_grid)
	
	def IMF_component(self, mass_grid):
		''' Returns the IMF vector computed at the mass grids'''
		return IMF(mass_grid)
	
	def dMdtauM_component(self, lifetime_grid, derlog = IN.derlog): #!!!!!!!
		''' computes the derivative of M(tauM) w.r.t. tauM '''
		if derlog == False:
			return lifetime_class.dMdtauM(np.log10(lifetime_grid), self.metallicity*np.ones(len(lifetime_grid)))#(lifetime_grid)
		if derlog == True:
			return 0.5	
				
	def yield_component(self, channel_switch, ZA_Symb, vel_idx=IN.LC18_vel_idx):
		Yield_i_birthtime = pick_yields(channel_switch, ZA_Symb, vel_idx=vel_idx)
		return Yield_i_birthtime

	def mass_component(self, channel_switch, ZA_Symb, vel_idx=IN.LC18_vel_idx): #
		''' Portinari+98, page 22, last eq. first column '''
		mass_grid = self.grid_picker(channel_switch, 'mass')
		lifetime_grid = self.grid_picker(channel_switch, 'lifetime')
		return self.IMF_component(mass_grid) * self.dMdtauM_component(np.log10(lifetime_grid)) #* self.yield_component(channel_switch, ZA_Symb, vel_idx=vel_idx) 

	def SNIa_FM1(self, M1):
		f_nu = lambda nu: 24 * (1 - nu)**2
		M1_min = 0.5 * IN.MBl
		M1_max = IN.MBu
		nu_min = np.max(0.5, M1 / IN.MBu)
		nu_max = np.min(1, M1 / IN.MBl)
		int_SNIa = lambda nu: f_nu(nu) * IMF(M1 / nu)
		integrand_SNIa = integr.quad(int_SNIa, nu_min, nu_max)[0]
		return integrand_SNIa, M1_min, M1_max

	#def compute_iso(self, channel_switch, ZA_Symb, vel_idx=IN.LC18_vel_idx): #
	def compute(self, channel_switch, ZA_Symb, vel_idx=IN.LC18_vel_idx): #
		'''Computes, using the Simpson rule, the integral Wi 
		elements of eq. (34) Portinari+98 -- for stars that die at tn, for every i'''		
		birthtime_grid = self.grid_picker(channel_switch, 'birthtime')
		SFR_comp = self.SFR_component(birthtime_grid)
		SFR_comp[SFR_comp<0] = 0.
		mass_comp = self.mass_component(channel_switch, ZA_Symb, vel_idx=vel_idx)# 
		integrand = np.multiply(SFR_comp, mass_comp)
		#if channel_switch == 'SNIa':
		# 	integrand_SNIa, M1_min, M1_max = SNIa_FM1(self, M1)
		#	return (1 - IN.A) * integr.simps(integrand) + IN.A * integr.simps(integrand_SNIa)
		#else:
		return integr.simps(integrand, x=birthtime_grid) 

	#def compute():
	#	''' Computes the vector to be added to Mass_i_v[:, tn] '''
	#	return None


class Evolution:
	'''
	Main GCE one-zone class 
	'''
	def f_RK4_Mi_Wi(self, t_n, y_n, n):
		'''
		Explicit general diff eq GCE function

		INPUT
			t_n		time_chosen[n]
			y_n		dependent variable at n
			n		index of the timestep

		Functions:
			Infall rate: [Msun/Gyr]
			SFR: [Msun/Gyr]
		'''
		Wi_class = Wi(n)
		#if n <= 29: # time_uniform dt 0.001
		if n <= 0:
			return Infall_rate[n] * Xi_inf  - np.multiply(SFR_tn(n), Xi_v[:,n])
		else:
			val = Infall_rate[n] * Xi_inf  - np.multiply(SFR_tn(n), Xi_v[:,n]) + Wi_class.compute("Massive", 'Na')
			val[val<0] = 0. # !!!!!!! if negative set to zero
			return val

	def f_RK4(self, t_n, y_n, n):
		'''
		Explicit general diff eq GCE function
		'''
		return Infall_rate[n] - SFR_tn(n)

	#@lru_cache(maxsize=4)
	def no_integral(self, n):
		SFR_v[n] = SFR_tn(n)
		Mstar_v[n+1] = Mstar_v[n] + SFR_v[n] * IN.nTimeStep
		Mstar_test[n+1] = Mtot[n-1] - Mgas_v[n]
		Mgas_v[n+1] = aux.RK4(self.f_RK4, time_chosen[n], Mgas_v[n], n, IN.nTimeStep)	

	def evolve(self):
		for n in range(len(time_chosen[:idx_age_Galaxy])):	
			print(f'n = {n}')
			self.no_integral(n)		
			Xi_v[:, n] = np.divide(Mass_i_v[:,n], Mgas_v[n]) 
			Mass_i_v[:, n+1] = aux.RK4(self.f_RK4_Mi_Wi, time_chosen[n], Mass_i_v[:,n], n, IN.nTimeStep)
		Xi_v[:,-1] = np.divide(Mass_i_v[:,-1], Mgas_v[-1]) 
		return None

tic.append(time.process_time())
package_loading_time = tic[-1]
print(f'Package lodaded in {1e0*(package_loading_time)} seconds.')

""""""""""""""""""""""""""""""""""""
"                                  "
"    One Zone evolution routine    "
"                                  "
""""""""""""""""""""""""""""""""""""

def main():
	tic.append(time.process_time())
	Evolution_class = Evolution()
	Evolution_class.evolve()
	plts.no_integral_plot()
	Z_v = np.divide(np.sum(Mass_i_v[elemZ_for_metallicity:,:]), Mgas_v)
	G_v = np.divide(Mgas_v, Mtot)
	S_v = 1 - G_v
	aux.tic_count(tic=tic)
	print("Saving the output...")
	np.savetxt('output/phys.dat', np.column_stack((time_chosen, Mtot, Mgas_v,
			   Mstar_v, SFR_v/1e9, Infall_rate/1e9, Z_v, G_v, S_v)), fmt='%-12.4e', #SFR is divided by 1e9 to get the /Gyr to /yr conversion 
			   header = ' (0) time_chosen [Gyr]    (1) Mtot [Msun]    (2) Mgas_v [Msun]    (3) Mstar_v [Msun]    (4) SFR_v [Msun/yr]    (5)Infall_v [Msun/yr]    (6) Z_v    (7) G_v    (8) S_v')
	np.savetxt('output/Mass_i.dat', np.column_stack((ZA_sorted, Mass_i_v)), fmt=' '.join(['%5.i']*2 + ['%12.4e']*Mass_i_v[0,:].shape[0]),
			   header = ' (0) elemZ,	(1) elemA,	(2) masses [Msun] of every isotope for every timestep')
	np.savetxt('output/X_i.dat', np.column_stack((ZA_sorted, Xi_v)), fmt=' '.join(['%5.i']*2 + ['%12.4e']*Xi_v[0,:].shape[0]),
			   header = ' (0) elemZ,	(1) elemA,	(2) abundance mass ratios of every isotope for every timestep (normalized to solar, Asplund et al., 2009)')
	aux.tic_count(string="Output saved in = ", tic=tic)
	return None