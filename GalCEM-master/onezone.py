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
		self.metallicity = metallicity #* np.ones(IN.num_MassGrid) # !!!!!!!
		self.age_idx = age_idx
		return None

	def grids(self, Ml_lim, Mu_lim):
		'''
		Ml_lim and Mu_lim are mass limits
		They are converted to lifetimes by integr_lim() in integration_grid()
		'''
		mass_grid = np.geomspace(Ml_lim, Mu_lim, num = IN.num_MassGrid)
		lifetime_grid = lifetime_class.interp_stellar_lifetimes(self.metallicity)(mass_grid) #np.power(10, lifetime_class.interp_stellar_lifetimes(mass_grid, self.metallicity))#np.column_stack([mass_grid, self.metallicity * np.ones(len(mass_grid))])))
		birthtime_grid = time_chosen[self.age_idx] - lifetime_grid 
		positive_idx = np.where(birthtime_grid > 0.)
		#print(f"For grids, {mass_grid[positive_idx]=}")
		#print(f"For grids, {lifetime_grid[positive_idx]=}")
		#print(f"For grids, {birthtime_grid[positive_idx]=}")
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
		#print('Massive channel')
		self.Massive_birthtime_grid, self.Massive_lifetime_grid, self.Massive_mass_grid = (
				self.Wi_grid_class.grids(IN.Ml_Massive, IN.Mu_Massive))
		#print('LIMs channel')
		self.LIMs_birthtime_grid, self.LIMs_lifetime_grid, self.LIMs_mass_grid = (
				self.Wi_grid_class.grids(IN.Ml_LIMs, IN.Mu_LIMs)) # !!!!!!! you should subtract SNIa fraction
		#print('SNIa channel')
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

	def _Z_component(self, birthtime_grid):
		''' Returns the interpolated SFR vector computed at the birthtime grids'''
		_Z_interp = interp.interp1d(time_chosen[:self.age_idx+1], Z_v[:self.age_idx+1], fill_value='extrapolate')
		return _Z_interp(birthtime_grid)

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
		
	def _yield_component(self, channel_switch, i, _Z_comp, mass_grid, vel_idx=IN.LC18_vel_idx):
		#Yield_i_birthtime = pick_yields(channel_switch, ZA_Symb, vel_idx=vel_idx)
		if channel_switch == 'Massive':
			Xi = np.array([_Z_comp, vel_idx * np.ones(len(mass_grid)), mass_grid]).T
			print(f'In yield_component: {i=}')
			print(f'In yield_component: {Xi.shape=}')
			return models_lc18[i](Xi)
		else:
			print(f'{channel_switch = } currently not included.')
			pass
		#elif channel_switch == 'LIMs':
		#	Xi = 
		#	return models_k10(Xi)#(mass_grid)
		#elif channel_switch == 'SNIa':
		#	idx = isotope_class.pick_by_Symb(yields_SNIa_class.elemZ, ZA_Symb)
		#	return yields_SNIa_class.yields[idx]
		#return Yield_i_birthtime

	def mass_component(self, channel_switch, ZA_Symb, vel_idx=IN.LC18_vel_idx): #
		''' Portinari+98, page 22, last eq. first column '''
		mass_grid = self.grid_picker(channel_switch, 'mass')
		lifetime_grid = self.grid_picker(channel_switch, 'lifetime')
		IMF_comp = self.IMF_component(mass_grid)
		return IMF_comp, IMF_comp * self.dMdtauM_component(np.log10(lifetime_grid)) #* self.yield_component(channel_switch, ZA_Symb, vel_idx=vel_idx) 
	def _mass_component(self, channel_switch, i, _Z_comp, vel_idx=IN.LC18_vel_idx): #
		''' Portinari+98, page 22, last eq. first column '''
		mass_grid = self.grid_picker(channel_switch, 'mass')
		lifetime_grid = self.grid_picker(channel_switch, 'lifetime')	
		return (self.IMF_component(mass_grid) * self.dMdtauM_component(np.log10(lifetime_grid))
		 		* self.yield_component(channel_switch, i, _Z_comp, mass_grid, vel_idx=vel_idx))

	#def compute_iso(self, channel_switch, ZA_Symb, vel_idx=IN.LC18_vel_idx): #
	def compute(self, channel_switch, ZA_Symb, vel_idx=IN.LC18_vel_idx): #
		'''Computes, using the Simpson rule, the integral Wi 
		elements of eq. (34) Portinari+98 -- for stars that die at tn, for every i'''		
		birthtime_grid = self.grid_picker(channel_switch, 'birthtime')
		SFR_comp = self.SFR_component(birthtime_grid)
		SFR_comp[SFR_comp<0] = 0.
		IMF_comp, mass_comp = self.mass_component(channel_switch, ZA_Symb, vel_idx=vel_idx)# 
		integrand = np.multiply(SFR_comp, mass_comp)
		#print(f"For compute, {birthtime_grid=}")
		#print(f"For compute, {SFR_comp=}")
		#print(f"For compute, {mass_comp=}")
		#print(f"For compute, {IMF_comp=}")
		#print(f"For compute, {integrand=}")
		#integrand_rateSNII = np.multiply(SFR_comp, IMF_comp)
		if len(self.grid_picker('Massive', 'birthtime')) > 0.:
			rateSNII = self.compute_rate(channel_switch='Massive')
		else:
			rateSNII = IN.epsilon
		if len(self.grid_picker('LIMs', 'birthtime')) > 0.:
			rateLIMs = self.compute_rate(channel_switch='LIMs')
		else:
			rateLIMs = IN.epsilon
		if len(self.grid_picker('SNIa', 'birthtime')) > 0.:
			R_SNIa = self.compute_rateSNIa()
		else:
			R_SNIa = IN.epsilon
		#print(f"For compute, {R_SNIa=}")
		#if channel_switch == 'SNIa':
		# 	integrand_SNIa, M1_min, M1_max = SNIa_FM1(self, M1)
		#	return (1 - IN.A) * integr.simps(integrand) + IN.A * integr.simps(integrand_SNIa)
		#else:
		return integr.simps(integrand, x=birthtime_grid), rateSNII, R_SNIa, rateLIMs

	#def compute():
	#	''' Computes the vector to be added to Mass_i_v[:, tn] '''
	#	return None
	def _compute(self, channel_switch, vel_idx=IN.LC18_vel_idx):
		''' Computes the vector to be added to Mass_i_v[:, tn] '''	
		mass_comp, integrand, integral = [], [], []
		birthtime_grid = self.grid_picker(channel_switch, 'birthtime')
		mass_grid = self.grid_picker(channel_switch, 'mass')
		lifetime_grid = self.grid_picker(channel_switch, 'lifetime')
		#mass_comp = np.empty(len(birthtime_grid))

		SFR_comp = self.SFR_component(birthtime_grid)
		SFR_comp[SFR_comp<0] = 0.
		_Z_comp = self._Z_component(birthtime_grid)
		_Z_comp[_Z_comp<0] = 0.
		IMF_comp = self.IMF_component(mass_grid)
		dMdtau_comp = self.dMdtauM_component(np.log10(lifetime_grid))
		M_comp = np.multiply(IMF_comp, dMdtau_comp)
		for i, val in enumerate(ZA_sorted):
			if X_lc18[i].size != 0:
				print(f'In compute: {i=}')
				print(f'In compute: {X_lc18[i].size=}')
				print(f'In compute: {X_lc18[i]=}')
				yield_comp = self.yield_component(channel_switch, i, _Z_comp, mass_grid, vel_idx=vel_idx)
				print('yield_component ok')
				print(f'{M_comp.shape=}')
				print(f'{yield_comp.shape=}')
				print(f'{type(M_comp)=}')
				print(f'{type(yield_comp)=}')
				print(f'{M_comp=}')
				print(f'{yield_comp=}')
				mass_comp.append(np.multiply(M_comp, yield_comp))
				print(f'{mass_comp[-1]=}')
				integrand.append(np.multiply(SFR_comp, mass_comp[-1]))
				print(f'{integrand[-1]=}')
				integral.append(integr.simps(integrand[-1], x=birthtime_grid))
			else:
				print(f'In compute else: {i=}')
				print(f'In compute else: {X_lc18[i].size=}')
				print(f'In compute else: {X_lc18[i]=}')
				mass_comp.append(0.)
				integrand.append(0.)
				integral.append(0.)
		#if channel_switch == 'SNIa':
		# 	integrand_SNIa, M1_min, M1_max = SNIa_FM1(self, M1)
		#	return (1 - IN.A) * integr.simps(integrand) + IN.A * integr.simps(integrand_SNIa)
		#else:
		return np.array(integral)

	def compute_rate(self, channel_switch='Massive'):
		''' Computes the Type II SNae rate '''	
		birthtime_grid = self.grid_picker(channel_switch, 'birthtime')
		mass_grid = self.grid_picker(channel_switch, 'mass')
		SFR_comp = self.SFR_component(birthtime_grid)
		SFR_comp[SFR_comp<0] = 0.
		IMF_comp = self.IMF_component(mass_grid)
		integrand = np.multiply(SFR_comp, IMF_comp)
		#print(f"For compute_rateSNII, {SFR_comp=}")
		#print(f"For compute_rateSNII, {IMF_comp=}")
		#print(f"For compute_rateSNII, {integrand=}")
		return integr.simps(integrand, x=mass_grid)
	
	def compute_rateSNIa(self, channel_switch='SNIa'):
		birthtime_grid = self.grid_picker(channel_switch, 'birthtime')
		mass_grid = self.grid_picker(channel_switch, 'mass')
		f_nu = lambda nu: 24 * (1 - nu)**2
		#M1_min = 0.5 * IN.Ml_SNIa
		#M1_max = IN.Mu_SNIa
		nu_min = np.max([0.5, np.max(np.divide(mass_grid, IN.Mu_SNIa))])
		nu_max = np.min([1, np.min(np.divide(mass_grid, IN.Ml_SNIa))])
		#print(f'{nu_min = },\t {nu_max=}')
		#print(f'nu_min = 0.5,\t nu_max= 1.0')
		#nu_test = np.linspace(nu_min, nu_max, num=len(mass_grid))
		nu_test = np.linspace(0.5, 1, num=len(mass_grid))
		IMF_v = np.divide(mass_grid, nu_test)
		int_SNIa = f_nu(nu_test) * self.IMF_component(IMF_v)
		F_SNIa = integr.simps(int_SNIa, x=nu_test)	
		SFR_comp = self.SFR_component(birthtime_grid)
		integrand = np.multiply(SFR_comp, F_SNIa)
		return integr.simps(integrand, x=birthtime_grid)

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
			#print(f"For compute, {time_chosen[:n]=}")
			#print(f"For compute, {SFR_v[:n]=}")
			Wi_val, rateSNII, rateSNIa, rateLIMs = Wi_class.compute("Massive", 'H')
			Rate_SNII[n] = rateSNII
			Rate_SNIa[n] = rateSNIa
			Rate_LIMs[n] = rateLIMs
			val = Infall_rate[n] * Xi_inf  - np.multiply(SFR_tn(n), Xi_v[:,n]) + Wi_val
			val[val<0] = 0. # !!!!!!! if negative set to zero
			return val

	def f_RK4(self, t_n, y_n, n):
		'''
		Explicit general diff eq GCE function
		'''
		return Infall_rate[n] - SFR_tn(n)

	def f_rateSNII(self, n):
		'''
		SNII Rate
		'''
		Wi_class = Wi(n)
		if n <= 10:
			return 0.
		else:
			return Wi_class.compute_rateSNII()

	#@lru_cache(maxsize=4)
	def no_integral(self, n):
		SFR_v[n] = SFR_tn(n)
		Mstar_v[n+1] = Mstar_v[n] + SFR_v[n] * IN.nTimeStep
		Mstar_test[n+1] = Mtot[n-1] - Mgas_v[n]
		Mgas_v[n+1] = aux.RK4(self.f_RK4, time_chosen[n], Mgas_v[n], n, IN.nTimeStep)	

	def evolve(self):
		for n in range(len(time_chosen[:idx_age_Galaxy])):	#320
			print(f'{n = }')
			self.no_integral(n)		
			Xi_v[:, n] = np.divide(Mass_i_v[:,n], Mgas_v[n]) 
			Mass_i_v[:, n+1] = aux.RK4(self.f_RK4_Mi_Wi, time_chosen[n], Mass_i_v[:,n], n, IN.nTimeStep)
			Z_v[n] = np.divide(np.sum(Mass_i_v[:,n]), Mgas_v[n])
			#G_v[n] = np.divide(Mgas_v[n], Mtot[n])
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
	#Z_v = np.divide(np.sum(Mass_i_v[elemZ_for_metallicity:,:]), Mgas_v)
	G_v = np.divide(Mgas_v, Mtot)
	S_v = 1 - G_v
	aux.tic_count(tic=tic)
	print("Saving the output...")
	np.savetxt('output/phys.dat', np.column_stack((time_chosen, Mtot, Mgas_v,
			   Mstar_v, SFR_v/1e9, Infall_rate/1e9, Z_v, G_v, S_v, Rate_SNII, Rate_SNIa)), fmt='%-12.4e', #SFR is divided by 1e9 to get the /Gyr to /yr conversion 
			   header = ' (0) time_chosen [Gyr]    (1) Mtot [Msun]    (2) Mgas_v [Msun]    (3) Mstar_v [Msun]    (4) SFR_v [Msun/yr]    (5)Infall_v [Msun/yr]    (6) Z_v    (7) G_v    (8) S_v 	(9) Rate_SNII 	(10) Rate_SNIa')
	np.savetxt('output/Mass_i.dat', np.column_stack((ZA_sorted, Mass_i_v)), fmt=' '.join(['%5.i']*2 + ['%12.4e']*Mass_i_v[0,:].shape[0]),
			   header = ' (0) elemZ,	(1) elemA,	(2) masses [Msun] of every isotope for every timestep')
	np.savetxt('output/X_i.dat', np.column_stack((ZA_sorted, Xi_v)), fmt=' '.join(['%5.i']*2 + ['%12.4e']*Xi_v[0,:].shape[0]),
			   header = ' (0) elemZ,	(1) elemA,	(2) abundance mass ratios of every isotope for every timestep (normalized to solar, Asplund et al., 2009)')
	aux.tic_count(string="Output saved in = ", tic=tic)
	return None