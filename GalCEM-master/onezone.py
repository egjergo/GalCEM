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
	"""Actual SFR employed within the integro-differential equation

	Args:
		timestep_n ([int]): [timestep index]

	Returns:
		[function]: [SFR as a function of Mgas]
	"""
	return SFR_class.SFR(Mgas=Mgas_v, Mtot=Mtot, timestep_n=timestep_n) # Function: SFR(Mgas)

def _pick_yields(channel_switch, ZA_Symb, n, stellar_mass_idx=None, metallicity_idx=None, vel_idx=IN.LC18_vel_idx):
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
		self.yield_load = None
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

	def Z_component(self, birthtime_grid):
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
		
	def yield_array(self, channel_switch, mass_grid, birthtime_grid, vel_idx=IN.LC18_vel_idx):
		len_X = len(mass_grid)
		Z_comp = Z_v[self.age_idx] * np.ones(len_X) #self.Z_component(birthtime_grid)
		y = []
		if channel_switch == 'Massive':
			#print(f'{vel_idx = } ')
			#print(f'{mass_grid=}')
			X_sample = np.column_stack([Z_comp, vel_idx * np.ones(len_X), mass_grid])
			X, Y, models = X_lc18, Y_lc18, models_lc18
		elif channel_switch == 'LIMs':
			X_sample = np.column_stack([Z_comp, mass_grid])
			X, Y, models = X_k10, Y_k10, models_k10
		else:
			print(f'{channel_switch = } currently not included.')
			pass

		for i, model in enumerate(models):
			if model != None:
				y.append(model(X_sample)) # !!!!!!! use asynchronicity to speed up the computation
			else:
				y.append(0.)
			#print(f'{y=}')
		return y # len consistent with ZA_sorted

	def mass_component(self, channel_switch, mass_grid, lifetime_grid): #
		''' Portinari+98, page 22, last eq. first column '''
		birthtime_grid = self.grid_picker(channel_switch, 'birthtime')
		IMF_comp = self.IMF_component(mass_grid) # overwrite continuously in __init__
		return IMF_comp, IMF_comp * self.dMdtauM_component(np.log10(lifetime_grid)) 

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
 
	def compute_rates(self):
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
		return rateSNII, rateLIMs, R_SNIa

	def compute(self, channel_switch, vel_idx=IN.LC18_vel_idx): 
		'''Computes, using the Simpson rule, the integral Wi 
		elements of eq. (34) Portinari+98 -- for stars that die at tn, for every i'''
		mass_grid = self.grid_picker(channel_switch, 'mass')
		lifetime_grid = self.grid_picker(channel_switch, 'lifetime')		
		birthtime_grid = self.grid_picker(channel_switch, 'birthtime')
		self.yield_load = self.yield_array(channel_switch, mass_grid, birthtime_grid, vel_idx=vel_idx)
		SFR_comp = self.SFR_component(birthtime_grid)
		SFR_comp[SFR_comp<0] = 0.
		IMF_comp, mass_comp = self.mass_component(channel_switch, mass_grid, lifetime_grid)# 
		#integrand = np.prod(np.vstack[SFR_comp, mass_comp, self.yield_load[i]])
		integrand = np.prod(np.vstack([SFR_comp, mass_comp]), axis=0)
		#return integr.simps(integrand, x=birthtime_grid)
		return [integrand, birthtime_grid]


class Evolution:
    '''
    Main GCE one-zone class 
    '''
    def __init__(self):
        return None
    	
    def f_RK4(self, t_n, y_n, n, i=None):
        ''' Explicit general diff eq GCE function '''
        return Infall_rate[n] - SFR_tn(n)

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
        if n <= 0:
            val = Infall_rate[n] * Xi_inf  - np.multiply(SFR_tn(n), Xi_v[:,n])
        else:
            rateSNII, rateLIMs, rateSNIa = Wi_class.compute_rates()
            Wi_comp = Wi_class.compute("Massive") # Wi_comp = [integrand, birthrate]
            Wi_val = []
            for i, yields in enumerate(Wi_class.yield_load):
                Wi_val.append(integr.simps(Wi_comp[0] * Wi_class.yield_load[i], x=Wi_comp[1]))
            Rate_SNII[n] = rateSNII
            Rate_SNIa[n] = rateSNIa
            Rate_LIMs[n] = rateLIMs
            val = Infall_rate[n] * Xi_inf  - np.multiply(SFR_tn(n), Xi_v[:,n]) + np.sum(Wi_val, axis=0)
            val[val<0] = 0. # !!!!!!! if negative set to zero
        return val
	
    def f_RK4_Mi_Wi_iso(self, t_n, y_n, n, **kwargs): #Wi_comp,
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
        #Wi_class = Wi(n) #how can I pass this from evolve()? used in aux.RK4 
        Wi_comp = kwargs['Wi_comp']
        Wi_class = kwargs['Wi_class']
        i = kwargs['i']
        if n <= 0:
            val = Infall_rate[n] * Xi_inf[i]  - np.multiply(SFR_v[n], Xi_v[i,n])
        else:
            Wi_val = integr.simps(Wi_comp[0] * Wi_class.yield_load[i], x=Wi_comp[1])
            infall_comp = Infall_rate[n] * Xi_inf[i]
            sfr_comp = SFR_v[n] * Xi_v[i,n]
            val = infall_comp  - sfr_comp + Wi_val #+ np.sum(Wi_val, axis=0)
            if i == 19:
                print(f'{n=}, {i=}, {infall_comp=}, {sfr_comp=}, {Wi_val=}')
                print(f'{val=}')
            if val < 0.:
                #print('val negative')
                val = 0.
        return val
	
    def _f_RK4_Mi_Wi_iso(self, t_n, y_n, n, **kwargs): #Wi_comp,
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
        #Wi_class = Wi(n) #how can I pass this from evolve()? used in aux.RK4 
        Wi_comp = kwargs['Wi_comp']
        Wi_class = kwargs['Wi_class']
        i = kwargs['i']
        if n == 0:
            val = Infall_rate[n] * Xi_inf[i] 
        elif n==1:
            val = Infall_rate[n-1] * Xi_inf[i]  - np.multiply(SFR_v[n], Xi_v[i,n])
        elif n>=2:
            Wi_val = integr.simps(Wi_comp[0] * Wi_class.yield_load[i], x=Wi_comp[1])
            infall_comp = Infall_rate[n-2] * Xi_inf[i]
            sfr_comp = SFR_v[n-1] * Xi_v[i,n-1]
            val = infall_comp  - sfr_comp + Wi_val #+ np.sum(Wi_val, axis=0)
            if i == 19:
                print(f'{n=}, {i=}, {infall_comp=}, {sfr_comp=}, {Wi_val=}')
                print(f'{val=}')
            if val < 0.:
                #print('val negative')
                val = 0.
        return val
	
    def phys_integral(self, n):
        SFR_v[n] = SFR_tn(n)
        Mstar_v[n+1] = Mstar_v[n] + SFR_v[n] * IN.nTimeStep
        Mstar_test[n+1] = Mtot[n-1] - Mgas_v[n]
        Mgas_v[n+1] = aux.RK4(self.f_RK4, time_chosen[n], Mgas_v[n], n, IN.nTimeStep)	

    def evolve(self):
        for n in range(len(time_chosen[:idx_age_Galaxy])):
            print(f'{n = }')
            self.phys_integral(n)		
            Xi_v[:, n] = np.divide(Mass_i_v[:,n], Mgas_v[n])
            if n > 0.: 
                Wi_class = Wi(n)
                rateSNII, rateLIMs, rateSNIa = Wi_class.compute_rates()
                Rate_SNII[n] = rateSNII
                Rate_SNIa[n] = rateSNIa
                Rate_LIMs[n] = rateLIMs
                Wi_comp = Wi_class.compute("Massive") # Wi_comp = [integrand, birthrate]
                for i, pair in enumerate(ZA_sorted): 
                    Mass_i_v[i, n+1] = aux.RK4(self.f_RK4_Mi_Wi_iso, time_chosen[n], Mass_i_v[i,n], n, IN.nTimeStep, Wi_class=Wi_class, i=i, Wi_comp=Wi_comp)
            Z_v[n] = np.divide(np.sum(Mass_i_v[:,n]), Mgas_v[n])
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
	plts.phys_integral_plot()
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
	aux.tic_count(string="Output saved in ", tic=tic)
	return None