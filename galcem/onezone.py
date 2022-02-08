""" I only achieve simplicity with enormous effort (Clarice Lispector) """
import time
import numpy as np
import scipy.integrate as integr
import scipy.interpolate as interp

from . import morphology as morph
from . import yields as Y

class OneZone:
    def __init__(self, inputs):
        self.tic = []
        self.tic.append(time.process_time())
        self.IN = inputs
        self.aux = morph.Auxiliary()
        lifetime_class = morph.Stellar_Lifetimes(self.IN)
        """ Setup """
        Ml = lifetime_class.s_mass[0] # Lower limit stellar masses [Msun] 
        Mu = lifetime_class.s_mass[-1] # Upper limit stellar masses [Msun]
        mass_uniform = np.linspace(Ml, Mu, num = self.IN.num_MassGrid)
        #time_uniform = np.arange(IN.time_start, IN.time_end, IN.nTimeStep)
        #time_logspace = np.logspace(np.log10(IN.time_start), np.log10(IN.time_end), num=IN.numTimeStep)
        time_uniform = np.arange(self.IN.time_start, self.IN.age_Galaxy, self.IN.nTimeStep)
        time_logspace = np.logspace(np.log10(self.IN.time_start), np.log10(self.IN.age_Galaxy), num=self.IN.numTimeStep)
        self.time_chosen = time_uniform
        self.idx_age_Galaxy = self.aux.find_nearest(self.time_chosen, self.IN.age_Galaxy)
        '''Surface density for the disk. The bulge goes as an inverse square law.'''
        surf_density_Galaxy = self.IN.sd / np.exp(self.IN.r / self.IN.Reff[self.IN.morphology]) #sigma(t_G) before eq(7) not used so far !!!!!!!

        infall_class = morph.Infall(self.IN, morphology=self.IN.morphology, time=self.time_chosen)
        infall = infall_class.inf()

        self.SFR_class = morph.Star_Formation_Rate(self.IN, self.IN.SFR_option, self.IN.custom_SFR)
        IMF_class = morph.Initial_Mass_Function(Ml, Mu, self.IN, self.IN.IMF_option, self.IN.custom_IMF)
        self.IMF = IMF_class.IMF() #() # Function @ input stellar mass

        isotope_class = Y.Isotopes(self.IN)
        self.yields_LIMs_class = Y.Yields_LIMs(self.IN)
        self.yields_LIMs_class.import_yields()
        yields_Massive_class = Y.Yields_Massive(self.IN)
        yields_Massive_class.import_yields()
        self.yields_SNIa_class = Y.Yields_SNIa(self.IN)
        self.yields_SNIa_class.import_yields()
        yields_BBN_class = Y.Yields_BBN(self.IN)
        yields_BBN_class.import_yields()

        c_class = Y.Concentrations(self.IN)
        ZA_LIMs = c_class.extract_ZA_pairs_LIMs(self.yields_LIMs_class)
        ZA_SNIa = c_class.extract_ZA_pairs_SNIa(self.yields_SNIa_class)
        ZA_Massive = c_class.extract_ZA_pairs_Massive(yields_Massive_class)
        ZA_all = np.vstack((ZA_LIMs, ZA_SNIa, ZA_Massive))

        """ Initialize Global tracked quantities """ 
        self.Infall_rate = infall(self.time_chosen)
        self.ZA_sorted = c_class.ZA_sorted(ZA_all) # [Z, A] VERY IMPORTANT! 321 isotopes with yields_SNIa_option = 'km20', 192 isotopes for 'i99' 
        self.ZA_sorted = self.ZA_sorted[1:,:]
        ZA_symb_list = self.IN.periodic['elemSymb'][self.ZA_sorted[:,0]] # name of elements for all isotopes
        asplund3_percent = c_class.abund_percentage(c_class.asplund3_pd, self.ZA_sorted)
        #ZA_symb_iso_list = np.asarray([ str(A) for A in self.IN.periodic['elemA'][self.ZA_sorted]])  # name of elements for all isotopes
        elemZ_for_metallicity = np.where(self.ZA_sorted[:,0]>2)[0][0] #  starting idx (int) that excludes H and He for the metallicity selection
        self.Mtot = np.insert(np.cumsum((self.Infall_rate[1:] + self.Infall_rate[:-1]) * self.IN.nTimeStep / 2), 0, self.IN.epsilon) # The total baryonic mass (i.e. the infall mass) is computed right away
        #Mtot_quad = [quad(infall, self.time_chosen[0], i)[0] for i in range(1,len(self.time_chosen)-1)] # slow loop, deprecate!!!!!!!
        self.Mstar_v = self.IN.epsilon * np.ones(len(self.time_chosen)) # Global
        self.Mstar_test = self.IN.epsilon * np.ones(len(self.time_chosen)) # Global
        self.Mgas_v = self.IN.epsilon * np.ones(len(self.time_chosen)) # Global
        self.SFR_v = self.IN.epsilon * np.ones(len(self.time_chosen)) #
        self.Mass_i_v = self.IN.epsilon * np.ones((len(self.ZA_sorted), len(self.time_chosen)))	# Gass mass (i,j) where the i rows are the isotopes and j are the timesteps, [:,j] follows the timesteps
        Xi_inf = isotope_class.construct_yield_vector(yields_BBN_class, self.ZA_sorted)
        Mass_i_inf = np.column_stack(([Xi_inf] * len(self.Mtot)))
        self.Xi_v = self.IN.epsilon * np.ones((len(self.ZA_sorted), len(self.time_chosen)))	# Xi 
        self.Z_v = self.IN.epsilon * np.ones(len(self.time_chosen)) # Metallicity 
        G_v = self.IN.epsilon * np.ones(len(self.time_chosen)) # G 
        S_v = self.IN.epsilon * np.ones(len(self.time_chosen)) # S = 1 - G 
        self.Rate_SNII = self.IN.epsilon * np.ones(len(self.time_chosen)) 
        self.Rate_LIMs = self.IN.epsilon * np.ones(len(self.time_chosen)) 
        self.Rate_SNIa = self.IN.epsilon * np.ones(len(self.time_chosen)) 
        self.Rate_NSM = self.IN.epsilon * np.ones(len(self.time_chosen)) 

        self.tic.append(time.process_time())
        package_loading_time = self.tic[-1]
        print(f'Package lodaded in {1e0*(package_loading_time)} seconds.')
        
    def main(self):
        self.tic.append(time.process_time())
        self.file1 = open("output/Terminal_output.txt", "w")
        self.evolve()
        self.aux.tic_count(string="Computation time", tic=self.tic)
        #Z_v = np.divide(np.sum(self.Mass_i_v[elemZ_for_metallicity:,:]), Mgas_v)
        G_v = np.divide(self.Mgas_v, self.Mtot)
        S_v = 1 - G_v
        print("Saving the output...")
        np.savetxt('output/phys.dat', np.column_stack((self.time_chosen, self.Mtot, self.Mgas_v,
                self.Mstar_v, self.SFR_v/1e9, self.Infall_rate/1e9, self.Z_v, G_v, S_v, self.Rate_SNII, self.Rate_SNIa, self.Rate_LIMs)), fmt='%-12.4e', #SFR is divided by 1e9 to get the /Gyr to /yr conversion 
                header = ' (0) time_chosen [Gyr]    (1) Mtot [Msun]    (2) Mgas_v [Msun]    (3) Mstar_v [Msun]    (4) SFR_v [Msun/yr]    (5)Infall_v [Msun/yr]    (6) Z_v    (7) G_v    (8) S_v     (9) Rate_SNII     (10) Rate_SNIa     (11) Rate_LIMs')
        np.savetxt('output/Mass_i.dat', np.column_stack((self.ZA_sorted, self.Mass_i_v)), fmt=' '.join(['%5.i']*2 + ['%12.4e']*self.Mass_i_v[0,:].shape[0]),
                header = ' (0) elemZ,    (1) elemA,    (2) masses [Msun] of every isotope for every timestep')
        np.savetxt('output/X_i.dat', np.column_stack((self.ZA_sorted, self.Xi_v)), fmt=' '.join(['%5.i']*2 + ['%12.4e']*self.Xi_v[0,:].shape[0]),
                header = ' (0) elemZ,    (1) elemA,    (2) abundance mass ratios of every isotope for every timestep (normalized to solar, Asplund et al., 2009)')
        self.aux.tic_count(string="Output saved in", tic=self.tic)
        print('Plotting...')
        self.file1.close()
        self.aux.tic_count(string="Plots saved in", tic=self.tic)
        return None
    
    ''' EVOLVE Methods '''
    def f_RK4(self, t_n, y_n, n, i=None):
        ''' Explicit general diff eq GCE function '''
        return self.Infall_rate[n] - self.SFR_tn(n)

    def solve_integral(self, t_n, y_n, n, **kwargs): #Wi_comp,
        '''
        Explicit general diff eq GCE function
        INPUT
        t_n        time_chosen[n]
        y_n        dependent variable at n
        n        index of the timestep
        Functions:
        Infall rate: [Msun/Gyr]
        SFR: [Msun/Gyr]
        '''
        Wi_comps = kwargs['Wi_comp'] # [list of 2-array lists] #Wi_comps[0][0].shape=(155,)
        Wi_SNIa = kwargs['Wi_SNIa']
        i = kwargs['i']
        yields = kwargs['yields']
        channel_switch = ['Massive', 'LIMs']
        if n <= 0:
            val = self.Infall_rate[n] * self.Xi_inf[i]  - np.multiply(self.SFR_v[n], self.Xi_v[i,n])
        else:
            Wi_vals = []
            for j,val in enumerate(Wi_comps):
                if len(val[1]) > 0.:
                    #print(f'{channel_switch[j]}, {ZA_sorted[i]}')
                    #file1.write(f'{channel_switch[j]}, {ZA_sorted[i]}\n') 
                    Wi_vals.append(integr.simps(val[0] * yields[j], x=val[1]))
            infall_comp = self.Infall_rate[n] * self.Xi_inf[i]
            sfr_comp = self.SFR_v[n] * self.Xi_v[i,n]
            val = infall_comp  - sfr_comp + np.sum(Wi_vals) + Wi_SNIa #+ np.sum(Wi_val, axis=0)
            if val < 0.:
                #print('val negative')
                val = 0.
        return val

    def phys_integral(self, n):
        self.SFR_v[n] = self.SFR_tn(n)
        self.Mstar_v[n+1] = self.Mstar_v[n] + self.SFR_v[n] * self.IN.nTimeStep
        self.Mstar_test[n+1] = self.Mtot[n-1] - self.Mgas_v[n]
        self.Mgas_v[n+1] = self.aux.RK4(self.f_RK4, self.time_chosen[n], self.Mgas_v[n], n, self.IN.nTimeStep)    

    def evolve(self):
        for n in range(len(self.time_chosen[:self.idx_age_Galaxy])):
            print(f'{n = }')
            self.file1.write(f'{n = }\n')
            self.phys_integral(n)        
            self.Xi_v[:, n] = np.divide(self.Mass_i_v[:,n], self.Mgas_v[n])
            if n > 0.: 
                Wi_class = Wi(n, self.IN, self.lifetime_class, self.time_chosen, self.Z_v, self.SFR_v, self.IMF, self.yields_SNIa_class, self.models_lc18, self.models_k10, self.ZA_sorted)
                self.Rate_SNII[n], self.Rate_LIMs[n], self.Rate_SNIa[n] = Wi_class.compute_rates()
                Wi_comp = [Wi_class.compute("Massive"), Wi_class.compute("LIMs")]
                #yield_SNII = Wi_class.yield_array('Massive', Wi_class.Massive_mass_grid, Wi_class.Massive_birthtime_grid)
                #yield_LIMs = Wi_class.yield_array('LIMs', Wi_class.LIMs_mass_grid, Wi_class.LIMs_birthtime_grid)
                for i, _ in enumerate(self.ZA_sorted): 
                    self.file1.write(f'{i=}\n')
                    Wi_SNIa = self.Rate_SNIa[n] * self.Y_i99[i]
                    if self.X_lc18[i].empty:
                        yields_lc18 = 0.
                    else:
                        idx_SNII = np.digitize(self.Z_v[n-1], self.averaged_lc18[i][1])
                        yields_lc18 = self.averaged_lc18[i][0][idx_SNII]
                    if self.X_k10[i].empty:
                        yields_k10 = 0.
                    else:
                        idx_LIMs = np.digitize(self.Z_v[n-1], self.averaged_k10[i][1])
                        yields_k10 = self.averaged_k10[i][0][idx_LIMs]
                    yields = [yields_lc18, yields_k10]
                    self.Mass_i_v[i, n+1] = self.aux.RK4(self.solve_integral, self.time_chosen[n], self.Mass_i_v[i,n], n, self.IN.nTimeStep, i=i, Wi_comp=Wi_comp, Wi_SNIa=Wi_SNIa, yields=yields)
            self.Z_v[n] = np.divide(np.sum(self.Mass_i_v[:,n]), self.Mgas_v[n])
        self.Xi_v[:,-1] = np.divide(self.Mass_i_v[:,-1], self.Mgas_v[-1]) 
        return None

    def SFR_tn(self, timestep_n):
        """Actual SFR employed within the integro-differential equation

        Args:
            timestep_n ([int]): [timestep index]

        Returns:
            [function]: [SFR as a function of Mgas]
        """
        return self.SFR_class.SFR(Mgas=self.Mgas_v, Mtot=self.Mtot, timestep_n=timestep_n) # Function: SFR(Mgas)


class Wi_grid:
    '''
    birthtime grid for Wi integral
    '''
    def __init__(self, metallicity, age_idx, IN, lifetime_class, time_chosen):
        self.metallicity = metallicity #* np.ones(IN.num_MassGrid) # !!!!!!!
        self.age_idx = age_idx
        self.IN = IN
        self.lifetime_class = lifetime_class
        self.time_chosen = time_chosen

    def grids(self, Ml_lim, Mu_lim):
        '''
        Ml_lim and Mu_lim are mass limits
        They are converted to lifetimes by integr_lim() in integration_grid()
        '''
        mass_grid = np.geomspace(Ml_lim, Mu_lim, num = self.IN.num_MassGrid)
        lifetime_grid = self.lifetime_class.interp_stellar_lifetimes(0.06)(mass_grid) #np.power(10, self.lifetime_class.interp_stellar_lifetimes(mass_grid, self.metallicity))#np.column_stack([mass_grid, self.metallicity * np.ones(len(mass_grid))])))
        birthtime_grid = self.time_chosen[self.age_idx] - lifetime_grid 
        positive_idx = np.where(birthtime_grid > 0.)
        #print(f"For grids, {mass_grid[positive_idx]=}")
        #print(f"For grids, {lifetime_grid[positive_idx]=}")
        #print(f"For grids, {birthtime_grid[positive_idx]=}")
        return birthtime_grid[positive_idx], lifetime_grid[positive_idx], mass_grid[positive_idx]

            
class Wi:
    '''
    Solves each integration item by integrating over birthtimes.
    
    Input upper and lower mass limits (to be mapped onto birthtimes)
    
    Gyr_age    (t)     is the Galactic age
    birthtime (t')     is the stellar birthtime
    lifetime (tau)    is the stellar lifetime
    '''
    def __init__(self, age_idx, IN, lifetime_class, time_chosen, Z_v, SFR_v, IMF, yields_SNIa_class, models_lc18, models_k10, ZA_sorted):
        self.IN = IN
        self.lifetime_class = lifetime_class
        self.time_chosen = time_chosen
        self.Z_v = Z_v
        self.SFR_v = SFR_v
        self.IMF = IMF
        self.yields_SNIa_class = yields_SNIa_class
        self.models_lc18 = models_lc18
        self.models_k10 = models_k10
        self.ZA_sorted = ZA_sorted
        self.metallicity = self.Z_v[age_idx]
        self.age_idx = age_idx
        self.Wi_grid_class = Wi_grid(self.metallicity, self.age_idx, self.IN, lifetime_class, self.time_chosen)
        #print('Massive channel')
        self.Massive_birthtime_grid, self.Massive_lifetime_grid, self.Massive_mass_grid = (
                self.Wi_grid_class.grids(self.IN.Ml_Massive, self.IN.Mu_Massive))
        #print('LIMs channel')
        self.LIMs_birthtime_grid, self.LIMs_lifetime_grid, self.LIMs_mass_grid = (
                self.Wi_grid_class.grids(self.IN.Ml_LIMs, self.IN.Mu_LIMs)) # !!!!!!! you should subtract SNIa fraction
        #print('SNIa channel')
        self.SNIa_birthtime_grid, self.SNIa_lifetime_grid, self.SNIa_mass_grid = (
                self.Wi_grid_class.grids(self.IN.Ml_SNIa, self.IN.Mu_SNIa))
        self.yield_load = None
        return None
        
    def grid_picker(self, channel_switch, grid_type):
        '''
        Selects e.g. "self.LIMs_birthtime_grid"
        
        channel_switch:        can be 'LIMs', 'SNIa', 'Massive'
        grid_type:            can be 'birthtime', 'lifetime', 'mass' 
        '''
        return self.__dict__[channel_switch+'_'+grid_type+'_grid']
    
    def SFR_component(self, birthtime_grid):
        ''' Returns the interpolated SFR vector computed at the birthtime grids'''
        SFR_interp = interp.interp1d(self.time_chosen[:self.age_idx+1], self.SFR_v[:self.age_idx+1], fill_value='extrapolate')
        return SFR_interp(birthtime_grid)

    def Z_component(self, birthtime_grid):
        ''' Returns the interpolated SFR vector computed at the birthtime grids'''
        _Z_interp = interp.interp1d(self.time_chosen[:self.age_idx+1], self.Z_v[:self.age_idx+1], fill_value='extrapolate')
        return _Z_interp(birthtime_grid)

    def IMF_component(self, mass_grid):
        ''' Returns the IMF vector computed at the mass grids'''
        return self.IMF(mass_grid)
    
    def dMdtauM_component(self, lifetime_grid, derlog=None): #!!!!!!!
        ''' computes the derivative of M(tauM) w.r.t. tauM '''
        derlog = self.IN.derlog if derlog is None else derlog
        if derlog == False:
            return self.lifetime_class.dMdtauM(np.log10(lifetime_grid), self.metallicity*np.ones(len(lifetime_grid)))#(lifetime_grid)
        if derlog == True:
            return 0.5    
 
    def _yield_array(self, channel_switch, mass_grid, birthtime_grid, vel_idx=None):
        vel_idx = self.IN.LC18_vel_idx if vel_idx is None else vel_idx
        len_X = len(mass_grid)
        Z_comp = self.Z_v[self.age_idx] * np.ones(len_X) #self.Z_component(birthtime_grid)
        y = []
        if channel_switch == 'SNIa': 
            y = self.yields_SNIa_class
        else:  
            if channel_switch == 'Massive':
                X_sample = np.column_stack([Z_comp, vel_idx * np.ones(len_X), mass_grid])
                models = self.models_lc18
            elif channel_switch == 'LIMs':
                Z_comp /= self.IN.solar_metallicity
                X_sample = np.column_stack([Z_comp, mass_grid])
                models = self.models_k10
            else:
                print(f'{channel_switch = } currently not included.')
                pass
            
            for i, model in enumerate(models):
                if model != None:
                    #print(f'{channel_switch=}, \t{i=}')
                    fit = model(X_sample)
                    #print(f"{len(fit)=}")
                    y.append(fit) # !!!!!!! use asynchronicity to speed up the computation
                else:
                    y.append(np.zeros(len_X))
        return 0.005 * np.ones(len(self.ZA_sorted)) #y # len consistent with ZA_sorted

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
        nu_min = np.max([0.5, np.max(np.divide(mass_grid, self.IN.Mu_SNIa))])
        nu_max = np.min([1, np.min(np.divide(mass_grid, self.IN.Ml_SNIa))])
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
            rateSNII = self.IN.epsilon
        if len(self.grid_picker('LIMs', 'birthtime')) > 0.:
            rateLIMs = self.compute_rate(channel_switch='LIMs')
        else:
            rateLIMs = self.IN.epsilon
        if len(self.grid_picker('SNIa', 'birthtime')) > 0.:
            R_SNIa = self.IN.A_SNIa * self.compute_rateSNIa()
        else:
            R_SNIa = self.IN.epsilon
        return rateSNII, rateLIMs, R_SNIa

    def compute(self, channel_switch, vel_idx=None):
        vel_idx = self.IN.LC18_vel_idx if vel_idx is None else vel_idx
        '''Computes, using the Simpson rule, the integral Wi 
        elements of eq. (34) Portinari+98 -- for stars that die at tn, for every i'''
        mass_grid = self.grid_picker(channel_switch, 'mass')
        lifetime_grid = self.grid_picker(channel_switch, 'lifetime')        
        birthtime_grid = self.grid_picker(channel_switch, 'birthtime')
        SFR_comp = self.SFR_component(birthtime_grid)
        SFR_comp[SFR_comp<0] = 0.
        IMF_comp, mass_comp = self.mass_component(channel_switch, mass_grid, lifetime_grid)# 
        #integrand = np.prod(np.vstack[SFR_comp, mass_comp, self.yield_load[i]])
        integrand = np.prod(np.vstack([SFR_comp, mass_comp]), axis=0)
        #return integr.simps(integrand, x=birthtime_grid)
        return [integrand, birthtime_grid]
