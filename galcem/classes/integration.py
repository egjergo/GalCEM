import numpy as np
import scipy.interpolate as interp
import scipy.integrate as integr


class Wi_grid:
    # birthtime grid for Wi integral
    def __init__(self, metallicity, age_idx, IN, lifetime_class, time_chosen):
        self.metallicity = metallicity #* np.ones(IN.num_MassGrid) # !!!!!!!
        self.age_idx = age_idx
        self.IN = IN
        self.lifetime_class = lifetime_class
        self.time_chosen = time_chosen

    def grids(self, Ml_lim, Mu_lim):
        # Ml_lim and Mu_lim are mass limits
        # They are converted to lifetimes by integr_lim() in integration_grid()
        mass_grid = np.geomspace(Ml_lim, Mu_lim, num = self.IN.num_MassGrid)
        lifetime_grid = self.lifetime_class.interp_stellar_lifetimes(0.06)(mass_grid) #np.power(10, self.lifetime_class.interp_stellar_lifetimes(mass_grid, self.metallicity))#np.column_stack([mass_grid, self.metallicity * np.ones(len(mass_grid))])))
        birthtime_grid = self.time_chosen[self.age_idx] - lifetime_grid 
        positive_idx = np.where(birthtime_grid > 0.)
        #print(f"For grids, {mass_grid[positive_idx]=}")
        #print(f"For grids, {lifetime_grid[positive_idx]=}")
        #print(f"For grids, {birthtime_grid[positive_idx]=}")
        return birthtime_grid[positive_idx], lifetime_grid[positive_idx], mass_grid[positive_idx]

            
class Wi:
    # Solves each integration item by integrating over birthtimes.
    # Input upper and lower mass limits (to be mapped onto birthtimes)
    # Gyr_age    (t)     is the Galactic age
    # birthtime (t')     is the stellar birthtime
    # lifetime (tau)    is the stellar lifetime
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
        self.Massive_birthtime_grid, self.Massive_lifetime_grid, self.Massive_mass_grid = self.Wi_grid_class.grids(self.IN.Ml_Massive, self.IN.Mu_Massive)
        #print('LIMs channel')
        self.LIMs_birthtime_grid, self.LIMs_lifetime_grid, self.LIMs_mass_grid = self.Wi_grid_class.grids(self.IN.Ml_LIMs, self.IN.Mu_LIMs) # !!!!!!! you should subtract SNIa fraction
        #print('SNIa channel')
        self.SNIa_birthtime_grid, self.SNIa_lifetime_grid, self.SNIa_mass_grid = self.Wi_grid_class.grids(self.IN.Ml_SNIa, self.IN.Mu_SNIa)
        self.yield_load = None
        
    def grid_picker(self, channel_switch, grid_type):
        # Selects e.g. "self.LIMs_birthtime_grid"
        # channel_switch:        can be 'LIMs', 'SNIa', 'Massive'
        # grid_type:            can be 'birthtime', 'lifetime', 'mass' 
        return self.__dict__[channel_switch+'_'+grid_type+'_grid']
    
    def SFR_component(self, birthtime_grid):
        # Returns the interpolated SFR vector computed at the birthtime grids
        SFR_interp = interp.interp1d(self.time_chosen[:self.age_idx+1], self.SFR_v[:self.age_idx+1], fill_value='extrapolate')
        return SFR_interp(birthtime_grid)

    def Z_component(self, birthtime_grid):
        # Returns the interpolated SFR vector computed at the birthtime grids
        _Z_interp = interp.interp1d(self.time_chosen[:self.age_idx+1], self.Z_v[:self.age_idx+1], fill_value='extrapolate')
        return _Z_interp(birthtime_grid)

    def IMF_component(self, mass_grid):
        # Returns the IMF vector computed at the mass grids
        return self.IMF(mass_grid)
    
    def dMdtauM_component(self, lifetime_grid, derlog=None): #!!!!!!!
        # computes the derivative of M(tauM) w.r.t. tauM
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
                print('%s currently not included.'%channel_switch)
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
        # Portinari+98, page 22, last eq. first column
        birthtime_grid = self.grid_picker(channel_switch, 'birthtime')
        IMF_comp = self.IMF_component(mass_grid) # overwrite continuously in __init__
        return IMF_comp, IMF_comp * self.dMdtauM_component(np.log10(lifetime_grid)) 

    def compute_rate(self, channel_switch='Massive'):
        # Computes the Type II SNae rate 
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
        # Computes, using the Simpson rule, the integral Wi 
        # elements of eq. (34) Portinari+98 -- for stars that die at tn, for every i
        vel_idx = self.IN.LC18_vel_idx if vel_idx is None else vel_idx
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