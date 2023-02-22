""""""""""""""""""""""""""""""""""""""""""""""""
"                                              "
"             INTEGRATION CLASSES              "
"   Contains classes that solve the integral   " 
"  part of the integro-differential equations  "
"                                              "
" LIST OF CLASSES:                             "
"    __        Wi_grid                         "
"    __        Wi                              "
"                                              "
""""""""""""""""""""""""""""""""""""""""""""""""
import time
import numpy as np
import pandas as pd
import scipy.interpolate as interp
import scipy.integrate as integr

from ..classes import morphology as morph
from ..classes.inputs import Auxiliary


class Wi_grid:
    ''' grids for the Wi integral '''
    def __init__(self, metallicity, age_idx, IN, lifetime_class, time_chosen, Z_v):
        self.metallicity = metallicity 
        self.age_idx = age_idx
        self.IN = IN
        self.Z_v = Z_v
        self.lifetime_class = lifetime_class
        self.time_chosen = time_chosen
    
    def __repr__(self):
        aux = Auxiliary()
        return aux.repr(self)

    def grids(self, Ml_lim, Mu_lim):
        '''
        Computes the mass, stellar lifetime and birthtime grids
        employed in the rate intragrals.
        
        Ml_lim and Mu_lim are mass limits for each channel.
        '''
        mass_grid = np.geomspace(Ml_lim, Mu_lim, num = self.IN.num_MassGrid)
        
        df_mz = pd.DataFrame(np.array([mass_grid, np.ones(len(mass_grid))*
                        self.metallicity]).T, columns=['mass', 'metallicity'])
        lifetime_grid0 = self.lifetime_class.interp_stellar_lifetimes(df_mz)
        birthtime_grid0 = self.time_chosen[self.age_idx] - lifetime_grid0 
        metallicity_grid0 = self.Z_component(birthtime_grid0)
        metallicity_grid = np.ones(len(metallicity_grid0))
        
        # # Make sure the grid points for the integration converge
        # to_sec = 30
        # timeout = time.time() + to_sec 
        # while not np.allclose(metallicity_grid, metallicity_grid0, equal_nan=False):
        #     metallicity_grid0 = metallicity_grid
        df_mz = pd.DataFrame(np.array([mass_grid, metallicity_grid0]).T,
                                  columns=['mass', 'metallicity'])
        lifetime_grid = self.lifetime_class.interp_stellar_lifetimes(df_mz)
        birthtime_grid = self.time_chosen[self.age_idx] - lifetime_grid
        metallicity_grid = self.Z_component(birthtime_grid)
        #     if time.time() > timeout:
        #         print(f"Error: The metallicity grid isn't converging within {to_sec} seconds.")
        # #         print(f"{metallicity_grid0=}")
        # #         print(f"{metallicity_grid=}")
        #         break     
        positive_idx = np.where(birthtime_grid > 0.)
        return (birthtime_grid[positive_idx], lifetime_grid[positive_idx], 
                mass_grid[positive_idx])

    def Z_component(self, birthtime_grid):
        # Returns the interpolated SFR vector computed at the birthtime grids
        _Z_interp = interp.interp1d(self.time_chosen[:self.age_idx+1],
                    self.Z_v[:self.age_idx+1], fill_value='extrapolate')
        return _Z_interp(birthtime_grid) # Linear metallicity
            
class Wi:
    '''
    Solves each integration item by integrating over birthtimes.
    Input upper and lower mass limits (to be mapped onto birthtimes)
    Gyr_age    (t)     is the Galactic age
    birthtime (t')     is the stellar birthtime
    lifetime (tau)    is the stellar lifetime
    '''
    def __init__(self, age_idx, IN, lifetime_class, time_chosen, Z_v, SFR_v, Greggio05_SD, IMF, ZA_sorted):
        self.IN = IN
        self.lifetime_class = lifetime_class
        self.time_chosen = time_chosen
        self.Z_v = Z_v
        self.SFR_v = SFR_v
        self.Greggio05_SD = Greggio05_SD
        self.IMF = IMF
        self.ZA_sorted = ZA_sorted
        self.metallicity = self.Z_v[age_idx]
        self.age_idx = age_idx
        self.Wi_grid_class = Wi_grid(self.metallicity, self.age_idx, self.IN, lifetime_class, self.time_chosen, self.Z_v)
        self.MRSN_birthtime_grid, self.MRSN_lifetime_grid, self.MRSN_mass_grid = self.Wi_grid_class.grids(self.IN.Ml_MRSN, self.IN.Mu_MRSN)
        self.NSM_birthtime_grid, self.NSM_lifetime_grid, self.NSM_mass_grid = self.Wi_grid_class.grids(self.IN.Ml_NSM, self.IN.Mu_NSM)
        self.SNCC_birthtime_grid, self.SNCC_lifetime_grid, self.SNCC_mass_grid = self.Wi_grid_class.grids(self.IN.Ml_SNCC, self.IN.Mu_SNCC)
        self.LIMs_birthtime_grid, self.LIMs_lifetime_grid, self.LIMs_mass_grid = self.Wi_grid_class.grids(self.IN.Ml_LIMs, self.IN.Mu_LIMs) # !!!!!!! you should subtract SNIa fraction
        self.SNIa_birthtime_grid, self.SNIa_lifetime_grid, self.SNIa_mass_grid = self.Wi_grid_class.grids(self.IN.Ml_SNIa, self.IN.Mu_SNIa)
        self.yield_load = None
    
    def __repr__(self):
        aux = Auxiliary()
        return aux.repr(self)
        
    def grid_picker(self, channel_switch, grid_type):
        '''
        Selects e.g. "self.LIMs_birthtime_grid"
        channel_switch:        can be 'LIMs', 'SNIa', 'SNCC'
        grid_type:            can be 'birthtime', 'lifetime', 'mass' 
        '''
        return self.__dict__[channel_switch+'_'+grid_type+'_grid']
    
    def Z_component(self, birthtime_grid):
        ''' Returns the interpolated metallicity vector
        computed at the birthtime grids'''
        _Z_interp = interp.interp1d(self.time_chosen[:self.age_idx+1], 
                    self.Z_v[:self.age_idx+1], fill_value='extrapolate')
        return _Z_interp(birthtime_grid) # Linear metallicity
    
    def SFR_component(self, birthtime_grid):
        '''Returns the interpolated SFR vector 
        computed at the birthtime grids'''
        SFR_interp = interp.interp1d(self.time_chosen[:self.age_idx+1],
                    self.SFR_v[:self.age_idx+1], fill_value='extrapolate')
        return SFR_interp(birthtime_grid)

    def IMF_component(self, mass_grid):
        # Returns the IMF vector computed at the mass grids
        return self.IMF(mass_grid)
    
    def dMdtauM_component(self, lifetime_grid, birthtime_grid, derlog=None):
        # computes the derivative of M(tauM) w.r.t. tauM
        derlog = self.IN.derlog if derlog is None else derlog
        if derlog == False:
            metallicity_grid = self.Z_component(birthtime_grid)
            df_lz = pd.DataFrame(np.array([lifetime_grid,
                            metallicity_grid]).T, columns=['lifetime_Gyr',
                                                           'metallicity'])
            return self.lifetime_class.dMdtauM(df_lz)
        if derlog == True:
            return 1   

    def yield_component(self, channel_switch, mass_grid, birthtime_grid, vel_idx=None):
        return interpolation(mass_grid, metallicity(birthtime_grid))
    
    def mass_component(self, channel_switch, mass_grid, lifetime_grid, birthtime_grid): #
        # Portinari+98, page 22, last eq. first column
        birthtime_grid = self.grid_picker(channel_switch, 'birthtime')
        IMF_comp = self.IMF_component(mass_grid) # overwrite continuously in __init__
        return IMF_comp, IMF_comp * self.dMdtauM_component(lifetime_grid, birthtime_grid) 

    def integr_SNIa(self, mass_grid):
        f_nu = lambda nu: 24 * (1 - nu)**2        
        nu_min = np.max([0.5, np.max(np.divide(mass_grid, self.IN.Mu_SNIa))])
        nu_max = np.min([1, np.min(np.divide(mass_grid, 0.5 * self.IN.Ml_SNIa))])
        nu_v = np.linspace(nu_min, nu_max, num=len(mass_grid))
        return [integr.simps(np.multiply(f_nu(nu_v), self.IMF_component(np.divide(m, nu_v))), x=nu_v) for m in mass_grid]

    def DTD_SNIa(self, mass_grid):
        #print(f'{nu_min = },\t {nu_max=}')
        #print(f'nu_min = 0.5,\t nu_max= 1.0')
        #nu_test = np.linspace(0.5, 1, num=len(mass_grid))
        F_SNIa = self.integr_SNIa(mass_grid)   
        #self.f_SNIa_v[self.age_idx] = F_SNIa
        return F_SNIa 
    
    def _compute_rateSNIa(self, channel_switch='SNIa'):
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
        self.f_SNIa_v[self.age_idx] = F_SNIa 
        SFR_comp = self.SFR_component(birthtime_grid)
        integrand = np.multiply(SFR_comp, F_SNIa)
        return integr.simps(integrand, x=birthtime_grid)
 
    def compute_rateSNIa(self, channel_switch='SNIa'):
        birthtime_grid = self.grid_picker(channel_switch, 'birthtime')
        lifetime_grid = self.grid_picker(channel_switch, 'lifetime')
        SFR_comp = self.SFR_component(birthtime_grid)
        F_SNIa = np.array([D.f_SD_Ia for D in self.Greggio05_SD(lifetime_grid)])
        #F_SNIa = [DTD_class.MaozMannucci12(t) for t in lifetime_grid]
        integrand = np.multiply(SFR_comp, F_SNIa)
        return integr.simps(integrand, x=lifetime_grid)
 
    def compute_rate(self, channel_switch='SNCC'):
        # Computes the Type II SNae rate 
        birthtime_grid = self.grid_picker(channel_switch, 'birthtime')
        mass_grid = self.grid_picker(channel_switch, 'mass')
        SFR_comp = self.SFR_component(birthtime_grid)
        SFR_comp[SFR_comp<0] = 0.
        IMF_comp = self.IMF_component(mass_grid)
        integrand = np.multiply(SFR_comp, IMF_comp)
        return integr.simps(integrand, x=mass_grid)
    
    def exec_compute_rate(self, channel_switch):
        if channel_switch == 'SNIa':
            if len(self.grid_picker('SNIa', 'birthtime')) > 0.:
                #R_SNIa = self.IN.A_SNIa * self.compute_rate(channel_switch='SNIa')
                return self.compute_rateSNIa()
            else:
                return self.IN.epsilon  
        else:
            if len(self.grid_picker(channel_switch, 'birthtime')) > 0.:
                return self.compute_rate(channel_switch=channel_switch)
            else:
                return self.IN.epsilon
    
    def compute_rates(self):
        rates = {}
        for ch in self.IN.include_channel:
            rates[ch] = self.exec_compute_rate(ch)
        return rates #[self.exec_compute_rate(ch) for ch in self.IN.include_channel]

    def compute(self, channel_switch, vel_idx=None):
        if channel_switch == "SNIa":
            return {'integrand': [0.5], 'birthtime_grid': [0.5], 'mass_grid': [0.5]}
        else:
            vel_idx = self.IN.LC18_vel_idx if vel_idx is None else vel_idx
            mass_grid = self.grid_picker(channel_switch, 'mass')
            lifetime_grid = self.grid_picker(channel_switch, 'lifetime')        
            birthtime_grid = self.grid_picker(channel_switch, 'birthtime')
            SFR_comp = self.SFR_component(birthtime_grid)
            SFR_comp[SFR_comp<0] = 0.
            IMF_comp, mass_comp = self.mass_component(channel_switch, mass_grid, lifetime_grid, birthtime_grid) 
            #integrand = np.prod(np.vstack[SFR_comp, mass_comp, self.yield_load[i]])
            integrand = np.prod(np.vstack([SFR_comp, mass_comp]), axis=0)
            #return integr.simps(integrand, x=birthtime_grid)
            return {'integrand': integrand, 'birthtime_grid': birthtime_grid, 'mass_grid': mass_grid}