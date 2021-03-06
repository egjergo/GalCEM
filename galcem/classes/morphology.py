import math
import time
import dill
import os
import numpy as np
import scipy.integrate as integr
import scipy.misc as sm 

""""""""""""""""""""""""""""""""""""""""""""""""
"                                              "
"              MORPHOLOGY CLASSES              "
"  Contains classes that shape the simulated   "
"    galaxies, or otherwise integrates the     "
"      contributions by individual yields      "
"                                              "
" LIST OF CLASSES:                             "
"    __        Auxiliary                       "
"    __        Stellar_Lifetimes               "
"    __        Infall                          "
"    __        Initial_Mass_Function           "
"    __        Star_Formation_Rate             "
"                                              "
""""""""""""""""""""""""""""""""""""""""""""""""

class Auxiliary:
    def varname(self, var, dir=locals()):
        return [ key for key, val in dir.items() if id( val) == id( var)][0]

    def is_monotonic(self, arr):
        #print ('for %s'%varname(arr)) 
        if all(arr[i] <= arr[i + 1] for i in range(len(arr) - 1)): 
            return "monotone increasing" 
        elif all(arr[i] >= arr[i + 1] for i in range(len(arr) - 1)):
            return "monotone decreasing"
        return "not monotonic array"
    
    def find_nearest(self, array, value):
        '''Returns the index in array s.t. array[idx] is closest to value(float)'''
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return idx
    
    def deriv(self, func, x, n=1):
        ''' Returns the nth order derivative of a function '''
        return sm.derivative(func, x)

    def tic_count(self, string="Computation time", tic=None):
        tic.append(time.process_time())
        m = math.floor((tic[-1] - tic[-2])/60.)
        s = ((tic[-1] - tic[-2])%60.)
        print('%s = %d minutes and %d seconds'%(string,m,s))

    def pick_ZA_sorted_idx(self, ZA_sorted, Z=1,A=1):
        return np.intersect1d(np.where(ZA_sorted[:,0]==Z), np.where(ZA_sorted[:,1]==A))[0]

    def age_from_z(self, zf, h = 0.7, OmegaLambda0 = 0.7, Omegam0 = 0.3, Omegar0 = 1e-4, lookback_time = False):
        '''
        Finds the age (or lookback time) given a redshift (or scale factor).
        Assumes flat LCDM. Std cosmology, zero curvature. z0 = 0, a0 = 1.
        Omegam0 = 0.28 # 0.24 DM + 0.04 baryonic
        
        INPUT:    
        (Deprecated input aem = Float. Scale factor.) #zf = np.reciprocal(np.float(aem))-1
        (OmegaLambda0 = 0.72, Omegam0 = 0.28)
        zf  = redshift
        
        OUTPUT
        lookback time.
        '''
        H0 = 100 * h * 3.24078e-20 * 3.15570e16 # [ km s^-1 Mpc^-1 * Mpc km^-1 * s Gyr^-1 ]
        age = integr.quad(lambda z: 1 / ( (z + 1) *np.sqrt(OmegaLambda0 + 
                                Omegam0 * (z+1)**3 + Omegar0 * (z+1)**4) ), 
                                zf, np.inf)[0] / H0 # Since BB [Gyr]
        if not lookback_time:
            return age
        else:
            age0 = scipy.ntegrate.quad(lambda z: 1 / ( (z + 1) *np.sqrt(OmegaLambda0 
                                + Omegam0 * (z+1)**3 + Omegar0 * (z+1)**4) ),
                                 0, np.inf)[0] / H0 # present time [Gyr]
            return age0 - age

    def RK4(self, f, t, y, n, h, **kwargs):
        '''
        Classic Runge-Kutta 4th order for solving:     dy/dt = f(t,y,n)
        
        INPUT
            f    explicit function
            t    independent variable
            y    dependent variable
            n    timestep index
            h    timestep width (delta t)
        
        RETURN
            next timestep
        '''
        k1 = f(t, y, n, **kwargs)
        k2 = f(t+0.5*h, y+0.5*h*k1, n, **kwargs)
        k3 = f(t+0.5*h, y+0.5*h*k2, n, **kwargs)
        k4 = f(t+h, y+h*k3, n, **kwargs)
        return y + h * (k1 + 2*k2 + 2*k3 + k4) / 6


class Stellar_Lifetimes:
    '''
    Interpolation of Portinari+98 Table 14
    
    The first column of s_lifetimes_p98 identifies the stellar mass
    All the other columns indicate the respective lifetimes, 
    evaluated at different metallicities.
    '''
    def __init__(self, IN):
        self.IN = IN
        s_mlz_root = os.path.dirname(__file__)+'/../../yield_interpolation/lifetime_mass_metallicity/'
        self.s_mass = self.IN.s_lifetimes_p98['M'].values
        self.lifetime_by_mass_metallicity_loaded = dill.load(open(s_mlz_root+'models/lifetime_by_mass_metallicity.pkl','rb'))
        self.mass_by_lifetime_metallicity_loaded = dill.load(open(s_mlz_root+'models/mass_by_lifetime_metallicity.pkl','rb'))
    
    def interp_stellar_lifetimes(self, mz):
        '''Picks the tau(M) interpolation at the appropriate metallicity'''
        return self.lifetime_by_mass_metallicity_loaded(mz)

    def interp_stellar_masses(self, lz):
        '''Picks the M(tau) interpolation at the appropriate metallicity'''
        return self.mass_by_lifetime_metallicity_loaded(lz)

    def dMdtauM(self, lz):#, time_chosen, n=1):
        '''
        Computes the first order derivative of the M(tau) function
        with respect to dtau, but multiplied by dtau/dt' = -1
        '''
        return - self.mass_by_lifetime_metallicity_loaded(lz,dwrt='lifetime_Gyr')
        
        
class Infall:
    '''
    CLASS
    Computes the infall as an exponential decay
    
    EXAMPLE:
        Infall_class = Infall(morphology)
        infall = Infall_class.inf()
    (infall will contain the infall rate at every time i in time)
    '''
    def __init__(self, IN, morphology=None, option=None, time=None):
        self.IN = IN
        self.morphology = self.IN.morphology if morphology is None else morphology
        self.time = time
        self.option = self.IN.inf_option if option is None else option
    
    def infall_func_simple(self):
        '''
        Analytical implicit function of an exponentially decaying infall.
        Only depends on time (in Gyr)
        '''
        return lambda t: np.exp(-t / self.IN.tau_inf)
        
    def two_infall(self):
        '''
        My version of Chiappini+01
        '''
        # After a radial dependence upgrade
        return None
    
    def infall_func(self):
        '''
        Picks the infall function based on the option
        '''
        if not self.option:
            return self.infall_func_simple()
        elif self.option == 'two-infall':
            return self.two_infall()
    
    def aInf(self):
        """
        Computes the infall normalization constant
        
        USED IN:
            inf() and SFR()
        """
        return np.divide(self.IN.M_inf, integr.quad(self.infall_func(),
                             self.time[0], self.IN.age_Galaxy)[0])

    def inf(self):
        '''
        Returns the infall array
        '''
        return lambda t: self.aInf() * self.infall_func()(t)
    
        
class DTD:
    '''
    For all delayed time distibutions
    '''
    def __init__(self):
        self.A_Ia = .35
        self.t0_Ia = 0.150 # [Gyr]
        self.tau_Ia = 1.1
        
    def MaozMannucci12(self, t):
        if t > self.t0_Ia:
            return t**(- self.tau_Ia)
        else:
            return 0.
        
    def DTD_select(self):
        if not self.custom:
            if self.option == 'mm12':
                    return self.MaozMannucci12()
        if self.custom:
            return self.custom
        
class Initial_Mass_Function:
    '''
    CLASS
    Instantiates the IMF
    
    You may define custom IMFs, paying attention to the fact that:
    
    integrand = IMF * Mstar
    $\int_{Ml}^Mu integrand dx = 1$
    
    REQUIRES
        Mass (float) as input
    
    RETURNS
        normalized IMF by calling .IMF() onto the instantiated class
    
    Accepts a custom function 'func'
    Defaults options are 'Salpeter55', 'Kroupa03', [...]
    '''
    def __init__(self, Ml, Mu, IN, option=None, custom_IMF=None):
        self.Ml = Ml
        self.Mu = Mu
        self.IN = IN
        self.option = self.IN.IMF_option if option is None else option
        self.custom = self.IN.custom_IMF if custom_IMF is None else custom_IMF
    
    def Salpeter55(self, plaw=None):
        plaw = self.IN.Salpeter_IMF_Plaw if plaw is None else plaw
        return lambda Mstar: Mstar ** (-(1 + plaw))
        
    def canonical_IMF(self, Mstar):
        if Mstar <= 0.5:
            plaw = -0.3
        elif self.mass <= 1.0:
            plaw = 1.2
        else:
            plaw = 1.7
        return Mstar **(-(1 + plaw))
        
    def Kroupa93(self):#, C = 0.31): #if m <= 0.5: lambda m: 0.58 * (m ** -0.30)/m
        '''
        Kroupa, Tout & Gilmore (1993)
        '''
        return lambda Mstar: self.canonical_IMF(Mstar)
        
    def IMF_select(self):
        if not self.custom:
            if self.option == 'Salpeter55':
                    return self.Salpeter55()
            if self.option == 'Kroupa03':
                    return self.Kroupa03()
        if self.custom:
            return self.custom
    
    def integrand(self, Mstar):
        return Mstar * self.IMF_select()(Mstar)
        
    def normalization(self): 
        return np.reciprocal(integr.quad(self.integrand, self.Ml, self.Mu)[0])

    def IMF(self): #!!!!!!!! might not be efficient with the IGIMF
        return lambda Mstar: self.integrand(Mstar) * self.normalization()
        
    def IMF_test(self):
        '''
        Returns the normalized integrand integral. If the IMF works, it should return 1.
        '''
        return self.normalization() * integr.quad(self.integrand, self.Ml, self.Mu)[0]
        
        
class Star_Formation_Rate:
    '''
    CLASS
    Instantiates the SFR
    
    Accepts a custom function 'func'
    Defaults options are 'SFRgal', 'CSFR', [...]
    '''
    def __init__(self, IN, option=None, custom=None, option_CSFR=None, morphology=None):
        self.IN = IN
        self.option = self.IN.SFR_option if option is None else option
        self.custom = self.IN.custom_SFR if custom is None else custom
        self.option_CSFR = self.IN.CSFR_option if option_CSFR is None else option_CSFR
        self.morphology = self.IN.morphology if morphology is None else morphology

    def SFRgal(self, k=None, Mgas=[], Mtot=[], timestep_n=0): 
        ''' Talbot & Arnett (1975)'''
        k = self.IN.k_SFR if k is None else k
        f_g = Mgas[timestep_n] / Mtot[timestep_n]
        return self.IN.nu  * (Mgas[timestep_n]) * f_g**(k-1) * self.IN.SFR_rescaling
    
    
    def SFR_G(self, k=None, G=[], Mtot=[], timestep_n=0): 
        ''' Talbot & Arnett (1975)'''
        k = self.IN.k_SFR if k is None else k
        return np.multiply(self.IN.nu  * (G[timestep_n])**(k))
            #/ (Mtot[timestep_n])**(k-1), self.IN.SFR_rescaling)
    
    def CSFR(self):
        '''
        Cosmic Star Formation rate dictionary
            'md14'        Madau & Dickinson (2014)
            'hb06'        Hopkins & Beacom (2006)
            'f07'        Fardal (2007)
            'w08'        Wilken (2008)
            'sh03'        Springel & Hernquist (2003)
        '''
        CSFR = {'md14': (lambda z: (0.015 * np.power(1 + z, 2.7)) 
                         / (1 + np.power((1 + z) / 2.9, 5.6))), 
                'hb06': (lambda z: 0.7 * (0.017 + 0.13 * z) / (1 + (z / 3.3)**5.3)), 
                'f07': (lambda z: (0.0103 + 0.088 * z) / (1 + (z / 2.4)**2.8)), 
                'w08': (lambda z: (0.014 + 0.11 * z) / (1 + (z / 1.4)**2.2)), 
                'sh03': (lambda z: (0.15 * (14. / 15) * np.exp(0.6 * (z - 5.4)) 
                         / (14.0 / 15) - 0.6 + 0.6 * np.exp((14. / 15) * (z - 5.4))))}
        return CSFR.get(self.option_CSFR, "Invalid CSFR option")
        
    def SFR(self, Mgas=None, Mtot=None, timestep_n=0):
        if not self.custom:
            if self.option == 'SFRgal':
                    return self.SFRgal(Mgas=Mgas, Mtot=Mtot, timestep_n=timestep_n)
            elif self.option == 'CSFR':
                if self.option_CSFR:
                    return self.CSFR(self.option_CSFR)
                else:
                    print('Please define the CSFR option "option_CSFR"')
        if self.custom:
            print('Using custom SFR')
            return self.custom
            
    def outflow(self, Mgas, morphology, SFR=SFRgal, wind_eff=None, k=1):
        wind_eff = self.IN.wind_efficiency if wind_eff is None else wind_eff
        return wind_eff * SFR(Mgas, morphology)