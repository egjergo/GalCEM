""""""""""""""""""""""""""""""""""""""""""""""""
"                                              "
"              MORPHOLOGY CLASSES              "
"         Contains classes that shape          "
"            the simulated galaxies            "
"                                              "
" LIST OF CLASSES:                             "
"    __        Infall                          "
"    __        Star_Formation_Rate             "
"    __        Initial_Mass_Function           "
"    __        Stellar_Lifetimes               "
"    __        Greggio05                       "
"    __        DTD                             "
"                                              "
""""""""""""""""""""""""""""""""""""""""""""""""

import math, time
import os
import dill
import numpy as np
import scipy.integrate
import scipy.misc as sm 
import scipy.interpolate as interp
import scipy.integrate as integr
import scipy.stats as ss


class Infall:
    '''
    CLASS
    Computes the gas infall. The default is an exponential decay
    
    INPUT:
        IN    [instance of a class] an instance of inputs
                                    (see examples/mwe.py)
    
    EXAMPLE:
    >>> import galcem as gc
    >>> import numpy as np
    >>> inputs = gc.Inputs()
    >>> time_v = np.arange(0.5,13.8, 0.01)
    >>> Infall_class = gc.morph.Infall(inputs)
    >>> infall_func = Infall_class.inf()
    >>> infall_v = infall_func(time_v)
    
    ("infall_v" will contain the infall rate [Msun/Gyr] at every timestep i)
    '''
    def __init__(self, IN, morph=None, option=None, time=None):
        self.IN = IN
        self.morphology = self.IN.morphology if morph is None else morph
        self.time = time
        self.option = self.IN.inf_option if option is None else option
    
    def __repr__(self):
        print('\nThese are the parameter values of the class\n')
        import json
        print(json.dumps(self.__dict__, default=str, indent=4))
        print('\nThese are the class contents\n')
        contents = [(x, type(x)) for x in self.__dir__() if not x.startswith('__')]
        return '\n'.join(contents)
        #return '\n'.join(self.__dir__())
        #return '\n'.join(self.__doc__)
    
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
        return np.divide(self.IN.M_inf, scipy.integrate.quad(self.infall_func(),
                             self.time[0], self.IN.age_Galaxy)[0])

    def inf(self):
        '''
        Returns the infall array
        '''
        return lambda t: self.aInf() * self.infall_func()(t)


class Star_Formation_Rate:
    '''
    CLASS
    Instantiates the SFR
    
    Accepts a custom function 'func'
    Defaults options are 'SFRgal', 'CSFR', [...]
    '''
    def __init__(self, IN, option=None, custom=None, option_CSFR=None, morph=None):
        self.IN = IN
        self.option = self.IN.SFR_option if option is None else option
        self.custom = self.IN.custom_SFR if custom is None else custom
        self.option_CSFR = (self.IN.CSFR_option if option_CSFR is None 
                                                else option_CSFR)
        self.morphology = self.IN.morphology if morph is None else morph

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
    Defaults options are 'Salpeter55', 'Kroupa01', [...]
    '''
    def __init__(self, Ml, Mu, IN, option=None, custom_IMF=None):
        self.Ml = Ml
        self.Mu = Mu
        self.IN = IN
        self.option = self.IN.IMF_option if option is None else option
        self.custom = self.IN.custom_IMF if custom_IMF is None else custom_IMF
        self.xi0 = self.normalization()
    
    def powerlaw(self, Mstar, alpha=2.3):
        return Mstar**(-alpha)
    
    def Salpeter55(self, plaw=None):
        plaw = self.IN.Salpeter_IMF_Plaw if plaw is None else plaw
        return lambda Mstar: self.powerlaw(Mstar, alpha=plaw)
        
    def Kroupa01(self, alpha1=1.3, alpha2=2.3, alpha3=2.3):
        return lambda Mstar: np.piecewise(Mstar, 
                            [np.logical_or(Mstar < self.IN.Ml_LIMs, Mstar >=  self.IN.Mu_SNCC),
                             np.logical_and(Mstar >=  self.IN.Ml_LIMs, Mstar < 0.5),
                             np.logical_and(Mstar >= 0.5, Mstar < 1.),
                             np.logical_and(Mstar >= 1., Mstar < self.IN.Mu_SNCC)],
                            [0., 
                             lambda M: 2 * self.powerlaw(M, alpha=alpha1), 
                             lambda M: self.powerlaw(M, alpha=alpha2), 
                             lambda M: self.powerlaw(M, alpha=alpha3)])
        
    def IMF_select(self):
        if not self.custom:
            if self.option == 'Salpeter55': return self.Salpeter55()
            if self.option == 'Kroupa01' or self.option == 'canonical': return self.Kroupa01()
        if self.custom:
            return self.custom
    
    def integrand(self, Mstar):
        return Mstar * self.IMF_select()(Mstar)
        
    def normalization(self): 
        return np.reciprocal(integr.quad(self.integrand, self.Ml, self.Mu)[0])

    def IMF(self): #!!!!!!!! it is missing the time dependence (for the IGIMF or custom IMFs)
        #return lambda Mstar: self.IMF_select()(Mstar) * self.normalization()
        return lambda Mstar: self.IMF_select()(Mstar) * self.normalization()
        
    def massweighted_IMF(self): #!!!!!!!! it is missing the time dependence (for the IGIMF or custom IMFs)
        #return lambda Mstar: self.IMF_select()(Mstar) * self.normalization()
        return lambda Mstar: self.integrand(Mstar) * self.normalization()
    
    def IMF_fraction(self, Mlow, Mhigh, massweighted=True):
        '''
        If massweighted==True, returns the fraction by mass of the stars 
        within [Mlow, Mhigh] w.r.t. the total mass-weighted IMF.
        
        If massweighted==False, computes the same fraction, by number,
        w.r.t. the IMF.
        '''
        if massweighted==True:
            function = self.massweighted_IMF()
        else:
            function = self.IMF()
        numerator = integr.quad(function, Mlow, Mhigh)[0]
        denominator = integr.quad(function, self.Ml, self.Mu)[0]
        return np.divide(numerator, denominator)
    
    def IMF_test(self):
        '''
        Returns the normalized integrand integral. If the IMF works, it should return 1.
        '''
        return self.normalization() * integr.quad(self.integrand, self.Ml, self.Mu)[0]


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
    
    def interp_stellar_lifetimes(self, df_mass_metallicity):
        '''Picks the tau(M) interpolation at the appropriate metallicity'''
        return self.lifetime_by_mass_metallicity_loaded(df_mass_metallicity)

    def interp_stellar_masses(self, df_lifetime_metallicity):
        '''Picks the M(tau) interpolation at the appropriate metallicity'''
        return self.mass_by_lifetime_metallicity_loaded(df_lifetime_metallicity)

    def dMdtauM(self, df_lifetime_metallicity):#, time_chosen, n=1):
        '''
        Computes the first order derivative of the M(tau) function
        with respect to dtau, but multiplied by dtau/dt' = -1
        '''
        return - self.mass_by_lifetime_metallicity_loaded(df_lifetime_metallicity,dwrt='lifetime_Gyr')


class Greggio05:
    '''Greggio (2005, A&A 441, 1055G) Single degenerate
    https://ui.adsabs.harvard.edu/abs/2005A%26A...441.1055G/abstract 
    
    tauMS in Gyr'''
    def __init__(self, tauMS):
        self.tauMS = tauMS # 
        self.K = 0.86 # Valid for Kroupa01, alpha=2.35, gamma=1 of Eq. (16)
        self.k_alpha = 1.55 # For Kroupa01, 2.83 for Salpeter55
        self.A_Ia = 1e-3 # For Kroupa01, 5e-4 for Salpeter55
        self.alpha = 2.35
        self.gamma = 1
        self.epsilon = 1 # Represented as solid and dashed lines in Fig. 2 for 1 and 0.5 respectively
        self.m2 = self.Girardi00_secondary_lifetime()
        self.m2c = self.m2c_func()
        self.m2e = self.m2e_func()
        self.mWDn = self.mWDn_func()
        self.m1n = self.m1n_func()
        self.m1i = self.m1i_func()
        self.n_SD = self.SD_n_m2()
        self.deriv_m2_abs = self.abs_deriv_m2()
        self.f_SD_Ia = 10**self.K * self.n_SD * self.deriv_m2_abs #self.f_SD_Ia_func()
        
    def f_SD_Ia_func(self):
        val = 10**self.K * self.n_SD * self.deriv_m2_abs
        if val > 0.:
            return val
        else:
            return 0.

    def Girardi00_secondary_lifetime(self):
        '''Eq. (12) returns m2'''
        logtauMS = np.log10(self.tauMS*1e9, where=self.tauMS>0.)
        if np.logical_and(self.tauMS> 0.04, self.tauMS< 25):
            return np.piecewise(logtauMS, [logtauMS==0., logtauMS > 0.], 
                            [1e-32, 10**(0.0471*logtauMS**2 - 1.2*logtauMS + 7.3)])
        else:
            return 1e-32
        
    def SD_n_m2(self):
        '''Distribution function of the secondaries in SNIa progenitor systems
        obtained by summing over all possible primaries, ranging from 
        a minimum value (m_{1,i}) to 8 Msun
        Eq. (16)'''
        if self.m2<=8:
            exponent = self.alpha + self.gamma
            return self.m2**(-self.alpha) * ((self.m2/self.m1i)**exponent - (self.m2/8)**exponent)
        else:
            return 0.
    
    def m1i_func(self):
        return np.amax([self.m2, self.m1n])
    
    def m1n_func(self):
        '''Eq. (19)'''
        return np.amax([2., 2. + 10.*(self.mWDn - 0.6)])
    
    def mWDn_func(self):
        '''Eq. (17)'''
        return 1.4 - self.epsilon * self.m2e
    
    def m2e_func(self):
        '''right after Eq. (18)'''
        return self.m2 - self.m2c
    
    def m2c_func(self):
        '''Eq. (18)'''
        return np.amax([0.3, 0.3 + 0.1*(self.m2-2), 0.15*(self.m2-4)])
    
    def abs_deriv_m2(self):
        '''Page 5 right after Eq. (14)
        $|\dot{m}_2| \propto \tau^{-1.44}$'''
        return np.power(self.tauMS,-1.44, where=self.tauMS>0.)
      
     
class DTD:
    '''
    For all delayed time distibutions (inactive for now)
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

