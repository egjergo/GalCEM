""""""""""""""""""""""""""""""""""""""""""""""""
"                                              "
"                  INPUT CLASS                 "
"     Contains the class that initializes      "
"    the input parameters to set up a run      " 
"                                              "
" LIST OF CLASSES:                             "
"    __        Inputs                          "
"    __        Auxiliary                       "
"                                              "
""""""""""""""""""""""""""""""""""""""""""""""""

import os
import time
import math
import numpy as np
import pandas as pd
import scipy.integrate as integr


class Inputs:
    """
    Configuration inputs for OneZone
    
    ags_Galaxy (float): age of the Galaxy (Gyr)
    """
    def __init__(self):
        '''	applies to the thick disk at 8 kpc '''        
        # Time parameters
        self.nTimeStep = 0.01 #0.002 #0.01 [Gyr]
        self.numTimeStep = 2000 #2000 # Like FM [how many timesteps]
        self.num_MassGrid = 200
        self.include_channel = ['SNCC', 'LIMs', 'SNIa']
        
        self.Galaxy_birthtime = 0.05 #0.1 # [Gyr]
        self.Galaxy_age = 13.8 # [Gyr]
        self.solar_age = 4.6 # [Gyr]
        self.solar_metallicity = 0.0134 # Asplund et al. (2009, Table 4)
        self.r = 8 # [kpc] Compute at the solar radius
        self.k_SFR = 1.4 # SFR power law exponent
        
        self.morphology = 'spiral'

        # Fraction of compact objects
        #self.A_SNIa = # Fixed inside morph.Greggio05() # Fraction of white dwarfs that underwent a SNIa
        self.A_NSM = 0.03 #0.06 # Fraction of neutron stars that underwent an episode of coalescence
        self.A_collapsars = 0.05 #0.06 # Fraction of massive stars that evolved into collapsars

        # Mass limits
        self.Ml_SNIa = 1.6 # Lower limit for total binary mass for SNIae [Msun]
        self.Mu_SNIa = 16 # Upper limit for total binary mass for SNIae [Msun]
        self.Ml_LIMs = 0.07 # [Msun] 
        self.Mu_LIMs = 9 # [Msun] 
        self.Ml_NSM = 9 # [Msun] 
        self.Mu_NSM = 50 # [Msun] 
        self.Ml_MRSN = 9 # [Msun] 
        self.Mu_MRSN = 50 # [Msun] 
        self.Ml_SNCC = 10 # [Msun] 
        self.Mu_SNCC = 120 # [Msun] 
        self.Ml_collapsars = 9 # [Msun] 
        self.Mu_collapsars = 150 # [Msun] 

        self.sd = 530.96618 # surf density coefficient for the disk (normalized to the MW mass?) !!!!!!!
        
        self.MW_SFR = 1.9 #+-0.4 [Msun/yr] from Chomiuk & Povich (2011) Galactic SFR (z=0)
        self.MW_RSNIa = np.empty(3) # Galactic SNIa rates (filled in Setup, from Mannucci+05)
        self.MW_RSNCC = np.empty(3) # Galactic SNCC rates (filled in Setup, from Mannucci+05)
        
        self.custom_IMF = None
        self.import_custom_SFR = None#np.loadtxt('galcem/input/custom_SFR/data.dat')
        if not self.import_custom_SFR.empty:
            self.custom_SFR = pd.DataFrame(
                {
                    'time': self.import_custom_SFR[:,0],
                    'SFR': self.import_custom_SFR[:,1]
                }
            )
        self.custom_SNIaDTD = None

        self.inf_option = None # None: default exponential decay, or 'two-infall'
        self.IMF_option = 'brokenplaw' #'canonical', 'Salpeter55', 'Kroupa01'
        if self.IMF_option == 'Salpeter55':
            self.IMF_single_slope = 2.35 # IMF Salpeter power law 
        if (self.IMF_option == 'Kroupa01') or (self.IMF_option == 'canonical') or (self.IMF_option == 'brokenplaw'):
            self.K01_params = { 
                'alpha0': 0.3,
                'alpha1': 1.3,
                'alpha2': 2.3,
                'alpha3': 2.3,
                'lim01': 0.08, # Keep it higher than the lower mass limit
                'lim12': 0.5,
                'lim23': 1.
            }
        self.SFR_option = 'SFRgal' # or 'CSFR'
        self.CSFR_option = None # !!!!!!! remove? None: no cosmic SFR, e.g. 'md14' for Madau & Dickinson (2014). This requires self.SFR_option='CSFR'
        self.SNIaDTD_option = 'Greggio05'# !!!!!!! 'GreggioRenzini83' # 'RuizMannucci01'

        self.yields_NSM_option = 'r14' # 'r14', 'tk'
        self.yields_MRSN_option = 'n17' # 'n17', 'tk'
        self.yields_Collapsar_option = 'tk' # 'n17', 'tk'
        self.yields_LIMs_option = 'c15' #'k10'
        self.yields_SNCC_option = 'lc18'
        self.LC18_vel_idx = 0 # !!!!!!! eventually you should write a function to compute this
        self.yields_SNIa_option = 'i99' # 'k20', 'i99'
        self.yields_BBN_option = 'gp13'

        self.epsilon = 1e-32 # Avoid numerical errors - consistent with BBN
        self.to_sec = .1 # Metallicity convergence break time
        
        _dir = os.path.join(os.path.dirname( __file__ ), '..')
        self.s_lifetimes_p98 = pd.read_csv(_dir+'/input/starlifetime/portinari98table14.dat')
        self.s_lifetimes_p98.columns = [name.replace('#M','M').replace('Z=0.','Z') for name in self.s_lifetimes_p98.columns]
        
        self.periodic = pd.read_csv(_dir+'/input/physics/periodicinfo.dat', sep=',', comment='#')
        
        self.asplund3 = pd.read_csv(_dir+'/input/physics/asplund09/table3.dat', sep=',', comment='#')
        self.asplund1 = pd.read_csv(_dir+'/input/physics/asplund09/table1.dat', sep=',', comment='#')
        for col in self.asplund1.columns[2:]:
            self.asplund1[col] = pd.to_numeric(self.asplund1[col], errors='coerce')
    
    def __repr__(self):
        aux = Auxiliary()
        return aux.repr(self)
    
    def default_params(self, choice, morphology):
        '''
        Dictionary of dictionaries. Picks default morphology parameters:
        M_inf              - Final baryonic mass of the galaxy [Msun]
        Reff               - effective radius [kpc]
        tau_inf            - infall timescale [Gyr]
        nu                 - star formation efficiency [Gyr^-1]
        wind_efficiency    - outflow parameter [dimensionless]
        
        Calibrations from Molero+21a and b
        
        Default options for morphologies include 
        characteristic values and specific values
        calibrated to a list of dwarf galaxies:
            'elliptical'
            'spiral'
            'irregular'
            'Fornax'
            'Sculptor'
            'ReticulumII'
            'BootesI'
            'Carina'
            'Sagittarius'
            'Sextan'
            'UrsaMinor'
        '''
        dictionary = {
            'M_inf' : {
                'elliptical': 5.0e11, 
                'spiral': 5.0e10,
                'irregular': 5.5e8,
                'Fornax': 5.0e8,
                'Sculptor': 1.0e8,
                'ReticulumII': 1.0e5,
                'BootesI': 1.1e7,
                'Carina': 5.0e8,
                'Sagittarius': 2.1e9,
                'Sextan': 5.0e8,
                'UrsaMinor': 5.0e8},
            'Reff' : {
                'elliptical': 7,
                'spiral': 3.5,
                'irregular': 1,
                'Fornax': 1, # !!!!!!! Ask!!!!!!! not on the paper
                'Sculptor': 1,
                'ReticulumII': 1,
                'BootesI': 1,
                'Carina': 1,
                'Sagittarius': 1,
                'Sextan': 1,
                'UrsaMinor': 1},
            'tau_inf' : {
                'elliptical': 0.2, 
            'spiral': 7.,
            'irregular': 7.,
            'Fornax': 3,
            'Sculptor': 0.5,
            'ReticulumII': 0.05,
            'BootesI': 0.05,
            'Carina': 0.5,
            'Sagittarius': 0.5,
            'Sextan': 0.5,
            'UrsaMinor': 0.5},
            'nu' : {
                'elliptical': 17., 
                'spiral': 1., 
                'irregular': 0.1,
                'Fornax': 0.1,
                'Sculptor': 0.2,
                'ReticulumII': 0.01,
                'BootesI': 0.005,
                'Carina': 0.15,
                'Sagittarius': 1,
                'Sextan': 0.005,
                'UrsaMinor': 0.05},
            'wind_efficiency' : {
                'elliptical': 10, 
                'spiral': 0.2,
                'irregular': 0.5,
                'Fornax': 1,
                'Sculptor': 9,
                'ReticulumII': 6,
                'BootesI': 12,
                'Carina': 5,
                'Sagittarius': 9,
                'Sextan': 11,
                'UrsaMinor': 11}
        }
        if morphology not in dictionary[choice]:
            par = (input(f"Morphology not in the dictionary. Enter your custom value for {choice} in {morphology}:\t"))
            return float(par)
        else:
            return dictionary[choice][morphology]
    
    
    def Mannucci05_SN_rate(self, SNtype, morphology):
        '''
        Observational constraint at present time
        
        SN rate per century per 10^10 Msun in Galaxy mass
        Note: The SNII and SNIb/c rate is only expressed as an upper limit
        
        From Table 2 of Mannucci et al. (2005) 
        https://ui.adsabs.harvard.edu/abs/2005A%26A...433..807M/abstract
        
        OPTIONS
            SNtype        'Ia', 'Ib/c', 'II'
            morphology    'elliptical', 'S0', 'spiral', 'irregular' 
            
        RETURNS
            list          [value, +err, -err]
        '''
        if morphology != 'elliptical':
            if morphology != 'spiral':
                if morphology != 'irregular':
                    print("Can't use Mannucci05 data")
                    morphology = 'irregular'
                    #break
                    print("WARNING: Morphology is not in the Mannucci05_SN_rate dictionary. using 'irregular' instead.") # !!!!!!! make sure that it breaks 
        dictionary = {
            'Ia': {
                'elliptical': [0.044, 0.016, 0.014],
                'S0':  [0.065, 0.027, 0.025],
                'spiral': [0.17, 0.068, 0.063],
                'irregular': [0.77, 0.42, 0.31]
            },
            'Ib/c': {
                'elliptical': [0.0093, 0., 2.],
                'S0': [0.036, 0.026, 0.018],
                'spiral': [0.12, 0.074, 0.059],
                'irregular': [0.54, 0.66, 0.38]
            },
            'II': {
                'elliptical': [0.013, 0., 2.],
                'S0': [0.12, 0.059, 0.054],
                'spiral': [0.74, 0.31, 0.3],
                'irregular': [1.7, 1.4, 1.0]
            }
        }
        return dictionary[SNtype][morphology]
    
    def Mannucci05_convert_to_SNrate_yr(self, SNtype, morphology, SNmassfrac=.1878, SNnfrac=4.381e-3, NtotvsMtot=1.808):
        '''
        Returns the number of SNae of "SNtype" type, per year,
        at present day, for the galaxy's final mass.
        
        Depends on the Galaxy's final mass, self.M_inf
        
        SNmassfrac is calibrated on Kroupa01 on the mass range 10-120Msun for SNtype='II',
        and it is the fraction, by mass, of the SN mass range over the whole IMF.
        Calculated with morph.Initial_Mass_Function.IMF_fraction
        
        SNnfrac is equivalent to SNmassfrac, but the fraction is computed by number.
        NtotvsMtot is the ratio of the integral of the IMF vs the mass-weighted IMF
        over the total mass range (Mtot = 1 by definition)
        
        From Mannucci05_SN_rate: N_SN / (century * 10^10 Msun)
        
        Mannucci... * (century/10^2yr) * (Mgal,fin/10^10Msun) * N_tot/N_SN * M_SN/M_tot * M_tot/N_tot  
        This operation is valid only for non-starbursty systems (like spiral disks) where the SFR
        has been near constant for several Gyr.
        
        OUTPUT
        M_SN/yr the supernova rate in solar mass units per year.  
        '''
        to_Gal_fin_mass = self.M_inf /1.e10
        to_yr = 1.e-2
        N_to_M = SNmassfrac / (SNnfrac * NtotvsMtot)
        rate = self.Mannucci05_SN_rate(SNtype, morphology)
        return np.multiply([rate[0], rate[0]+rate[1], rate[0]-rate[2]], 
                           N_to_M * to_Gal_fin_mass * to_yr)
        self.M_inf /1.e10

class Auxiliary:
    '''
    A collection of auxiliary static methods, 
    but they're not defined as such because for some reason
    it slows down the runs.
    '''
    def __repr__(self):
        import json
        print('\nThese are the class functions:')
        contents = [x for x in self.__dir__() if not x.startswith('__')]
        return '\n'.join(contents)
    
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
        from scipy.misc import derivative
        return derivative(func, x)

    def tic_count(self, string="Computation time", tic=None):
        tic.append(time.process_time())
        m = math.floor((tic[-1] - tic[-2])/60.)
        s = ((tic[-1] - tic[-2])%60.)
        print('%s = %d minutes and %d seconds'%(string,m,s))

    def repr(self, class_self):
        import json
        #print('\nThese are the parameter values of the class:')
        #print(json.dumps(class_self.__dict__, default=str, indent=4))
        print('\nThese are the class functions:')
        contents = [x for x in class_self.__dir__() if not x.startswith('__')
                                              if x not in class_self.__dict__]
        print(json.dumps(contents, default=str, indent=4))
        dict_keys = list(class_self.__dict__.keys())
        dictionary = {}
        for d in dict_keys: dictionary[d] = type(class_self.__dict__[d])
        for c in contents: dictionary[c] = type(getattr(class_self, c))
        repr_dict = {}#d: str(dictionary[d])
        for i in dictionary.keys(): repr_dict[i] = str(dictionary[i])
        print("There are the types for all the contents in the class")
        print(json.dumps(repr_dict, default=str, indent=4))
        return '\n'.join(repr_dict)

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
        age = integr.simps(integrand, x=birthtime_grid).quad(lambda z: 1 / ( (z + 1) *np.sqrt(OmegaLambda0 + 
                                Omegam0 * (z+1)**3 + Omegar0 * (z+1)**4) ), 
                                zf, np.inf)[0] / H0 # Since BB [Gyr]
        if not lookback_time:
            return age
        else:
            age0 = integr.quad(lambda z: 1 / ( (z + 1) *np.sqrt(OmegaLambda0 
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

    def fastquad(self):
        "https://stackoverflow.com/questions/65269540/how-can-i-speed-up-scipy-integrate-quad"
