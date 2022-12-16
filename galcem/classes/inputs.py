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


class Inputs:
    """
    Configuration inputs for OneZone
    
    ags_Galaxy (float): age of the Galaxy (Gyr)
    """
    def __init__(self):
        '''	applies to the thick disk at 8 kpc '''        
        # Time parameters
        self.nTimeStep = 0.01 #0.002 #0.01 
        self.numTimeStep = 2000 # Like FM
        self.num_MassGrid = 200
        self.include_channel = ['SNCC', 'LIMs', 'SNIa']
        
        self.age_Galaxy = 13.8 # [Gyr]
        self.age_Sun = 4.6 # [Gyr]
        self.solar_metallicity = 0.0134 # Asplund et al. (2009, Table 4)
        self.r = 8 # [kpc] Compute around the solar neighborhood
        self.k_SFR = 1
        
        self.morphology = 'spiral'
        self.M_inf = self.default_params('M_inf', self.morphology)
        self.Reff = self.default_params('Reff', self.morphology)
        self.tau_inf = self.default_params('tau_inf', self.morphology)
        self.nu = self.default_params('nu', self.morphology)
        self.wind_efficiency = self.default_params('wind_efficiency', 
                                                   self.morphology)
        self.wind_efficiency = 0 # override: no overflow

        # Fraction of compact objects
        self.A_SNIa = 1. # Fixed inside morph.Greggio05() #0.06 # Fraction of white dwarfs that underwent a SNIa
        self.A_NSM = 0.03 #0.06 # Fraction of white dwarfs that underwent a SNIa
        self.A_collapsars = 0.05 #0.06 # Fraction of white dwarfs that underwent a SNIa

        # Mass limits
        self.Ml_SNIa = 1. # Lower limit for total binary mass for SNIae [Msun]
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

        self.sd = 530.96618 # surf density coefficient for the disk (normalized to the MW mass?) 
        self.MW_SFR = 1.9 #+-0.4 [Msun/yr] from Chomiuk & Povich (2011) Galactic SFR (z=0)
        self.MW_RSNIa = np.divide([1699.5622597959612, 2348.4781118615615, 1013.0199016364531], 1e6/2.8) # 1.4*2 Msun, average SNIa mass
        self.MW_RSNCC = np.divide([7446.483293967046, 10430.201123624402, 4367.610510548821], 1e6/15) # 15 Msun, IMF-averaged mass
        self.Salpeter_IMF_Plaw = 1.35 # IMF Salpeter power law

        self.custom_IMF = None
        self.custom_SFR = None
        self.custom_SNIaDTD = None

        self.inf_option = None # or 'two-infall'
        self.IMF_option = 'Kroupa01' #'canonical' #'Salpeter55'  
        self.SFR_option = 'SFRgal' # or 'CSFR'
        self.CSFR_option = None # e.g., 'md14'. 
        self.SNIaDTD_option = 'GreggioRenzini83' # 'RuizMannucci01'

        self.yields_NSM_option = 'r14'
        self.yields_MRSN_option = 'n17'
        self.yields_LIMs_option = 'c15'
        self.yields_SNCC_option = 'lc18'
        self.LC18_vel_idx = 0 # !!!!!!! eventually you should write a function to compute this
        self.yields_SNIa_option = 'i99' # 'k20' 
        self.yields_BBN_option = 'gp13'

        self.delta_max = 8e-2 # Convergence limit for eq. 28, Portinari+98
        self.epsilon = 1e-32 # Avoid numerical errors - consistent with BBN
        self.SFR_rescaling = 1 # !!!!!!! Constrained by present-day observations of the galaxy of interest
        self.derlog = False
        
        _dir = os.path.join(os.path.dirname( __file__ ), '..')
        p98_t14_df = pd.read_csv(_dir+'/input/starlifetime/portinari98table14.dat')
        p98_t14_df.columns = [name.replace('#M','mass').replace('Z=','') 
                                            for name in p98_t14_df.columns]
        p98_t14_df = pd.melt(p98_t14_df, id_vars='mass', 
                                        value_vars=list(p98_t14_df.columns[1:]), var_name='metallicity', value_name='lifetimes_yr')
        p98_t14_df['mass_log10'] = np.log10(p98_t14_df['mass'])
        p98_t14_df['metallicity'] = p98_t14_df['metallicity'].astype(float)
        p98_t14_df['lifetimes_log10_Gyr'] = np.log10(p98_t14_df['lifetimes_yr']/1e9)
        p98_t14_df['lifetimes_Gyr'] = p98_t14_df['lifetimes_yr']/1e9
        s_lifetimes_p98 = pd.read_csv(_dir+'/input/starlifetime/portinari98table14.dat')
        s_lifetimes_p98.columns = [name.replace('#M','M').replace('Z=0.','Z') for name in s_lifetimes_p98.columns]
        self.time_start = np.min([s_lifetimes_p98[Z] for Z in ['Z0004', 'Z008', 'Z02', 'Z05']]) / 1e9 # [Gyr]
        self.time_end = np.max([s_lifetimes_p98[Z] for Z in ['Z0004', 'Z008', 'Z02', 'Z05']]) / 1e9 # [Gyr]
        self.s_lifetimes_p98 = s_lifetimes_p98
        self.p98_t14_df = p98_t14_df
        
        self.asplund1 = pd.read_csv(_dir+'/input/physics/asplund09/table1.dat', sep=',', comment='#')
        self.asplund1['photospheric'] = pd.to_numeric(self.asplund1['photospheric'], errors='coerce')
        self.asplund1['meteoric'] = pd.to_numeric(self.asplund1['meteoric'], errors='coerce')
        self.asplund1['P-err'] = pd.to_numeric(self.asplund1['P-err'], errors='coerce')
        self.asplund1['M-err'] = pd.to_numeric(self.asplund1['M-err'], errors='coerce')
        self.asplund3 = pd.read_csv(_dir+'/input/physics/asplund09/table3.dat', sep=',', comment='#')
        self.periodic = pd.read_csv(_dir+'/input/physics/periodicinfo.dat', sep=',', comment='#')
    
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
        'M_inf' : {'elliptical': 5.0e11, 
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
        'Reff' : {'elliptical': 7,
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
        'tau_inf' : {'elliptical': 0.2, 
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
        'nu' : {'elliptical': 17., 
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
        'wind_efficiency' : {'elliptical': 10, 
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
        return dictionary[choice][morphology]
    

class Auxiliary:
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
        return sm.derivative(func, x)

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
        age = scipy.integrate.quad(lambda z: 1 / ( (z + 1) *np.sqrt(OmegaLambda0 + 
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

    def fastquad(self):
        "https://stackoverflow.com/questions/65269540/how-can-i-speed-up-scipy-integrate-quad"
