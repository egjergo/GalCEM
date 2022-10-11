import numpy as np
import pandas as pd
import os

""""""""""""""""""""""""""""""""""""""""""""""""
"                                              "
"                  INPUT CLASS                 "
"     Contains the class that initializes      "
"    the input parameters to set up a run      " 
"                                              "
" LIST OF CLASSES:                             "
"    __        INPUTS                          "
"                                              "
""""""""""""""""""""""""""""""""""""""""""""""""

class Inputs:
    """
    Configuration inputs for OneZone
    
    ags_Galaxy (float): age of the Galaxy (Gyr)
    """
    def __init__(self):
        '''	applies to the thick disk at 8 kpc '''        
        # Time parameters
        self.nTimeStep = 0.01 #0.002 #0.01 # Picked to smooth the mapping between stellar masses and lifetimes
        self.numTimeStep = 2000 # Like FM
        self.num_MassGrid = 200
        self.include_channel = ['SNII', 'LIMs', 'SNIa']#, 'NSM', 'MRSN']
        
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
        self.wind_efficiency = self.default_params('wind_efficiency', self.morphology)
        #self.wind_efficiency = 0 # override: no overflow #self.default_params('wind_efficiency', self.morphology)

        # Fraction of compact objects
        self.A_SNIa = 0.06 #0.35 # Fraction of white dwarfs that underwent a SNIa
        self.A_NSM = 0.03 #0.06 # Fraction of neutron star that coalesced
        self.A_MRSN = 0.01 #0.06 # Fraction of the IMF that underwent a MRSN

        # Mass limits
        self.Ml_SNIa = 3./2 # Lower limit for total binary mass for SNIae [Msun]
        self.Mu_SNIa = 12 # Upper limit for total binary mass for SNIae [Msun]
        self.Ml_LIMs = 0.07 # [Msun] !!!!!!! temporary. Import from yield tables
        self.Mu_LIMs = 9 # [Msun] !!!!!!! temporary. Import from yield tables
        self.Ml_NSM = 9 # [Msun] !!!!!!! temporary. Import from yield tables
        self.Mu_NSM = 50 # [Msun] !!!!!!! temporary. Import from yield tables
        self.Ml_MRSN = 25 # [Msun] !!!!!!! temporary. Import from yield tables
        self.Mu_MRSN = 100 # [Msun] !!!!!!! temporary. Import from yield tables
        self.Ml_SNII = 10 # [Msun] !!!!!!! temporary. Import from yield tables
        self.Mu_SNII = 120 # [Msun] !!!!!!! temporary. Import from yield tables
        self.Ml_collapsars = 9 # [Msun] !!!!!!! temporary. Import from yield tables
        self.Mu_collapsars = 120 # [Msun] !!!!!!! temporary. Import from yield tables

        self.sd = 530.96618 # surf density coefficient for the disk (normalized to the MW mass?) 
        self.MW_SFR = 1.9 #+-0.4 [Msun/yr] from Chomiuk & Povich (2011) Galactic SFR (z=0)
        self.MW_RSNIa = np.divide([1699.5622597959612, 2348.4781118615615, 1013.0199016364531], 1e6/2.8) # 1.4*2 Msun, average SNIa mass
        self.MW_RSNII = np.divide([7446.483293967046, 10430.201123624402, 4367.610510548821], 1e6/15) # 15 Msun, IMF-averaged mass
        self.Salpeter_IMF_Plaw = 1.35 # IMF Salpeter power law

        self.custom_IMF = None
        self.custom_SFR = None
        self.custom_SNIaDTD = None

        self.inf_option = None # or 'two-infall'
        self.IMF_option = 'Salpeter55' #'Kroupa03' #'Kroupa01'  
        self.SFR_option = 'SFRgal' # or 'CSFR'
        self.CSFR_option = None # e.g., 'md14'. 
        self.SNIaDTD_option = 'GreggioRenzini83' # 'RuizMannucci01'

        self.yields_NSM_option = 'r14'
        self.yields_MRSN_option = 'n17'
        self.yields_LIMs_option = 'c15'
        self.yields_SNII_option = 'lc18'
        self.LC18_vel_idx = 0 # !!!!!!! eventually you should write a function to compute this
        self.yields_SNIa_option = 'i99' # 'k20' 
        self.yields_BBN_option = 'gp13'

        self.delta_max = 8e-2 # Convergence limit for eq. 28, Portinari+98
        self.epsilon = 1e-32 # Avoid numerical errors - consistent with BBN
        self.SFR_rescaling = 1 # !!!!!!! Constrained by present-day observations of the galaxy of interest
        self.derlog = True
        
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