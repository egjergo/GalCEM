import numpy as np
import pandas as pd
import os
# Romano+10 uses Portinari's code, with Kroupa+03 and the two-infall model
# So far, I'm using Salpeter+55 for the IMF. Also check the Schmidt power exponent

class Inputs:
    """
    Configuration inputs for OneZone
    
    ags_Galaxy (float): age of the Galaxy (Gyr)
    """
    def __init__(self):
        '''	applies to thick disk at 8 kpc '''
        self.age_Galaxy = 13.8 # [Gyr]
        self.solar_metallicity = 0.0181 # Asplund et al. (2009, Table 4)
        self.morphology = 'spiral'
        self.r = 8 # Compute around the solar neighborhood [kpc]
        self.k_SFR = 1
        # For SNIa
        self.Ml_SNIa = 3./2 # Lower limit for total binary mass for SNIae [Msun]
        self.Mu_SNIa = 12 # Upper limit for total binary mass for SNIae [Msun]
        self.A_SNIa = 0.35 #0.06 # Fraction of white dwarfs that underwent a SNIa

        self.Ml_LIMs = 0.6 # [Msun] !!!!!!! temporary. Import from yield tables
        self.Mu_LIMs = 9 # [Msun] !!!!!!! temporary. Import from yield tables
        self.Ml_NSM = 9 # [Msun] !!!!!!! temporary. Import from yield tables
        self.Mu_NSM = 50 # [Msun] !!!!!!! temporary. Import from yield tables
        self.A_NSM = 0.03 #0.06 # Fraction of white dwarfs that underwent a SNIa
        self.A_collapsars = 0.05 #0.06 # Fraction of white dwarfs that underwent a SNIa
        self.Ml_Massive = 10 # [Msun] !!!!!!! temporary. Import from yield tables
        self.Mu_Massive = 120 # [Msun] !!!!!!! temporary. Import from yield tables
        self.Ml_collapsars = 9 # [Msun] !!!!!!! temporary. Import from yield tables
        self.Mu_collapsars = 120 # [Msun] !!!!!!! temporary. Import from yield tables

        self.nTimeStep = 0.01 # Picked to smooth the mapping between stellar masses and lifetimes
        self.numTimeStep = 2000 # Like FM
        self.num_MassGrid = 200

        self.sd = 530.96618 # surf density coefficient for the disk (normalized to the MW mass?) 
        self.MW_SFR = 1.9 #+-0.4 [Msun/yr] from Chomiuk & Povich (2011) Galactic SFR (z=0)
        self.Salpeter_IMF_Plaw = 1.35 # IMF Salpeter power law

        self.delta_max = 8e-2 # Convergence limit for eq. 28, Portinari+98
        self.epsilon = 1e-32 # Avoid numerical errors - consistent with BBN
        self.SFR_rescaling = 1 # !!!!!!! Constrained by observations at z=0 of the galaxy of interest
        self.derlog = True

        self.custom_IMF = None
        self.custom_SFR = None
        self.custom_SNIaDTD = None

        self.inf_option = None # or 'two-infall'
        self.IMF_option = 'Salpeter55' #'Kroupa03' or 'Kroupa01' 
        self.SFR_option = 'SFRgal' # or 'CSFR'
        self.CSFR_option = None # e.g., 'md14'. 
        self.SNIaDTD_option = 'GreggioRenzini83' # 'RuizMannucci01'

        self.yields_LIMs_option = 'k10'
        self.yields_massive_option = 'lc18'
        self.LC18_vel_idx = 0 # !!!!!!! eventually you should write a function about this
        self.yields_SNIa_option = 'i99' #'km20'
        self.yields_BBN_option = 'gp13'

        # Calibrations from Molero+21a and b
        self.M_inf = {'elliptical': 5.0e11, # Final baryonic mass of the galaxy in [Msun]
            'spiral': 5.0e10,
            'irregular': 5.5e8,
            'Fornax': 5.0e8,
            'Sculptor': 1.0e8,
            'ReticulumII': 1.0e5,
            'BootesI': 1.1e7,
            'Carina': 5.0e8,
            'Sagittarius': 2.1e9,
            'Sextan': 5.0e8,
            'UrsaMinor': 5.0e8}
        self.Reff = {'elliptical': 7, # effective radius in [kpc]
		    'spiral': 3.5,
		    'irregular': 1,
            'Fornax': 1, # !!!!!!! Ask!!!!!!! not on the paper
            'Sculptor': 1,
            'ReticulumII': 1,
            'BootesI': 1,
            'Carina': 1,
            'Sagittarius': 1,
            'Sextan': 1,
            'UrsaMinor': 1}
        self.tau_inf = {'elliptical': 0.2, # infall timescale [Gyr]
           'spiral': 7.,
           'irregular': 7.,
           'Fornax': 3,
           'Sculptor': 0.5,
           'ReticulumII': 0.05,
           'BootesI': 0.05,
           'Carina': 0.5,
           'Sagittarius': 0.5,
           'Sextan': 0.5,
           'UrsaMinor': 0.5}
        self.nu = {'elliptical': 17.,  # nu is the SF efficiency in [Gyr^-1]
	        'spiral': 1., 
	        'irregular': 0.1,
            'Fornax': 0.1,
            'Sculptor': 0.2,
            'ReticulumII': 0.01,
            'BootesI': 0.005,
            'Carina': 0.15,
            'Sagittarius': 1,
            'Sextan': 0.005,
            'UrsaMinor': 0.05}
        self.wind_efficiency = {'elliptical': 10, # ouflow parameter [dimensionless]
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
        self.wind_efficiency = 0 # override: no overflow in this run
        
        _dir = os.path.dirname(__file__)
        self.p98_t14_df = pd.read_csv(_dir+'/input/starlifetime/portinari98table14.dat')
        self.p98_t14_df.columns = [name.replace('#M','mass').replace('Z=','') 
                                            for name in self.p98_t14_df.columns]
        self.p98_t14_df = pd.melt(self.p98_t14_df, id_vars='mass', 
                                        value_vars=list(self.p98_t14_df.columns[1:]), var_name='metallicity', value_name='lifetimes_yr')
        self.p98_t14_df['mass_log10'] = np.log10(self.p98_t14_df['mass'])
        self.p98_t14_df['metallicity'] = self.p98_t14_df['metallicity'].astype(float)
        self.p98_t14_df['lifetimes_log10_Gyr'] = np.log10(self.p98_t14_df['lifetimes_yr']/1e9)
        self.p98_t14_df['lifetimes_Gyr'] = self.p98_t14_df['lifetimes_yr']/1e9

        self.s_lifetimes_p98 = np.genfromtxt(_dir+'/input/starlifetime/portinari98table14.dat', 
                                delimiter = ',', # padded to evaluate at boundary masses
                                names = ['M','Z0004', 'Z008', 'Z02', 'Z05'])
        self.time_start = np.min([self.s_lifetimes_p98[Z] for Z in ['Z0004', 'Z008', 'Z02', 'Z05']]) / 1e9 # [Gyr]
        self.time_end = np.max([self.s_lifetimes_p98[Z] for Z in ['Z0004', 'Z008', 'Z02', 'Z05']]) / 1e9 # [Gyr]

        self.asplund1 = np.genfromtxt(_dir+'/input/physics/asplund09/table1.dat', 
						 names=['elemZ','elemN','photospheric','perr','meteoric','merr'], 
						 delimiter=',', dtype=[('elemZ', '<f8'), ('elemN', '<U5'), 
						 ('photospheric','<f8'), ('perr', '<f8'), ('meteoric', '<f8'),
						  ('merr', '<f8')])                     

        self.asplund3 = np.genfromtxt(_dir+'/input/physics/asplund09/table3.dat', 
						 names=['elemN','elemZ','elemA','percentage'], 
						 dtype=[('elemN', '<U2'), ('elemZ', '<i8'), ('elemA', '<i8'),
						  ('percentage', 'float')], delimiter=',')

        self.periodic = np.genfromtxt(_dir+'/input/physics/periodicinfo.dat', 
						 names=['elemZ','_','elemName','-','elemSymb','--','elemA'], 
						 delimiter=',', dtype=[('elemZ', '<f8'), ('_', '<U5'), 
						 ('elemName', '<U13'), ('-', '<U5'), ('elemSymb', '<U5'), 
						 ('--', '<U5'), ('elemA','<f8')])