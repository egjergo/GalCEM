import numpy as np
# Romano+10 uses Portinari's code, with Kroupa+03 and the two-infall model
# So far, I'm using Salpeter+55 for the IMF. Also check the Schmidt power exponent

'''	applies to thick disk at 8 kpc '''
age_Galaxy = 13.8 # [Gyr]
morphology = 'spiral'
r = 8 # Compute around the solar neighborhood [kpc]
k_SFR = 1
# For SNIa
Ml_SNIa = 3 # Lower limit for total binary mass for SNIae [Msun]
Mu_SNIa = 12 # Upper limit for total binary mass for SNIae [Msun]
A = 0.035 # Fraction of white dwarfs that underwent a SNIa

Ml_LIMs = 0.6 # [Msun] !!!!!!! temporary. Import from yield tables
Mu_LIMs = 9 # [Msun] !!!!!!! temporary. Import from yield tables
Ml_Massive = 10 # [Msun] !!!!!!! temporary. Import from yield tables
Mu_Massive = 120 # [Msun] !!!!!!! temporary. Import from yield tables

nTimeStep = 0.002#0.001 # Picked to smooth the mapping between stellar masses and lifetimes
numTimeStep = 2000 # Like FM
num_MassGrid = 200

sd = 530.96618 # surf density coefficient for the disk (normalized to the MW mass?) 
MW_SFR = 1.9 #+-0.4 [Msun/yr] from Chomiuk & Povich (2011) Galactic SFR (z=0)

delta_max = 8e-2 # Convergence limit for eq. 28, Portinari+98
epsilon = 1e-32 # Avoid numerical errors - consistent with BBN
SFR_rescaling = 1e+10 # !!!!!!! Constrained by observations at z=0 of the galaxy of interest

custom_IMF = None
custom_SFR = None
custom_SNIaDTD = None

inf_option = None # or 'two-infall'
IMF_option = 'Salpeter55' # or 'Kroupa03'
SFR_option = 'SFRgal' # or 'CSFR'
CSFR_option = None # e.g., 'md14'. 
SNIaDTD_option = 'GreggioRenzini83' # 'RuizMannucci01'

yields_LIMs_option = 'k10'
yields_massive_option = 'lc18'
LC18_vel_idx = 0 # !!!!!!! eventually you should write a function about this
yields_SNIa_option = 'i99' #'km20'
yields_BBN_option = 'gp13'

# Calibrations from Molero+21a and b
M_inf = {'elliptical': 5.0e11, # Final baryonic mass of the galaxy in [Msun]
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
Reff = {'elliptical': 7, # effective radius in [kpc]
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
tau_inf = {'elliptical': 0.2, # infall timescale [Gyr]
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
nu = {'elliptical': 17.,  # nu is the SF efficiency in [Gyr^-1]
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
wind_efficiency = {'elliptical': 10, # ouflow parameter [dimensionless]
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
wind_efficiency = 0 # override: no overflow in this run

s_lifetimes_p98 = np.genfromtxt('input/starlifetime/portinari98table14.dat', 
                                delimiter = ',', # padded to evaluate at boundary masses
                                names = ['M','Z0004', 'Z008', 'Z02', 'Z05'])
time_start = np.min([s_lifetimes_p98[Z] for Z in ['Z0004', 'Z008', 'Z02', 'Z05']]) / 1e9 # [Gyr]
time_end = np.max([s_lifetimes_p98[Z] for Z in ['Z0004', 'Z008', 'Z02', 'Z05']]) / 1e9 # [Gyr]

asplund1 = np.genfromtxt('input/physics/asplund09/table1.dat', 
						 names=['elemZ','elemN','photospheric','perr','meteoric','merr'], 
						 delimiter=',', dtype=[('elemZ', '<f8'), ('elemN', '<U5'), 
						 ('photospheric','<f8'), ('perr', '<f8'), ('meteoric', '<f8'),
						  ('merr', '<f8')])                     

asplund3 = np.genfromtxt('input/physics/asplund09/table3.dat', 
						 names=['elemN','elemZ','elemA','percentage'], 
						 dtype=[('elemN', '<U2'), ('elemZ', '<i8'), ('elemA', '<i8'),
						  ('percentage', 'float')], delimiter=',')

periodic = np.genfromtxt('input/physics/periodicinfo.dat', 
						 names=['elemZ','_','elemName','-','elemSymb','--','elemA'], 
						 delimiter=',', dtype=[('elemZ', '<f8'), ('_', '<U5'), 
						 ('elemName', '<U13'), ('-', '<U5'), ('elemSymb', '<U5'), 
						 ('--', '<U5'), ('elemA','<f8')])
