import numpy as np
# Chiappini+10 uses Portinari's code, with Kroupa+03 and the two-infall model
# So far, I'm using Salpeter+55 for the IMF. Also check the Schmidt power exponent

age_Galaxy = 13.8 # [Gyr]
morphology = 'spiral'
r = 8 # Compute around the solar neighborhood [kpc]
iTimeStep  = 0.01#0.002 # Picked to smooth the mapping between stellar masses and lifetimes
num_MassGrid = 200
time_start = 0.00311 # Gyr (Calibrated on Portinari+98)
time_end   = 79.2 # Gyr (Calibrated on Portinari+98) to ensure tau can be mapped to masses and vice versa
sd = 530.96618 # surf density coefficient for the disk (normalized to the MW mass?) 
k_SFR = 1
# For SNIa
MBl = 3 # Lower limit for total binary mass for SNIae [Msun]
Mbu = 12 # Upper limit for total binary mass for SNIae [Msun]
A = 0.035 # Fraction of white dwarfs that underwent a SNIa

delta_max = 8e-2 # Convergence limit for (e. 28, Portinari+98)
epsilon = 1e-8 # Avoid numerical errors - consistent with BBN

custom_IMF = None
custom_SFR = None
custom_SNIaDTD = None
inf_option = None # or 'two-infall'
IMF_option = 'Salpeter55' # or 'Kroupa03'
SFR_option = 'SFRgal' # or 'CSFR'
CSFR_option = None # e.g., 'md14'. 
SNIaDTD_option = 'GreggioRenzini83' # 'RuizMannucci01'

yields_LIMS_option = 'k10'
yields_massive_option = 'lc18'
yields_SNIa_option = 'i99' #'km20'
yields_BBN_option = 'gp13'

# Calibrations from Molero+21a
M_inf = {'elliptical': 5.0e11, # Final baryonic mass of the galaxy in [Msun]
         'spiral': 5.0e10,
         'irregular': 5.5e8}
Reff = {'elliptical': 7, # effective radius in [kpc]
		'spiral': 3.5,
		'irregular': 1}
tau_inf = {'elliptical': 0.2, # infall timescale [Gyr]
           'spiral': 7.,
           'irregular': 7.}
nu = {'elliptical': 17.,  # nu is the SF efficiency in [Gyr^-1]
	  'spiral': 1., 
	  'irregular': 0.1}
wind_efficiency = {'elliptical': 10, # ouflow parameter [dimensionless]
				   'spiral': 0.2,
				   'irregular': 0.5}
wind_efficiency = 0 # override: no overflow in this run

s_lifetimes_p98 = np.genfromtxt('input/starlifetime/portinari98table14.dat', 
                                delimiter = ',', # padded to evaluate at boundary masses
                                names = ['M','Z0004', 'Z008', 'Z02', 'Z05'])

asplund1 = np.genfromtxt('input/physics/asplund09/table1.dat', 
						 names=['elemZ','elemN','photospheric','perr','meteoric','merr'], 
						 delimiter=',', dtype=[('elemZ', '<f8'), ('elemN', '<U5'), 
						 ('photospheric','<f8'), ('perr', '<f8'), ('meteoric', '<f8'),
						  ('merr', '<f8')])                     
                                
periodic = np.genfromtxt('input/physics/periodicinfo.dat', 
						 names=['elemZ','_','elemName','-','elemSymb','--','elemA'], 
						 delimiter=',', dtype=[('elemZ', '<f8'), ('_', '<U5'), 
						 ('elemName', '<U13'), ('-', '<U5'), ('elemSymb', '<U5'), 
						 ('--', '<U5'), ('elemA','<f8')])