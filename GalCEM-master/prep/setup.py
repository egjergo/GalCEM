""" 
You can change the variables, but don't remove or rename any
"""
import numpy as np
import scipy.interpolate as interp

import prep.inputs as INp
IN = INp.Inputs()
import classes.morphology as morph
import classes.yields as Y
import test.yield_interpolation_test as yt 


""" Setup """
#class Setup:
#    def __init__(self):
aux = morph.Auxiliary()
lifetime_class = morph.Stellar_Lifetimes()
Ml = lifetime_class.s_mass[0] # Lower limit stellar masses [Msun] 
Mu = lifetime_class.s_mass[-1] # Upper limit stellar masses [Msun]
mass_uniform = np.linspace(Ml, Mu, num = IN.num_MassGrid)
#time_uniform = np.arange(IN.time_start, IN.time_end, IN.nTimeStep)
#time_logspace = np.logspace(np.log10(IN.time_start), np.log10(IN.time_end), num=IN.numTimeStep)
time_uniform = np.arange(IN.time_start, IN.age_Galaxy, IN.nTimeStep)
time_logspace = np.logspace(np.log10(IN.time_start), np.log10(IN.age_Galaxy), num=IN.numTimeStep)
time_chosen = time_uniform
idx_age_Galaxy = aux.find_nearest(time_chosen, IN.age_Galaxy)
'''Surface density for the disk. The bulge goes as an inverse square law.'''
surf_density_Galaxy = IN.sd / np.exp(IN.r / IN.Reff[IN.morphology]) #sigma(t_G) before eq(7) not used so far !!!!!!!

infall_class = morph.Infall(morphology=IN.morphology, time=time_chosen)
infall = infall_class.inf()

SFR_class = morph.Star_Formation_Rate(IN.SFR_option, IN.custom_SFR)
IMF_class = morph.Initial_Mass_Function(Ml, Mu, IN.IMF_option, IN.custom_IMF)
IMF = IMF_class.IMF() # Function @ input stellar mass

isotope_class = Y.Isotopes()
yields_LIMs_class = Y.Yields_LIMs()
yields_LIMs_class.import_yields()
yields_Massive_class = Y.Yields_Massive()
yields_Massive_class.import_yields()
yields_SNIa_class = Y.Yields_SNIa()
yields_SNIa_class.import_yields()
yields_BBN_class = Y.Yields_BBN()
yields_BBN_class.import_yields()

c_class = Y.Concentrations()
ZA_LIMs = c_class.extract_ZA_pairs_LIMs(yields_LIMs_class)
ZA_SNIa = c_class.extract_ZA_pairs_SNIa(yields_SNIa_class)
ZA_Massive = c_class.extract_ZA_pairs_Massive(yields_Massive_class)
ZA_all = np.vstack((ZA_LIMs, ZA_SNIa, ZA_Massive))

""" Initialize Global tracked quantities """ 
Infall_rate = infall(time_chosen)
ZA_sorted = c_class.ZA_sorted(ZA_all) # [Z, A] VERY IMPORTANT! 321 isotopes with yields_SNIa_option = 'km20', 192 isotopes for 'i99' 
ZA_sorted = ZA_sorted[1:,:]
ZA_symb_list = IN.periodic['elemSymb'][ZA_sorted[:,0]] # name of elements for all isotopes
asplund3_percent = c_class.abund_percentage(c_class.asplund3_pd, ZA_sorted)
#ZA_symb_iso_list = np.asarray([ str(A) for A in IN.periodic['elemA'][ZA_sorted]])  # name of elements for all isotopes
elemZ_for_metallicity = np.where(ZA_sorted[:,0]>2)[0][0] #  starting idx (int) that excludes H and He for the metallicity selection
Mtot = np.insert(np.cumsum((Infall_rate[1:] + Infall_rate[:-1]) * IN.nTimeStep / 2), 0, IN.epsilon) # The total baryonic mass (i.e. the infall mass) is computed right away
#Mtot_quad = [quad(infall, time_chosen[0], i)[0] for i in range(1,len(time_chosen)-1)] # slow loop, deprecate!!!!!!!
Mstar_v = IN.epsilon * np.ones(len(time_chosen)) # Global
Mstar_test = IN.epsilon * np.ones(len(time_chosen)) # Global
Mgas_v = IN.epsilon * np.ones(len(time_chosen)) # Global
SFR_v = IN.epsilon * np.ones(len(time_chosen)) #
Mass_i_v = IN.epsilon * np.ones((len(ZA_sorted), len(time_chosen)))	# Gass mass (i,j) where the i rows are the isotopes and j are the timesteps, [:,j] follows the timesteps
Xi_inf = isotope_class.construct_yield_vector(yields_BBN_class, ZA_sorted)
Mass_i_inf = np.column_stack(([Xi_inf] * len(Mtot)))
Xi_v = IN.epsilon * np.ones((len(ZA_sorted), len(time_chosen)))	# Xi 
Z_v = IN.epsilon * np.ones(len(time_chosen)) # Metallicity 
G_v = IN.epsilon * np.ones(len(time_chosen)) # G 
S_v = IN.epsilon * np.ones(len(time_chosen)) # S = 1 - G 
Rate_SNII = IN.epsilon * np.ones(len(time_chosen)) 
Rate_LIMs = IN.epsilon * np.ones(len(time_chosen)) 
Rate_SNIa = IN.epsilon * np.ones(len(time_chosen)) 

""" load yield tables """
X_lc18, Y_lc18, models_lc18 = yt.load_processed_yields(func_name='lc18', loc='input/yields/snii/lc18/tab_R')
X_k10, Y_k10, models_k10 = yt.load_processed_yields(func_name='k10', loc='input/yields/lims/k10')
#yields_SNIa = yt.test_for_ZA_sorted_nomodel()

def _pick_yields(channel_switch, ZA_Symb, n, stellar_mass_idx=None, metallicity_idx=None, vel_idx=IN.LC18_vel_idx):
    ''' !!!!!!! this function must be edited if you import yields from other authors
    channel_switch    [str] can be 'LIMs', 'Massive', or 'SNIa'
    ZA_Symb            [str] is the element symbol, e.g. 'Na'
    
    'LIMs' requires metallicity_idx and stellar mass_idx
    'Massive' requires metallicity_idx, stellar mass_idx, and vel_idx
    'SNIa' and 'BBN' require None
    '''
    if channel_switch == 'LIMs':
        if (stellar_mass_idx == None or metallicity_idx == None):
            raise Exception('You must import the stellar mass and metallicity grids')
        idx = isotope_class.pick_by_Symb(yields_LIMs_class.elemZ, ZA_Symb)
        return yields_LIMs_class.yields[metallicity_idx][idx, stellar_mass_idx]
    elif channel_switch == 'Massive':
        if (stellar_mass_idx == None or metallicity_idx == None or vel_idx == None):
            raise Exception('You must import the stellar mass, metallicity, and velocity grids')
        metallicity_idx = np.digitize(Z_v[n], yields_Massive_class.metallicity_bins)
        vel_idx = IN.LC18_vel_idx
        idx = isotope_class.pick_by_Symb(yields_Massive_class.elemZ, ZA_Symb)
        return idx#yields_Massive_class.yields[metallicity_idx, vel_idx, idx, stellar_mass_idx]
    elif channel_switch == 'SNIa':
        idx = isotope_class.pick_by_Symb(yields_SNIa_class.elemZ, ZA_Symb)
        return yields_SNIa_class.yields[idx]