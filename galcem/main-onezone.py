""" I only achieve simplicity with enormous effort (Clarice Lispector) """
import time
#from functools import cache, lru_cache
tic = []
tic.append(time.process_time())
import numpy as np
import scipy.integrate as integr
import scipy.interpolate as interp



""""""""""""""""""""""""""""""""""""
"                                  "
"    One Zone evolution routine    "
"                                  "
""""""""""""""""""""""""""""""""""""

def main():
    tic.append(time.process_time())
    global file1 
    file1 = open("output/Terminal_output.txt", "w")
    Evolution_class = Evolution()
    Evolution_class.evolve()
    aux.tic_count(string="Computation time", tic=tic)
    #Z_v = np.divide(np.sum(Mass_i_v[elemZ_for_metallicity:,:]), Mgas_v)
    G_v = np.divide(Mgas_v, Mtot)
    S_v = 1 - G_v
    print("Saving the output...")
    np.savetxt('output/phys.dat', np.column_stack((time_chosen, Mtot, Mgas_v,
               Mstar_v, SFR_v/1e9, Infall_rate/1e9, Z_v, G_v, S_v, Rate_SNII, Rate_SNIa, Rate_LIMs)), fmt='%-12.4e', #SFR is divided by 1e9 to get the /Gyr to /yr conversion 
               header = ' (0) time_chosen [Gyr]    (1) Mtot [Msun]    (2) Mgas_v [Msun]    (3) Mstar_v [Msun]    (4) SFR_v [Msun/yr]    (5)Infall_v [Msun/yr]    (6) Z_v    (7) G_v    (8) S_v     (9) Rate_SNII     (10) Rate_SNIa     (11) Rate_LIMs')
    np.savetxt('output/Mass_i.dat', np.column_stack((ZA_sorted, Mass_i_v)), fmt=' '.join(['%5.i']*2 + ['%12.4e']*Mass_i_v[0,:].shape[0]),
               header = ' (0) elemZ,    (1) elemA,    (2) masses [Msun] of every isotope for every timestep')
    np.savetxt('output/X_i.dat', np.column_stack((ZA_sorted, Xi_v)), fmt=' '.join(['%5.i']*2 + ['%12.4e']*Xi_v[0,:].shape[0]),
               header = ' (0) elemZ,    (1) elemA,    (2) abundance mass ratios of every isotope for every timestep (normalized to solar, Asplund et al., 2009)')
    aux.tic_count(string="Output saved in", tic=tic)
    print('Plotting...')
    plts.phys_integral_plot()
    file1.close()
    aux.tic_count(string="Plots saved in", tic=tic)
    return None

def plots():
    tic.append(time.process_time())
    plts.elem_abundance()
    plts.iso_evolution()
    plts.iso_abundance()
    aux.tic_count(string="Plots saved in", tic=tic)
    return None
