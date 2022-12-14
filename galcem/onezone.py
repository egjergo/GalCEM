# I only achieve simplicity with enormous effort (Clarice Lispector)
import time
import numpy as np
import pandas as pd
import scipy.integrate as integr
from scipy.interpolate import *
import os
import pickle

from .classes import morphology as morph
from .classes import yields as yi
from .classes import integration as gcint

""""""""""""""""""""""""""""""""""""""""""""""""
"                                              "
"      MAIN CLASSES FOR SINGLE-ZONE RUNS       "
"   Contains classes that solve the integral   " 
"  part of the integro-differential equations  "
"                                              "
" LIST OF CLASSES:                             "
"    __        Setup (parent)                  "
"    __        OneZone (subclass)              "
"    __        Plots (subclass)                "
"                                              "
""""""""""""""""""""""""""""""""""""""""""""""""

class Setup:
    """
    shared initial setup for both the OneZone and Plots classes
    """
    def __init__(self, IN, outdir='runs/myrun/'):
        self._dir_out = outdir if outdir[-1]=='/' else outdir+'/'
        print('Output directory: ', self._dir_out)
        self._dir_out_figs = self._dir_out + 'figs/'
        os.makedirs(self._dir_out,exist_ok=True)
        os.makedirs(self._dir_out_figs,exist_ok=True)
        self.IN = IN
        self.aux = morph.Auxiliary()
        self.lifetime_class = morph.Stellar_Lifetimes(self.IN)
        
        # Setup
        Ml = self.lifetime_class.s_mass[0] # Lower limit stellar masses [Msun] 
        Mu = self.lifetime_class.s_mass[-1] # Upper limit stellar masses [Msun]
        self.mass_uniform = np.linspace(Ml, Mu, num = self.IN.num_MassGrid)
        self.time_logspace = np.logspace(np.log10(IN.time_start), np.log10(IN.time_end), num=IN.numTimeStep)
        self.time_uniform = np.arange(self.IN.time_start, self.IN.age_Galaxy, self.IN.nTimeStep) # np.arange(IN.time_start, IN.time_end, IN.nTimeStep)
        self.time_logspace = np.logspace(np.log10(self.IN.time_start), np.log10(self.IN.age_Galaxy), num=self.IN.numTimeStep)
        self.time_chosen = self.time_uniform
        self.idx_age_Galaxy = self.aux.find_nearest(self.time_chosen, self.IN.age_Galaxy)
        # Surface density for the disk. The bulge goes as an inverse square law
        surf_density_Galaxy = self.IN.sd / np.exp(self.IN.r / self.IN.Reff) #sigma(t_G) before eq(7) not used so far !!!!!!!
        self.infall_class = morph.Infall(self.IN, morphology=self.IN.morphology, time=self.time_chosen)
        self.infall = self.infall_class.inf()
        self.SFR_class = morph.Star_Formation_Rate(self.IN, self.IN.SFR_option, self.IN.custom_SFR)
        IMF_class = morph.Initial_Mass_Function(Ml, Mu, self.IN, self.IN.IMF_option, self.IN.custom_IMF)
        self.IMF = IMF_class.IMF() #() # Function @ input stellar mass
        
        # Initialize Yields
        self.iso_class = yi.Isotopes(self.IN)
        #self.yields_MRSN_class = yi.Yields_MRSN(self.IN)
        #self.yields_MRSN_class.import_yields()
        #self.yields_NSM_class = yi.Yields_NSM(self.IN)
        #self.yields_NSM_class.import_yields()
        self.yields_LIMs_class = yi.Yields_LIMs(self.IN)
        self.yields_LIMs_class.import_yields()
        self.yields_SNCC_class = yi.Yields_SNCC(self.IN)
        self.yields_SNCC_class.import_yields()
        self.yields_SNIa_class = yi.Yields_SNIa(self.IN)
        self.yields_SNIa_class.import_yields()
        self.yields_BBN_class = yi.Yields_BBN(self.IN)
        self.yields_BBN_class.import_yields()
        
        # Initialize ZA_all
        self.c_class = yi.Concentrations(self.IN)
        self.ZA_BBN = self.c_class.extract_ZA_pairs(self.yields_BBN_class)
        self.ZA_LIMs = self.c_class.extract_ZA_pairs(self.yields_LIMs_class) #self.c_class.extract_ZA_pairs_LIMs(self.yields_LIMs_class)
        self.yields_LIMs_class.elemZ, self.yields_LIMs_class.elemA = self.ZA_LIMs[:,0], self.ZA_LIMs[:,1] # !!!!!!! remove eventually
        self.ZA_SNIa = self.c_class.extract_ZA_pairs(self.yields_SNIa_class)
        self.yields_SNIa_class.elemZ, self.yields_SNIa_class.elemA = self.ZA_SNIa[:,0], self.ZA_SNIa[:,1] # !!!!!!! remove eventually
        self.ZA_SNCC = self.c_class.extract_ZA_pairs(self.yields_SNCC_class)
        self.yields_SNCC_class.elemZ, self.yields_SNCC_class.elemA = self.ZA_SNCC[:,0], self.ZA_SNCC[:,1] # !!!!!!! remove eventually
        #self.ZA_NSM = self.c_class.extract_ZA_pairs(self.yields_NSM_class)
        #self.yields_NSM_class.elemZ, self.yields_NSM_class.elemA = self.ZA_NSM[:,0], self.ZA_NSM[:,1] # !!!!!!! remove eventually
        #self.ZA_MRSN = self.c_class.extract_ZA_pairs(self.yields_MRSN_class)
        #self.yields_MRSN_class.elemZ, self.yields_MRSN_class.elemA = self.ZA_MRSN[:,0], self.ZA_MRSN[:,1] # !!!!!!! remove eventually
        ZA_all = np.vstack((self.ZA_LIMs, self.ZA_SNIa, self.ZA_SNCC))#, self.ZA_NSM, self.ZA_MRSN))
        
        self.Infall_rate = self.infall(self.time_chosen)
        self.ZA_sorted = self.c_class.ZA_sorted(ZA_all) # [Z, A] VERY IMPORTANT! 321 isotopes with yields_SNIa_option = 'km20', 192 isotopes for 'i99' 
        #self.ZA_sorted = self.ZA_sorted[1:,:]
        self.ZA_symb_list = self.IN.periodic['elemSymb'][self.ZA_sorted[:,0]] # name of elements for all isotopes
        
        # Load Interpolation Models
        self._dir = os.path.dirname(__file__)
        self.yields_BBN_class.construct_yields(self.ZA_sorted)
        self.models_BBN = self.yields_BBN_class.yields
        self.yields_SNCC_class.construct_yields(self.ZA_sorted)
        self.models_SNCC = self.yields_SNCC_class.yields
        self.yields_LIMs_class.construct_yields(self.ZA_sorted)
        self.models_LIMs = self.yields_LIMs_class.yields
        self.yields_SNIa_class.construct_yields(self.ZA_sorted)
        self.models_SNIa = self.yields_SNIa_class.yields
        #self.yields_NSM_class.construct_yields(self.ZA_sorted)
        #models_NSM = self.yields_NSM_class.yields
        #self.yields_MRSN_class.construct_yields(self.ZA_sorted)
        #models_MRSN = self.yields_MRSN_class.yields
        self.yield_models = {ch: self.__dict__['models_'+ch] for ch in self.IN.include_channel}
        
        # Initialize Global tracked quantities
        self.asplund3_percent = self.c_class.abund_percentage(self.ZA_sorted)
        #self.ZA_symb_iso_list = np.asarray([ str(A) for A in self.IN.periodic['elemA'][self.ZA_sorted]])  # name of elements for all isotopes
        self.i_Z = np.where(self.ZA_sorted[:,0]>2)[0][0] #  starting idx (int) that excludes H and He for the metallicity selection
        self.Mtot = np.insert(np.cumsum((self.Infall_rate[1:] + self.Infall_rate[:-1]) * self.IN.nTimeStep / 2), 0, self.IN.epsilon) # The total baryonic mass (i.e. the infall mass) is computed right away
        #Mtot_quad = [quad(infall, self.time_chosen[0], i)[0] for i in range(1,len(self.time_chosen)-1)] # slow loop, deprecate!!!!!!!
        self.Mstar_v = self.IN.epsilon * np.ones(len(self.time_chosen)) 
        self.Mgas_v = self.IN.epsilon * np.ones(len(self.time_chosen)) 
        self.returned_Mass_v = self.IN.epsilon * np.ones(len(self.time_chosen)) 
        self.SFR_v = self.IN.epsilon * np.ones(len(self.time_chosen)) #
        self.f_SNIa_v = self.IN.epsilon * np.ones(len(self.time_chosen))
        self.Mass_i_v = self.IN.epsilon * np.ones((len(self.ZA_sorted), len(self.time_chosen)))    # Gass mass (i,j) where the i rows are the isotopes and j are the timesteps, [:,j] follows the timesteps
        self.W_i_comp = {ch: self.IN.epsilon * np.ones((len(self.ZA_sorted), len(self.time_chosen)), dtype=float) for ch in self.IN.include_channel}#dtype=object   # Gass mass (i,j) where the i rows are the isotopes and j are the timesteps, [:,j] follows the timesteps
        self.W_i_comp['BBN'] = self.IN.epsilon * np.ones((len(self.ZA_sorted), len(self.time_chosen)), dtype=float)
        #self.Xi_inf = self.yield_models['BBN']
        #self.Mass_i_inf = np.column_stack(([self.Xi_inf] * len(self.Mtot)))
        self.Xi_v = self.IN.epsilon * np.ones((len(self.ZA_sorted), len(self.time_chosen)))    # Xi 
        self.Z_v = self.IN.epsilon * np.ones(len(self.time_chosen)) # Metallicity 
        #self.G_v = self.IN.epsilon * np.ones(len(self.time_chosen)) # G 
        #self.S_v = self.IN.epsilon * np.ones(len(self.time_chosen)) # S = 1 - G 
        self.Rate_SNCC = self.IN.epsilon * np.ones(len(self.time_chosen)) 
        self.Rate_LIMs = self.IN.epsilon * np.ones(len(self.time_chosen)) 
        self.Rate_SNIa = self.IN.epsilon * np.ones(len(self.time_chosen)) 
        self.Rate_NSM = self.IN.epsilon * np.ones(len(self.time_chosen)) 
        self.Rate_MRSN = self.IN.epsilon * np.ones(len(self.time_chosen)) 
   
        
class OneZone(Setup):
    """
    OneZone class
    
    In (Input): an Input configuration instance 
    """
    def __init__(self, IN, outdir = 'runs/mygcrun/'):
        self.tic = []
        super().__init__(IN, outdir=outdir)
        self.tic.append(time.process_time())
        # Record load time
        self.tic.append(time.process_time())
        package_loading_time = self.tic[-1]
        print('Package lodaded in %.1e seconds.'%package_loading_time)   
    
    def main(self):
        ''' Run the OneZone program '''
        self.tic.append(time.process_time())
        self.file1 = open(self._dir_out + "Terminal_output.txt", "w")
        pickle.dump(self.IN,open(self._dir_out + 'inputs.pkl','wb'))
        with open(self._dir_out + 'inputs.txt', 'w') as f: 
            for key, value in self.IN.__dict__.items(): 
                if type(value) is not pd.DataFrame:
                    f.write('%s:\t%s\n' % (key, value))
        with open(self._dir_out + 'inputs.txt', 'a') as f: 
            for key, value in self.IN.__dict__.items(): 
                if type(value) is pd.DataFrame:
                    with open(self._dir_out + 'inputs.txt', 'a') as ff:
                        ff.write('\n %s:\n'%(key))
                    value.to_csv(self._dir_out + 'inputs.txt', mode='a', sep='\t', index=True, header=True)
        self.evolve()
        self.aux.tic_count(string="Computation time", tic=self.tic)
        G_v = np.divide(self.Mgas_v, self.Mtot)
        S_v = 1 - G_v
        print("Saving the output...")
        np.savetxt(self._dir_out + 'phys.dat', np.column_stack((self.time_chosen, self.Mtot, self.Mgas_v, self.Mstar_v, self.SFR_v/1e9, self.Infall_rate/1e9, self.Z_v, G_v, S_v, self.Rate_SNCC, self.Rate_SNIa, self.Rate_LIMs, self.f_SNIa_v)), fmt='%-12.4e', #SFR is divided by 1e9 to get the /Gyr to /yr conversion 
                header = ' (0) time_chosen [Gyr]    (1) Mtot [Msun]    (2) Mgas_v [Msun]    (3) Mstar_v [Msun]    (4) SFR_v [Msun/yr]    (5)Infall_v [Msun/yr]    (6) Z_v    (7) G_v    (8) S_v     (9) Rate_SNCC     (10) Rate_SNIa     (11) Rate_LIMs     (12) DTD SNIa')
        np.savetxt(self._dir_out +  'Mass_i.dat', np.column_stack((self.ZA_sorted, self.Mass_i_v)), fmt=' '.join(['%5.i']*2 + ['%12.4e']*self.Mass_i_v[0,:].shape[0]),
                header = ' (0) elemZ,    (1) elemA,    (2) masses [Msun] of every isotope for every timestep')
        np.savetxt(self._dir_out +  'X_i.dat', np.column_stack((self.ZA_sorted, self.Xi_v)), fmt=' '.join(['%5.i']*2 + ['%12.4e']*self.Xi_v[0,:].shape[0]),
                header = ' (0) elemZ,    (1) elemA,    (2) abundance mass ratios of every isotope for every timestep (normalized to solar, Asplund et al., 2009)')
        pickle.dump(self.W_i_comp,open(self._dir_out + 'W_i_comp.pkl','wb'))
        self.aux.tic_count(string="Output saved in", tic=self.tic)
        self.file1.close()
    
    def evolve(self):
        '''Evolution routine'''
        #self.file1.write('A list of the proton/isotope number pairs for all the nuclides included in this run.\n ZA_sorted =\n\n')
        #self.file1.write(self.ZA_sorted)
        self.Mass_i_v[:,0] = np.multiply(self.Mtot[0], self.models_BBN)
        self.Mass_i_v[:,1] = np.multiply(self.Mtot[1], self.models_BBN)
        for n in range(len(self.time_chosen[:self.idx_age_Galaxy])):
            print('time [Gyr] = %.2f'%self.time_chosen[n])
            self.file1.write('n = %d\n'%n)
            self.phys_integral(n)        
            self.Xi_v[:, n] = np.divide(self.Mass_i_v[:,n], self.Mgas_v[n])
            self.Z_v[n] = np.divide(np.sum(self.Mass_i_v[self.i_Z:,n]), self.Mgas_v[n])
            self.file1.write(' sum X_i at n %d= %.3f\n'%(n, np.sum(self.Xi_v[:,n])))
            if n > 0.: 
                Wi_class = gcint.Wi(n, self.IN, self.lifetime_class, self.time_chosen, self.Z_v, self.SFR_v, 
                              self.f_SNIa_v, self.IMF, self.ZA_sorted)
                self.Rate_SNCC[n], self.Rate_LIMs[n], self.Rate_SNIa[n] = Wi_class.compute_rates()
                Wi_comp = {ch: Wi_class.compute(ch) for ch in self.IN.include_channel}
                Z_comp = {}
                for ch in self.IN.include_channel:
                    if len(Wi_comp[ch]['birthtime_grid']) > 1.:
                        Z_comp[ch] = pd.DataFrame(Wi_class.Z_component(Wi_comp[ch]['birthtime_grid']), columns=['metallicity']) 
                    else:
                        Z_comp[ch] = pd.DataFrame(columns=['metallicity']) 
                for i, _ in enumerate(self.ZA_sorted): 
                    self.Mass_i_v[i, n+1] = self.aux.RK4(self.solve_integral, self.time_chosen[n], self.Mass_i_v[i,n], n, self.IN.nTimeStep, i=i, Wi_comp=Wi_comp, Z_comp=Z_comp)
            #self.Xi_v[:, n] = np.divide(self.Mass_i_v[:,n], self.Mgas_v[n])
        self.Z_v[-1] = np.divide(np.sum(self.Mass_i_v[self.i_Z:,-1]), self.Mgas_v[-1])
        self.Xi_v[:,-1] = np.divide(self.Mass_i_v[:,-1], self.Mgas_v[-1]) 

    def Mgas_func(self, t_n, y_n, n, i=None):
        # Explicit general diff eq GCE function
        # Mgas(t)
        #print(f'{self.SFR_tn(n)==self.SFR_v[n]=}')
        return self.Infall_rate[n] + np.sum([self.W_i_comp[ch][:,n] for ch in self.IN.include_channel]) - self.SFR_tn(n) #* np.sum(self.Xi_v[:,n])
    
    def Mstar_func(self, t_n, y_n, n, i=None):
        # Mstar(t)
        return - np.sum([self.W_i_comp[ch][:,n] for ch in self.IN.include_channel]) + self.SFR_tn(n) #* np.sum(self.Xi_v[:,n])

    def SFR_tn(self, timestep_n):
        '''
        Actual SFR employed within the integro-differential equation
        Args:
            timestep_n ([int]): [timestep index]
        Returns:
            [function]: [SFR as a function of Mgas]
        '''
        return self.SFR_class.SFR(Mgas=self.Mgas_v, Mtot=self.Mtot, timestep_n=timestep_n) # Function: SFR(Mgas)
    
    def phys_integral(self, n):
        '''Integral for the total physical quantities'''
        self.SFR_v[n] = self.SFR_tn(n)
        self.Mstar_v[n+1] = self.aux.RK4(self.Mstar_func, self.time_chosen[n], self.Mstar_v[n], n, self.IN.nTimeStep) 
        self.Mgas_v[n+1] = self.aux.RK4(self.Mgas_func, self.time_chosen[n], self.Mgas_v[n], n, self.IN.nTimeStep)    

    def solve_integral(self, t_n, y_n, n, **kwargs):
        '''
        Explicit general diff eq GCE function
        INPUT
        t_n        time_chosen[n]
        y_n        dependent variable at n
        n        index of the timestep
        Functions:
        Infall rate: [Msun/Gyr]
        SFR: [Msun/Gyr]
        '''
        i = kwargs['i']
        Wi_comps = kwargs['Wi_comp'] 
        Z_comps = kwargs['Z_comp'] 
        infall_comp = self.Infall_rate[n] * self.models_BBN[i]
        self.W_i_comp['BBN'][i,n] = infall_comp
        sfr_comp = self.SFR_v[n] * self.Xi_v[i,n] 
        if n <= 0:
            val = infall_comp - sfr_comp
        else:
            Wi_vals = {}
            for ch in self.IN.include_channel:
                if ch == 'SNIa':
                    Wi_vals[ch] = self.Rate_SNIa[n] * self.yield_models['SNIa'][i] 
                else:
                    if len(Wi_comps[ch]['birthtime_grid']) > 1.:
                        if not self.yield_models[ch][i].empty:
                            yield_grid = Z_comps[ch]
                            yield_grid['mass'] = Wi_comps[ch]['mass_grid']
                            Wi_vals[ch] = integr.simps(np.multiply(Wi_comps[ch]['integrand'], self.yield_models[ch][i](yield_grid)), x=Wi_comps[ch]['mass_grid'])
                        else:
                            Wi_vals[ch] = 0.
                    else:
                        Wi_vals[ch] = 0.
                self.W_i_comp[ch][i,n] = Wi_vals[ch] 
            val = infall_comp - sfr_comp + np.sum([Wi_vals[ch] for ch in self.IN.include_channel])
        return val
