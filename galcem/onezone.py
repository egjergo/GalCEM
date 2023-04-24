""""""""""""""""""""""""""""""""""""""""""""""""
"                                              "
"      MAIN CLASSES FOR SINGLE-ZONE RUNS       "
"       Contains classes that solve the        " 
"     integro-differential equations of GCE    "
"                                              "
" LIST OF CLASSES:                             "
"    __        Setup (parent)                  "
"    __        OneZone (subclass)              "
"                                              "
""""""""""""""""""""""""""""""""""""""""""""""""

import os
import pickle
import time
import numpy as np
import pandas as pd
import scipy.integrate as integr
#from scipy.interpolate import *
from .classes import morphology as morph
from .classes import yields as yi
from .classes import integration as gcint
from .classes.inputs import Auxiliary

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
        self.morph = morph
        self.yi = yi
        self.gcint = gcint
        self.IN = IN
        self.aux = Auxiliary()
        self.lifetime_class = morph.Stellar_Lifetimes(self.IN)
        
        self.IN.M_inf = self.IN.default_params('M_inf', self.IN.morphology)
        self.IN.Reff = self.IN.default_params('Reff', self.IN.morphology)
        self.IN.tau_inf = self.IN.default_params('tau_inf', self.IN.morphology)
        self.IN.nu = self.IN.default_params('nu', self.IN.morphology)
        self.IN.wind_efficiency = 0 # !!!!!!! override: no outflow
        
        # Setup
        self.Ml = self.IN.Ml_LIMs # Lower limit stellar mass [Msun] 
        self.Mu = self.IN.Mu_collapsars # Upper limit stellar mass [Msun]
        self.time_uniform = np.arange(self.IN.Galaxy_birthtime, 
                            self.IN.Galaxy_age+self.IN.nTimeStep,
                                      self.IN.nTimeStep)
        self.time_logspace = np.logspace(np.log10(self.IN.Galaxy_birthtime),
                    np.log10(self.IN.Galaxy_age), num=self.IN.numTimeStep)
        # For now, leave to time_chosen equal to time_uniform. 
        # Some computations depend on a uniform timestep
        self.time_chosen = self.time_uniform
        self.idx_Galaxy_age = self.aux.find_nearest(self.time_chosen, 
                                                    self.IN.Galaxy_age)
        # Surface density for the disk. 
        # The bulge goes as an inverse square law
        #sigma(t_G) before eq(7). Not used so far !!!!!!!
        surf_density_Galaxy = self.IN.sd / np.exp(self.IN.r / self.IN.Reff)
        
        self.infall_class = morph.Infall(self.IN, time=self.time_chosen)
        self.infall = self.infall_class.inf()
        self.SFR_class = morph.Star_Formation_Rate(self.IN, self.IN.SFR_option,
                                                   self.IN.custom_SFR)
        self.IMF_class = morph.Initial_Mass_Function(self.Ml, self.Mu, self.IN,
                                    self.IN.IMF_option, self.IN.custom_IMF)
        self.IMF = self.IMF_class.IMF() #() # Function @ input stellar mass
        
        #normalization
        self.Greggio05_SD = np.vectorize(morph.Greggio05) # [func(lifetime)]
        gal_time = np.logspace(-3,1.5, num=1000)
        DTD_SNIa = [D.f_SD_Ia for D in self.Greggio05_SD(gal_time)]
        K = 1/integr.simpson(DTD_SNIa, x=gal_time)
        self.f_SNIa_v = np.array([K * D.f_SD_Ia for D in self.Greggio05_SD(self.time_chosen)])
        
        # Comparison of rates with observations from Mannucci+05
        SNmassfrac = self.IMF_class.IMF_fraction(self.IN.Ml_SNCC, self.IN.Mu_SNCC, massweighted=True)
        SNnfrac = self.IMF_class.IMF_fraction(self.IN.Ml_SNCC, self.IN.Mu_SNCC, massweighted=False)
        N_IMF = integr.quad(self.IMF, self.Ml, self.Mu)[0]
        self.IN.MW_RSNCC = self.IN.Mannucci05_convert_to_SNrate_yr('II', self.IN.morphology, 
                                                           SNmassfrac=SNmassfrac, SNnfrac=SNnfrac, NtotvsMtot=N_IMF)
        N_RSNIa = np.multiply(self.IN.Mannucci05_SN_rate('Ia', self.IN.morphology),
                              1.4 * self.IN.M_inf /1.e10 * 1e-2) # 1.4 Msun for Chandrasekhar's limit (SD scenario) 
        self.IN.MW_RSNIa = np.array([N_RSNIa[0], N_RSNIa[0]+ N_RSNIa[1], N_RSNIa[0] - N_RSNIa[2]])
        
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
        self.ZA_LIMs = self.c_class.extract_ZA_pairs(self.yields_LIMs_class)
        self.yields_LIMs_class.elemZ = self.ZA_LIMs[:,0]
        self.yields_LIMs_class.elemA = self.ZA_LIMs[:,1]
        self.ZA_SNIa = self.c_class.extract_ZA_pairs(self.yields_SNIa_class)
        self.yields_SNIa_class.elemZ = self.ZA_SNIa[:,0]
        self.yields_SNIa_class.elemA = self.ZA_SNIa[:,1]
        self.ZA_SNCC = self.c_class.extract_ZA_pairs(self.yields_SNCC_class)
        self.yields_SNCC_class.elemZ = self.ZA_SNCC[:,0]
        self.yields_SNCC_class.elemA = self.ZA_SNCC[:,1]
        #self.ZA_NSM = self.c_class.extract_ZA_pairs(self.yields_NSM_class)
        #self.yields_NSM_class.elemZ = self.ZA_NSM[:,0]
        #self.yields_NSM_class.elemA = self.ZA_NSM[:,1]
        #self.ZA_MRSN = self.c_class.extract_ZA_pairs(self.yields_MRSN_class)
        #self.yields_MRSN_class.elemZ = self.ZA_MRSN[:,0]
        #self.yields_MRSN_class.elemA = self.ZA_MRSN[:,1]
        ZA_all = np.vstack((self.ZA_LIMs, self.ZA_SNIa,
                            self.ZA_SNCC))#, self.ZA_NSM, self.ZA_MRSN))
        
        self.Infall_rate = self.infall(self.time_chosen)
        # Sorted list of unique [Z,A] pairs which include all isotopes
        self.ZA_sorted = self.c_class.ZA_sorted(ZA_all) 
        # name of elements for all isotopes
        self.ZA_symb_list = self.IN.periodic['elemSymb'][self.ZA_sorted[:,0]]
        
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
        #self.models_NSM = self.yields_NSM_class.yields
        #self.yields_MRSN_class.construct_yields(self.ZA_sorted)
        #self.models_MRSN = self.yields_MRSN_class.yields
        self.yield_models = {ch: self.__dict__['models_'+ch] 
                            for ch in self.IN.include_channel}
        
        # Initialize Global tracked quantities
        self.asplund3_percent = self.c_class.abund_percentage(self.ZA_sorted)
        # starting idx [int]. Excludes H and He for the metallicity selection
        self.i_Z = np.where(self.ZA_sorted[:,0]>2)[0][0] 
        
        # The total baryonic mass (i.e. the infall mass) is computed right away
        self.Mtot = np.insert(np.cumsum((self.Infall_rate[1:]
                            + self.Infall_rate[:-1]) * self.IN.nTimeStep / 2),
                              0, self.IN.epsilon) # !!!!!!! edit this with non-uniform timesteps
        self.Mstar_v = self.initialize()
        self.Mgas_v = self.initialize() 
        self.returned_Mass_v = self.initialize() 
        self.SFR_v = self.initialize()
        # Mass_i_v is the gass mass (i,j) where the i rows are the isotopes 
        # and j are the timesteps, [:,j] follows the timesteps
        self.Mass_i_v = self.initialize(matrix=True)
        self.W_i_comp = {ch: self.initialize(matrix=True) 
                         for ch in self.IN.include_channel} #dtype=object
        self.W_i_comp['BBN'] = self.initialize(matrix=True) 
        self.Xi_v = self.initialize(matrix=True) 
        self.Z_v = self.initialize() # Metallicity 
        #self.G_v = self.initialize() # G 
        #self.S_v = self.initialize() # S = 1 - G 
        self.Rate_SNCC = self.initialize() 
        self.Rate_LIMs = self.initialize() 
        self.Rate_SNIa = self.initialize() 
        self.Rate_NSM = self.initialize()
        self.Rate_MRSN = self.initialize() 
    
    def __repr__(self):
        aux = Auxiliary()
        return aux.repr(self)
   
    def initialize(self,matrix=False):
        if matrix==True:
            return self.IN.epsilon * np.ones((len(self.ZA_sorted),
                                        len(self.time_chosen)), dtype=float)
        else:
            return self.IN.epsilon * np.ones(len(self.time_chosen)) 
        
class OneZone(Setup):
    """
    OneZone class
    
    In (Input): an Input configuration instance 
    """
    def __init__(self, IN, outdir = 'runs/mygcrun/'):
        self.tic = []
        self.tic.append(time.process_time())
        super().__init__(IN, outdir=outdir)
        self.tic.append(time.process_time())
        package_loading_time = self.tic[-1]
        print('Package lodaded in %.1e seconds.'%package_loading_time)
    
    def __repr__(self):
        aux = Auxiliary()
        return aux.repr(self)
    
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
                        ff.write('\n %s type %s\n'%(key, str(type(value))))
                    #value.to_csv(self._dir_out + 'inputs.txt', mode='a',
                    #             sep='\t', index=True, header=True)
        self.evolve()
        self.aux.tic_count(string="Computation time", tic=self.tic)
        G_v = np.divide(self.Mgas_v, self.Mtot)
        S_v = 1 - G_v
        print("Saving the output...")
        phys_dat = {
            'time[Gyr]'   : self.time_chosen,
            'Mtot[Msun]'  : self.Mtot, 
            'Mgas[Msun]'  : self.Mgas_v,
            'Mstar[Msun]' : self.Mstar_v, 
            'SFR[Msun/yr]': self.SFR_v/1e9 * self.IN.M_inf, # rescale to Msun/yr from Msun/Gyr/galMass
            'Inf[Msun/yr]': self.Infall_rate/1e9, #/Gyr to /yr conversion
            'Zfrac'       : self.Z_v,
            'Gfrac'       : G_v, 
            'Sfrac'       : S_v, 
            'R_CC[M/yr]'  : self.Rate_SNCC/1e9,
            'R_Ia[M/yr]'  : self.Rate_SNIa/1e9,
            'R_LIMs[M/y]' : self.Rate_LIMs/1e9,
            'DTD_Ia[N/yr]': self.f_SNIa_v
        }
        phys_df = pd.DataFrame(phys_dat)
        phys_df.to_csv(self._dir_out+'phys.dat', index=False, 
                       header="phys = pd.read_csv(run_path+'phys.dat', sep=',', comment='#')")
        np.savetxt(self._dir_out+'Mass_i.dat', np.column_stack((
            self.ZA_sorted, self.Mass_i_v)), fmt=' '.join(['%5.i']*2+['%12.4e']
                                                 *self.Mass_i_v[0,:].shape[0]),
                header = 'elemZ    elemA    masses[Msun] # of every isotope for every timestep')
        np.savetxt(self._dir_out+'X_i.dat', np.column_stack((
            self.ZA_sorted, self.Xi_v)), fmt=' '.join(['%5.i']*2 + ['%12.4e']*
                                                    self.Xi_v[0,:].shape[0]),
                header = 'elemZ    elemA    X_i # abundance mass ratios of every isotope for every timestep (normalized to solar, Asplund et al., 2009)')
        pickle.dump(self.W_i_comp,open(self._dir_out + 'W_i_comp.pkl','wb'))
        self.aux.tic_count(string="Output saved in", tic=self.tic)
        self.file1.close()
    
    def evolve(self):
        '''Evolution routine'''
        #self.file1.write('A list of the proton/isotope number pairs for all the nuclides included in this run.\n ZA_sorted =\n\n')
        #self.file1.write(self.ZA_sorted)
        # First timestep: the galaxy is empty
        self.Mass_i_v[:,0] = np.multiply(self.Mtot[0], self.models_BBN)
        self.Mgas_v[0] = self.Mtot[0]
        # Second timestep: infall only
        self.Mass_i_v[:,1] = np.multiply(self.Mtot[1], self.models_BBN)
        self.Mgas_v[1] = self.Mtot[1]
        for n in range(len(self.time_chosen[:self.idx_Galaxy_age])+1):
            print('time [Gyr] = %.2f'%self.time_chosen[n])
            self.file1.write('n = %d\n'%n)
            self.total_evolution(n)        
            self.Xi_v[:, n] = np.divide(self.Mass_i_v[:,n], self.Mgas_v[n])
            self.Z_v[n] = np.divide(np.sum(self.Mass_i_v[self.i_Z:,n]),
                                    self.Mgas_v[n])
            self.file1.write(' sum X_i at n %d= %.3f\n'%(n, np.sum(
                             self.Xi_v[:,n])))
            
            if n > 0.: 
                Wi_class = gcint.Wi(n, self.IN, self.lifetime_class, 
                                    self.time_chosen, self.Z_v, self.SFR_v,
                            self.Greggio05_SD, self.IMF, self.ZA_sorted)
                _rates = Wi_class.compute_rates()
                self.Rate_SNCC[n] = _rates['SNCC']
                self.Rate_LIMs[n] = _rates['LIMs']
                self.Rate_SNIa[n] = _rates['SNIa']
                Wi_comp = {ch: Wi_class.compute(ch)
                        for ch in self.IN.include_channel}
                Z_comp = {}
                for ch in self.IN.include_channel:
                    if len(Wi_comp[ch]['birthtime_grid']) > 1.:
                        Z_comp[ch] = pd.DataFrame(
                        Wi_class.Z_component(Wi_comp[ch]['birthtime_grid']),
                                            columns=['metallicity']) 
                    else:
                        Z_comp[ch] = pd.DataFrame(columns=['metallicity']) 
                for i, _ in enumerate(self.ZA_sorted): 
                    self.Mass_i_v[i, n+1] = self.aux.RK4(
                        self.isotopes_evolution,self.time_chosen[n],
                        self.Mass_i_v[i,n], n, self.IN.nTimeStep,
                        i=i, Wi_comp=Wi_comp, Z_comp=Z_comp)
            self.Mass_i_v[:, n] = np.multiply(self.Mass_i_v[:,n], #!!!!!!!
                                              self.Mgas_v[n]/np.sum(self.Mass_i_v[:,n]))
        self.Z_v[-1] = np.divide(np.sum(self.Mass_i_v[self.i_Z:,-1]), 
                                self.Mgas_v[-1])
        self.Xi_v[:,-1] = np.divide(self.Mass_i_v[:,-1], self.Mgas_v[-1]) 

    def Mgas_func(self, t_n, y_n, n, i=None):
        # Explicit general diff eq GCE function
        # Mgas(t)
        #print(f'{self.SFR_tn(n)==self.SFR_v[n]=}')
        return self.Infall_rate[n] - self.SFR_tn(n) * self.IN.M_inf + np.sum([
                self.W_i_comp[ch][:,n-1] for ch in self.IN.include_channel])
    
    def Mstar_func(self, t_n, y_n, n, i=None):
        # Mstar(t)
        return self.SFR_tn(n) * self.IN.M_inf - np.sum([
               self.W_i_comp[ch][:,n-1] for ch in self.IN.include_channel])

    def SFR_tn(self, timestep_n):
        '''
        Actual SFR employed within the integro-differential equation
        Args:
            timestep_n ([int]): [timestep index]
        Returns:
            [function]: [SFR as a function of Mgas] units of [Gyr^-1]
        '''
        return self.SFR_class.SFR(Mgas=self.Mgas_v, Mtot=self.Mtot, 
                                  timestep_n=timestep_n) 
    
    def total_evolution(self, n):
        '''Integral for the total physical quantities'''
        self.SFR_v[n] = self.SFR_tn(n)
        self.Mstar_v[n+1] = self.aux.RK4(self.Mstar_func, self.time_chosen[n],
                                        self.Mstar_v[n], n, self.IN.nTimeStep)
        self.Mgas_v[n+1] = self.aux.RK4(self.Mgas_func, self.time_chosen[n], 
                                        self.Mgas_v[n], n, self.IN.nTimeStep)   

    def isotopes_evolution(self, t_n, y_n, n, **kwargs):
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
        sfr_comp = self.SFR_v[n] * self.Xi_v[i,n] # astration
        if n <= 0:
            val = infall_comp - sfr_comp
        else:
            Wi_vals = {}
            for ch in self.IN.include_channel:
                if ch == 'SNIa':
                    Wi_vals[ch] = 0.5 * (self.Rate_SNIa[n] * # Don't count SNIas twice
                                   self.yield_models['SNIa'][i])
                else:
                    if len(Wi_comps[ch]['birthtime_grid']) > 1.:
                        if not self.yield_models[ch][i].empty:
                            yield_grid = Z_comps[ch]
                            yield_grid['mass'] = Wi_comps[ch]['mass_grid']
                            Wi_vals[ch] = self.IN.factor * integr.simps(np.multiply(
                                Wi_comps[ch]['integrand'], 
                                self.yield_models[ch][i](yield_grid)), 
                                        x=Wi_comps[ch]['birthtime_grid'])
                        else:
                            Wi_vals[ch] = 0.
                    else:
                        Wi_vals[ch] = 0.
                self.W_i_comp[ch][i,n] = Wi_vals[ch] 
            val = infall_comp - sfr_comp + np.sum([Wi_vals[ch] 
                                        for ch in self.IN.include_channel])
        return val
