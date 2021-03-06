# I only achieve simplicity with enormous effort (Clarice Lispector)
import time
import numpy as np
import pandas as pd
import scipy.integrate as integr
from scipy.interpolate import *
import os
import pickle

from .classes.morphology import Auxiliary,Stellar_Lifetimes,Infall,Star_Formation_Rate,Initial_Mass_Function, DTD
from .classes.yields import Isotopes,Concentrations,Yields_BBN,Yields_SNIa,Yields_SNII,Yields_LIMs,Yields_MRSN,Yields_NSM
from .classes.integration import Wi

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
        self.aux = Auxiliary()
        self.lifetime_class = Stellar_Lifetimes(IN)
        
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
        infall_class = Infall(self.IN, morphology=self.IN.morphology, time=self.time_chosen)
        self.infall = infall_class.inf()
        self.SFR_class = Star_Formation_Rate(self.IN, self.IN.SFR_option, self.IN.custom_SFR)
        IMF_class = Initial_Mass_Function(Ml, Mu, self.IN, self.IN.IMF_option, self.IN.custom_IMF)
        self.IMF = IMF_class.IMF() #() # Function @ input stellar mass
        
        # Initialize Yields
        isotope_class = Isotopes(self.IN)
        self.yields_MRSN_class = Yields_MRSN(self.IN)
        self.yields_MRSN_class.import_yields()
        self.yields_NSM_class = Yields_NSM(self.IN)
        self.yields_NSM_class.import_yields()
        self.yields_LIMs_class = Yields_LIMs(self.IN)
        self.yields_LIMs_class.import_yields()
        self.yields_SNII_class = Yields_SNII(self.IN)
        self.yields_SNII_class.import_yields()
        self.yields_SNIa_class = Yields_SNIa(self.IN)
        self.yields_SNIa_class.import_yields()
        self.yields_BBN_class = Yields_BBN(self.IN)
        self.yields_BBN_class.import_yields()
        
        # Initialize isotope list
        self.c_class = Concentrations(self.IN)
        self.ZA_BBN = self.c_class.extract_ZA_pairs(self.yields_BBN_class)
        self.ZA_LIMs = self.c_class.extract_ZA_pairs(self.yields_LIMs_class) #self.c_class.extract_ZA_pairs_LIMs(self.yields_LIMs_class)
        self.yields_LIMs_class.elemZ, self.yields_LIMs_class.elemA = self.ZA_LIMs[:,0], self.ZA_LIMs[:,1] # !!!!!!! remove eventually
        self.ZA_SNIa = self.c_class.extract_ZA_pairs(self.yields_SNIa_class)
        self.yields_SNIa_class.elemZ, self.yields_SNIa_class.elemA = self.ZA_SNIa[:,0], self.ZA_SNIa[:,1] # !!!!!!! remove eventually
        self.ZA_SNII = self.c_class.extract_ZA_pairs(self.yields_SNII_class)
        self.yields_SNII_class.elemZ, self.yields_SNII_class.elemA = self.ZA_SNII[:,0], self.ZA_SNII[:,1] # !!!!!!! remove eventually
        self.ZA_NSM = self.c_class.extract_ZA_pairs(self.yields_NSM_class)
        self.yields_NSM_class.elemZ, self.yields_NSM_class.elemA = self.ZA_NSM[:,0], self.ZA_NSM[:,1] # !!!!!!! remove eventually
        self.ZA_MRSN = self.c_class.extract_ZA_pairs(self.yields_MRSN_class)
        self.yields_MRSN_class.elemZ, self.yields_MRSN_class.elemA = self.ZA_MRSN[:,0], self.ZA_MRSN[:,1] # !!!!!!! remove eventually
        ZA_all = np.vstack((self.ZA_LIMs, self.ZA_SNIa, self.ZA_SNII))#, self.ZA_MRSN, self.ZA_NSM))
        
        self.Infall_rate = self.infall(self.time_chosen)
        self.ZA_sorted = self.c_class.ZA_sorted(ZA_all) # [Z, A] VERY IMPORTANT! 321 isotopes with yields_SNIa_option = 'km20', 192 isotopes for 'i99' 
        #self.ZA_sorted = self.ZA_sorted[1:,:]
        self.ZA_symb_list = self.IN.periodic['elemSymb'][self.ZA_sorted[:,0]] # name of elements for all isotopes
        
        # Load Interpolation Models
        self._dir = os.path.dirname(__file__)
        self.yields_BBN_class.construct_yields(self.ZA_sorted)
        self.models_BBN = self.yields_BBN_class.yields
        self.yields_SNII_class.construct_yields(self.ZA_sorted)
        self.models_SNII = self.yields_SNII_class.yields
        self.yields_LIMs_class.construct_yields(self.ZA_sorted)
        self.models_LIMs = self.yields_LIMs_class.yields
        self.yields_SNIa_class.construct_yields(self.ZA_sorted)
        self.models_SNIa = self.yields_SNIa_class.yields
        self.yields_NSM_class.construct_yields(self.ZA_sorted)
        self.models_NSM = self.yields_NSM_class.yields
        self.yields_MRSN_class.construct_yields(self.ZA_sorted)
        self.models_MRSN = self.yields_MRSN_class.yields
        self.yield_models = {ch: self.__dict__['models_'+ch] for ch in self.IN.include_channel}
        
        # Initialize Global tracked quantities
        self.asplund3_percent = self.c_class.abund_percentage(self.ZA_sorted)
        #self.ZA_symb_iso_list = np.asarray([ str(A) for A in self.IN.periodic['elemA'][self.ZA_sorted]])  # name of elements for all isotopes
        self.i_Z = np.where(self.ZA_sorted[:,0]>2)[0][0] #  starting idx (int) that excludes H and He for the metallicity selection
        self.Mtot = np.insert(np.cumsum((self.Infall_rate[1:] + self.Infall_rate[:-1]) * self.IN.nTimeStep / 2), 0, self.IN.epsilon) # The total baryonic mass (i.e. the infall mass) is computed right away
        #Mtot_quad = [quad(infall, self.time_chosen[0], i)[0] for i in range(1,len(self.time_chosen)-1)] # slow loop, deprecate!!!!!!!
        self.Mstar_v = self.IN.epsilon * np.ones(len(self.time_chosen)) 
        self.Mgas_v = self.IN.epsilon * np.ones(len(self.time_chosen)) 
        self.Mgas_i_v = self.IN.epsilon * np.ones(len(self.time_chosen)) 
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
        self.Rate_SNII = self.IN.epsilon * np.ones(len(self.time_chosen)) 
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
        np.savetxt(self._dir_out + 'phys.dat', np.column_stack((self.time_chosen, self.Mtot, self.Mgas_v, self.Mstar_v, self.SFR_v/1e9, self.Infall_rate/1e9, self.Z_v, G_v, S_v, self.Rate_SNII, self.Rate_SNIa, self.Rate_LIMs, self.f_SNIa_v, self.Rate_MRSN)), fmt='%-12.4e', #SFR is divided by 1e9 to get the /Gyr to /yr conversion 
                header = ' (0) time_chosen [Gyr]    (1) Mtot [Msun]    (2) Mgas_v [Msun]    (3) Mstar_v [Msun]    (4) SFR_v [Msun/yr]    (5)Infall_v [Msun/yr]    (6) Z_v    (7) G_v    (8) S_v     (9) Rate_SNII     (10) Rate_SNIa     (11) Rate_LIMs     (12) DTD SNIa     (13) Rate_MRSN')
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
        self.ini_setup(0)
        self.ini_setup(1)
        for n in range(len(self.time_chosen[:self.idx_age_Galaxy])):
            print('time [Gyr] = %.2f'%self.time_chosen[n])
            self.file1.write('n = %d\n'%n)
            self.total_evolution(n)        
            self.file1.write(' sum X_i at n %d= %.3f\n'%(n, np.sum(self.Xi_v[:,n])))
            if n > 0.: 
                Wi_class = Wi(n, self.IN, self.lifetime_class, self.time_chosen, self.Z_v, self.SFR_v, 
                              self.f_SNIa_v, self.IMF, self.ZA_sorted)
                #self.Rate_SNII[n], self.Rate_LIMs[n], self.Rate_SNIa[n], self.Rate_MRSN[n] = Wi_class.compute_rates()
                self.Rate_SNII[n], self.Rate_LIMs[n], self.Rate_SNIa[n] = Wi_class.compute_rates()
                Wi_comp = {ch: Wi_class.compute(ch) for ch in self.IN.include_channel}
                Z_comp = {}
                for ch in self.IN.include_channel:
                    if len(Wi_comp[ch]['birthtime_grid']) > 1.:
                        Z_comp[ch] = pd.DataFrame(Wi_class.Z_component(Wi_comp[ch]['birthtime_grid']), columns=['metallicity']) 
                    else:
                        Z_comp[ch] = pd.DataFrame(columns=['metallicity']) 
                for i, _ in enumerate(self.ZA_sorted): 
                    self.Mass_i_v[i, n+1] = self.aux.RK4(self.isotopes_evolution, self.time_chosen[n], self.Mass_i_v[i,n], n, self.IN.nTimeStep, i=i, Wi_comp=Wi_comp, Z_comp=Z_comp)
                self.Mass_i_v[:,n+1] *= self.Mgas_v[n+1]/np.sum(self.Mass_i_v[:,n+1]) #!!!!!!! renorm numerical error propagation
            #self.Xi_v[:, n] = np.divide(self.Mass_i_v[:,n], self.Mgas_v[n])
        self.Mgas_i_v[-1] = np.sum(self.Mass_i_v[:,-1])
        self.Z_v[-1] = np.divide(np.sum(self.Mass_i_v[self.i_Z:,-1]), self.Mgas_v[-1])
        self.Xi_v[:,-1] = np.divide(self.Mass_i_v[:,-1], self.Mgas_v[-1]) 

    def ini_setup(self, n):
        self.Mass_i_v[:,n] = np.multiply(self.Mtot[n], self.models_BBN)
        self.Mgas_v[n] = self.Mtot[n]
        
    def Mgas_func(self, t_n, y_n, n, i=None):
        # Explicit general diff eq GCE function
        # Mgas(t)
        #print(f'{self.SFR_tn(n)==self.SFR_v[n]=}')
        return self.Infall_rate[n] + np.sum([self.W_i_comp[ch][:,n] for ch in self.IN.include_channel]) - self.SFR_v[n] #* np.sum(self.Xi_v[:,n])
    
    def Mstar_func(self, t_n, y_n, n, i=None):
        # Mstar(t)
        return - np.sum([self.W_i_comp[ch][:,n] for ch in self.IN.include_channel]) + self.SFR_v[n] #* np.sum(self.Xi_v[:,n])

    def SFR_tn(self, timestep_n):
        '''
        Actual SFR employed within the integro-differential equation
        Args:
            timestep_n ([int]): [timestep index]
        Returns:
            [function]: [SFR as a function of Mgas]
        '''
        return (1 - self.IN.wind_efficiency) * self.SFR_class.SFR(Mgas=self.Mgas_v, Mtot=self.Mtot, timestep_n=timestep_n) # Function: SFR(Mgas)
    
    def total_evolution(self, n):
        '''Integral for the total physical quantities'''
        self.SFR_v[n] = self.SFR_tn(n)
        self.Mgas_i_v[n] = np.sum(self.Mass_i_v[:,n])
        self.Xi_v[:, n] = np.divide(self.Mass_i_v[:,n], self.Mgas_v[n])
        self.Z_v[n] = np.divide(np.sum(self.Mass_i_v[self.i_Z:,n]), self.Mgas_v[n])
        self.Mstar_v[n+1] = self.aux.RK4(self.Mstar_func, self.time_chosen[n], self.Mstar_v[n], n, self.IN.nTimeStep) 
        self.Mgas_v[n+1] = self.aux.RK4(self.Mgas_func, self.time_chosen[n], self.Mgas_v[n], n, self.IN.nTimeStep)    

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
        sfr_comp = self.SFR_v[n] * self.Xi_v[i,n] 
        if n <= 0:
            val = infall_comp - sfr_comp
        else:
            Wi_vals = {}
            for ch in self.IN.include_channel:
                if ch == 'SNIa':
                    Wi_vals[ch] = self.Rate_SNIa[n] * self.yield_models['SNIa'][i] 
                elif ch == 'MRSN':
                    Wi_vals[ch] = self.IN.A_MRSN * self.Rate_MRSN[n] * self.yield_models['MRSN'][i] 
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


class Plots(Setup):
    """
    PLOTTING
    """    
    def __init__(self, outdir = 'runs/mygcrun/'):
        self.tic = []
        IN = pickle.load(open(outdir + 'inputs.pkl','rb'))
        super().__init__(IN, outdir=outdir)
        self.tic.append(time.process_time())
        # Record load time
        self.tic.append(time.process_time())
        package_loading_time = self.tic[-1]
        print('Lodaded the plotting class in %.1e seconds.'%package_loading_time)   
        
    def plots(self):
        self.tic.append(time.process_time())
        print('Starting to plot')
        self.FeH_evolution(logAge=True)
        self.OH_evolution(logAge=True)
        self.FeH_evolution(logAge=False)
        self.OH_evolution(logAge=False)
        self.total_evolution_plot(logAge=False)
        self.total_evolution_plot(logAge=True)
        #self.DTD_plot()
        #self.iso_abundance()
        ## self.iso_evolution()
        self.iso_evolution_comp(logAge=False)
        self.iso_evolution_comp(logAge=True)
        self.observational()
        self.observational_lelemZ()
        #self.obs_lZ()
        self.lifetimeratio_test_plot()
        self.tracked_elements_3D_plot()
        ## self.elem_abundance() # compares and requires multiple runs (IMF & SFR variations)
        self.aux.tic_count(string="Plots saved in", tic=self.tic)
        
    def _ZA_sorted_plot(self, cmap_name='magma_r', cbins=10): # angle = 2 * np.pi / np.arctan(0.4) !!!!!!!
        print('Starting ZA_sorted_plot()')
        from matplotlib import cm,pyplot as plt
        #plt.style.use(self._dir+'/galcem.mplstyle')
        import matplotlib.colors as colors
        import matplotlib.ticker as ticker
        x = self.ZA_sorted[:,1]#- ZA_sorted[:,0]
        y = self.ZA_sorted[:,0]
        z = self.asplund3_percent
        cmap_ = cm.get_cmap(cmap_name, cbins)
        binning = np.digitize(z, np.linspace(0,9.*100/cbins,num=cbins-1))
        percent_colors = [cmap_.colors[c] for c in binning]
        fig, ax = plt.subplots(figsize =(11,5))
        ax.grid(True, which='major', linestyle='--', linewidth=0.5, color='purple', alpha=0.5)
        ax.grid(True, which='minor', linestyle=':', linewidth=0.5, color='purple', alpha=0.5)
        ax.set_axisbelow(True)
        smap = ax.scatter(x,y, marker='s', alpha=0.95, edgecolors='none', s=5, cmap=cmap_name, c=percent_colors) 
        smap.set_clim(0, 100)
        norm = colors.Normalize(vmin=0, vmax=100)
        cb = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap_name), orientation='vertical', pad=0.0)
        cb.set_label(label=r'Isotope $\odot$ abundance %', fontsize=17)
        ax.set_ylabel(r'Proton (Atomic) Number Z', fontsize=20)
        ax.set_xlabel(r'Atomic Mass $A$', fontsize=20)
        ax.set_title(r'Tracked isotopes', fontsize=20)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(20))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(20))
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(5))
        ax.yaxis.set_minor_locator(ticker.MultipleLocator(5))
        ax.tick_params(width=2, length=10)
        ax.tick_params(width=1, length = 5, which='minor')
        ax.set_xlim(np.min(x)-2.5, np.max(x)+2.5)
        ax.set_ylim(np.min(y)-2.5, np.max(y)+2.5)
        plt.tight_layout()
        plt.show(block=False)
        plt.savefig(self._dir_out_figs + 'tracked_elements.pdf', bbox_inches='tight')

    def tracked_elements_3D_plot(self, cmap_name='magma_r', cbins=10): # angle = 2 * np.pi / np.arctan(0.4) !!!!!!!
        print('Starting ZA_sorted_plot()')
        from matplotlib import cm,pyplot as plt
        #plt.style.use(self._dir+'/galcem.mplstyle')
        import matplotlib.colors as colors
        import matplotlib.ticker as ticker
        x = self.ZA_sorted[:,1]#- ZA_sorted[:,0]
        y = self.ZA_sorted[:,0]
        z = self.asplund3_percent
        cmap_ = cm.get_cmap(cmap_name, cbins)
        binning = np.digitize(z, np.linspace(0,9.*100/cbins,num=cbins-1))
        percent_colors = [cmap_.colors[c] for c in binning]
        fig, ax = plt.subplots(figsize =(11,5))
        ax.grid(True, which='major', linestyle='--', linewidth=0.5, color='purple', alpha=0.5)
        ax.grid(True, which='minor', linestyle=':', linewidth=0.5, color='purple', alpha=0.5)
        ax.set_axisbelow(True)
        smap = ax.scatter(x,y, marker='s', alpha=0.95, edgecolors='none', s=5, cmap=cmap_name, c=percent_colors) 
        smap.set_clim(0, 100)
        norm = colors.Normalize(vmin=0, vmax=100)
        cb = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap_name), orientation='vertical', pad=0.0)
        cb.set_label(label=r'Isotope $\odot$ abundance %', fontsize=17)
        ax.set_ylabel(r'Proton (Atomic) Number Z', fontsize=20)
        ax.set_xlabel(r'Atomic Mass $A$', fontsize=20)
        ax.set_title(r'Tracked isotopes', fontsize=20)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(20))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(20))
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(5))
        ax.yaxis.set_minor_locator(ticker.MultipleLocator(5))
        ax.tick_params(width=2, length=10)
        ax.tick_params(width=1, length = 5, which='minor')
        ax.set_xlim(np.min(x)-2.5, np.max(x)+2.5)
        ax.set_ylim(np.min(y)-2.5, np.max(y)+2.5)
        ax3 = fig.add_axes([.02, 0.5, .4, .4], projection='3d')
        #ax3.azim=-90
        ax3.bar3d(x, y, 0, 1, 1, z, color=percent_colors, zsort='average')
        ax3.tick_params(axis='x', labelsize= 8, labeltop=True, labelbottom=False)
        ax3.tick_params(axis='y', labelsize= 8, labelright=True, labelbottom=False, labelleft=False)
        ax3.tick_params(axis='z', labelsize= 8)
        ax3.set_zlabel(r'$\odot$ isotopic %', fontsize=8, labelpad=0)
        ax3.set_ylabel(r'Atomic Number $Z$', fontsize=8, labelpad=0)
        ax3.set_xlabel(r'Atomic Mass $A$', fontsize=8, labelpad=0)
        #ax3.set_zticklabels([])
        plt.tight_layout()
        plt.show(block=False)
        plt.savefig(self._dir_out_figs + 'tracked_elements.pdf', bbox_inches='tight')
        
    def _tracked_elements_3D_indiv_plot(self, cmap_name='magma_r', cbins=10): # angle = 2 * np.pi / np.arctan(0.4) !!!!!!!
        print('Starting ZA_sorted_plot()')
        from matplotlib import cm,pyplot as plt
        from mpl_toolkits.mplot3d.axes3d import Axes3D
        #plt.style.use(self._dir+'/galcem.mplstyle')
        import matplotlib.colors as colors
        import matplotlib.ticker as ticker
        x = self.ZA_sorted[:,1]#- ZA_sorted[:,0]
        y = self.ZA_sorted[:,0]
        z = self.asplund3_percent
        #hist, xedges, yedges = np.histogram2d(x, y, bins=(np.max(x)-np.min(x), np.max(y)-np.min(y)))
        #xpos, ypos = np.meshgrid(xedges[:-1]+xedges[1:], yedges[:-1]+yedges[1:])
        #xpos = xpos.flatten()/2
        #ypos = ypos.flatten()/2
        #zpos = np.zeros_like(xpos)
        #dx = 1
        #dy = 1
        #dz = z
        cmap_ = cm.get_cmap(cmap_name, cbins)
        binning = np.digitize(z, np.linspace(0,9.*100/cbins,num=cbins-1))
        percent_colors = [cmap_.colors[c] for c in binning]
        fig = plt.figure(figsize =(11,5))
        ax = fig.add_subplot(111, projection='3d')
        #ax.grid(True, which='major', linestyle='--', linewidth=0.5, color='purple', alpha=0.5)
        #ax.grid(True, which='minor', linestyle=':', linewidth=0.5, color='purple', alpha=0.5)
        #ax.set_axisbelow(True)
        #smap = ax.scatter(x,y, marker='s', alpha=0.95, edgecolors='none', s=5, cmap=cmap_name, c=percent_colors) 
        #smap.set_clim(0, 100)
        #norm = colors.Normalize(vmin=0, vmax=100)
        #cb = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap_name), orientation='vertical', pad=0.0)
        #cb.set_label(label=r'Isotope $\odot$ abundance %', fontsize=17)
        ax.bar3d(x, y, 0, 1, 1, z, color=percent_colors, zsort='average')
        ax.azim=-90
        ax.set_zlabel(r'$\odot$ isotopic %', fontsize=12)
        ax.set_ylabel(r'Atomic Number $Z$', fontsize=12)
        ax.set_xlabel(r'Atomic Mass $A$', fontsize=12)
        ax.set_title(r'Tracked isotopes', fontsize=15)
        #ax.xaxis.set_major_locator(ticker.MultipleLocator(20))
        #ax.yaxis.set_major_locator(ticker.MultipleLocator(20))
        #ax.xaxis.set_minor_locator(ticker.MultipleLocator(5))
        #ax.yaxis.set_minor_locator(ticker.MultipleLocator(5))
        #ax.tick_params(width=2, length=10)
        #ax.tick_params(width=1, length = 5, which='minor')
        #ax.set_xlim(np.min(x)-2.5, np.max(x)+2.5)
        #ax.set_ylim(np.min(y)-2.5, np.max(y)+2.5)
        plt.tight_layout()
        plt.show(block=False)
        plt.savefig(self._dir_out_figs + 'tracked_elements_3D.pdf', bbox_inches='tight')

    def DTD_plot(self):
        print('Starting DTD_plot()')
        from matplotlib import pyplot as plt
        #plt.style.use(self._dir+'/galcem.mplstyle')
        phys = np.loadtxt(self._dir_out + 'phys.dat')
        time = phys[:-1,0]
        DTD_SNIa = phys[:-1,12]
        fig, ax = plt.subplots(1,1, figsize=(7,5))
        ax.loglog(time, DTD_SNIa, color='blue', label='SNIa')
        ax.legend(loc='best', frameon=False, fontsize=13)
        ax.set_ylabel(r'Normalized DTD', fontsize=15)
        ax.set_xlabel('Age [Gyr]', fontsize=15)
        ax.set_ylim(1e-3,1e0)
        ax.set_xlim(1e-2, 1.9e1)
        fig.tight_layout()
        plt.savefig(self._dir_out_figs + 'DTD.pdf', bbox_inches='tight')
        
    def lifetimeratio_test_plot(self,colormap='Paired'):
        print('Starting lifetimeratio_test_plot()')
        from matplotlib import pyplot as plt
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        #plt.style.use(self._dir+'/galcem.mplstyle')
        fig, ax = plt.subplots(2,2, figsize=(7,6), gridspec_kw={'width_ratios': [15, 1], 'height_ratios':[1,2]})
        divid05 = np.divide(self.IN.s_lifetimes_p98['Z05'], self.IN.s_lifetimes_p98['Z0004'])
        divid02 = np.divide(self.IN.s_lifetimes_p98['Z02'], self.IN.s_lifetimes_p98['Z0004'])
        divid008 = np.divide(self.IN.s_lifetimes_p98['Z008'], self.IN.s_lifetimes_p98['Z0004'])
        divid004 = np.divide(self.IN.s_lifetimes_p98['Z004'], self.IN.s_lifetimes_p98['Z0004'])
        ax[1,0].semilogx(self.IN.s_lifetimes_p98['M'], divid05, color='black', label='Z = 0.05')
        ax[1,0].semilogx(self.IN.s_lifetimes_p98['M'], divid02, color='black', linestyle='--', label='Z = 0.02')
        ax[1,0].semilogx(self.IN.s_lifetimes_p98['M'], divid008, color='black', linestyle='-.', label='Z = 0.008')
        ax[1,0].semilogx(self.IN.s_lifetimes_p98['M'], divid004, color='black', linestyle=':', label='Z = 0.004')
        ax[0,0].hlines(1, 0.6,120, color='white', label=' ', alpha=0.)
        ax[0,0].hlines(1, 0.6,120, color='white', label='  ', alpha=0.)
        ax[1,0].hlines(1, 0.6,120, color='white', label='  ', alpha=0.)
        ax[0,0].vlines(3, 0.001,120, color='red', label=r'$3 M_{\odot}$')
        ax[0,0].vlines(6, 0.001,120, color='red', alpha=0.6, linestyle='--', label=r'$6 M_{\odot}$')
        ax[0,0].vlines(9, 0.001,120, color='red', alpha=0.3, linestyle = ':', label=r'$9 M_{\odot}$')
        ax[1,0].vlines(3, 0.6,2.6, color='red', label=r'$3 M_{\odot}$')
        ax[1,0].vlines(6, 0.6,2.6, color='red', alpha=0.6, linestyle='--', label=r'$6 M_{\odot}$')
        ax[1,0].vlines(9, 0.6,2.6, color='red', alpha=0.3, linestyle = ':', label=r'$9 M_{\odot}$')
        cm = plt.cm.get_cmap(colormap)
        sc=ax[1,0].scatter(self.IN.s_lifetimes_p98['M'], divid05, c=np.log10(self.IN.s_lifetimes_p98['Z05']), cmap=cm, s=50)
        sc=ax[1,0].scatter(self.IN.s_lifetimes_p98['M'], divid02, c=np.log10(self.IN.s_lifetimes_p98['Z02']), cmap=cm, s=50)
        sc=ax[1,0].scatter(self.IN.s_lifetimes_p98['M'], divid008, c=np.log10(self.IN.s_lifetimes_p98['Z008']), cmap=cm, s=50)
        sc=ax[1,0].scatter(self.IN.s_lifetimes_p98['M'], divid004, c=np.log10(self.IN.s_lifetimes_p98['Z004']), cmap=cm, s=50)
        ax[1,0].legend(loc='best', ncol=2, frameon=False, fontsize=10)
        ax[1,0].set_ylabel(r'$\tau(X)/\tau(Z=0.0004)$', fontsize=15)
        ax[1,0].set_xlabel('Mass', fontsize=15)
        ax[1,0].set_ylim(0.6,1.95)
        ax[1,0].set_xlim(0.6, 120)
        #divider = make_axes_locatable(ax[2])
        #cax = divider.append_axes('right', size='5%', pad=0.05)
        cbar = fig.colorbar(sc, cax=ax[1,1], label=r'$\tau(M_*)$')
        cbar.ax.tick_params(labelsize=10) 
        cbar.set_label(r'$\tau(M_*)$',fontsize=13)
        ax[0,0].loglog(self.IN.s_lifetimes_p98['M'], self.IN.s_lifetimes_p98['Z05']/1e9, color='#ffbf00', label='Z = 0.05')
        ax[0,0].loglog(self.IN.s_lifetimes_p98['M'], self.IN.s_lifetimes_p98['Z02']/1e9, color='#00ff80', linestyle='--', label='Z = 0.02')
        ax[0,0].loglog(self.IN.s_lifetimes_p98['M'], self.IN.s_lifetimes_p98['Z008']/1e9, color='#ff4000', linestyle='-.', label='Z = 0.008')
        ax[0,0].loglog(self.IN.s_lifetimes_p98['M'], self.IN.s_lifetimes_p98['Z004']/1e9, color='#4000ff', linestyle=':', label='Z = 0.004')
        ax[0,0].loglog(self.IN.s_lifetimes_p98['M'], self.IN.s_lifetimes_p98['Z0004']/1e9, color='black', linestyle='-', label='Z = 0.0004')
        ax[0,0].legend(loc='best', ncol=2, frameon=False, fontsize=11)
        #labels=ax[0,0].get_label()
        handles, labels = ax[0,0].get_legend_handles_labels()
        ax[0,0].legend(reversed(handles), reversed(labels), ncol=2, frameon=False, fontsize=10)
        handles, labels = ax[1,0].get_legend_handles_labels()
        ax[1,0].legend(reversed(handles), reversed(labels), ncol=2, frameon=False, fontsize=10)
        ax[0,0].set_ylabel(r'$\tau(M_*, Z)$', fontsize=15)
        #ax[0,0].set_xlabel('Mass', fontsize=15)
        ax[0,0].set_ylim(0.001,200)
        ax[0,0].set_xlim(0.6, 120)
        #divider = make_axes_locatable(ax[1])
        #cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.delaxes(ax[0,1])
        fig.tight_layout()
        plt.savefig(self._dir_out_figs + 'tauratio.pdf', bbox_inches='tight')
     
    def total_evolution_plot(self, figsiz=(12,7), logAge=False): #(12,6)
        print('Starting total_evolution_plot()')
        from matplotlib import pyplot as plt
        import matplotlib.ticker as ticker
        #plt.style.use(self._dir+'/galcem.mplstyle')
        phys = np.loadtxt(self._dir_out + 'phys.dat')
        Mass_i = np.loadtxt(self._dir_out + 'Mass_i.dat')
        time_chosen = phys[:,0]
        Mtot = phys[:,1]
        Mgas_v = phys[:,2]
        Mstar_v = phys[:,3]
        SFR_v = phys[:,4]
        Infall_rate = phys[:,5] 
        Z_v = phys[:,6]
        G_v = phys[:,7]
        S_v = phys[:,8] 
        Rate_SNII = phys[:,9]
        Rate_SNIa = phys[:,10]
        Rate_LIMs = phys[:,11]
        fig, axs = plt.subplots(1, 2, figsize=figsiz)
        axt = axs[1].twinx()
        time_plot = time_chosen
        xscale = '_lin'
        MW_SFR_xcoord = 13.7
        axs[0].hlines(self.IN.M_inf, 0, self.IN.age_Galaxy, label=r'$M_{gal,f}$', linewidth=1, linestyle = '-.', color='#8c00ff')
        axt.vlines(MW_SFR_xcoord, self.IN.MW_SFR-.4, self.IN.MW_SFR+0.4, label=r'SFR$_{MW}$ CP11', linewidth = 6, linestyle = '-', color='#ff8c00', alpha=0.8)
        axt.vlines(MW_SFR_xcoord, self.IN.MW_RSNII[2], self.IN.MW_RSNII[1], label=r'R$_{SNII,MW}$ M05', linewidth = 6, linestyle = '-', color='#0034ff', alpha=0.8)
        axt.vlines(MW_SFR_xcoord, self.IN.MW_RSNIa[2], self.IN.MW_RSNIa[1], label=r'R$_{SNIa,MW}$ M05', linewidth = 6, linestyle = '-', color='#00b3ff', alpha=0.8)
        axs[0].semilogy(time_plot, Mstar_v, label= r'$M_{star}$', linewidth=3, color='#ff8c00')
        axs[0].semilogy(time_plot, Mgas_v, label= r'$M_{gas}$', linewidth=3, color='#0d00ff')
        axs[0].semilogy(time_plot, np.sum(Mass_i[:,2:], axis=0), label = r'$M_{g,tot,i}$', linewidth=2, linestyle=':', color='#00b3ff')
        axs[0].semilogy(time_plot, np.sum(Mass_i[:2,2:], axis=0), label = r'$M_{H,g}$', linewidth=1, linestyle='-.', color='#0033ff')
        axs[0].semilogy(time_plot, np.sum(Mass_i[4:,2:], axis=0), label = r'$M_{Z,g}$', linewidth=2, linestyle=':', color='#ff0073')
        axs[0].semilogy(time_plot, Mtot, label=r'$M_{tot}$', linewidth=4, color='black')
        axs[0].semilogy(time_plot, Mstar_v + Mgas_v, label= r'$M_g + M_s$', linewidth=3, linestyle = '--', color='#a9a9a9')
        axs[0].semilogy(time_plot, np.sum(Mass_i[2:4,2:], axis=0), label = r'$M_{He,g}$', linewidth=1, linestyle='--', color='#0073ff')
        axs[1].semilogy(time_plot[:-1], np.divide(Rate_SNII[:-1],1e9), label= r'$R_{SNII}$', color = '#0034ff', linestyle=':', linewidth=3)
        axs[1].semilogy(time_plot[:-1], np.divide(Rate_SNIa[:-1],1e9), label= r'$R_{SNIa}$', color = '#00b3ff', linestyle=':', linewidth=3)
        axs[1].semilogy(time_plot[:-1], np.divide(Rate_LIMs[:-1],1e9), label= r'$R_{LIMs}$', color = '#ff00b3', linestyle=':', linewidth=3)
        axs[1].semilogy(time_plot[:-1], Infall_rate[:-1], label= r'Infall', color = 'black', linestyle='-', linewidth=3)
        axs[1].semilogy(time_plot[:-1], SFR_v[:-1], label= r'SFR', color = '#ff8c00', linestyle='--', linewidth=3)
        axs[0].set_ylim(1e6, 1e11)
        axs[1].set_ylim(1e-3, 1e2)
        axt.set_ylim(1e-3, 1e2)
        axt.set_yscale('log')
        if not logAge:
            axs[0].set_xlim(0,13.8)
            axs[1].set_xlim(0,13.8)
            axt.set_xlim(0,13.8)
            axs[0].xaxis.set_minor_locator(ticker.MultipleLocator(base=1))
            axs[0].tick_params(width=1, length=10, axis='x', which='minor', bottom=True, top=True, direction='in')
            axs[0].xaxis.set_major_locator(ticker.MultipleLocator(base=5))
            axs[0].tick_params(width=2, length=15, axis='x', which='major', bottom=True, top=True, direction='in')
            axs[1].xaxis.set_minor_locator(ticker.MultipleLocator(base=1))
            axs[1].tick_params(width=1, length=10, axis='x', which='minor', bottom=True, top=True, direction='in')
            axs[1].xaxis.set_major_locator(ticker.MultipleLocator(base=5))
            axs[1].tick_params(width=2, length=15, axis='x', which='major', bottom=True, top=True, direction='in')
        else:
            axs[0].set_xscale('log')
            axs[1].set_xscale('log')
            axt.set_xscale('log')
            xscale = '_log'
        axs[0].tick_params(right=True, which='both', direction='in')
        axs[1].tick_params(right=True, which='both', direction='in')
        axt.tick_params(right=True, which='both', direction='in')
        axs[0].set_xlabel(r'Age [Gyr]', fontsize = 20)
        axs[1].set_xlabel(r'Age [Gyr]', fontsize = 20)
        axs[0].set_ylabel(r'Masses [$M_{\odot}$]', fontsize = 20)
        axs[1].set_ylabel(r'Rates [$M_{\odot}/yr$]', fontsize = 20)
        #axs[0].set_title(r'$f_{SFR} = $ %.2f' % (self.IN.SFR_rescaling), fontsize=15)
        axs[0].legend(fontsize=18, loc='lower center', ncol=2, frameon=True, framealpha=0.8)
        axs[1].legend(fontsize=15, loc='upper center', ncol=2, frameon=True, framealpha=0.8)
        axt.legend(fontsize=15, loc='lower center', ncol=1, frameon=True, framealpha=0.8)
        plt.tight_layout()
        plt.show(block=False)
        plt.savefig(self._dir_out_figs + 'total_physical'+str(xscale)+'.pdf', bbox_inches='tight')
        
    def age_observations(self):
        observ = pd.read_table(self._dir + '/input/observations/age/meusinger91.txt', sep=',')
        observ_SA = np.genfromtxt(self._dir + '/input/observations/age/silva-aguirre18.txt', names=['KIC', 'Mass', 'e_Mass', 'Rad', 'e_Rad', 'logg', 'e_logg', 'Age', 'e_Age', 'Lum', 'e_Lum', 'Dist', 'e_Dist', 'Prob'])
        observ_P14_2 = np.genfromtxt(self._dir + '/input/observations/age/pinsonneault14/table2.dat', names=['KIC', 'Teff', 'FeH', 'log(g)', 'e_log(g)'])
        observ_P14_5 = np.genfromtxt(self._dir + '/input/observations/age/pinsonneault14/table5.dat', names=['KIC', 'Teff2', 'e_Teff2', 'MH2', 'e_MH2', 'M2', 'E_M2', 'e_M2', 'R2', 'E_R2', 'e_R2', 'log.g2', 'E_log.g2', 'e_log.g2', 'rho2', 'E_rho2', 'e_rho2'])
        
        id_KIC = observ_SA['KIC']
        id_match2 = np.intersect1d(observ_P14_2['KIC'], id_KIC, return_indices=True)
        id_match5 = np.intersect1d(observ_P14_5['KIC'], id_KIC, return_indices=True)
        
        ages = observ_SA['Age']
        FeH_value = observ_P14_2['FeH'][id_match2[1]]
        FeH_age = observ_SA['Age'][id_match2[2]]
        metallicity_value = observ_P14_5['MH2'][id_match5[1]]
        metallicity_age = observ_SA['Age'][id_match5[2]]
        return FeH_value, FeH_age, metallicity_value, metallicity_age
        
    def FeH_evolution(self, c=2, elemZ=26, logAge=True):
        print('Starting FeH_evolution()')
        from matplotlib import pyplot as plt
        #plt.style.use(self._dir+'/galcem.mplstyle')
        Z_list = np.unique(self.ZA_sorted[:,0])
        phys = np.loadtxt(self._dir_out + 'phys.dat')
        time = phys[c:,0]
        solar_norm_H = self.c_class.solarA09_vs_H_bymass[Z_list]
        solar_norm_Fe = self.c_class.solarA09_vs_Fe_bymass[Z_list]
        Mass_i = np.loadtxt(self._dir_out + 'Mass_i.dat')
        FeH_value, FeH_age, _, _ = self.age_observations()
        a, b = np.polyfit(FeH_age, FeH_value, 1)
        #Fe = np.sum(Mass_i[np.intersect1d(np.where(ZA_sorted[:,0]==26)[0], np.where(ZA_sorted[:,1]==56)[0]), c+2:], axis=0)
        Fe = np.sum(Mass_i[self.select_elemZ_idx(elemZ), c+2:], axis=0)
        H = np.sum(Mass_i[self.select_elemZ_idx(1), c+2:], axis=0)
        FeH = np.log10(np.divide(Fe, H)) - solar_norm_H[elemZ]
        fig, ax = plt.subplots(1,1, figsize=(7,5))
        ax.plot(time, FeH, color='black', label='[Fe/H]', linewidth=3) 
        ax.axvline(x=self.IN.age_Galaxy-self.IN.age_Sun, linewidth=2, color='orange', label=r'Age$_{\odot}$')
        ax.plot(self.IN.age_Galaxy +0.5 - FeH_age, a*FeH_age+b, color='red', alpha=1, linewidth=3, label='linear fit on [Fe/H]')
        ax.scatter(self.IN.age_Galaxy +0.5 - FeH_age, FeH_value, color='red', marker='*', alpha=0.3, label='Silva Aguirre et al. (2018)')
        ax.axhline(y=0, linewidth=1, color='orange', linestyle='--')
        #ax.errorbar(self.IN.age_Galaxy - observ['age'], observ['FeH'], yerr=observ['FeHerr'], marker='s', label='Meusinger+91', mfc='gray', ecolor='gray', ls='none')
        ax.legend(loc='lower right', frameon=False, fontsize=17)
        ax.set_ylabel(r'['+np.unique(self.ZA_symb_list[elemZ].values)[0]+'/H]', fontsize=20)
        ax.set_xlabel('Galaxy Age [Gyr]', fontsize=20)
        ax.set_ylim(-2,1)
        xscale = '_lin'
        if not logAge:
            ax.set_xlim(0,self.IN.age_Galaxy)
        else:
            ax.set_xscale('log')
            xscale = '_log'
        #ax.set_xlim(1e-2, 1.9e1)
        fig.tight_layout()
        plt.savefig(self._dir_out_figs + 'FeH_evolution'+str(xscale)+'.pdf', bbox_inches='tight')

    def OH_evolution(self, c=2, elemZ=8, logAge=False):
        print('Starting OH_evolution()')
        from matplotlib import pyplot as plt
        #plt.style.use(self._dir+'/galcem.mplstyle')
        Z_list = np.unique(self.ZA_sorted[:,0])
        phys = np.loadtxt(self._dir_out + 'phys.dat')
        time = phys[c:,0]
        _, _, metallicity_value, metallicity_age = self.age_observations()
        a, b = np.polyfit(metallicity_age, metallicity_value, 1)
        solar_norm_H = self.c_class.solarA09_vs_H_bymass[Z_list]
        Mass_i = np.loadtxt(self._dir_out + 'Mass_i.dat')
        O = np.sum(Mass_i[self.select_elemZ_idx(elemZ), c+2:], axis=0)
        H = np.sum(Mass_i[self.select_elemZ_idx(1), c+2:], axis=0)
        OH = np.log10(np.divide(O, H)) - solar_norm_H[elemZ]
        fig, ax = plt.subplots(1,1, figsize=(7,5))
        ax.plot(time, OH, color='blue', label='[O/H]', linewidth=3)
        ax.axvline(x=self.IN.age_Galaxy-self.IN.age_Sun, linewidth=2, color='orange', label=r'Age$_{\odot}$')
        ax.axhline(y=0, linewidth=1, color='orange', linestyle='--')
        ax.plot(self.IN.age_Galaxy +0.5 - metallicity_age, a*metallicity_age+b, color='red', alpha=1, linewidth=3, label='linear fit on [M/H]')
        ax.scatter(self.IN.age_Galaxy +0.5 - metallicity_age, metallicity_value, color='red', marker='*', alpha=0.3, label='Silva Aguirre et al. (2018)')
        #ax.errorbar(self.IN.age_Galaxy - observ['age'], observ['FeH'], yerr=observ['FeHerr'], marker='s', label='Meusinger+91', mfc='gray', ecolor='gray', ls='none')
        ax.legend(loc='lower right', frameon=False, fontsize=17)
        ax.set_ylabel(r'['+np.unique(self.ZA_symb_list[elemZ].values)[0]+'/H]', fontsize=20)
        ax.set_xlabel('Galaxy Age [Gyr]', fontsize=20)
        ax.set_ylim(-2,1)
        xscale = '_lin'
        if not logAge:
            ax.set_xlim(0,self.IN.age_Galaxy)
        else:
            ax.set_xscale('log')
            xscale = '_log'
        #ax.set_xlim(1e-2, 1.9e1)
        fig.tight_layout()
        plt.savefig(self._dir_out_figs + 'OH_evolution'+str(xscale)+'.pdf', bbox_inches='tight')
        
    def ind_evolution(self, c=5, elemZ=8, logAge=False):
        elemZ1 = 7
        elemZ2 = 12 
        print('Starting ind_evolution()')
        from matplotlib import pyplot as plt
        #plt.style.use(self._dir+'/galcem.mplstyle')
        Z_list = np.unique(self.ZA_sorted[:,0])
        phys = np.loadtxt(self._dir_out + 'phys.dat')
        time = phys[c:,0]
       # _, _, metallicity_value, metallicity_age = self.age_observations()
        #a, b = np.polyfit(metallicity_age, metallicity_value, 1)
        solar_norm_Fe = self.c_class.solarA09_vs_Fe_bymass[Z_list]
        Mass_i = np.loadtxt(self._dir_out + 'Mass_i.dat')
        N = np.sum(Mass_i[self.select_elemZ_idx(elemZ1), c+2:], axis=0)
        Mg = np.sum(Mass_i[self.select_elemZ_idx(elemZ2), c+2:], axis=0)
        Fe = np.sum(Mass_i[self.select_elemZ_idx(26), c+2:], axis=0)
        NFe = np.log10(np.divide(N, Fe)) - solar_norm_Fe[elemZ1]
        MgFe = np.log10(np.divide(Mg, Fe)) - solar_norm_Fe[elemZ2]
        fig, ax = plt.subplots(1,1, figsize=(7,5))
        ax.plot(time, NFe, color='magenta', label='[N/Fe]', linewidth=3)
        ax.plot(time, MgFe, color='teal', label='[Mg/Fe]', linewidth=3)
        ax.axvline(x=self.IN.age_Galaxy-self.IN.age_Sun, linewidth=2, color='orange', label=r'Age$_{\odot}$')
        #ax.axhline(y=0, linewidth=1, color='orange', linestyle='--')
        #ax.plot(self.IN.age_Galaxy +0.5 - metallicity_age, a*metallicity_age+b, color='red', alpha=1, linewidth=3, label='linear fit on [M/H]')
        #ax.scatter(self.IN.age_Galaxy +0.5 - metallicity_age, metallicity_value, color='red', marker='*', alpha=0.3, label='Silva Aguirre et al. (2018)')
        #ax.errorbar(self.IN.age_Galaxy - observ['age'], observ['FeH'], yerr=observ['FeHerr'], marker='s', label='Meusinger+91', mfc='gray', ecolor='gray', ls='none')
        ax.legend(loc='best', frameon=False, fontsize=17)
        ax.set_ylabel(r'[X/Fe]', fontsize=20)
        ax.set_xlabel('Galaxy Age [Gyr]', fontsize=20)
        ax.set_ylim(-2,1)
        xscale = '_lin'
        if not logAge:
            ax.set_xlim(0,self.IN.age_Galaxy)
        else:
            ax.set_xscale('log')
            xscale = '_log'
            ax.set_xlim(2e-2,self.IN.age_Galaxy)
        #ax.set_xlim(1e-2, 1.9e1)
        fig.tight_layout()
        plt.savefig(self._dir_out_figs + 'ind_evolution'+str(xscale)+'.pdf', bbox_inches='tight')
           
    def iso_evolution(self, figsize=(40,13)):
        print('Starting iso_evolution()')
        from matplotlib import pyplot as plt
        #plt.style.use(self._dir+'/galcem.mplstyle')
        import matplotlib.ticker as ticker
        Mass_i = np.loadtxt(self._dir_out + 'Mass_i.dat')
        Masses = np.log10(Mass_i[:,2:])
        phys = np.loadtxt(self._dir_out + 'phys.dat')
        timex = phys[:,0]
        Z = self.ZA_sorted[:,0]
        A = self.ZA_sorted[:,1]
        ncol = self.aux.find_nearest(np.power(np.arange(20),2), len(Z))
        if len(self.ZA_sorted) > ncol:
            nrow = ncol
        else:
            nrow = ncol + 1
        fig, axs = plt.subplots(nrow, ncol, figsize=figsize)#, sharex=True)
        for i, ax in enumerate(axs.flat):
            if i < len(Z):
                ax.plot(timex, Masses[i])
                ax.annotate('%s(%d,%d)'%(self.ZA_symb_list.values[i],Z[i],A[i]), xy=(0.5, 0.92), xycoords='axes fraction', horizontalalignment='center', verticalalignment='top', fontsize=12, alpha=0.7)
                ax.set_ylim(-7.5, 10.5)
                ax.set_xlim(0.1,13.8)
                ax.xaxis.set_minor_locator(ticker.MultipleLocator(base=1))
                ax.tick_params(width=1, length=2, axis='x', which='minor', bottom=True, top=True, direction='in')
                ax.yaxis.set_minor_locator(ticker.MultipleLocator(base=1))
                ax.tick_params(width=1, length=2, axis='y', which='minor', left = True, right = True, direction='in')
                ax.xaxis.set_major_locator(ticker.MultipleLocator(base=5))
                ax.tick_params(width=1, length = 5, axis='x', which='major', bottom=True, top=True, direction='in')
                ax.yaxis.set_major_locator(ticker.MultipleLocator(base=5))
                ax.tick_params(width=1, length = 5, axis='y', which='major', left = True, right = True, direction='in')
            else:
                fig.delaxes(ax)
        for i in range(nrow):
            for j in range(ncol):
                if j != 0:
                    axs[i,j].set_yticklabels([])
                if i != nrow-1:
                    axs[i,j].set_xticklabels([])
        axs[nrow//2,0].set_ylabel(r'Masses [$M_{\odot}$]', fontsize = 15)
        axs[nrow-1, ncol//2].set_xlabel('Age [Gyr]', fontsize = 15)
        plt.tight_layout(rect = [0.05, 0, 1, .98])
        plt.subplots_adjust(wspace=0., hspace=0.)
        plt.show(block=False)
        plt.savefig(self._dir_out_figs + 'iso_evolution.pdf', bbox_inches='tight')

    def iso_evolution_comp(self, figsize=(12,18), logAge=True, ncol=15):
        import math
        print('Starting iso_evolution_comp()')
        from matplotlib import pyplot as plt
        plt.style.use(self._dir+'/galcem.mplstyle')
        import matplotlib.ticker as ticker
        Mass_i = np.loadtxt(self._dir_out + 'Mass_i.dat')
        Masses = np.log10(Mass_i[:,2:])
        phys = np.loadtxt(self._dir_out + 'phys.dat')
        W_i_comp = pickle.load(open(self._dir_out + 'W_i_comp.pkl','rb'))
        #Mass_MRSN = np.log10(W_i_comp['MRSN'])
        Mass_BBN = np.log10(W_i_comp['BBN'])
        Mass_SNII = np.log10(W_i_comp['SNII'])
        Mass_AGB = np.log10(W_i_comp['LIMs'])
        Mass_SNIa = np.log10(W_i_comp['SNIa'])
        timex = phys[:,0]
        Z = self.ZA_sorted[:,0]
        A = self.ZA_sorted[:,1]
        if ncol==None: ncol = np.floor(np.sqrt(lenA)).astype('int')
        nrow = np.ceil(len(A)/ncol).astype('int')
        #print('(# nuclides, nrow, ncol) = (%d, %d, %d)'%(len(Z), nrow, ncol))
        fig, axs = plt.subplots(nrow, ncol, figsize=figsize)#, sharex=True)
        for i, ax in enumerate(axs.flat):
            if i < len(Z):
                #print('i %d'%(i))
                #print('%s(%d,%d)'%(self.ZA_symb_list.values[i],Z[i],A[i]))
                ax.annotate('%d%s'%(A[i],self.ZA_symb_list.values[i]), xy=(0.5, 0.3), xycoords='axes fraction', horizontalalignment='center', verticalalignment='top', fontsize=7, alpha=0.7)
                ax.set_ylim(-4.9, 10.9)
                ax.set_xlim(0.01,13.8)
                ax.yaxis.set_minor_locator(ticker.MultipleLocator(base=1))
                ax.tick_params(width=1, length=2, axis='y', which='minor', left=True, right=True, direction='in')
                ax.yaxis.set_major_locator(ticker.MultipleLocator(base=5))
                ax.tick_params(width=1, length=3, axis='y', which='major', left=True, right=True, direction='in')
                ax.plot(timex[:-1], Mass_BBN[i][:-1], color='black', linestyle='-.', linewidth=3, alpha=0.8, label='BBN')
                ax.plot(timex[:-1], Mass_SNII[i][:-1], color='#0034ff', linestyle='-.', linewidth=3, alpha=0.8, label='SNII')
                ax.plot(timex[:-1], Mass_AGB[i][:-1], color='#ff00b3', linestyle='--', linewidth=3, alpha=0.8, label='LIMs')
                ax.plot(timex[:-1], Mass_SNIa[i][:-1], color='#00b3ff', linestyle=':', linewidth=3, alpha=0.8, label='SNIa')
                #ax.plot(timex[:-1], Mass_MRSN[i][:-1], color='#000c3b', linestyle=':', linewidth=3, alpha=0.8, label='MRSN')
                if not logAge:
                    ax.xaxis.set_minor_locator(ticker.MultipleLocator(base=1))
                    ax.tick_params(width=1, length=2, axis='x', which='minor', bottom=True, top=True, direction='in')
                    ax.xaxis.set_major_locator(ticker.MultipleLocator(base=5))
                    ax.tick_params(width=1, length=3, axis='x', which='major', bottom=True, top=True, direction='in')
                else:
                    ax.set_xscale('log')
                    #ax.set_xticks([0.01, 1])
                    #ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
                    ax.xaxis.set_major_locator(ticker.LogLocator(base=100, numticks=3))
                    ax.tick_params(width=1, length=3, axis='x', which='major', bottom=True, top=True, direction='in')
                    ax.xaxis.set_minor_locator(ticker.LogLocator(base=10.0,subs=(0.2,0.4,0.6,0.8,1.),numticks=5))
                    ax.xaxis.set_minor_formatter(ticker.NullFormatter())
                    ax.tick_params(width=1, length=2, axis='x', which='minor', bottom=True, top=True, direction='in')
            else:
                fig.delaxes(ax)
            last_idx =i
        for i in range(nrow):
            for j in range(ncol):
                if j != 0:
                    axs[i,j].set_yticklabels([])
                if i < nrow-2:
                    axs[i,j].set_xticklabels([])
                    
        #axs[nrow//2,0].set_ylabel(r'Masses [ $\log_{10}$($M_{\odot}$/yr)]', fontsize = 15)
        axs[nrow//2,0].set_ylabel(r'Returned masses [ $\log_{10}$($M_{\odot}$)]', fontsize = 15)
        axs[0, ncol//2].legend(ncol=len(W_i_comp), loc='upper center', bbox_to_anchor=(0.5, 1.8), frameon=False, fontsize=12)
        if not logAge:
            xscale = '_lin'
            axs[nrow-2, ncol//2].set_xlabel('Age [Gyr]', fontsize = 15)
            #axs.flat[last_idx].set_xlabel('Age [Gyr]', fontsize = 15)
            #plt.xlabel('Age [Gyr]', fontsize = 15)
        else:
            xscale = '_log'
            axs[nrow-2, ncol//2].set_xlabel('Log  Age [Gyr]', fontsize = 15)
        plt.subplots_adjust(wspace=0., hspace=0.)
        plt.tight_layout(rect = [0.03, 0.03, 1, .90])
        plt.show(block=False)
        plt.savefig(self._dir_out_figs + 'iso_evolution_comp'+str(xscale)+'.pdf', bbox_inches='tight')

    def iso_evolution_comp_lelemz(self, figsize=(12,15), logAge=True, ncol=10):
        import math
        import pickle
        IN = pickle.load(open(self._dir_out + 'inputs.pkl','rb'))
        print('Starting iso_evolution_comp()')
        from matplotlib import pyplot as plt
        plt.style.use(self._dir+'/galcem.mplstyle')
        import matplotlib.ticker as ticker
        Mass_i = np.loadtxt(self._dir_out + 'Mass_i.dat')
        Masses = np.log10(Mass_i[:,2:])
        phys = np.loadtxt(self._dir_out + 'phys.dat')
        W_i_comp = pickle.load(open(self._dir_out + 'W_i_comp.pkl','rb'))
        #Mass_MRSN = np.log10(W_i_comp['MRSN'])
        yr_rate = IN.nTimeStep * 1e9
        Mass_BBN = np.log10(W_i_comp['BBN'] / yr_rate)
        Mass_SNII = np.log10(W_i_comp['SNII'] / yr_rate)
        Mass_AGB = np.log10(W_i_comp['LIMs'] / yr_rate)
        Mass_SNIa = np.log10(W_i_comp['SNIa'] / yr_rate)
        timex = phys[:,0]
        Z = self.ZA_sorted[:,0]
        A = self.ZA_sorted[:,1]
        if ncol==None: ncol = np.floor(np.sqrt(lenA)).astype('int')
        nrow = 12# np.ceil(len(A)/ncol).astype('int')
        #print('(# nuclides, nrow, ncol) = (%d, %d, %d)'%(len(Z), nrow, ncol))
        fig, axs = plt.subplots(nrow, ncol, figsize=figsize)#, sharex=True)
        for i, ax in enumerate(axs.flat):
            if i < len(Z):
                #print('i %d'%(i))
                #print('%s(%d,%d)'%(self.ZA_symb_list.values[i],Z[i],A[i]))
                ax.annotate('%d%s'%(A[i],self.ZA_symb_list.values[i]), xy=(0.5, 0.3), xycoords='axes fraction', horizontalalignment='center', verticalalignment='top', fontsize=7, alpha=0.7)
                #ax.set_ylim(-4.9, 10.9)
                ax.set_ylim(-16.9, 5.9)
                ax.set_xlim(0.01,13.8)
                ax.yaxis.set_minor_locator(ticker.MultipleLocator(base=1))
                ax.tick_params(width=1, length=2, axis='y', which='minor', left=True, right=True, direction='in')
                ax.yaxis.set_major_locator(ticker.MultipleLocator(base=5))
                ax.tick_params(width=1, length=3, axis='y', which='major', left=True, right=True, direction='in')
                ax.plot(timex[:-1], Mass_BBN[i][:-1], color='black', linestyle='-.', linewidth=3, alpha=0.8, label='BBN')
                ax.plot(timex[:-1], Mass_SNII[i][:-1], color='#0034ff', linestyle='-.', linewidth=3, alpha=0.8, label='SNII')
                ax.plot(timex[:-1], Mass_AGB[i][:-1], color='#ff00b3', linestyle='--', linewidth=3, alpha=0.8, label='LIMs')
                ax.plot(timex[:-1], Mass_SNIa[i][:-1], color='#00b3ff', linestyle=':', linewidth=3, alpha=0.8, label='SNIa')
                #ax.plot(timex[:-1], Mass_MRSN[i][:-1], color='#000c3b', linestyle=':', linewidth=3, alpha=0.8, label='MRSN')
                if not logAge:
                    ax.xaxis.set_minor_locator(ticker.MultipleLocator(base=1))
                    ax.tick_params(width=1, length=2, axis='x', which='minor', bottom=True, top=True, direction='in')
                    ax.xaxis.set_major_locator(ticker.MultipleLocator(base=5))
                    ax.tick_params(width=1, length=3, axis='x', which='major', bottom=True, top=True, direction='in')
                else:
                    ax.set_xscale('log')
                    #ax.set_xticks([0.01, 1])
                    #ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
                    ax.xaxis.set_major_locator(ticker.LogLocator(base=100, numticks=3))
                    ax.tick_params(width=1, length=3, axis='x', which='major', bottom=True, top=True, direction='in')
                    ax.xaxis.set_minor_locator(ticker.LogLocator(base=10.0,subs=(0.2,0.4,0.6,0.8,1.),numticks=5))
                    ax.xaxis.set_minor_formatter(ticker.NullFormatter())
                    ax.tick_params(width=1, length=2, axis='x', which='minor', bottom=True, top=True, direction='in')
            else:
                fig.delaxes(ax)
            last_idx =i
        for i in range(nrow):
            for j in range(ncol):
                if j != 0:
                    axs[i,j].set_yticklabels([])
                if i < nrow-2:
                    axs[i,j].set_xticklabels([])
                    
        #axs[nrow//2,0].set_ylabel(r'Masses [ $\log_{10}$($M_{\odot}$/yr)]', fontsize = 15)
        axs[nrow//2,0].set_ylabel(r'Returned masses [ $\log_{10}$($M_{\odot}/$ yr)]', fontsize = 15)
        axs[0, ncol//2].legend(ncol=len(W_i_comp), loc='upper center', bbox_to_anchor=(0.5, 1.8), frameon=False, fontsize=12)
        if not logAge:
            xscale = '_lin'
            axs[nrow-1, ncol//2].set_xlabel('Age [Gyr]', fontsize = 15)
            #axs.flat[last_idx].set_xlabel('Age [Gyr]', fontsize = 15)
            #plt.xlabel('Age [Gyr]', fontsize = 15)
        else:
            xscale = '_log'
            axs[nrow-1, ncol//2].set_xlabel('Log  Age [Gyr]', fontsize = 15)
        plt.subplots_adjust(wspace=0., hspace=0.)
        plt.tight_layout(rect = [0.03, 0.03, 1, .95])
        plt.show(block=False)
        plt.savefig(self._dir_out_figs + 'iso_evolution_comp_lz'+str(xscale)+'.pdf', bbox_inches='tight')

    def iso_abundance(self, figsize=(20,13), c=3): 
        print('Starting iso_abundance()')
        from matplotlib import pyplot as plt
        #plt.style.use(self._dir+'/galcem.mplstyle')
        import matplotlib.ticker as ticker
        Mass_i = np.loadtxt(self._dir_out + 'Mass_i.dat')
        Fe = np.sum(Mass_i[self.select_elemZ_idx(26), c+2:], axis=0)
        Masses = np.log10(np.divide(Mass_i[:,c+2:], Fe))
        XH = np.log10(np.divide(Fe, Mass_i[0,c+2:])) 
        Z = self.ZA_sorted[:,0]
        A = self.ZA_sorted[:,1]
        ncol = self.aux.find_nearest(np.power(np.arange(20),2), len(Z))
        if len(self.ZA_sorted) < ncol:
            nrow = ncol
        else:
            nrow = ncol + 1
        fig, axs = plt.subplots(nrow, ncol, figsize=figsize)#, sharex=True)
        for i, ax in enumerate(axs.flat):
            if i < len(Z):
                ax.plot(XH, Masses[i])
                ax.annotate('%s(%d,%d)'%(self.ZA_symb_list.values[i],Z[i],A[i]), xy=(0.5, 0.92), xycoords='axes fraction', horizontalalignment='center', verticalalignment='top', fontsize=12, alpha=0.7)
                ax.set_ylim(-15, 0.5)
                ax.set_xlim(-11, 0.5)
                ax.xaxis.set_minor_locator(ticker.MultipleLocator(base=1))
                ax.tick_params(width=1, length=2, axis='x', which='minor', bottom=True, top=True, direction='in')
                ax.yaxis.set_minor_locator(ticker.MultipleLocator(base=1))
                ax.tick_params(width=1, length=2, axis='y', which='minor', left = True, right = True, direction='in')
                ax.xaxis.set_major_locator(ticker.MultipleLocator(base=5))
                ax.tick_params(width=1, length = 5, axis='x', which='major', bottom=True, top=True, direction='in')
                ax.yaxis.set_major_locator(ticker.MultipleLocator(base=5))
                ax.tick_params(width=1, length = 5, axis='y', which='major', left = True, right = True, direction='in')
            else:
                fig.delaxes(ax)
        for i in range(nrow):
            for j in range(ncol):
                if j != 0:
                    axs[i,j].set_yticklabels([])
                if i != nrow-1:
                    axs[i,j].set_xticklabels([])
        axs[nrow//2,0].set_ylabel('Absolute Abundances', fontsize = 15)
        #axs[nrow-1, ncol//2].set_xlabel('[%s%s/H]'%(A[elem_idx][0], self.ZA_symb_list[elem_idx][0]), fontsize = 15)
        axs[nrow-1, ncol//2].set_xlabel('[%s/H]'%(self.ZA_symb_list[26].values[0]), fontsize = 15)
        plt.tight_layout(rect = [0.05, 0, 1, 1])
        plt.subplots_adjust(wspace=0., hspace=0.)
        plt.show(block=False)
        plt.savefig(self._dir_out_figs + 'iso_abundance.pdf', bbox_inches='tight')
        
    def extract_normalized_abundances(self, Z_list, Mass_i_loc, c=3):
        solar_norm_H = self.c_class.solarA09_vs_H_bymass[Z_list]
        solar_norm_Fe = self.c_class.solarA09_vs_Fe_bymass[Z_list]
        Mass_i = np.loadtxt(Mass_i_loc)
        #Fe = np.sum(Mass_i[np.intersect1d(np.where(ZA_sorted[:,0]==26)[0], np.where(ZA_sorted[:,1]==56)[0]), c+2:], axis=0)
        Fe = np.sum(Mass_i[self.select_elemZ_idx(26), c+2:], axis=0)
        H = np.sum(Mass_i[self.select_elemZ_idx(1), c+2:], axis=0)
        FeH = np.log10(np.divide(Fe, H)) - solar_norm_H[26]
        abund_i = []
        for i,val in enumerate(Z_list):
            mass = np.sum(Mass_i[self.select_elemZ_idx(val), c+2:], axis=0)
            abund_i.append(np.log10(np.divide(mass,Fe)) - solar_norm_Fe[val])
        normalized_abundances = np.array(abund_i)
        return normalized_abundances, FeH

    def elem_abundance(self, figsiz = (32,10), c=3, setylim = (-6, 6), setxlim=(-6.5, 0.5)):
        print('Starting elem_abundance()')
        from matplotlib import pyplot as plt
        #plt.style.use(self._dir+'/galcem.mplstyle')
        import matplotlib.ticker as ticker
        Z_list = np.unique(self.ZA_sorted[:,0])
        ncol = self.aux.find_nearest(np.power(np.arange(20),2), len(Z_list))
        if len(Z_list) < ncol:
            nrow = ncol
        else:
            nrow = ncol + 1
        Z_symb_list = self.IN.periodic['elemSymb'][Z_list] # name of elements for all isotopes
        
        normalized_abundances, FeH = self.extract_normalized_abundances(Z_list, Mass_i_loc='runs/baseline/Mass_i.dat', c=c+2)
        normalized_abundances_lowZ, FeH_lowZ = self.extract_normalized_abundances(Z_list, Mass_i_loc='runs/lifetimeZ0003/Mass_i.dat', c=c+2)
        normalized_abundances_highZ, FeH_highZ = self.extract_normalized_abundances(Z_list, Mass_i_loc='runs/lifetimeZ06/Mass_i.dat', c=c+2)
        normalized_abundances_lowIMF, FeH_lowIMF = self.extract_normalized_abundances(Z_list, Mass_i_loc='runs/IMF1pt2/Mass_i.dat', c=c+2)
        normalized_abundances_highIMF, FeH_highIMF = self.extract_normalized_abundances(Z_list, Mass_i_loc='runs/IMF1pt7/Mass_i.dat', c=c+2)
        normalized_abundances_lowSFR, FeH_lowSFR = self.extract_normalized_abundances(Z_list, Mass_i_loc='runs/k_SFR_0.5/Mass_i.dat', c=c+2)
        normalized_abundances_highSFR, FeH_highSFR = self.extract_normalized_abundances(Z_list, Mass_i_loc='runs/k_SFR_2/Mass_i.dat', c=c+2)
        
        fig, axs = plt.subplots(nrow, ncol, figsize =figsiz)#, sharex=True)
        for i, ax in enumerate(axs.flat):
            if i < len(Z_list):
                #ax.plot(FeH, Masses[i], color='blue')
                #ax.plot(FeH, Masses2[i], color='orange', linewidth=2)
                ax.fill_between(FeH, normalized_abundances_lowIMF[i], normalized_abundances_highIMF[i], alpha=0.2, color='blue')
                ax.fill_between(FeH, normalized_abundances_lowSFR[i], normalized_abundances_highSFR[i], alpha=0.2, color='red')
                ax.plot(FeH, normalized_abundances[i], color='red', alpha=0.3)
                ax.plot(FeH_lowZ, normalized_abundances_lowZ[i], color='red', linestyle=':', alpha=0.3)
                ax.plot(FeH_highZ, normalized_abundances_highZ[i], color='red', linestyle='--', alpha=0.3)
                ax.axhline(y=0, color='grey', linestyle='--', linewidth=1, alpha=0.5)
                ax.axvline(x=0, color='grey', linestyle='--', linewidth=1, alpha=0.5)
                ax.annotate('%s%d'%(Z_list[i],Z_symb_list[i]), xy=(0.5, 0.92), xycoords='axes fraction', horizontalalignment='center', verticalalignment='top', fontsize=12, alpha=0.7)
                ax.set_ylim(setylim) #(-2, 2) #(-1.5, 1.5)
                ax.set_xlim(setxlim) #(-11, -2) #(-8.5, 0.5)
                ax.xaxis.set_minor_locator(ticker.MultipleLocator(base=1))
                ax.tick_params(width=1, length=2, axis='x', which='minor', bottom=True, top=True, direction='in')
                ax.yaxis.set_minor_locator(ticker.MultipleLocator(base=1))
                ax.tick_params(width=1, length=2, axis='y', which='minor', left = True, right = True, direction='in')
                ax.xaxis.set_major_locator(ticker.MultipleLocator(base=5))
                ax.tick_params(width=1, length = 5, axis='x', which='major', bottom=True, top=True, direction='in')
                ax.yaxis.set_major_locator(ticker.MultipleLocator(base=5))
                ax.tick_params(width=1, length = 5, axis='y', which='major', left = True, right = True, direction='in')
            else:
                fig.delaxes(ax)
        for i in range(nrow):
            for j in range(ncol):
                if j != 0:
                    axs[i,j].set_yticklabels([])
                if i != nrow-1:
                    axs[i,j].set_xticklabels([])
        axs[nrow//2,0].set_ylabel('[X/Fe]', fontsize = 15)
        axs[nrow-1, ncol//2].set_xlabel('[Fe/H]', fontsize = 15)
        fig.tight_layout(rect = [0.03, 0, 1, 1])
        fig.subplots_adjust(wspace=0., hspace=0.)
        plt.show(block=False)
        plt.savefig(self._dir_out_figs + 'elem_abundance.pdf', bbox_inches='tight')
    
    def select_elemZ_idx(self, elemZ):
        ''' auxiliary function that selects the isotope indexes where Z=elemZ '''
        return np.where(self.ZA_sorted[:,0]==elemZ)[0]
   
    def observational(self, figsiz = (15,10), c=3):
        print('Starting observational()')
        import glob
        import itertools
        from matplotlib import pyplot as plt
        import matplotlib.ticker as ticker
        plt.style.use(self._dir+'/galcem.mplstyle')
        Mass_i = np.loadtxt(self._dir_out+'Mass_i.dat')
        Z_list = np.unique(self.ZA_sorted[:,0])
        Z_symb_list = self.IN.periodic['elemSymb'][Z_list] # name of elements for all isotopes
        solar_norm_H = self.c_class.solarA09_vs_H_bymass[Z_list]
        solar_norm_Fe = self.c_class.solarA09_vs_Fe_bymass[Z_list]
        Masses2_i = []
        Fe = np.sum(Mass_i[self.select_elemZ_idx(26), c+2:], axis=0)
        H = np.sum(Mass_i[self.select_elemZ_idx(1), c+2:], axis=0)
        for i,val in enumerate(Z_list):
            mass = np.sum(Mass_i[self.select_elemZ_idx(val), c+2:], axis=0)
            Masses2_i.append(np.log10(np.divide(mass,Fe)) - solar_norm_Fe[val])
        Masses2 = np.array(Masses2_i) 
        FeH = np.log10(np.divide(Fe, H)) - solar_norm_H[26]
        ncol = self.aux.find_nearest(np.power(np.arange(20),2), len(Z_list))
        if len(Z_list) < ncol:
            nrow = ncol
        else:
            nrow = ncol + 1
        fig, axs = plt.subplots(nrow, ncol, figsize =figsiz)#, sharex=True)
    
        path = self._dir + r'/input/observations/abund' # use your path
        all_files = glob.glob(path + "/*.txt")
        all_files = sorted(all_files, key=len)

        li = []
        linames = []
        elemZmin = 12
        elemZmax = 12

        for filename in all_files:
            df = pd.read_table(filename, sep=',')
            elemZmin0 = np.min(df.iloc[:,0])
            elemZmax0 = np.max(df.iloc[:,0])
            elemZmin = np.min([elemZmin0, elemZmin])
            elemZmax = np.max([elemZmax0, elemZmax])
            li.append(df)
            linames.append(df['paperName'][0])

        lenlist = len(li)
        listmarkers = [r"$\mathcal{A}$",  r"$\mathcal{B}$",  r"$\mathcal{C}$",
                                    r"$\mathcal{D}$", r"$\mathcal{E}$", r"$\mathcal{F}$",
                                    r"$\mathcal{G}$", r"$\mathcal{H}$", r"$\mathcal{I}$",
                                    r"$\mathcal{J}$", r"$\mathcal{K}$", r"$\mathcal{L}$",
                                    r"$\mathcal{M}$", r"$\mathcal{N}$", r"$\mathcal{O}$",
                                    r"$\mathcal{P}$", r"$\mathcal{Q}$", r"$\mathcal{R}$",
                                    r"$\mathcal{S}$", r"$\mathcal{T}$", r"$\mathcal{U}$",
                                    r"$\mathcal{V}$", r"$\mathcal{X}$", r"$\mathcal{Y}$",
                                    "$1$", "$2$", "$3$", "$4$", "$5$", "$6$", 
                                    "$7$", "$8$", "$9$", "$f$", "$\u266B$",
                                    r"$\frac{1}{2}$",  'o', '+', 'x', 'v', '^', '<', '>',
                                    'P', '*', 'd', 'X',  "_", '|']
        listcolors = ['#cc6c00', '#ff8800', '#ffbb33', '#ffe564', '#2c4c00', '#436500',
        '#669900', '#99cc00', '#d2fe4c', '#3c1451', '#6b238e', '#9933cc',
        '#aa66cc', '#bc93d1', '#004c66', '#007299', '#0099cc', '#33b5e5',
        '#8ed5f0', '#660033', '#b20058', '#e50072', '#ff3298', '#ff7fbf',
        '#252525', '#525252', '#737373', '#969696', '#bdbdbd', '#d9d9d9',
        '#7f0000', '#cc0000', '#ff4444', '#ff7f7f', '#ffb2b2', '#995100']
                     #['#ff3399', '#5d8aa8', '#e32636', '#ffbf00', '#9966cc', '#a4c639',
                     # '#cd9575', '#008000', '#fbceb1', '#00ffff', '#4b5320', '#a52a2a',
                     # '#007fff', '#ff2052', '#21abcd', '#e97451', '#592720', '#fad6a5',
                     # '#36454f', '#e4d00a', '#ff3800', '#ffbcd9', '#008b8b', '#8b008b',
                     # '#03c03c', '#00009c', '#ccff00', '#673147', '#0f0f0f', '#324ab2',
                     # '#ffcc33', '#ffcccc', '#ff66ff', '#ff0033', '#ccff33', '#ccccff']
        lenlist = len(li)
    
        for i, ax in enumerate(axs.flat):
            colorlist = itertools.cycle(listcolors)
            markerlist =itertools.cycle(listmarkers)
            for j, ll in enumerate(li):
                ip = i+2 # Shift to skip H and He
                idx_obs = np.where(ll.iloc[:,0] == ip+1)[0]
                ax.scatter(ll.iloc[idx_obs,1], ll.iloc[idx_obs,2], label=linames[j], alpha=0.3, marker=next(markerlist), c=next(colorlist), s=20)
            if i == len(Z_list)-3:
                    ax.legend(ncol=7, loc='upper left', bbox_to_anchor=(1, 1), frameon=False, fontsize=7)
                    ax.set_xlabel(f'[Fe/H]', fontsize = 15)
            if i < len(Z_list)-2:
                ip = i+2 # Shift to skip H and He
                ax.plot(FeH, Masses2[ip], color='black', linewidth=2)
                ax.annotate(f"{Z_list[ip]}{Z_symb_list[Z_list[ip]]}", xy=(0.5, 0.92), xycoords='axes fraction', horizontalalignment='center', verticalalignment='top', fontsize=12, alpha=0.7)
                ax.set_ylim(-5.9, 5.9)
                ax.set_xlim(-6.5, 1.5)
                ax.xaxis.set_minor_locator(ticker.MultipleLocator(base=.5))
                ax.tick_params(width=1, length=2, axis='x', which='minor', bottom=True, top=True, direction='in')
                ax.yaxis.set_minor_locator(ticker.MultipleLocator(base=.5))
                ax.tick_params(width=1, length=2, axis='y', which='minor', left = True, right = True, direction='in')
                ax.xaxis.set_major_locator(ticker.MultipleLocator(base=2))
                ax.tick_params(width=1, length = 5, axis='x', which='major', bottom=True, top=True, direction='in')
                ax.yaxis.set_major_locator(ticker.MultipleLocator(base=3))
                ax.tick_params(width=1, length = 5, axis='y', which='major', left = True, right = True, direction='in')
            else:
                fig.delaxes(ax)
        for i in range(nrow):
            for j in range(ncol):
                if j != 0:
                    axs[i,j].set_yticklabels([])
                if i != nrow-1:
                    axs[i,j].set_xticklabels([])
        axs[nrow//2,0].set_ylabel('[X/Fe]', fontsize = 15)
        #axs[nrow-1, ncol//2].set_xlabel(f'[Fe/H]', fontsize = 15)
        fig.tight_layout(rect = [0.03, 0, 1, 1])
        fig.subplots_adjust(wspace=0., hspace=0.)
        plt.show(block=False)
        plt.savefig(self._dir_out_figs + 'elem_obs.pdf', bbox_inches='tight')
        return None

    def obs_table(self, up_to_elemZ=30):
        import glob
        elemZ = np.arange(3,up_to_elemZ+1)
        Z_symb_list = self.IN.periodic['elemSymb'][elemZ]
        
        path = self._dir + r'/input/observations/abund' # use your path
        all_files = glob.glob(path + "/*.txt")
        all_files = sorted(all_files, key=len)#list(np.sort(all_files))

        li = []
        linames = []
        for filename in all_files:
            df = pd.read_table(filename, sep=',')
            li.append(df)
            df['paperName'] = df['paperName'].str.replace('&','-and-')
            linames.append(df['paperName'][0])
        
        obs_dict = {}
        for en, paperName in enumerate(linames):
            elemZ_yn = []
            for eZ in elemZ:
                if eZ in np.unique(li[en]['elemZ']):
                    elemZ_yn.append(' $\\times$ ')
                else:
                    if not eZ == 26:
                        elemZ_yn.append(' $\\bigcirc$ ')
                    else:
                        elemZ_yn.append(' $\\times$ ')
            obs_dict[paperName] = elemZ_yn
        save_obs_dict = pd.DataFrame(obs_dict)
        save_obs_dict['elemZ'] = Z_symb_list.to_numpy()
        save_obs_dict_to_csv = save_obs_dict.T.iloc[::-1]
        save_obs_dict_to_csv['return'] = ' \\\\'
        save_obs_dict_to_csv.to_csv(self._dir_out + 'observationtable.csv', sep='&')

    def _observational_lelemZ(self, figsiz = (15,10), c=3, yrange='zoom', romano10=False):
        ''' yrange full to include all observational points'''
        print('Starting observational_lelemZ()')
        import glob
        import itertools
        from matplotlib import pyplot as plt
        import matplotlib.ticker as ticker
        #plt.style.use(self._dir+'/galcem.mplstyle')
        Mass_i = np.loadtxt(self._dir_out+'Mass_i.dat')
        Z_list = np.unique(self.ZA_sorted[:,0])
        Z_symb_list = self.IN.periodic['elemSymb'][Z_list] # name of elements for all isotopes
        solar_norm_H = self.c_class.solarA09_vs_H_bymass[Z_list]
        solar_norm_Fe = self.c_class.solarA09_vs_Fe_bymass[Z_list]
        Masses_i = []
        Masses2_i = []
        Fe = np.sum(Mass_i[self.select_elemZ_idx(26), c+2:], axis=0)
        H = np.sum(Mass_i[self.select_elemZ_idx(1), c+2:], axis=0)
        print(f'{Z_list}')
        for i,val in enumerate(Z_list):
            print(f'{val=}')
            print(f'{self.select_elemZ_idx(val)=}')
            print(f'{Mass_i[self.select_elemZ_idx(val), c+2:]=}')
            mass = np.sum(Mass_i[self.select_elemZ_idx(val), c+2:], axis=0)
            Masses2_i.append(np.log10(np.divide(mass,Fe)) - solar_norm_Fe[val])
            Masses_i.append(mass)
        Masses = np.log10(np.divide(Masses_i, Fe))
        Masses2 = np.array(Masses2_i) 
        FeH = np.log10(np.divide(Fe, H)) - solar_norm_H[26]
        nrow = 5
        ncol = 6
        fig, axs = plt.subplots(nrow, ncol, figsize =figsiz)#, sharex=True)

        path = self._dir + r'/input/observations/abund' # use your path
        all_files = glob.glob(path + "/*.txt")
        all_files = sorted(all_files, key=len)#list(np.sort(all_files))

        li = []
        linames = []
        elemZmin = 12
        elemZmax = 12

        for filename in all_files:
            df = pd.read_table(filename, sep=',')
            elemZmin0 = np.min(df.iloc[:,0])
            elemZmax0 = np.max(df.iloc[:,0])
            elemZmin = np.min([elemZmin0, elemZmin])
            elemZmax = np.max([elemZmax0, elemZmax])
            li.append(df)
            linames.append(df['paperName'][0])

        lenlist = len(li)
        listmarkers = [r"$\mathcal{A}$",  r"$\mathcal{B}$",  r"$\mathcal{C}$",
                                    r"$\mathcal{D}$", r"$\mathcal{E}$", r"$\mathcal{F}$",
                                    r"$\mathcal{G}$", r"$\mathcal{H}$", r"$\mathcal{I}$",
                                    r"$\mathcal{J}$", r"$\mathcal{K}$", r"$\mathcal{L}$",
                                    r"$\mathcal{M}$", r"$\mathcal{N}$", r"$\mathcal{O}$",
                                    r"$\mathcal{P}$", r"$\mathcal{Q}$", r"$\mathcal{R}$",
                                    r"$\mathcal{S}$", r"$\mathcal{T}$", r"$\mathcal{U}$",
                                    r"$\mathcal{V}$", r"$\mathcal{X}$", r"$\mathcal{Y}$",
                                    "$1$", "$2$", "$3$", "$4$", "$5$", "$6$", 
                                    "$7$", "$8$", "$9$", "$f$", "$\u266B$",
                                    r"$\frac{1}{2}$",  'o', '+', 'x', 'v', '^', '<', '>',
                                    'P', '*', 'd', 'X',  "_", '|']
        listcolors = ['#cc6c00', '#ff8800', '#ffbb33', '#ffe564', '#2c4c00', '#436500',
        '#669900', '#99cc00', '#d2fe4c', '#3c1451', '#6b238e', '#9933cc',
        '#aa66cc', '#bc93d1', '#004c66', '#007299', '#0099cc', '#33b5e5',
        '#8ed5f0', '#660033', '#b20058', '#e50072', '#ff3298', '#ff7fbf',
        '#252525', '#525252', '#737373', '#969696', '#bdbdbd', '#d9d9d9',
        '#7f0000', '#cc0000', '#ff4444', '#ff7f7f', '#ffb2b2', '#995100']
                     #['#ff3399', '#5d8aa8', '#e32636', '#ffbf00', '#9966cc', '#a4c639',
                     # '#cd9575', '#008000', '#fbceb1', '#00ffff', '#4b5320', '#a52a2a',
                     # '#007fff', '#ff2052', '#21abcd', '#e97451', '#592720', '#fad6a5',
                     # '#36454f', '#e4d00a', '#ff3800', '#ffbcd9', '#008b8b', '#8b008b',
                     # '#03c03c', '#00009c', '#ccff00', '#673147', '#0f0f0f', '#324ab2',
                     # '#ffcc33', '#ffcccc', '#ff66ff', '#ff0033', '#ccff33', '#ccccff']

        if romano10 == True:
            r10_path = self._dir + r'/input/r10/'
            abundb = np.loadtxt(r10_path+'romano10b.dat')
            abundc = np.loadtxt(r10_path+'romano10c.dat')
            r10_FeH = abundc[:,1]
            r10_time_Gyr = abundc[:,0]
            cu = np.loadtxt(r10_path+'cu.dat')
            r10_labels = ['C', 'N', 'O', 'Na', 'Mg', 'Al', 'Si', 'S', 'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Co', 'Ni', 'Cu', 'Zn', '-']
            r10_elem = [abundc[:,3], abundb[:,7], abundc[:,2], np.add(abundc[:,4], 1.5), abundc[:,5], abundc[:,9], abundb[:,6], abundc[:,6], abundc[:,11], abundb[:,8], abundb[:,5], abundb[:,9], abundb[:,10], abundb[:,11], abundb[:,12], abundc[:,10], abundb[:,4], cu[:,1], abundb[:,3], np.zeros(len(abundb[:,0]))]
            r10_elem_dict = dict(zip(r10_labels, r10_elem))

        obs_paper_list = {}
        for i, ax in enumerate(axs.flat):
            colorlist = itertools.cycle(listcolors)
            markerlist =itertools.cycle(listmarkers)
            paper_list = []
            for j, ll in enumerate(li):
                ip = i+2 # Shift to skip H and He
                idx_obs = np.where(ll.iloc[:,0] == ip+1)[0]
                ax.scatter(ll.iloc[idx_obs,1], ll.iloc[idx_obs,2], label=linames[j], alpha=0.3, marker=next(markerlist), c=next(colorlist), s=20)
                paper_list.append(np.unique(ll.iloc[idx_obs,-1].to_numpy())[0])
            obs_paper_list[ip] = paper_list
            if i == 0:
                    ax.legend(ncol=7, loc='lower left', bbox_to_anchor=(-.2, 1.), frameon=False, fontsize=9)
            if i < nrow*ncol:
                ip = i+2 # Shift to skip H and He
                #ax.plot(FeH, Masses[i], color='blue')
                ax.plot(FeH, Masses2[ip], color='black', linewidth=2)
                ax.annotate(f"{Z_list[ip]}{Z_symb_list[Z_list[ip]]}", xy=(0.5, 0.92), xycoords='axes fraction', horizontalalignment='center', verticalalignment='top', fontsize=12, alpha=0.7)
                if romano10 == True:
                    if Z_symb_list[Z_list[ip]] in r10_labels:
                        ax.plot(r10_FeH, r10_elem_dict[Z_symb_list[Z_list[ip]]], color='red', linewidth=2)
                ax.set_ylim(-2.5, 2.5)
                if yrange=='full': ax.set_ylim(-5.9, 5.9)
                ax.set_xlim(-6.5, 1.5)
                ax.xaxis.set_minor_locator(ticker.MultipleLocator(base=.5))
                ax.tick_params(width=1, length = 5, axis='x', which='minor', bottom=True, top=True, direction='in')
                ax.yaxis.set_minor_locator(ticker.MultipleLocator(base=.5))
                ax.tick_params(width=1, length = 5, axis='y', which='minor', left = True, right = True, direction='in')
                ax.xaxis.set_major_locator(ticker.MultipleLocator(base=2))
                ax.tick_params(width=1, length = 7, axis='x', which='major', bottom=True, top=True, direction='in')
                ax.yaxis.set_major_locator(ticker.MultipleLocator(base=2))
                ax.tick_params(width=1, length = 7, axis='y', which='major', left = True, right = True, direction='in')
            else:
                fig.delaxes(ax)
        for i in range(nrow):
            for j in range(ncol):
                if j != 0:
                    axs[i,j].set_yticklabels([])
                if i != nrow-1:
                    axs[i,j].set_xticklabels([])
        axs[nrow//2,0].set_ylabel('[X/Fe]', fontsize = 15)
        axs[nrow-1, ncol//2].set_xlabel(f'[Fe/H]', fontsize = 15)
        fig.tight_layout(rect=[0., 0, 1, .9])
        fig.subplots_adjust(wspace=0., hspace=0.)
        plt.show(block=False)
        #pickle.dump(obs_paper_list, open(self._dir+ '/elem_obs.pkl', 'wb'))
        plt.savefig(self._dir_out_figs + 'elem_obs_lelemZ.pdf', bbox_inches='tight')
        return None
    
    def observational_lelemZ(self, figsiz = (15,10), c=3, yrange='zoom', romano10=False):
        ''' yrange full to include all observational points'''
        print('Starting observational_lelemZ()')
        import glob
        import itertools
        from matplotlib import pyplot as plt
        import matplotlib.ticker as ticker
        #plt.style.use(self._dir+'/galcem.mplstyle')
        Mass_i = np.loadtxt(self._dir_out+'Mass_i.dat')
        Z_list = np.unique(self.ZA_sorted[:,0])
        Z_symb_list = self.IN.periodic['elemSymb'][Z_list] # name of elements for all isotopes
        solar_norm_H = self.c_class.solarA09_vs_H_bymass[Z_list]
        solar_norm_Fe = self.c_class.solarA09_vs_Fe_bymass[Z_list]
        Masses_i = []
        Masses2_i = []
        Fe = np.sum(Mass_i[self.select_elemZ_idx(26), c+2:], axis=0)
        H = np.sum(Mass_i[self.select_elemZ_idx(1), c+2:], axis=0)
        for i,val in enumerate(Z_list):
            mass = np.sum(Mass_i[self.select_elemZ_idx(val), c+2:], axis=0)
            Masses2_i.append(np.log10(np.divide(mass,Fe)) - solar_norm_Fe[val])
            Masses_i.append(mass)
        Masses = np.log10(np.divide(Masses_i, Fe))
        Masses2 = np.array(Masses2_i) 
        FeH = np.log10(np.divide(Fe, H)) - solar_norm_H[26]
        nrow = 5
        ncol = 6
        fig, axs = plt.subplots(nrow, ncol, figsize =figsiz)#, sharex=True)

        path = self._dir + r'/input/observations/abund' # use your path
        all_files = glob.glob(path + "/*.txt")
        all_files = sorted(all_files, key=len)#list(np.sort(all_files))

        li = []
        linames = []
        elemZmin = 12
        elemZmax = 12

        for filename in all_files:
            df = pd.read_table(filename, sep=',')
            elemZmin0 = np.min(df.iloc[:,0])
            elemZmax0 = np.max(df.iloc[:,0])
            elemZmin = np.min([elemZmin0, elemZmin])
            elemZmax = np.max([elemZmax0, elemZmax])
            li.append(df)
            linames.append(df['paperName'][0])

        lenlist = len(li)
        listmarkers = [r"$\mathcal{A}$",  r"$\mathcal{B}$",  r"$\mathcal{C}$",
                                    r"$\mathcal{D}$", r"$\mathcal{E}$", r"$\mathcal{F}$",
                                    r"$\mathcal{G}$", r"$\mathcal{H}$", r"$\mathcal{I}$",
                                    r"$\mathcal{J}$", r"$\mathcal{K}$", r"$\mathcal{L}$",
                                    r"$\mathcal{M}$", r"$\mathcal{N}$", r"$\mathcal{O}$",
                                    r"$\mathcal{P}$", r"$\mathcal{Q}$", r"$\mathcal{R}$",
                                    r"$\mathcal{S}$", r"$\mathcal{T}$", r"$\mathcal{U}$",
                                    r"$\mathcal{V}$", r"$\mathcal{X}$", r"$\mathcal{Y}$",
                                    "$1$", "$2$", "$3$", "$4$", "$5$", "$6$", 
                                    "$7$", "$8$", "$9$", "$f$", "$\u266B$",
                                    r"$\frac{1}{2}$",  'o', '+', 'x', 'v', '^', '<', '>',
                                    'P', '*', 'd', 'X',  "_", '|']
        listcolors = ['#cc6c00', '#ff8800', '#ffbb33', '#ffe564', '#2c4c00', '#436500',
        '#669900', '#99cc00', '#d2fe4c', '#3c1451', '#6b238e', '#9933cc',
        '#aa66cc', '#bc93d1', '#004c66', '#007299', '#0099cc', '#33b5e5',
        '#8ed5f0', '#660033', '#b20058', '#e50072', '#ff3298', '#ff7fbf',
        '#252525', '#525252', '#737373', '#969696', '#bdbdbd', '#d9d9d9',
        '#7f0000', '#cc0000', '#ff4444', '#ff7f7f', '#ffb2b2', '#995100']
                     #['#ff3399', '#5d8aa8', '#e32636', '#ffbf00', '#9966cc', '#a4c639',
                     # '#cd9575', '#008000', '#fbceb1', '#00ffff', '#4b5320', '#a52a2a',
                     # '#007fff', '#ff2052', '#21abcd', '#e97451', '#592720', '#fad6a5',
                     # '#36454f', '#e4d00a', '#ff3800', '#ffbcd9', '#008b8b', '#8b008b',
                     # '#03c03c', '#00009c', '#ccff00', '#673147', '#0f0f0f', '#324ab2',
                     # '#ffcc33', '#ffcccc', '#ff66ff', '#ff0033', '#ccff33', '#ccccff']

        if romano10 == True:
            r10_path = self._dir + r'/input/r10/'
            abundb = np.loadtxt(r10_path+'abundb8h.mwgk09')
            abundc = np.loadtxt(r10_path+'abundc8h.mwgk09')
            r10_FeH = abundc[:,1]
            r10_time_Gyr = abundc[:,0]
            cu = np.loadtxt(r10_path+'cu.dat')
            r10_labels = ['C', 'N', 'O', 'Na', 'Mg', 'Al', 'Si', 'S', 'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Co', 'Ni', 'Cu', 'Zn', '-']
            r10_elem = [abundc[:,3], abundb[:,7], abundc[:,2], np.add(abundc[:,4], 1.5), abundc[:,5], abundc[:,9], abundb[:,6], abundc[:,6], abundc[:,11], abundb[:,8], abundb[:,5], abundb[:,9], abundb[:,10], abundb[:,11], abundb[:,12], abundc[:,10], abundb[:,4], cu[:,1], abundb[:,3], np.zeros(len(abundb[:,0]))]
            r10_elem_dict = dict(zip(r10_labels, r10_elem))

        for i, ax in enumerate(axs.flat):
            colorlist = itertools.cycle(listcolors)
            markerlist =itertools.cycle(listmarkers)
            for j, ll in enumerate(li):
                ip = i+2 # Shift to skip H and He
                idx_obs = np.where(ll.iloc[:,0] == ip+1)[0]
                ax.scatter(ll.iloc[idx_obs,1], ll.iloc[idx_obs,2], label=linames[j], alpha=0.3, marker=next(markerlist), c=next(colorlist), s=20)
            if i == 0:
                    ax.legend(ncol=7, loc='lower left', bbox_to_anchor=(-.2, 1.), frameon=False, fontsize=9)
            if i < nrow*ncol:
                ip = i+2 # Shift to skip H and He
                #ax.plot(FeH, Masses[i], color='blue')
                ax.plot(FeH, Masses2[ip], color='black', linewidth=2)
                ax.annotate(f"{Z_list[ip]}{Z_symb_list[Z_list[ip]]}", xy=(0.5, 0.92), xycoords='axes fraction', horizontalalignment='center', verticalalignment='top', fontsize=12, alpha=0.7)
                if romano10 == True:
                    if Z_symb_list[Z_list[ip]] in r10_labels:
                        ax.plot(r10_FeH, r10_elem_dict[Z_symb_list[Z_list[ip]]], color='red', linewidth=2)
                ax.set_ylim(-2.5, 2.5)
                if yrange=='full': ax.set_ylim(-5.9, 5.9)
                ax.set_xlim(-6.5, 1.5)
                ax.xaxis.set_minor_locator(ticker.MultipleLocator(base=.5))
                ax.tick_params(width=1, length = 5, axis='x', which='minor', bottom=True, top=True, direction='in')
                ax.yaxis.set_minor_locator(ticker.MultipleLocator(base=.5))
                ax.tick_params(width=1, length = 5, axis='y', which='minor', left = True, right = True, direction='in')
                ax.xaxis.set_major_locator(ticker.MultipleLocator(base=2))
                ax.tick_params(width=1, length = 7, axis='x', which='major', bottom=True, top=True, direction='in')
                ax.yaxis.set_major_locator(ticker.MultipleLocator(base=2))
                ax.tick_params(width=1, length = 7, axis='y', which='major', left = True, right = True, direction='in')
            else:
                fig.delaxes(ax)
        for i in range(nrow):
            for j in range(ncol):
                if j != 0:
                    axs[i,j].set_yticklabels([])
                if i != nrow-1:
                    axs[i,j].set_xticklabels([])
        axs[nrow//2,0].set_ylabel('[X/Fe]', fontsize = 15)
        axs[nrow-1, ncol//2].set_xlabel(f'[Fe/H]', fontsize = 15)
        fig.tight_layout(rect=[0., 0, 1, .9])
        fig.subplots_adjust(wspace=0., hspace=0.)
        plt.show(block=False)
        plt.savefig(self._dir_out_figs + 'elem_obs_lelemZ.pdf', bbox_inches='tight')
        return None
    
    def extract_comparison(self, dir_val, select_elemZ_idx, solar_norm_Fe, Z_list):
        directory = 'runs/'+dir_val+'/'
        Mass_i = np.loadtxt(directory+'Mass_i.dat')
        Masses_i = []
        Fe = np.sum(Mass_i[self.select_elemZ_idx(26), c+2:], axis=0)
        H = np.sum(Mass_i[self.select_elemZ_idx(1), c+2:], axis=0)
        FeH = np.log10(np.divide(Fe, H)) - solar_norm_H[26]
        for i,val in enumerate(Z_list):
            mass = np.sum(Mass1_i[self.select_elemZ_idx(val), c+2:], axis=0)
            Masses_i.append(np.log10(np.divide(mass,Fe)) - solar_norm_Fe[val])
        Masses = np.array(Masses_i) 
        return FeH, Masses
    
    def observational_helemZ_dir_comparison(self, figsiz = (15,10), c=3, yrange='full', 
                                            romano10=False, directories={'SMBH zap':'20220623_zap_2Myr','MRSN':'20220614_MRSN_massrange_2Myr'},
                                            Z_list=[26,38,39,40,41,42,44,45,46,47,56,57,58,
                                                    59,60,62,63,64,66,67,68,70,72,76,77,79,82]):
        ''' yrange full to include all observational points'''
        print('observational_helemZ_dir_comparison()')
        import glob
        import itertools
        from matplotlib import pyplot as plt
        import matplotlib.ticker as ticker
        #plt.style.use(self._dir+'/galcem.mplstyle')
        #Z_list = np.unique(self.ZA_sorted[:,0])
        Z_list = np.array(Z_list)
        Z_symb_list = self.IN.periodic['elemSymb'][Z_list] # name of elements for all isotopes
        solar_norm_H = self.c_class.solarA09_vs_H_bymass[Z_list]
        solar_norm_Fe = self.c_class.solarA09_vs_Fe_bymass[Z_list]
        plot_pairs = {}
        for d in directories:
            plot_pairs[d] = self.extract_comparison(directories[d], self.select_elemZ_idx, solar_norm_Fe, Z_list)

        path = self._dir + r'/input/observations/abund' # use your path
        all_files = glob.glob(path + "/*.txt")
        all_files = sorted(all_files, key=len)#list(np.sort(all_files))

        li = []
        linames = []
        elemZmin = 12
        elemZmax = 12

        for filename in all_files:
            df = pd.read_table(filename, sep=',')
            elemZmin0 = np.min(df.iloc[:,0])
            elemZmax0 = np.max(df.iloc[:,0])
            elemZmin = np.min([elemZmin0, elemZmin])
            elemZmax = np.max([elemZmax0, elemZmax])
            li.append(df)
            linames.append(df['paperName'][0])

        lenlist = len(li)
        listmarkers = [r"$\mathcal{A}$",  r"$\mathcal{B}$",  r"$\mathcal{C}$",
                                    r"$\mathcal{D}$", r"$\mathcal{E}$", r"$\mathcal{F}$",
                                    r"$\mathcal{G}$", r"$\mathcal{H}$", r"$\mathcal{I}$",
                                    r"$\mathcal{J}$", r"$\mathcal{K}$", r"$\mathcal{L}$",
                                    r"$\mathcal{M}$", r"$\mathcal{N}$", r"$\mathcal{O}$",
                                    r"$\mathcal{P}$", r"$\mathcal{Q}$", r"$\mathcal{R}$",
                                    r"$\mathcal{S}$", r"$\mathcal{T}$", r"$\mathcal{U}$",
                                    r"$\mathcal{V}$", r"$\mathcal{X}$", r"$\mathcal{Y}$",
                                    "$1$", "$2$", "$3$", "$4$", "$5$", "$6$", 
                                    "$7$", "$8$", "$9$", "$f$", "$\u266B$",
                                    r"$\frac{1}{2}$",  'o', '+', 'x', 'v', '^', '<', '>',
                                    'P', '*', 'd', 'X',  "_", '|']
        listcolors = ['#cc6c00', '#ff8800', '#ffbb33', '#ffe564', '#2c4c00', '#436500',
        '#669900', '#99cc00', '#d2fe4c', '#3c1451', '#6b238e', '#9933cc',
        '#aa66cc', '#bc93d1', '#004c66', '#007299', '#0099cc', '#33b5e5',
        '#8ed5f0', '#660033', '#b20058', '#e50072', '#ff3298', '#ff7fbf',
        '#252525', '#525252', '#737373', '#969696', '#bdbdbd', '#d9d9d9',
        '#7f0000', '#cc0000', '#ff4444', '#ff7f7f', '#ffb2b2', '#995100']

        if romano10 == True:
            r10_path = self._dir + r'/input/r10/'
            abundb = np.loadtxt(r10_path+'abundb8h.mwgk09')
            abundc = np.loadtxt(r10_path+'abundc8h.mwgk09')
            r10_FeH = abundc[:,1]
            r10_time_Gyr = abundc[:,0]
            cu = np.loadtxt(r10_path+'cu.dat')
            r10_labels = ['C', 'N', 'O', 'Na', 'Mg', 'Al', 'Si', 'S', 'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Co', 'Ni', 'Cu', 'Zn', '-']
            r10_elem = [abundc[:,3], abundb[:,7], abundc[:,2], np.add(abundc[:,4], 1.5), abundc[:,5], abundc[:,9], abundb[:,6], abundc[:,6], abundc[:,11], abundb[:,8], abundb[:,5], abundb[:,9], abundb[:,10], abundb[:,11], abundb[:,12], abundc[:,10], abundb[:,4], cu[:,1], abundb[:,3], np.zeros(len(abundb[:,0]))]
            r10_elem_dict = dict(zip(r10_labels, r10_elem))

        nrow = 5
        ncol = 6
        fig, axs = plt.subplots(nrow, ncol, figsize =figsiz)#, sharex=True)
        
        for i, ax in enumerate(axs.flat):
            colorlist = itertools.cycle(listcolors)
            markerlist =itertools.cycle(listmarkers)
            print(f'{FeH1==FeH2}')
            for j, ll in enumerate(li):
                if i < len(Z_list):
                    ip = Z_list[i]#+2 # Shift to skip H and He
                    idx_obs = np.where(ll.iloc[:,0] == ip)[0]
                    ax.scatter(ll.iloc[idx_obs,1], ll.iloc[idx_obs,2], label=linames[j], alpha=0.3, marker=next(markerlist), c=next(colorlist), s=20)
            if i == 0:
                    ax.legend(ncol=7, loc='lower left', bbox_to_anchor=(-.2, 1.), frameon=False, fontsize=9)
            if i < len(Z_list):
                ip = i#+2 # Shift to skip H and He
                for d in dictionaries:
                    ax.plot(plot_pairs[d][0], plot_pairs[d][1], color='black', linewidth=2, linestyle=tools.cylcle(['-','--',':','-.']), label=d)
                ax.plot(FeH2, Masses2[ip], color='black', linewidth=2)
                ax.plot(FeH1, Masses1[ip], color='black', linewidth=2, linestyle='--')
                ax.fill_between(FeH1, Masses1[ip], Masses2[ip], where=(Masses1[ip] > Masses2[ip]), color='blue', alpha=0.2,
                 interpolate=True)
                ax.fill_between(FeH1, Masses1[ip], Masses2[ip], where=(Masses1[ip] <= Masses2[ip]), color='red', alpha=0.2,
                 interpolate=True)
                ax.annotate(f"{Z_list[ip]}{Z_symb_list[Z_list[ip]]}", xy=(0.5, 0.92), xycoords='axes fraction', horizontalalignment='center', verticalalignment='top', fontsize=12, alpha=0.7)
                if romano10 == True:
                    if Z_symb_list[Z_list[ip]] in r10_labels:
                        ax.plot(r10_FeH, r10_elem_dict[Z_symb_list[Z_list[ip]]], color='red', linewidth=2)
                ax.set_ylim(-2.5, 2.5)
                if yrange=='full': ax.set_ylim(-5.9, 5.9)
                ax.set_xlim(-6.5, 1.5)
                ax.xaxis.set_minor_locator(ticker.MultipleLocator(base=.5))
                ax.tick_params(width=1, length = 5, axis='x', which='minor', bottom=True, top=True, direction='in')
                ax.yaxis.set_minor_locator(ticker.MultipleLocator(base=.5))
                ax.tick_params(width=1, length = 5, axis='y', which='minor', left = True, right = True, direction='in')
                ax.xaxis.set_major_locator(ticker.MultipleLocator(base=2))
                ax.tick_params(width=1, length = 7, axis='x', which='major', bottom=True, top=True, direction='in')
                ax.yaxis.set_major_locator(ticker.MultipleLocator(base=2))
                ax.tick_params(width=1, length = 7, axis='y', which='major', left = True, right = True, direction='in')
            else:
                fig.delaxes(ax)
        for i in range(nrow):
            for j in range(ncol):
                if j != 0:
                    axs[i,j].set_yticklabels([])
                if i != nrow-1:
                    axs[i,j].set_xticklabels([])
        axs[nrow//2,0].set_ylabel('[X/Fe]', fontsize = 15)
        axs[nrow-1, ncol//2].set_xlabel(f'[Fe/H]', fontsize = 15)
        fig.tight_layout(rect=[0., 0, 1, .9])
        fig.subplots_adjust(wspace=0., hspace=0.)
        plt.show(block=False)
        plt.savefig(self._dir_out_figs + 'elem_obs_helemZ_dir_comparison.pdf', bbox_inches='tight')
        return None
    
    
    def _obs_lZ(self, figsiz = (21,7), c=3):
        print('Starting observational_lZ()')
        import glob
        import itertools
        from matplotlib import pyplot as plt
        import matplotlib.ticker as ticker
        #plt.style.use(self._dir+'/galcem.mplstyle')
        Mass_i = np.loadtxt(self._dir_out+'Mass_i.dat')
        Z_list = np.unique(self.ZA_sorted[:,0])
        Z_symb_list = self.IN.periodic['elemSymb'][Z_list] # name of elements for all isotopes
        solar_norm_H = self.c_class.solarA09_vs_H_bymass[Z_list]
        solar_norm_Fe = self.c_class.solarA09_vs_Fe_bymass[Z_list]
        Masses_i = []
        Masses2_i = []
        Fe = np.sum(Mass_i[self.select_elemZ_idx(26), c+2:], axis=0)
        H = np.sum(Mass_i[self.select_elemZ_idx(1), c+2:], axis=0)
        for i,val in enumerate(Z_list):
            mass = np.sum(Mass_i[self.select_elemZ_idx(val), c+2:], axis=0)
            Masses2_i.append(np.log10(np.divide(mass,Fe)) - solar_norm_Fe[val])
            Masses_i.append(mass)
        Masses = np.log10(np.divide(Masses_i, Fe))
        Masses2 = np.array(Masses2_i) 
        FeH = np.log10(np.divide(Fe, H)) - solar_norm_H[26]
        nrow = 4
        ncol = 7
        fig, axs = plt.subplots(nrow, ncol, figsize =figsiz)#, sharex=True)

        path = self._dir + r'/input/observations/abund' # use your path
        all_files = glob.glob(path + "/*.txt")
        all_files = sorted(all_files, key=len)

        li = []
        linames = []
        elemZmin = 12
        elemZmax = 12

        for filename in all_files:
            df = pd.read_table(filename, sep=',')
            elemZmin0 = np.min(df.iloc[:,0])
            elemZmax0 = np.max(df.iloc[:,0])
            elemZmin = np.min([elemZmin0, elemZmin])
            elemZmax = np.max([elemZmax0, elemZmax])
            li.append(df)
            linames.append(df['paperName'][0])

        lenlist = len(li)
        listmarkers = [r"$\mathcal{A}$",  r"$\mathcal{B}$",  r"$\mathcal{C}$",
                                    r"$\mathcal{D}$", r"$\mathcal{E}$", r"$\mathcal{F}$",
                                    r"$\mathcal{G}$", r"$\mathcal{H}$", r"$\mathcal{I}$",
                                    r"$\mathcal{J}$", r"$\mathcal{K}$", r"$\mathcal{L}$",
                                    r"$\mathcal{M}$", r"$\mathcal{N}$", r"$\mathcal{O}$",
                                    r"$\mathcal{P}$", r"$\mathcal{Q}$", r"$\mathcal{R}$",
                                    r"$\mathcal{S}$", r"$\mathcal{T}$", r"$\mathcal{U}$",
                                    r"$\mathcal{V}$", r"$\mathcal{X}$", r"$\mathcal{Y}$",
                                    "$1$", "$2$", "$3$", "$4$", "$5$", "$6$", 
                                    "$7$", "$8$", "$9$", "$f$", "$\u266B$",
                                    r"$\frac{1}{2}$",  'o', '+', 'x', 'v', '^', '<', '>',
                                    'P', '*', 'd', 'X',  "_", '|']
        listcolors = ['#cc6c00', '#ff8800', '#ffbb33', '#ffe564', '#2c4c00', '#436500',
        '#669900', '#99cc00', '#d2fe4c', '#3c1451', '#6b238e', '#9933cc',
        '#aa66cc', '#bc93d1', '#004c66', '#007299', '#0099cc', '#33b5e5',
        '#8ed5f0', '#660033', '#b20058', '#e50072', '#ff3298', '#ff7fbf',
        '#252525', '#525252', '#737373', '#969696', '#bdbdbd', '#d9d9d9',
        '#7f0000', '#cc0000', '#ff4444', '#ff7f7f', '#ffb2b2', '#995100']
                     #['#ff3399', '#5d8aa8', '#e32636', '#ffbf00', '#9966cc', '#a4c639',
                     # '#cd9575', '#008000', '#fbceb1', '#00ffff', '#4b5320', '#a52a2a',
                     # '#007fff', '#ff2052', '#21abcd', '#e97451', '#592720', '#fad6a5',
                     # '#36454f', '#e4d00a', '#ff3800', '#ffbcd9', '#008b8b', '#8b008b',
                     # '#03c03c', '#00009c', '#ccff00', '#673147', '#0f0f0f', '#324ab2',
                     # '#ffcc33', '#ffcccc', '#ff66ff', '#ff0033', '#ccff33', '#ccccff']
        lenlist = len(li)

        for i, ax in enumerate(axs.flat):
            colorl = itertools.cycle(listcolors)
            markerl = itertools.cycle(listmarkers)
            for j, ll in enumerate(li):
                ip = i+2 # Shift to skip H and He
                idx_obs = np.where(ll.iloc[:,0] == ip+1)[0]
                ax.scatter(ll.iloc[idx_obs,1], ll.iloc[idx_obs,2], label=linames[j], alpha=0.3, marker=next(markerl), c=next(colorl), s=20)
            if i == 0:
                    ax.legend(ncol=7, loc='lower left', bbox_to_anchor=(-0.2, 1.05), frameon=False, fontsize=9)
            if i < nrow*ncol:
                ip = i+2 # Shift to skip H and He
                #ax.plot(FeH, Masses[i], color='blue')
                ax.plot(FeH, Masses2[ip], color='black', linewidth=2)
                ax.annotate(f"{Z_list[ip]}{Z_symb_list[Z_list[ip]]}", xy=(0.5, 0.92), xycoords='axes fraction', horizontalalignment='center', verticalalignment='top', fontsize=12, alpha=0.7)
                #ax.set_ylim(-5.9, 5.9)
                ax.set_ylim(-4.9, 4.9)
                ax.set_xlim(-6.5, 0.5)
                ax.xaxis.set_minor_locator(ticker.MultipleLocator(base=.5))
                ax.tick_params(width=1, length=2, axis='x', which='minor', bottom=True, top=True, direction='in')
                ax.yaxis.set_minor_locator(ticker.MultipleLocator(base=.5))
                ax.tick_params(width=1, length=2, axis='y', which='minor', left = True, right = True, direction='in')
                ax.xaxis.set_major_locator(ticker.MultipleLocator(base=2))
                ax.tick_params(width=1, length = 5, axis='x', which='major', bottom=True, top=True, direction='in')
                ax.yaxis.set_major_locator(ticker.MultipleLocator(base=2))
                ax.tick_params(width=1, length = 5, axis='y', which='major', left = True, right = True, direction='in')
            else:
                fig.delaxes(ax)
        for i in range(nrow):
            for j in range(ncol):
                if j != 0:
                    axs[i,j].set_yticklabels([])
                if i != nrow-1:
                    axs[i,j].set_xticklabels([])
        axs[nrow//2,0].set_ylabel('[X/Fe]', fontsize=15, loc='top')
        axs[nrow-1, ncol//2].set_xlabel(f'[Fe/H]', fontsize=15, loc='center')
        fig.tight_layout(rect=[0.0, 0, 1, .8])
        fig.subplots_adjust(wspace=0., hspace=0.)
        plt.show(block=False)
        plt.savefig(self._dir_out_figs + 'elem_obs_lZ.pdf', bbox_inches='tight')
        return None
    