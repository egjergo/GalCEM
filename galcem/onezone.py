# I only achieve simplicity with enormous effort (Clarice Lispector)
import time
import numpy as np
import scipy.integrate as integr
from scipy.interpolate import *
import os
import pickle

from .classes.morphology import Auxiliary,Stellar_Lifetimes,Infall,Star_Formation_Rate,Initial_Mass_Function
from .classes.yields import Isotopes,Yields_LIMs,Yields_SNII,Yields_SNIa,Yields_BBN,Concentrations
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
        self.lifetime_class = Stellar_Lifetimes(self.IN)
        
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
        self.yields_LIMs_class = Yields_LIMs(self.IN)
        self.yields_LIMs_class.import_yields()
        yields_SNII_class = Yields_SNII(self.IN)
        yields_SNII_class.import_yields()
        self.yields_SNIa_class = Yields_SNIa(self.IN)
        self.yields_SNIa_class.import_yields()
        yields_BBN_class = Yields_BBN(self.IN)
        yields_BBN_class.import_yields()
        
        # Initialize ZA_all
        self.c_class = Concentrations(self.IN)
        ZA_LIMs = self.c_class.extract_ZA_pairs_LIMs(self.yields_LIMs_class)
        ZA_SNIa = self.c_class.extract_ZA_pairs_SNIa(self.yields_SNIa_class)
        ZA_SNII = self.c_class.extract_ZA_pairs_SNII(yields_SNII_class)
        ZA_all = np.vstack((ZA_LIMs, ZA_SNIa, ZA_SNII))
        
        # Initialize Global tracked quantities
        self.Infall_rate = self.infall(self.time_chosen)
        self.ZA_sorted = self.c_class.ZA_sorted(ZA_all) # [Z, A] VERY IMPORTANT! 321 isotopes with yields_SNIa_option = 'km20', 192 isotopes for 'i99' 
        self.ZA_sorted = self.ZA_sorted[1:,:]
        self.ZA_symb_list = self.IN.periodic['elemSymb'][self.ZA_sorted[:,0]] # name of elements for all isotopes
        self.asplund3_percent = self.c_class.abund_percentage(self.ZA_sorted)
        #ZA_symb_iso_list = np.asarray([ str(A) for A in self.IN.periodic['elemA'][self.ZA_sorted]])  # name of elements for all isotopes
        self.elemZ_for_metallicity = np.where(self.ZA_sorted[:,0]>2)[0][0] #  starting idx (int) that excludes H and He for the metallicity selection
        self.Mtot = np.insert(np.cumsum((self.Infall_rate[1:] + self.Infall_rate[:-1]) * self.IN.nTimeStep / 2), 0, self.IN.epsilon) # The total baryonic mass (i.e. the infall mass) is computed right away
        #Mtot_quad = [quad(infall, self.time_chosen[0], i)[0] for i in range(1,len(self.time_chosen)-1)] # slow loop, deprecate!!!!!!!
        self.Mstar_v = self.IN.epsilon * np.ones(len(self.time_chosen)) 
        self.Mgas_v = self.IN.epsilon * np.ones(len(self.time_chosen)) 
        self.returned_Mass_v = self.IN.epsilon * np.ones(len(self.time_chosen)) 
        self.SFR_v = self.IN.epsilon * np.ones(len(self.time_chosen)) #
        self.f_SNIa_v = self.IN.epsilon * np.ones(len(self.time_chosen))
        self.Mass_i_v = self.IN.epsilon * np.ones((len(self.ZA_sorted), len(self.time_chosen)))    # Gass mass (i,j) where the i rows are the isotopes and j are the timesteps, [:,j] follows the timesteps
        self.W_i_comp = self.IN.epsilon * np.ones((len(self.ZA_sorted), len(self.time_chosen), 3), dtype=object)    # Gass mass (i,j) where the i rows are the isotopes and j are the timesteps, [:,j] follows the timesteps
        self.Xi_inf = isotope_class.construct_yield_vector(yields_BBN_class, self.ZA_sorted)
        Mass_i_inf = np.column_stack(([self.Xi_inf] * len(self.Mtot)))
        self.Xi_v = self.IN.epsilon * np.ones((len(self.ZA_sorted), len(self.time_chosen)))    # Xi 
        self.Z_v = self.IN.epsilon * np.ones(len(self.time_chosen)) # Metallicity 
        G_v = self.IN.epsilon * np.ones(len(self.time_chosen)) # G 
        S_v = self.IN.epsilon * np.ones(len(self.time_chosen)) # S = 1 - G 
        self.Rate_SNII = self.IN.epsilon * np.ones(len(self.time_chosen)) 
        self.Rate_LIMs = self.IN.epsilon * np.ones(len(self.time_chosen)) 
        self.Rate_SNIa = self.IN.epsilon * np.ones(len(self.time_chosen)) 
        self.Rate_NSM = self.IN.epsilon * np.ones(len(self.time_chosen)) 
        
        # Load Interpolation Models
        self._dir = os.path.dirname(__file__)
        self.X_lc18, self.Y_lc18, self.models_lc18, self.averaged_lc18 = self.load_processed_yields(func_name=self.IN.yields_SNII_option, loc=self._dir + '/input/yields/snii/'+ self.IN.yields_SNII_option + '/tab_R', df_list=['X', 'Y', 'models', 'avgmassfrac'])
        self.X_k10, self.Y_k10, self.models_k10, self.averaged_k10 = self.load_processed_yields(func_name=self.IN.yields_LIMs_option, loc=self._dir + '/input/yields/lims/' + self.IN.yields_LIMs_option, df_list=['X', 'Y', 'models', 'avgmassfrac'])
        self.Y_snia = self.load_processed_yields_snia(func_name=self.IN.yields_SNIa_option, loc=self._dir + '/input/yields/snia/' + self.IN.yields_SNIa_option, df_list='Y')
        
    def load_processed_yields(self,func_name, loc, df_list):
        df_dict = {}
        for df_l in df_list:
            with open('%s/processed/%s.pkl'%(loc,df_l), 'rb') as pickle_file:
                df_dict[df_l] = pickle.load(pickle_file)
        return [df_dict[d] for d in df_list]#df_dict[df_list[0]], df_dict[df_list[1]]#, df_dict[df_list[2]]

    def load_processed_yields_snia(self, func_name, loc, df_list):#, 'models']):
        df_dict = {}
        for df_l in df_list:
            with open('%s/processed/%s.pkl'%(loc,df_l), 'rb') as pickle_file:
                df_dict[df_l] = pickle.load(pickle_file)
        return df_dict[df_list[0]]
   
        
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
        import pandas as pd
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
        np.savetxt(self._dir_out + 'phys.dat', np.column_stack((self.time_chosen, self.Mtot, self.Mgas_v, self.Mstar_v, self.SFR_v/1e9, self.Infall_rate/1e9, self.Z_v, G_v, S_v, self.Rate_SNII, self.Rate_SNIa, self.Rate_LIMs, self.f_SNIa_v)), fmt='%-12.4e', #SFR is divided by 1e9 to get the /Gyr to /yr conversion 
                header = ' (0) time_chosen [Gyr]    (1) Mtot [Msun]    (2) Mgas_v [Msun]    (3) Mstar_v [Msun]    (4) SFR_v [Msun/yr]    (5)Infall_v [Msun/yr]    (6) Z_v    (7) G_v    (8) S_v     (9) Rate_SNII     (10) Rate_SNIa     (11) Rate_LIMs     (12) DTD SNIa')
        np.savetxt(self._dir_out +  'Mass_i.dat', np.column_stack((self.ZA_sorted, self.Mass_i_v)), fmt=' '.join(['%5.i']*2 + ['%12.4e']*self.Mass_i_v[0,:].shape[0]),
                header = ' (0) elemZ,    (1) elemA,    (2) masses [Msun] of every isotope for every timestep')
        np.savetxt(self._dir_out +  'X_i.dat', np.column_stack((self.ZA_sorted, self.Xi_v)), fmt=' '.join(['%5.i']*2 + ['%12.4e']*self.Xi_v[0,:].shape[0]),
                header = ' (0) elemZ,    (1) elemA,    (2) abundance mass ratios of every isotope for every timestep (normalized to solar, Asplund et al., 2009)')
        pickle.dump(self.W_i_comp,open(self._dir_out + 'W_i_comp.pkl','wb'))
        self.aux.tic_count(string="Output saved in", tic=self.tic)
        self.file1.close()
    
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
        Wi_comps = kwargs['Wi_comp'] # [list of 2-array lists] #Wi_comps[0][0].shape=(155,)
        Wi_SNIa = kwargs['Wi_SNIa']
        i = kwargs['i']
        yields = kwargs['yields']
        channel_switch = ['SNII', 'LIMs']
        infall_comp = self.Infall_rate[n] * self.Xi_inf[i]
        sfr_comp = self.SFR_v[n] * self.Xi_v[i,n] 
        if n <= 0:
            val = infall_comp - sfr_comp
        else:
            Wi_vals = []
            for j,val in enumerate(Wi_comps):
                if len(val[1]) > 0.:
                    Wi_vals.append(integr.simps(np.multiply(val[0], yields[j]), x=val[1]))
                    #Wi_vals.append(integr.simps(val[0] * yield_interp[j](mass_grid, metallicity_func[birthtime_grid]), x=val[1]))   
                else:
                    Wi_vals.append(0.)
            returned = [Wi_vals[0], Wi_vals[1], Wi_SNIa]
            self.W_i_comp[i,n,0] = returned[0] 
            self.W_i_comp[i,n,1] = returned[1] 
            self.W_i_comp[i,n,2] = returned[2]
            val = infall_comp - sfr_comp + np.sum(returned)
        return val

    def Mgas_func(self, t_n, y_n, n, i=None):
        # Explicit general diff eq GCE function
        # Mgas(t)
        #print(f'{self.SFR_tn(n)==self.SFR_v[n]=}')
        return self.Infall_rate[n] + np.sum(self.W_i_comp[:,n,:]) - self.SFR_tn(n) #* np.sum(self.Xi_v[:,n])
    
    def Mstar_func(self, t_n, y_n, n, i=None):
        # Mstar(t)
        return - np.sum(self.W_i_comp[:,n,:]) + self.SFR_tn(n) #* np.sum(self.Xi_v[:,n])

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

    def evolve(self):
        '''Evolution routine'''
        self.Mass_i_v[:,1] = np.multiply(self.Mtot[1], self.Xi_inf)
        for n in range(len(self.time_chosen[:self.idx_age_Galaxy])):
            print('time [Gyr] = %.2f'%self.time_chosen[n])
            self.file1.write('n = %d\n'%n)
            self.phys_integral(n)        
            self.Xi_v[:, n] = np.divide(self.Mass_i_v[:,n], self.Mgas_v[n])
            self.file1.write(' sum X_i at n %d= %.3f\n'%(n, np.sum(self.Xi_v[:,n])))
            if n > 0.: 
                Wi_class = Wi(n, self.IN, self.lifetime_class, self.time_chosen, self.Z_v, self.SFR_v, self.f_SNIa_v,
                              self.IMF, self.yields_SNIa_class, self.models_lc18, self.models_k10, self.ZA_sorted)
                self.Rate_SNII[n], self.Rate_LIMs[n], self.Rate_SNIa[n] = Wi_class.compute_rates()
                Wi_comp = [Wi_class.compute("SNII"), Wi_class.compute("LIMs")]
                #yield_SNII = Wi_class.yield_array('SNII', Wi_class.SNII_mass_grid, Wi_class.SNII_birthtime_grid)
                #yield_LIMs = Wi_class.yield_array('LIMs', Wi_class.LIMs_mass_grid, Wi_class.LIMs_birthtime_grid)
                for i, _ in enumerate(self.ZA_sorted): 
                    Wi_SNIa = self.Rate_SNIa[n] * self.Y_snia[i]
                    if self.X_lc18[i].empty:
                        yields_lc18 = 0.
                    else:
                        idx_SNII = np.digitize(self.Z_v[n-1] - self.IN.solar_metallicity, self.averaged_lc18[i][1])
                        yields_lc18 = self.averaged_lc18[i][0][idx_SNII]
                    if self.X_k10[i].empty:
                        yields_k10 = 0.
                    else:
                        idx_LIMs = np.digitize(self.Z_v[n-1] - self.IN.solar_metallicity, self.averaged_k10[i][1])
                        yields_k10 = self.averaged_k10[i][0][idx_LIMs]
                    yields = [yields_lc18, yields_k10]
                    self.Mass_i_v[i, n+1] = self.aux.RK4(self.solve_integral, self.time_chosen[n], self.Mass_i_v[i,n], n, self.IN.nTimeStep, i=i, Wi_comp=Wi_comp, Wi_SNIa=Wi_SNIa, yields=yields)
            #self.Z_v[n] = np.divide(np.sum(self.Mass_i_v[self.elemZ_for_metallicity:,n]), self.Mgas_v[n])
            self.Xi_v[:, n] = np.divide(self.Mass_i_v[:,n], self.Mgas_v[n])
            self.Z_v[n] = np.divide(np.sum(self.Mass_i_v[:,n]), self.Mgas_v[n])
        self.Z_v[-1] = np.divide(np.sum(self.Mass_i_v[:,-1]), self.Mgas_v[-1])
        self.Xi_v[:,-1] = np.divide(self.Mass_i_v[:,-1], self.Mgas_v[-1]) 


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
        self.FeH_evolution()
        #self.DTD_plot()
        self.iso_abundance()
        #self.iso_evolution()
        self.iso_evolution_comp()
        self.observational()
        self.observational_lelemZ()
        self.obs_lelemZ()
        self.phys_integral_plot()
        self.phys_integral_plot(logAge=True)
        self.lifetimeratio_test_plot()
        self.ZA_sorted_plot()
        # self.elem_abundance() # compares and requires multiple runs (IMF & SFR variations)
        self.aux.tic_count(string="Plots saved in", tic=self.tic)
        
    def ZA_sorted_plot(self, cmap_name='magma_r', cbins=10): # angle = 2 * np.pi / np.arctan(0.4) !!!!!!!
        print('Starting ZA_sorted_plot()')
        from matplotlib import cm,pyplot as plt
        plt.style.use(self._dir+'/galcem.mplstyle')
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
        ax.tick_params(width = 2, length = 10)
        ax.tick_params(width = 1, length = 5, which = 'minor')
        ax.set_xlim(np.min(x)-2.5, np.max(x)+2.5)
        ax.set_ylim(np.min(y)-2.5, np.max(y)+2.5)
        plt.tight_layout()
        plt.show(block=False)
        plt.savefig(self._dir_out_figs + 'tracked_elements.pdf')

    def DTD_plot(self):
        print('Starting DTD_plot()')
        from matplotlib import pyplot as plt
        plt.style.use(self._dir+'/galcem.mplstyle')
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
        plt.savefig(self._dir_out_figs + 'DTD.pdf')
        
    def lifetimeratio_test_plot(self,colormap='Paired'):
        print('Starting lifetimeratio_test_plot()')
        from matplotlib import pyplot as plt
        plt.style.use(self._dir+'/galcem.mplstyle')
        fig, ax = plt.subplots(1,1, figsize=(7,5))
        divid05 = np.divide(self.IN.s_lifetimes_p98['Z05'], self.IN.s_lifetimes_p98['Z0004'])
        divid02 = np.divide(self.IN.s_lifetimes_p98['Z02'], self.IN.s_lifetimes_p98['Z0004'])
        divid008 = np.divide(self.IN.s_lifetimes_p98['Z008'], self.IN.s_lifetimes_p98['Z0004'])
        ax.semilogx(self.IN.s_lifetimes_p98['M'], divid05, color='blue', label='Z = 0.05')
        ax.semilogx(self.IN.s_lifetimes_p98['M'], divid02, color='blue', linestyle='--', label='Z = 0.02')
        ax.semilogx(self.IN.s_lifetimes_p98['M'], divid008, color='blue', linestyle=':', label='Z = 0.008')
        ax.hlines(1, 0.6,120, color='orange', label='ratio=1')
        ax.vlines(3, 0.6,2.6, color='red', label=r'$3 M_{\odot}$')
        ax.vlines(6, 0.6,2.6, color='red', alpha=0.6, linestyle='--', label=r'$6 M_{\odot}$')
        ax.vlines(9, 0.6,2.6, color='red', alpha=0.3, linestyle = ':', label=r'$9 M_{\odot}$')
        cm = plt.cm.get_cmap(colormap)
        sc=ax.scatter(self.IN.s_lifetimes_p98['M'], divid05, c=np.log10(self.IN.s_lifetimes_p98['Z05']), cmap=cm, s=50)
        sc=ax.scatter(self.IN.s_lifetimes_p98['M'], divid02, c=np.log10(self.IN.s_lifetimes_p98['Z05']), cmap=cm, s=50)
        sc=ax.scatter(self.IN.s_lifetimes_p98['M'], divid008, c=np.log10(self.IN.s_lifetimes_p98['Z05']), cmap=cm, s=50)
        fig.colorbar(sc, label=r'$\tau(M_*)$')
        ax.legend(loc='best', frameon=False, fontsize=13)
        ax.set_ylabel(r'$\tau(X)/\tau(Z=0.0004)$', fontsize=15)
        ax.set_xlabel('Mass', fontsize=15)
        ax.set_ylim(0.6,1.95)
        ax.set_xlim(0.6, 120)
        fig.tight_layout()
        plt.savefig(self._dir_out_figs + 'tauratio.pdf')
     
    def phys_integral_plot(self, logAge=False):
        print('Starting phys_integral_plot()')
        # Requires running "phys_integral()" in onezone.py beforehand
        from matplotlib import pyplot as plt
        import matplotlib.ticker as ticker
        plt.style.use(self._dir+'/galcem.mplstyle')
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
        fig, axs = plt.subplots(1, 2, figsize=(12,7))
        axt = axs[1].twinx()
        time_plot = time_chosen
        xscale = '_lin'
        MW_SFR_xcoord = 13.7
        axs[0].hlines(self.IN.M_inf, 0, self.IN.age_Galaxy, label=r'$M_{gal,f}$', linewidth = 1, linestyle = '-.', color='#8c00ff')
        axt.vlines(MW_SFR_xcoord, self.IN.MW_SFR-.4, self.IN.MW_SFR+0.4, label=r'SFR$_{MW}$ CP11', linewidth = 6, linestyle = '-', color='#ff8c00', alpha=0.8)
        axt.vlines(MW_SFR_xcoord, self.IN.MW_RSNII[2], self.IN.MW_RSNII[1], label=r'R$_{SNII,MW}$ M05', linewidth = 6, linestyle = '-', color='#0034ff', alpha=0.8)
        axt.vlines(MW_SFR_xcoord, self.IN.MW_RSNIa[2], self.IN.MW_RSNIa[1], label=r'R$_{SNIa,MW}$ M05', linewidth = 6, linestyle = '-', color='#00b3ff', alpha=0.8)
        axs[0].semilogy(time_plot, Mstar_v, label= r'$M_{star}$', linewidth=3, color='#ff8c00')
        axs[0].semilogy(time_plot, Mgas_v, label= r'$M_{gas}$', linewidth=3, color='#0d00ff')
        axs[0].semilogy(time_plot, np.sum(Mass_i[:,2:], axis=0), label = r'$M_{g,tot,i}$', linewidth=2, linestyle=':', color='#00b3ff')
        axs[0].semilogy(time_plot, np.sum(Mass_i[:2,2:], axis=0), label = r'$M_{H,g}$', linewidth=1, linestyle='-.', color='#0033ff')
        axs[0].semilogy(time_plot, np.sum(Mass_i[4:,2:], axis=0), label = r'$M_{Z,g}$', linewidth=2, linestyle=':', color='#ff0073')
        axs[0].semilogy(time_plot, Mtot, label=r'$M_{tot}$', linewidth=4, color='black')
        axs[0].semilogy(time_plot, Mstar_v + Mgas_v, label= r'$M_g + M_s$', linewidth = 3, linestyle = '--', color='#a9a9a9')
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
            axs[0].tick_params(width = 1, length = 10, axis = 'x', which = 'minor', bottom = True, top = True, direction = 'in')
            axs[0].xaxis.set_major_locator(ticker.MultipleLocator(base=5))
            axs[0].tick_params(width = 2, length = 15, axis = 'x', which = 'major', bottom = True, top = True, direction = 'in')
            axs[1].xaxis.set_minor_locator(ticker.MultipleLocator(base=1))
            axs[1].tick_params(width = 1, length = 10, axis = 'x', which = 'minor', bottom = True, top = True, direction = 'in')
            axs[1].xaxis.set_major_locator(ticker.MultipleLocator(base=5))
            axs[1].tick_params(width = 2, length = 15, axis = 'x', which = 'major', bottom = True, top = True, direction = 'in')
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
        plt.savefig(self._dir_out_figs + 'total_physical'+str(xscale)+'.pdf')
        
    def FeH_evolution(self, c=2, logAge=False):
        print('Starting FeH_evolution()')
        from matplotlib import pyplot as plt
        import pandas as pd
        plt.style.use(self._dir+'/galcem.mplstyle')
        Z_list = np.unique(self.ZA_sorted[:,0])
        phys = np.loadtxt(self._dir_out + 'phys.dat')
        time = phys[c:,0]
        observ = pd.read_table(self._dir + '/input/observations/meusinger91.txt', sep=',')
        solar_norm_H = self.c_class.solarA09_vs_H_bymass[Z_list]
        solar_norm_Fe = self.c_class.solarA09_vs_Fe_bymass[Z_list]
        Mass_i = np.loadtxt(self._dir_out + 'Mass_i.dat')
        #Fe = np.sum(Mass_i[np.intersect1d(np.where(ZA_sorted[:,0]==26)[0], np.where(ZA_sorted[:,1]==56)[0]), c+2:], axis=0)
        Fe = np.sum(Mass_i[self.select_elemZ_idx(26), c+2:], axis=0)
        H = np.sum(Mass_i[self.select_elemZ_idx(1), c+2:], axis=0)
        FeH = np.log10(np.divide(Fe, H)) - solar_norm_H[26]
        fig, ax = plt.subplots(1,1, figsize=(7,5))
        ax.plot(time, FeH, color='black', label='SNIa', linewidth=3)
        ax.errorbar(self.IN.age_Galaxy - observ['age'], observ['FeH'], yerr=observ['FeHerr'], marker='s', label='Meusinger+91', mfc='gray', ecolor='gray', ls='none')
        ax.legend(loc='lower right', frameon=False, fontsize=17)
        ax.set_ylabel(r'[Fe/H]', fontsize=20)
        ax.set_xlabel('Age [Gyr]', fontsize=20)
        ax.set_ylim(-2,0.5)
        xscale = '_lin'
        if not logAge:
            ax.set_xlim(0,13.8)
        else:
            ax.set_xscale('log')
            xscale = '_log'
        #ax.set_xlim(1e-2, 1.9e1)
        fig.tight_layout()
        plt.savefig(self._dir_out_figs + 'FeH_evolution'+str(xscale)+'.pdf')

    def iso_evolution(self, figsize=(40,13)):
        print('Starting iso_evolution()')
        from matplotlib import pyplot as plt
        plt.style.use(self._dir+'/galcem.mplstyle')
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
                ax.tick_params(width = 1, length = 2, axis = 'x', which = 'minor', bottom = True, top = True, direction = 'in')
                ax.yaxis.set_minor_locator(ticker.MultipleLocator(base=1))
                ax.tick_params(width = 1, length = 2, axis = 'y', which = 'minor', left = True, right = True, direction = 'in')
                ax.xaxis.set_major_locator(ticker.MultipleLocator(base=5))
                ax.tick_params(width = 1, length = 5, axis = 'x', which = 'major', bottom = True, top = True, direction = 'in')
                ax.yaxis.set_major_locator(ticker.MultipleLocator(base=5))
                ax.tick_params(width = 1, length = 5, axis = 'y', which = 'major', left = True, right = True, direction = 'in')
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
        plt.savefig(self._dir_out_figs + 'iso_evolution.pdf')

    def iso_evolution_comp(self, figsize=(40,13)):
        print('Starting iso_evolution_comp()')
        from matplotlib import pyplot as plt
        plt.style.use(self._dir+'/galcem.mplstyle')
        import matplotlib.ticker as ticker
        Mass_i = np.loadtxt(self._dir_out + 'Mass_i.dat')
        Masses = np.log10(Mass_i[:,2:])
        phys = np.loadtxt(self._dir_out + 'phys.dat')
        W_i_comp = pickle.load(open(self._dir_out + 'W_i_comp.pkl','rb'))
        #W_i_comp +=  100.
        W_i_comp = np.asarray(W_i_comp, dtype=float)
        W_i_comp = np.log10(W_i_comp)
        Mass_SNII = W_i_comp[:,:,0]
        Mass_AGB = W_i_comp[:,:,1]
        Mass_SNIa = W_i_comp[:,:,2]
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
                ax.plot(timex, Mass_SNII[i], color='#0034ff', linestyle='-.', linewidth=3, alpha=0.8, label='SNII')
                ax.plot(timex, Mass_AGB[i], color='#ff00b3', linestyle='--', linewidth=3, alpha=0.8, label='LIMs')
                ax.plot(timex, Mass_SNIa[i], color='#00b3ff', linestyle=':', linewidth=3, alpha=0.8, label='SNIa')
                ax.annotate('%s(%d,%d)'%(self.ZA_symb_list.values[i],Z[i],A[i]), xy=(0.5, 0.92), xycoords='axes fraction', horizontalalignment='center', verticalalignment='top', fontsize=12, alpha=0.7)
                ax.set_ylim(-4.9, 9.9)
                ax.set_xlim(0.01,13.8)
                ax.xaxis.set_minor_locator(ticker.MultipleLocator(base=1))
                ax.tick_params(width = 1, length = 2, axis = 'x', which = 'minor', bottom = True, top = True, direction = 'in')
                ax.yaxis.set_minor_locator(ticker.MultipleLocator(base=1))
                ax.tick_params(width = 1, length = 2, axis = 'y', which = 'minor', left = True, right = True, direction = 'in')
                ax.xaxis.set_major_locator(ticker.MultipleLocator(base=5))
                ax.tick_params(width = 1, length = 5, axis = 'x', which = 'major', bottom = True, top = True, direction = 'in')
                ax.yaxis.set_major_locator(ticker.MultipleLocator(base=5))
                ax.tick_params(width = 1, length = 5, axis = 'y', which = 'major', left = True, right = True, direction = 'in')
            else:
                fig.delaxes(ax)
        for i in range(nrow):
            for j in range(ncol):
                if j != 0:
                    axs[i,j].set_yticklabels([])
                if i != nrow-1:
                    axs[i,j].set_xticklabels([])
                    
        axs[nrow//2,0].set_ylabel(r'Masses [ $\log_{10}$($M_{\odot}$/yr)]', fontsize = 15)
        axs[nrow-1, ncol//2].set_xlabel('Age [Gyr]', fontsize = 15)
        axs[0, ncol//2].legend(ncol=3, loc='upper center', bbox_to_anchor=(0.5, 1.5), frameon=False, fontsize=15)
        plt.tight_layout(rect = [0.03, 0, 1, .98])
        plt.subplots_adjust(wspace=0., hspace=0.)
        plt.show(block=False)
        plt.savefig(self._dir_out_figs + 'iso_evolution_comp.pdf')

    def iso_abundance(self, figsize=(40,13), c=3): 
        print('Starting iso_abundance()')
        from matplotlib import pyplot as plt
        plt.style.use(self._dir+'/galcem.mplstyle')
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
                ax.tick_params(width = 1, length = 2, axis = 'x', which = 'minor', bottom = True, top = True, direction = 'in')
                ax.yaxis.set_minor_locator(ticker.MultipleLocator(base=1))
                ax.tick_params(width = 1, length = 2, axis = 'y', which = 'minor', left = True, right = True, direction = 'in')
                ax.xaxis.set_major_locator(ticker.MultipleLocator(base=5))
                ax.tick_params(width = 1, length = 5, axis = 'x', which = 'major', bottom = True, top = True, direction = 'in')
                ax.yaxis.set_major_locator(ticker.MultipleLocator(base=5))
                ax.tick_params(width = 1, length = 5, axis = 'y', which = 'major', left = True, right = True, direction = 'in')
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
        plt.savefig(self._dir_out_figs + 'iso_abundance.pdf')
        
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
        plt.style.use(self._dir+'/galcem.mplstyle')
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
                ax.tick_params(width = 1, length = 2, axis = 'x', which = 'minor', bottom = True, top = True, direction = 'in')
                ax.yaxis.set_minor_locator(ticker.MultipleLocator(base=1))
                ax.tick_params(width = 1, length = 2, axis = 'y', which = 'minor', left = True, right = True, direction = 'in')
                ax.xaxis.set_major_locator(ticker.MultipleLocator(base=5))
                ax.tick_params(width = 1, length = 5, axis = 'x', which = 'major', bottom = True, top = True, direction = 'in')
                ax.yaxis.set_major_locator(ticker.MultipleLocator(base=5))
                ax.tick_params(width = 1, length = 5, axis = 'y', which = 'major', left = True, right = True, direction = 'in')
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
        plt.savefig(self._dir_out_figs + 'elem_abundance.pdf')
    
    def select_elemZ_idx(self, elemZ):
        ''' auxiliary function that selects the isotope indexes where Z=elemZ '''
        return np.where(self.ZA_sorted[:,0]==elemZ)[0]
   
    def observational(self, figsiz = (32,10), c=3):
        print('Starting observational()')
        import glob
        import itertools
        import pandas as pd
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

        li = []
        elemZmin = 12
        elemZmax = 12

        for filename in all_files:
            df = pd.read_table(filename, sep=',')
            elemZmin0 = np.min(df.iloc[:,0])
            elemZmax0 = np.max(df.iloc[:,0])
            elemZmin = np.min([elemZmin0, elemZmin])
            elemZmax = np.max([elemZmax0, elemZmax])
            li.append(df)

        markerlist = itertools.cycle(('o', 'v', '^', '<', '>', 'P', '*', 'd', 'X'))
        colorlist = itertools.cycle(('#00ccff', '#ffb300', '#ff004d'))
        colorlist_websafe = itertools.cycle(('#ff3399', '#5d8aa8', '#e32636', '#ffbf00', '#9966cc', '#a4c639',
                                             '#cd9575', '#008000', '#fbceb1', '#00ffff', '#4b5320', '#a52a2a',
                                             '#007fff', '#ff2052', '#21abcd', '#e97451', '#592720', '#fad6a5',
                                             '#36454f', '#e4d00a', '#ff3800', '#ffbcd9', '#008b8b', '#8b008b',
                                             '#03c03c', '#00009c', '#ccff00', '#673147', '#0f0f0f', '#324ab2'))
        lenlist = len(li)
    
        for i, ax in enumerate(axs.flat):
            for j, ll in enumerate(li):
                idx_obs = np.where(ll.iloc[:,0] == i+1)[0]
                ax.scatter(ll.iloc[idx_obs,1], ll.iloc[idx_obs,2], label=ll.iloc[idx_obs, 4], alpha=0.1, marker=next(markerlist), color=next(colorlist_websafe))
            if i < len(Z_list):
                ax.plot(FeH, Masses2[i], color='black', linewidth=2)
                ax.annotate(f"{Z_list[i]}{Z_symb_list[Z_list[i]]}", xy=(0.5, 0.92), xycoords='axes fraction', horizontalalignment='center', verticalalignment='top', fontsize=12, alpha=0.7)
                ax.set_ylim(-5.9, 5.9)
                #ax.set_ylim(-1.5, 1.5)
                #ax.set_xlim(-11, -2)
                #ax.set_xlim(-6.5, 0.5)
                ax.set_xlim(-6.5, 0.5)
                ax.xaxis.set_minor_locator(ticker.MultipleLocator(base=.5))
                ax.tick_params(width = 1, length = 2, axis = 'x', which = 'minor', bottom = True, top = True, direction = 'in')
                ax.yaxis.set_minor_locator(ticker.MultipleLocator(base=.5))
                ax.tick_params(width = 1, length = 2, axis = 'y', which = 'minor', left = True, right = True, direction = 'in')
                ax.xaxis.set_major_locator(ticker.MultipleLocator(base=2))
                ax.tick_params(width = 1, length = 5, axis = 'x', which = 'major', bottom = True, top = True, direction = 'in')
                ax.yaxis.set_major_locator(ticker.MultipleLocator(base=3))
                ax.tick_params(width = 1, length = 5, axis = 'y', which = 'major', left = True, right = True, direction = 'in')
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
        fig.tight_layout(rect = [0.03, 0, 1, 1])
        fig.subplots_adjust(wspace=0., hspace=0.)
        plt.show(block=False)
        plt.savefig(self._dir_out_figs + 'elem_obs.pdf')
        return None

    def observational_lelemZ(self, figsiz = (32,10), c=3):
        print('Starting observational_lelemZ()')
        import glob
        import itertools
        import pandas as pd
        from matplotlib import pyplot as plt
        import matplotlib.ticker as ticker
        plt.style.use(self._dir+'/galcem.mplstyle')
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

        li = []
        elemZmin = 12
        elemZmax = 12

        for filename in all_files:
            df = pd.read_table(filename, sep=',')
            elemZmin0 = np.min(df.iloc[:,0])
            elemZmax0 = np.max(df.iloc[:,0])
            elemZmin = np.min([elemZmin0, elemZmin])
            elemZmax = np.max([elemZmax0, elemZmax])
            li.append(df)

        markerlist =itertools.cycle(('o', 'v', '^', '<', '>', 'P', '*', 'd', 'X'))
        #colorlist = itertools.cycle(('#00ccff', '#ffb300', '#ff004d', '#003662', '#620005', '#366200'))
        colorlist = itertools.cycle(('#00ccff', '#ffb300', '#ff004d'))
        colorlist_websafe = itertools.cycle(('#ff3399', '#5d8aa8', '#e32636', '#ffbf00', '#9966cc', '#a4c639',
                                             '#cd9575', '#008000', '#fbceb1', '#00ffff', '#4b5320', '#a52a2a',
                                             '#007fff', '#ff2052', '#21abcd', '#e97451', '#592720', '#fad6a5',
                                             '#36454f', '#e4d00a', '#ff3800', '#ffbcd9', '#008b8b', '#8b008b',
                                             '#03c03c', '#00009c', '#ccff00', '#673147', '#0f0f0f', '#324ab2'))
        lenlist = len(li)

        for i, ax in enumerate(axs.flat):
            for j, ll in enumerate(li):
                idx_obs = np.where(ll.iloc[:,0] == i+1)[0]
                ax.scatter(ll.iloc[idx_obs,1], ll.iloc[idx_obs,2], label=ll.iloc[idx_obs, 4], alpha=0.1, marker=next(markerlist), color=next(colorlist_websafe))
            if i < nrow*ncol:
                #ax.plot(FeH, Masses[i], color='blue')
                ax.plot(FeH, Masses2[i], color='black', linewidth=2)
                ax.annotate(f"{Z_list[i]}{Z_symb_list[Z_list[i]]}", xy=(0.5, 0.92), xycoords='axes fraction', horizontalalignment='center', verticalalignment='top', fontsize=12, alpha=0.7)
                #ax.set_ylim(-6, 6)
                ax.set_ylim(-1.5, 1.5)
                #ax.set_xlim(-11, -2)
                #ax.set_xlim(-6.5, 0.5)
                ax.set_xlim(-6.5, 0.5)
                ax.xaxis.set_minor_locator(ticker.MultipleLocator(base=.5))
                ax.tick_params(width = 1, length = 2, axis = 'x', which = 'minor', bottom = True, top = True, direction = 'in')
                ax.yaxis.set_minor_locator(ticker.MultipleLocator(base=.2))
                ax.tick_params(width = 1, length = 2, axis = 'y', which = 'minor', left = True, right = True, direction = 'in')
                ax.xaxis.set_major_locator(ticker.MultipleLocator(base=2))
                ax.tick_params(width = 1, length = 5, axis = 'x', which = 'major', bottom = True, top = True, direction = 'in')
                ax.yaxis.set_major_locator(ticker.MultipleLocator(base=1))
                ax.tick_params(width = 1, length = 5, axis = 'y', which = 'major', left = True, right = True, direction = 'in')
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
        fig.tight_layout(rect = [0.03, 0, 1, 1])
        fig.subplots_adjust(wspace=0., hspace=0.)
        plt.show(block=False)
        plt.savefig(self._dir_out_figs + 'elem_obs_lelemZ.pdf')
        return None
    
    def obs_lelemZ(self, figsiz = (32,6), c=3):
        print('Starting observational_lelemZ()')
        import glob
        import itertools
        import pandas as pd
        from matplotlib import pyplot as plt
        import matplotlib.ticker as ticker
        plt.style.use(self._dir+'/galcem.mplstyle')
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

        li = []
        elemZmin = 12
        elemZmax = 12

        for filename in all_files:
            df = pd.read_table(filename, sep=',')
            elemZmin0 = np.min(df.iloc[:,0])
            elemZmax0 = np.max(df.iloc[:,0])
            elemZmin = np.min([elemZmin0, elemZmin])
            elemZmax = np.max([elemZmax0, elemZmax])
            li.append(df)

        markerlist =itertools.cycle(('o', 'v', '^', '<', '>', 'P', '*', 'd', 'X'))
        #colorlist = itertools.cycle(('#00ccff', '#ffb300', '#ff004d', '#003662', '#620005', '#366200'))
        colorlist = itertools.cycle(('#00ccff', '#ffb300', '#ff004d'))
        colorlist_websafe = itertools.cycle(('#ff3399', '#5d8aa8', '#e32636', '#ffbf00', '#9966cc', '#a4c639',
                                             '#cd9575', '#008000', '#fbceb1', '#00ffff', '#4b5320', '#a52a2a',
                                             '#007fff', '#ff2052', '#21abcd', '#e97451', '#592720', '#fad6a5',
                                             '#36454f', '#e4d00a', '#ff3800', '#ffbcd9', '#008b8b', '#8b008b',
                                             '#03c03c', '#00009c', '#ccff00', '#673147', '#0f0f0f', '#324ab2'))
        lenlist = len(li)

        for i, ax in enumerate(axs.flat):
            for j, ll in enumerate(li):
                idx_obs = np.where(ll.iloc[:,0] == i+1)[0]
                ax.scatter(ll.iloc[idx_obs,1], ll.iloc[idx_obs,2], label=ll.iloc[idx_obs, 4], alpha=0.1, marker=next(markerlist), color=next(colorlist_websafe))
            if i < nrow*ncol:
                #ax.plot(FeH, Masses[i], color='blue')
                ax.plot(FeH, Masses2[i], color='black', linewidth=2)
                ax.annotate(f"{Z_list[i]}{Z_symb_list[Z_list[i]]}", xy=(0.5, 0.92), xycoords='axes fraction', horizontalalignment='center', verticalalignment='top', fontsize=12, alpha=0.7)
                ax.set_ylim(-5.9, 5.9)
                ax.set_xlim(-6.5, 0.5)
                ax.xaxis.set_minor_locator(ticker.MultipleLocator(base=.5))
                ax.tick_params(width = 1, length = 2, axis = 'x', which = 'minor', bottom = True, top = True, direction = 'in')
                ax.yaxis.set_minor_locator(ticker.MultipleLocator(base=.5))
                ax.tick_params(width = 1, length = 2, axis = 'y', which = 'minor', left = True, right = True, direction = 'in')
                ax.xaxis.set_major_locator(ticker.MultipleLocator(base=2))
                ax.tick_params(width = 1, length = 5, axis = 'x', which = 'major', bottom = True, top = True, direction = 'in')
                ax.yaxis.set_major_locator(ticker.MultipleLocator(base=3))
                ax.tick_params(width = 1, length = 5, axis = 'y', which = 'major', left = True, right = True, direction = 'in')
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
        fig.tight_layout(rect = [0.03, 0, 1, 1])
        fig.subplots_adjust(wspace=0., hspace=0.)
        plt.show(block=False)
        plt.savefig(self._dir_out_figs + 'elem_obs_lZ.pdf')
        return None
    