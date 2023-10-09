""""""""""""""""""""""""""""""""""""""""""""""""
"                                              "
"       PLOT CLASS FOR SINGLE-ZONE RUNS        "
"  Contains the plot class to be paired with   " 
"        the Setup class of onezone.py         "
"                                              "
" LIST OF CLASSES:                             "
"    __        Plots (subclass)                "
"                                              "
""""""""""""""""""""""""""""""""""""""""""""""""

import pickle
import time
import numpy as np
import pandas as pd
from .gcsetup import Setup
from .classes.inputs import Auxiliary
import warnings
warnings.filterwarnings("ignore")
np.seterr(divide='ignore') 

class Plots(Setup):
    """
    PLOTTING
    """    
    def __init__(self, outdir = 'runs/mygcrun/'):
        self.tic = []
        self.tic.append(time.process_time())
        self.IN = pickle.load(open(outdir + 'inputs.pkl','rb'))
        super().__init__(self.IN, outdir=outdir)
        self.tic.append(time.process_time())
        package_loading_time = self.tic[-1]
        print('Lodaded the plotting class in %.1e seconds.'%package_loading_time)  
    
    def __repr__(self):
        aux = Auxiliary()
        return aux.repr(self) 
        
    def plots(self):
        self.tic.append(time.process_time())
        print('Starting to plot')
        self.FeH_evolution_plot(logAge=True)
        self.Z_evolution_plot(logAge=True)
        self.FeH_evolution_plot(logAge=False)
        self.Z_evolution_plot(logAge=False)
        self.total_evolution_plot(logAge=False)
        self.total_evolution_plot(logAge=True)
        self.lifetimeratio_test_plot()
        self.tracked_elements_3D_plot()
        self.observational_plot()
        self.observational_lelemZ_plot()
        self.obs_lZ_plot()
        self.iso_evolution_comp_plot(logAge=False)
        self.iso_evolution_comp_plot(logAge=True)
        self.iso_evolution_comp_lelemz_plot()
        self.obs_table()
        #self.ind_evolution_plot()
        self.DTD_plot()
        ## self.elem_abundance() # compares and requires multiple runs (IMF & SFR variations)
        self.aux.tic_count(string="Plots saved in", tic=self.tic)
      
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
        ax3 = fig.add_axes([.01, .42, .45, .45], projection='3d')
        #ax3.azim=-90
        ax3.bar3d(x, y, 0, 1, 1, z, color=percent_colors, zsort='average')
        ax3.tick_params(axis='x', labelsize= 8, labeltop=True, labelbottom=False)
        ax3.tick_params(axis='y', labelsize= 8, labelright=True, labelbottom=False, labelleft=False)
        ax3.tick_params(axis='z', labelsize= 8)
        ax3.set_zlabel(r'$\odot$ isotopic %', fontsize=8, labelpad=0)
        ax3.set_ylabel(r'Atomic Number $Z$', fontsize=8, labelpad=0)
        ax3.set_xlabel(r'Atomic Mass $A$', fontsize=8, labelpad=0)
        #ax3.set_zticklabels([])
        #plt.tight_layout()
        plt.show(block=False)
        plt.savefig(self._dir_out_figs + 'tracked_elements.pdf', bbox_inches='tight')
   
    def DTD_plot(self):
        print('Starting DTD_plot()')
        from matplotlib import pyplot as plt
        plt.style.use(self._dir+'/galcem.mplstyle')
        phys = pd.read_csv(self._dir_out+'phys.dat', sep=',', comment='#')
        gal_time = phys['time[Gyr]'].iloc[:-1]
        DTD_SNIa = phys['DTD_Ia[N/yr]'].iloc[:-1]
        fig, ax = plt.subplots(1,1, figsize=(7,5))
        ax.plot(np.log10(gal_time*1e9), np.log10(DTD_SNIa), color='blue', label=r'Eq. (16) $\gamma=1$ IMF: Kroupa (2001)')
        ax.legend(loc='best', frameon=False, fontsize=13)
        ax.set_ylabel(r'Normalized DTD', fontsize=15)
        ax.set_xlabel(r'SSP $\tau$ [yr]', fontsize=15)
        #ax.set_ylim(-4.5,1)
        ax.set_xlim(7.6, 10.2)
        fig.suptitle(r"Greggio (2005) SD SNIa model (Fig. 3)", fontsize=15)
        fig.tight_layout()
        plt.savefig(self._dir_out_figs + 'DTD_SNIa.pdf', bbox_inches='tight')
        
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
        ax[0,0].set_ylim(0.001,200)
        ax[0,0].set_xlim(0.6, 120)
        fig.delaxes(ax[0,1])
        fig.tight_layout()
        plt.savefig(self._dir_out_figs + 'tauratio.pdf', bbox_inches='tight')
     
    def total_evolution_plot(self, figsiz=(12,7), logAge=False): #(12,6)
        print('Starting total_evolution_plot()')
        from matplotlib import pyplot as plt
        import matplotlib.ticker as ticker
        Mfin = self.IN.M_inf
        #plt.style.use(self._dir+'/galcem.mplstyle')
        phys = pd.read_csv(self._dir_out+'phys.dat', sep=',', comment='#')
        Mass_i = np.loadtxt(self._dir_out + 'Mass_i.dat')
        time_chosen = phys['time[Gyr]']#.iloc[:-1]
        Mtot = phys['Mtot[Msun]']
        Mgas_v = phys['Mgas[Msun]']
        Mstar_v = phys['Mstar[Msun]']
        SFR_v = phys['SFR[Msun/yr]']
        Infall_rate = phys['Inf[Msun/yr]'] 
        Z_v = phys['Zfrac']
        G_v = phys['Gfrac']
        S_v = phys['Sfrac'] 
        Rate_SNCC = phys['R_CC[M/yr]']
        Rate_SNIa = phys['R_Ia[M/yr]']
        Rate_LIMs = phys['R_LIMs[M/y]']
        fig, axs = plt.subplots(1, 2, figsize=figsiz)
        axt = axs[1].twinx()
        time_plot = time_chosen
        xscale = '_lin'
        axs[0].hlines(self.IN.M_inf, 0, self.IN.Galaxy_age, label=r'$M_{gal,f}$', linewidth=1, linestyle = '-.', color='#8c00ff')
        axt.vlines(self.IN.Galaxy_age, self.IN.MW_SFR-.4, self.IN.MW_SFR+0.4, label=r'SFR$_{MW}$ CP11', linewidth = 7, linestyle = '-', color='#ff8c00', alpha=0.8)
        axt.vlines(self.IN.Galaxy_age, self.IN.MW_RSNCC[2], self.IN.MW_RSNCC[1], label=r'R$_{SNCC,MW}$ M05', linewidth = 6, linestyle = '-', color='#0034ff', alpha=0.8)
        axt.vlines(self.IN.Galaxy_age, self.IN.MW_RSNIa[2], self.IN.MW_RSNIa[1], label=r'R$_{SNIa,MW}$ M05', linewidth = 6, linestyle = '-', color='#00b3ff', alpha=0.8)
        axs[0].semilogy(time_plot, Mtot, label=r'$M_{tot}$', linewidth=4, color='black')
        axs[0].semilogy(time_plot, Mstar_v + Mgas_v, label= r'$M_g + M_s$', linewidth=3, linestyle = '--', color='#a9a9a9')
        axs[0].semilogy(time_plot, Mstar_v, label= r'$M_{star}$', linewidth=3, color='#ff8c00')
        axs[0].semilogy(time_plot, Mgas_v, label= r'$M_{gas}$', linewidth=3, color='#0d00ff')
        axs[0].semilogy(time_plot, np.sum(Mass_i[:,2:], axis=0), label = r'$M_{g,tot,i}$', linewidth=2, linestyle=':', color='#00b3ff')
        axs[0].semilogy(time_plot, np.sum(Mass_i[:2,2:], axis=0), label = r'$M_{H,g}$', linewidth=1, linestyle='-.', color='#0033ff')
        axs[0].semilogy(time_plot, np.sum(Mass_i[4:,2:], axis=0), label = r'$M_{Z,g}$', linewidth=2, linestyle=':', color='#ff0073')
        axs[0].semilogy(time_plot, np.sum(Mass_i[2:4,2:], axis=0), label = r'$M_{He,g}$', linewidth=1, linestyle='--', color='#0073ff')
        axs[1].semilogy(time_plot[:-1], Rate_SNCC[:-1], label= r'$R_{SNCC}$', color = '#0034ff', linestyle=':', linewidth=3)
        axs[1].semilogy(time_plot[:-1], Rate_SNIa[:-1], label= r'$R_{SNIa}$', color = '#00b3ff', linestyle=':', linewidth=3)
        axs[1].semilogy(time_plot[:-1], Rate_LIMs[:-1], label= r'$R_{LIMs}$', color = '#ff00b3', linestyle=':', linewidth=3)
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
            axs[0].set_xlim(self.IN.Galaxy_birthtime, 1.5e1)
            axs[1].set_xlim(self.IN.Galaxy_birthtime, 1.5e1)
            axt.set_xscale('log')
            xscale = '_log'
        axs[0].tick_params(right=True, which='both', direction='in')
        axs[1].tick_params(right=True, which='both', direction='in')
        axt.tick_params(right=True, which='both', direction='in')
        axs[0].set_xlabel(r'Age [Gyr]', fontsize = 20)
        axs[1].set_xlabel(r'Age [Gyr]', fontsize = 20)
        axs[0].set_ylabel(r'Masses [$M_{\odot}$]', fontsize = 20)
        axs[1].set_ylabel(r'Rates [$M_{\odot}/yr$]', fontsize = 20)
        axs[0].legend(fontsize=18, loc='lower center', ncol=2, frameon=True, framealpha=0.8)
        axs[1].legend(fontsize=15, loc='upper center', ncol=2, frameon=True, framealpha=0.8)
        axt.legend(fontsize=15, loc='lower center', ncol=1, frameon=True, framealpha=0.8)
        plt.tight_layout()
        plt.show(block=False)
        plt.savefig(self._dir_out_figs + 'total_physical'+str(xscale)+'.pdf', bbox_inches='tight')
        
    def _age_observations(self):
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
        FeH_id_sort = np.argsort(FeH_age)
        metallicity_value = observ_P14_5['MH2'][id_match5[1]]
        metallicity_age = observ_SA['Age'][id_match5[2]]
        Z_id_sort = np.argsort(metallicity_age)
        return FeH_value[FeH_id_sort], FeH_age[FeH_id_sort], metallicity_value[Z_id_sort], metallicity_age[Z_id_sort]
        
    def FeH_evolution_plot(self, c=2, elemZ=26, logAge=True):
        '''Skip the first two timestep (0 empty Galaxy, 1 only infall)'''
        print('Starting FeH_evolution()')
        from matplotlib import pyplot as plt
        #plt.style.use(self._dir+'/galcem.mplstyle')
        Z_list = np.unique(self.ZA_sorted[:,0])
        phys = pd.read_csv(self._dir_out+'phys.dat', sep=',', comment='#')
        gal_time = phys['time[Gyr]'].iloc[c:]
        solar_norm_H = self.c_class.solarA09_vs_H_bymass[Z_list]
        solar_norm_Fe = self.c_class.solarA09_vs_Fe_bymass[Z_list]
        Mass_i = np.loadtxt(self._dir_out + 'Mass_i.dat')
        FeH_value, FeH_age, _, _ = self._age_observations()
        a, b = np.polyfit(FeH_age, FeH_value, 1)
        Fe = np.sum(Mass_i[self._select_elemZ_idx(elemZ), c+2:], axis=0)
        H = np.sum(Mass_i[self._select_elemZ_idx(1), c+2:], axis=0)
        FeH = np.log10(np.divide(Fe, H)) - solar_norm_H[elemZ]
        fig, ax = plt.subplots(1,1, figsize=(7,5))
        ax.plot(gal_time, FeH, color='black', label='[Fe/H]', linewidth=3) 
        ax.axvline(x=self.IN.Galaxy_age-self.IN.solar_age, linewidth=2, color='orange', label=r'Age$_{\odot}$')
        ax.plot(self.IN.Galaxy_age - FeH_age, a*FeH_age+b, color='red', alpha=1, linewidth=3, label='linear fit on [Fe/H]')
        ax.scatter(self.IN.Galaxy_age - FeH_age, FeH_value, color='red', marker='*', alpha=0.3, label='Silva Aguirre et al. (2018)')
        ax.axhline(y=0, linewidth=1, color='orange', linestyle='--')
        #ax.errorbar(self.IN.Galaxy_age - observ['age'], observ['FeH'], yerr=observ['FeHerr'], marker='s', label='Meusinger+91', mfc='gray', ecolor='gray', ls='none')
        ax.legend(loc='lower right', frameon=False, fontsize=17)
        ax.set_ylabel(r'['+np.unique(self.ZA_symb_list[elemZ].values)[0]+'/H]', fontsize=20)
        ax.set_xlabel('Galaxy Age [Gyr]', fontsize=20)
        ax.set_ylim(-2,1)
        xscale = '_lin'
        if not logAge:
            ax.set_xlim(0,self.IN.Galaxy_age)
        else:
            ax.set_xscale('log')
            xscale = '_log'
        #ax.set_xlim(1e-2, 1.9e1)
        fig.tight_layout()
        plt.savefig(self._dir_out_figs + 'FeH_evolution'+str(xscale)+'.pdf', bbox_inches='tight')

    def Z_evolution_plot(self, c=2, logAge=False):
        '''Skip the first two timestep (0 empty Galaxy, 1 only infall)'''
        print('Starting Z_evolution()')
        from matplotlib import pyplot as plt
        #plt.style.use(self._dir+'/galcem.mplstyle')
        Z_list = np.unique(self.ZA_sorted[:,0])
        phys = pd.read_csv(self._dir_out+'phys.dat', sep=',', comment='#')
        gal_time = phys['time[Gyr]'].iloc[c:]
        _, _, metallicity_value, metallicity_age = self._age_observations()
        a, b = np.polyfit(metallicity_age, metallicity_value, 1)
        solar_norm_H = self.c_class.solarA09_vs_H_bymass[Z_list]
        Mass_i = np.loadtxt(self._dir_out + 'Mass_i.dat')
        Z = np.sum(Mass_i[self.i_Z:, c+2:], axis=0)
        H = np.sum(Mass_i[self._select_elemZ_idx(1), c+2:], axis=0)
        ZH = np.log10(np.divide(Z, H)/self.IN.solar_metallicity)
        fig, ax = plt.subplots(1,1, figsize=(7,5))
        ax.plot(gal_time, ZH, color='blue', label='Z', linewidth=3)
        ax.axvline(x=self.IN.Galaxy_age-self.IN.solar_age, linewidth=2, color='orange', label=r'Age$_{\odot}$')
        ax.axhline(y=0, linewidth=1, color='orange', linestyle='--')
        ax.plot(self.IN.Galaxy_age - metallicity_age, a*metallicity_age+b, color='red', alpha=1, linewidth=3, label='linear fit on [M/H]')
        ax.scatter(self.IN.Galaxy_age - metallicity_age, metallicity_value, color='red', marker='*', alpha=0.3, label='Silva Aguirre et al. (2018)')
        #ax.errorbar(self.IN.Galaxy_age - observ['age'], observ['FeH'], yerr=observ['FeHerr'], marker='s', label='Meusinger+91', mfc='gray', ecolor='gray', ls='none')
        ax.legend(loc='lower right', frameon=False, fontsize=17)
        ax.set_ylabel(r'metallicity', fontsize=20)
        ax.set_xlabel('Galaxy Age [Gyr]', fontsize=20)
        ax.set_ylim(-2,1)
        xscale = '_lin'
        if not logAge:
            ax.set_xlim(0,self.IN.Galaxy_age)
        else:
            ax.set_xscale('log')
            xscale = '_log'
        fig.tight_layout()
        plt.savefig(self._dir_out_figs + 'Z_evolution'+str(xscale)+'.pdf', bbox_inches='tight')
        
    def ind_evolution(self, c=5, elemZ=8, logAge=False):
        elemZ1 = 7
        elemZ2 = 12 
        print('Starting ind_evolution()')
        from matplotlib import pyplot as plt
        #plt.style.use(self._dir+'/galcem.mplstyle')
        Z_list = np.unique(self.ZA_sorted[:,0])
        phys = pd.read_csv(self._dir_out+'phys.dat', sep=',', comment='#')
        gal_time = phys['time[Gyr]'].iloc[c:]
       # _, _, metallicity_value, metallicity_age = self._age_observations()
        #a, b = np.polyfit(metallicity_age, metallicity_value, 1)
        solar_norm_Fe = self.c_class.solarA09_vs_Fe_bymass[Z_list]
        Mass_i = np.loadtxt(self._dir_out + 'Mass_i.dat')
        N = np.sum(Mass_i[self._select_elemZ_idx(elemZ1), c+2:], axis=0)
        Mg = np.sum(Mass_i[self._select_elemZ_idx(elemZ2), c+2:], axis=0)
        Fe = np.sum(Mass_i[self._select_elemZ_idx(26), c+2:], axis=0)
        NFe = np.log10(np.divide(N, Fe)) - solar_norm_Fe[elemZ1]
        MgFe = np.log10(np.divide(Mg, Fe)) - solar_norm_Fe[elemZ2]
        fig, ax = plt.subplots(1,1, figsize=(7,5))
        ax.plot(time, NFe, color='magenta', label='[N/Fe]', linewidth=3)
        ax.plot(time, MgFe, color='teal', label='[Mg/Fe]', linewidth=3)
        ax.axvline(x=self.IN.Galaxy_age-self.IN.solar_age, linewidth=2, color='orange', label=r'Age$_{\odot}$')
        #ax.axhline(y=0, linewidth=1, color='orange', linestyle='--')
        #ax.plot(self.IN.Galaxy_age +0.5 - metallicity_age, a*metallicity_age+b, color='red', alpha=1, linewidth=3, label='linear fit on [M/H]')
        #ax.scatter(self.IN.Galaxy_age +0.5 - metallicity_age, metallicity_value, color='red', marker='*', alpha=0.3, label='Silva Aguirre et al. (2018)')
        #ax.errorbar(self.IN.Galaxy_age - observ['age'], observ['FeH'], yerr=observ['FeHerr'], marker='s', label='Meusinger+91', mfc='gray', ecolor='gray', ls='none')
        ax.legend(loc='best', frameon=False, fontsize=17)
        ax.set_ylabel(r'[X/Fe]', fontsize=20)
        ax.set_xlabel('Galaxy Age [Gyr]', fontsize=20)
        ax.set_ylim(-2,1)
        xscale = '_lin'
        if not logAge:
            ax.set_xlim(0,self.IN.Galaxy_age)
        else:
            ax.set_xscale('log')
            xscale = '_log'
            ax.set_xlim(2e-2,self.IN.Galaxy_age)
        #ax.set_xlim(1e-2, 1.9e1)
        fig.tight_layout()
        plt.savefig(self._dir_out_figs + 'ind_evolution'+str(xscale)+'.pdf', bbox_inches='tight')
        
    def iso_evolution_comp_plot(self, figsize=(12,18), logAge=True, ncol=15):
        import math
        print('Starting iso_evolution_comp()')
        from matplotlib import pyplot as plt
        plt.style.use(self._dir+'/galcem.mplstyle')
        import matplotlib.ticker as ticker
        Mass_i = np.loadtxt(self._dir_out + 'Mass_i.dat')
        Masses = np.log10(Mass_i[:,2:])#, where=Mass_i[:,2:]>0.)
        phys = pd.read_csv(self._dir_out+'phys.dat', sep=',', comment='#')
        timex = phys['time[Gyr]']
        W_i_comp = pickle.load(open(self._dir_out + 'W_i_comp.pkl','rb'))
        #Mass_MRSN = np.log10(W_i_comp['MRSN'])
        Mass_BBN = np.log10(W_i_comp['BBN'])#, where=W_i_comp['BBN']>0.)
        Mass_SNCC = np.log10(W_i_comp['SNCC'])#, where=W_i_comp['SNCC']>0.)
        Mass_AGB = np.log10(W_i_comp['LIMs'])#, where=W_i_comp['LIMs']>0.)
        Mass_SNIa = np.log10(W_i_comp['SNIa'])#, where=W_i_comp['SNIa']>0.)
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
                ax.annotate('%d%s(%d)'%(Z[i],self.ZA_symb_list.values[i],A[i]), xy=(0.5, 0.3), xycoords='axes fraction', horizontalalignment='center', verticalalignment='top', fontsize=7, alpha=0.7)
                ax.set_ylim(-4.9, 10.9)
                ax.set_xlim(0.01,13.8)
                ax.yaxis.set_minor_locator(ticker.MultipleLocator(base=1))
                ax.tick_params(width=1, length=2, axis='y', which='minor', left=True, right=True, direction='in')
                ax.yaxis.set_major_locator(ticker.MultipleLocator(base=5))
                ax.tick_params(width=1, length=3, axis='y', which='major', left=True, right=True, direction='in')
                ax.plot(timex[:-1], Mass_BBN[i][:-1], color='black', linestyle='-.', linewidth=3, alpha=0.8, label='BBN')
                ax.plot(timex[:-1], Mass_SNCC[i][:-1], color='#0034ff', linestyle='-.', linewidth=3, alpha=0.8, label='SNCC')
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
                    
        axs[nrow//2,0].set_ylabel(r'Returned masses [ $\log_{10}$($M_{\odot}$)]', fontsize = 15)
        axs[0, ncol//2].legend(ncol=len(W_i_comp), loc='upper center', bbox_to_anchor=(0.5, 1.8), frameon=False, fontsize=12)
        if not logAge:
            xscale = '_lin'
            axs[nrow-2, ncol//2].set_xlabel('Age [Gyr]', fontsize = 15)
        else:
            xscale = '_log'
            axs[nrow-2, ncol//2].set_xlabel('Log  Age [Gyr]', fontsize = 15)
        plt.subplots_adjust(wspace=0., hspace=0.)
        #plt.tight_layout(rect = [0.03, 0.03, 1, .90])
        plt.show(block=False)
        plt.savefig(self._dir_out_figs + 'iso_evolution_comp'+str(xscale)+'.pdf', bbox_inches='tight')

    def iso_evolution_comp_lelemz_plot(self, figsize=(12,15), logAge=True, ncol=10):
        import math
        import pickle
        IN = pickle.load(open(self._dir_out + 'inputs.pkl','rb'))
        print('Starting iso_evolution_comp_lelemz()')
        from matplotlib import pyplot as plt
        plt.style.use(self._dir+'/galcem.mplstyle')
        import matplotlib.ticker as ticker
        Mass_i = np.loadtxt(self._dir_out + 'Mass_i.dat')
        Masses = np.log10(Mass_i[:,2:])#, where=Mass_i[:,2:]>0.)
        phys = pd.read_csv(self._dir_out+'phys.dat', sep=',', comment='#')
        timex = phys['time[Gyr]']
        W_i_comp = pickle.load(open(self._dir_out + 'W_i_comp.pkl','rb'))
        #Mass_MRSN = np.log10(W_i_comp['MRSN'], where=W_i_comp['MRSN']>0.)
        yr_rate = IN.nTimeStep * 1e9
        Mass_BBN = np.log10(W_i_comp['BBN']/yr_rate)#, where=W_i_comp['BBN']>0., out=np.zeros((W_i_comp['BBN']).shape))
        Mass_SNCC = np.log10(W_i_comp['SNCC']/yr_rate)#, where=W_i_comp['SNCC']>0., out=np.zeros((W_i_comp['SNCC']).shape))
        Mass_AGB = np.log10(W_i_comp['LIMs']/yr_rate)#, where=W_i_comp['LIMs']>0., out=np.zeros((W_i_comp['LIMs']).shape))
        Mass_SNIa = np.log10(W_i_comp['SNIa']/yr_rate)#, where=W_i_comp['SNIa']>0., out=np.zeros((W_i_comp['SNIa']).shape))
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
                ax.annotate('%d%s(%d)'%(Z[i],self.ZA_symb_list.values[i],A[i]), xy=(0.5, 0.3), xycoords='axes fraction', horizontalalignment='center', verticalalignment='top', fontsize=7, alpha=0.7)
                ax.set_ylim(-16.9, 5.9)
                ax.set_xlim(0.01,13.8)
                ax.yaxis.set_minor_locator(ticker.MultipleLocator(base=1))
                ax.tick_params(width=1, length=2, axis='y', which='minor', left=True, right=True, direction='in')
                ax.yaxis.set_major_locator(ticker.MultipleLocator(base=5))
                ax.tick_params(width=1, length=3, axis='y', which='major', left=True, right=True, direction='in')
                ax.plot(timex[:-1], Mass_BBN[i][:-1], color='black', linestyle='-.', linewidth=3, alpha=0.8, label='BBN')
                ax.plot(timex[:-1], Mass_SNCC[i][:-1], color='#0034ff', linestyle='-.', linewidth=3, alpha=0.8, label='SNCC')
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
                    
        axs[nrow//2,0].set_ylabel(r'Returned masses [ $\log_{10}$($M_{\odot}/$ yr)]', fontsize = 15)
        axs[0, ncol//2].legend(ncol=len(W_i_comp), loc='upper center', bbox_to_anchor=(0.5, 1.8), frameon=False, fontsize=12)
        if not logAge:
            xscale = '_lin'
            axs[nrow-1, ncol//2].set_xlabel('Age [Gyr]', fontsize = 15)
        else:
            xscale = '_log'
            axs[nrow-1, ncol//2].set_xlabel('Log  Age [Gyr]', fontsize = 15)
        plt.subplots_adjust(wspace=0., hspace=0.)
        #plt.tight_layout(rect = [0.03, 0.03, 1, .95])
        plt.show(block=False)
        plt.savefig(self._dir_out_figs + 'iso_evolution_comp_lz'+str(xscale)+'.pdf', bbox_inches='tight')

    def _extract_normalized_abundances(self, Z_list, Mass_i_loc, c=3):
        solar_norm_H = self.c_class.solarA09_vs_H_bymass[Z_list]
        solar_norm_Fe = self.c_class.solarA09_vs_Fe_bymass[Z_list]
        Mass_i = np.loadtxt(Mass_i_loc)
        #Fe = np.sum(Mass_i[np.intersect1d(np.where(ZA_sorted[:,0]==26)[0], np.where(ZA_sorted[:,1]==56)[0]), c+2:], axis=0)
        Fe = np.sum(Mass_i[self._select_elemZ_idx(26), c+2:], axis=0)
        H = np.sum(Mass_i[self._select_elemZ_idx(1), c+2:], axis=0)
        FeH = np.log10(np.divide(Fe, H)) - solar_norm_H[26]
        abund_i = []
        for i,val in enumerate(Z_list):
            mass = np.sum(Mass_i[self._select_elemZ_idx(val), c+2:], axis=0)
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
        
        normalized_abundances, FeH = self._extract_normalized_abundances(Z_list, Mass_i_loc='runs/baseline/Mass_i.dat', c=c+2)
        normalized_abundances_lowZ, FeH_lowZ = self._extract_normalized_abundances(Z_list, Mass_i_loc='runs/lifetimeZ0003/Mass_i.dat', c=c+2)
        normalized_abundances_highZ, FeH_highZ = self._extract_normalized_abundances(Z_list, Mass_i_loc='runs/lifetimeZ06/Mass_i.dat', c=c+2)
        normalized_abundances_lowIMF, FeH_lowIMF = self._extract_normalized_abundances(Z_list, Mass_i_loc='runs/IMF1pt2/Mass_i.dat', c=c+2)
        normalized_abundances_highIMF, FeH_highIMF = self._extract_normalized_abundances(Z_list, Mass_i_loc='runs/IMF1pt7/Mass_i.dat', c=c+2)
        normalized_abundances_lowSFR, FeH_lowSFR = self._extract_normalized_abundances(Z_list, Mass_i_loc='runs/k_SFR_0.5/Mass_i.dat', c=c+2)
        normalized_abundances_highSFR, FeH_highSFR = self._extract_normalized_abundances(Z_list, Mass_i_loc='runs/k_SFR_2/Mass_i.dat', c=c+2)
        
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
    
    def _select_elemZ_idx(self, elemZ):
        ''' auxiliary function that selects the isotope indexes where Z=elemZ '''
        return np.where(self.ZA_sorted[:,0]==elemZ)[0]
   
    def observational_plot(self, figsiz = (15,10), c=3):
        print('Starting observational_plot()')
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
        Fe = np.sum(Mass_i[self._select_elemZ_idx(26), c+2:], axis=0)
        H = np.sum(Mass_i[self._select_elemZ_idx(1), c+2:], axis=0)
        for i,val in enumerate(Z_list):
            mass = np.sum(Mass_i[self._select_elemZ_idx(val), c+2:], axis=0)
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
                    ax.set_xlabel('[Fe/H]', fontsize = 15)
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
        #fig.tight_layout(rect = [0.03, 0, 1, 1])
        fig.subplots_adjust(wspace=0., hspace=0.)
        plt.show(block=False)
        plt.savefig(self._dir_out_figs + 'elem_obs.pdf', bbox_inches='tight')
        return None

    def observational_lelemZ_plot(self, figsiz = (15,10), c=3, yrange='zoom', romano10=False):
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
        Fe = np.sum(Mass_i[self._select_elemZ_idx(26), c+2:], axis=0)
        H = np.sum(Mass_i[self._select_elemZ_idx(1), c+2:], axis=0)
        for i,val in enumerate(Z_list):
            mass = np.sum(Mass_i[self._select_elemZ_idx(val), c+2:], axis=0)
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
        axs[nrow-1, ncol//2].set_xlabel('[Fe/H]', fontsize = 15)
        #fig.tight_layout(rect=[0., 0, 1, .9])
        fig.subplots_adjust(wspace=0., hspace=0.)
        plt.show(block=False)
        plt.savefig(self._dir_out_figs + 'elem_obs_lelemZ.pdf', bbox_inches='tight')
        return None
    
    def _extract_comparison(self, dir_val, _select_elemZ_idx, solar_norm_H, solar_norm_Fe, Z_list, c):
        directory = 'runs/'+dir_val+'/'
        Mass_i = np.loadtxt(directory+'Mass_i.dat')
        Masses_i = []
        Fe = np.sum(Mass_i[self._select_elemZ_idx(26), c+2:], axis=0)
        H = np.sum(Mass_i[self._select_elemZ_idx(1), c+2:], axis=0)
        FeH = np.log10(np.divide(Fe, H)) - solar_norm_H[26]
        for i,val in enumerate(Z_list):
            mass = np.sum(Mass_i[self._select_elemZ_idx(val), c+2:], axis=0)
            Masses_i.append(np.log10(np.divide(mass,Fe)) - solar_norm_Fe[val])
        Masses = np.array(Masses_i) 
        return FeH, Masses
    
    def observational_helemZ_dir_comparison_plot(self, figsiz = (15,10), c=3, yrange='full', 
                                            romano10=False, directories={'SMBH zap':'20220623_zap_2Myr','MRSN':'20220614_MRSN_massrange_2Myr'},
                                            Z_list=[26,38,39,40,41,42,44,46,47,56,57,58,59,
                                                    60,62,63,64,66,67,68,70,72,76,77,79,82]):
        ''' yrange full to include all observational points'''
        print('observational_helemZ_dir_comparison()')
        import glob
        import itertools
        from matplotlib import pyplot as plt
        import matplotlib.ticker as ticker
        #plt.style.use(self._dir+'/galcem.mplstyle')
        Z_list = np.array(Z_list)
        Z_symb_list = self.IN.periodic['elemSymb'][Z_list] # name of elements for all isotopes
        solar_norm_H = self.c_class.solarA09_vs_H_bymass[Z_list]
        solar_norm_Fe = self.c_class.solarA09_vs_Fe_bymass[Z_list]
        plot_pairs = {}
        for d in directories:
            plot_pairs[d] = self._extract_comparison(directories[d], self._select_elemZ_idx, solar_norm_H, solar_norm_Fe, Z_list, c)

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
            markerlist = itertools.cycle(listmarkers)
            linestylelist = itertools.cycle(['-','--',':','-.']) 
            for j, ll in enumerate(li):
                if i < len(Z_list):
                    ip = Z_list[i]#+2 # Shift to skip H and He
                    idx_obs = np.where(ll.iloc[:,0] == ip)[0]
                    ax.scatter(ll.iloc[idx_obs,1], ll.iloc[idx_obs,2], label=linames[j], alpha=0.3, marker=next(markerlist), c=next(colorlist), s=20)
            if i == 0:
                    ax.legend(ncol=7, loc='lower left', bbox_to_anchor=(-.2, 1.), frameon=False, fontsize=9)
            if i < len(Z_list):
                ip = i#+2 # Shift to skip H and He
                for d in directories:
                    ax.plot(plot_pairs[d][0], plot_pairs[d][1][ip,:], color='black', linewidth=2, linestyle=next(linestylelist), label=d)
                #ax.fill_between(FeH1, Masses1[ip], Masses2[ip], where=(Masses1[ip] > Masses2[ip]), color='blue', alpha=0.2,
                # interpolate=True)
                #ax.fill_between(FeH1, Masses1[ip], Masses2[ip], where=(Masses1[ip] <= Masses2[ip]), color='red', alpha=0.2,
                # interpolate=True)
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
        axs[nrow-1, ncol//2].set_xlabel('[Fe/H]', fontsize = 15)
        fig.tight_layout(rect=[0., 0, 1, .9])
        fig.subplots_adjust(wspace=0., hspace=0.)
        plt.show(block=False)
        plt.savefig(self._dir_out_figs + 'elem_obs_helemZ_dir_comparison.pdf', bbox_inches='tight')
        return None
    
    def obs_lZ_plot(self, figsiz = (21,7), c=3):
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
        Fe = np.sum(Mass_i[self._select_elemZ_idx(26), c+2:], axis=0)
        H = np.sum(Mass_i[self._select_elemZ_idx(1), c+2:], axis=0)
        for i,val in enumerate(Z_list):
            mass = np.sum(Mass_i[self._select_elemZ_idx(val), c+2:], axis=0)
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
                ax.plot(FeH, Masses2[ip], color='black', linewidth=2)
                ax.annotate(f"{Z_list[ip]}{Z_symb_list[Z_list[ip]]}", xy=(0.5, 0.92), xycoords='axes fraction', horizontalalignment='center', verticalalignment='top', fontsize=12, alpha=0.7)
                ax.set_ylim(-4.9, 4.9)
                ax.set_xlim(-6.5, 1.5)
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
        axs[nrow-1, ncol//2].set_xlabel('[Fe/H]', fontsize=15, loc='center')
        #fig.tight_layout(rect=[0.0, 0, 1, .8])
        fig.subplots_adjust(wspace=0., hspace=0.)
        plt.show(block=False)
        plt.savefig(self._dir_out_figs + 'elem_obs_lZ.pdf', bbox_inches='tight')
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