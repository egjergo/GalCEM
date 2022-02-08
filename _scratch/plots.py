""" I only achieve simplicity with enormous effort (Clarice Lispector) """
import time
import numpy as np

import galcem.inputs as INp
IN = INp.Inputs()
from galcem.prep.setup import *

from matplotlib import cm
import matplotlib.pylab as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as colors
import matplotlib.ticker as ticker
#cmap = cm.get_cmap('plasma', 100)
supported_cmap = ['Accent', 'Accent_r', 'Blues', 'Blues_r', 'BrBG', 'BrBG_r', 'BuGn', 'BuGn_r', 'BuPu', 'BuPu_r', 'CMRmap', 'CMRmap_r', 'Dark2', 'Dark2_r', 'GnBu', 'GnBu_r', 'Greens', 'Greens_r', 'Greys', 'Greys_r', 'OrRd', 'OrRd_r', 'Oranges', 'Oranges_r', 'PRGn', 'PRGn_r', 'Paired', 'Paired_r', 'Pastel1', 'Pastel1_r', 'Pastel2', 'Pastel2_r', 'PiYG', 'PiYG_r', 'PuBu', 'PuBuGn', 'PuBuGn_r', 'PuBu_r', 'PuOr', 'PuOr_r', 'PuRd', 'PuRd_r', 'Purples', 'Purples_r', 'RdBu', 'RdBu_r', 'RdGy', 'RdGy_r', 'RdPu', 'RdPu_r', 'RdYlBu', 'RdYlBu_r', 'RdYlGn', 'RdYlGn_r', 'Reds', 'Reds_r', 'Set1', 'Set1_r', 'Set2', 'Set2_r', 'Set3', 'Set3_r', 'Spectral', 'Spectral_r', 'Wistia', 'Wistia_r', 'YlGn', 'YlGnBu', 'YlGnBu_r', 'YlGn_r', 'YlOrBr', 'YlOrBr_r', 'YlOrRd', 'YlOrRd_r', 'afmhot', 'afmhot_r', 'autumn', 'autumn_r', 'binary', 'binary_r', 'bone', 'bone_r', 'brg', 'brg_r', 'bwr', 'bwr_r', 'cividis', 'cividis_r', 'cool', 'cool_r', 'coolwarm', 'coolwarm_r', 'copper', 'copper_r', 'cubehelix', 'cubehelix_r', 'flag', 'flag_r', 'gist_earth', 'gist_earth_r', 'gist_gray', 'gist_gray_r', 'gist_heat', 'gist_heat_r', 'gist_ncar', 'gist_ncar_r', 'gist_rainbow', 'gist_rainbow_r', 'gist_stern', 'gist_stern_r', 'gist_yarg', 'gist_yarg_r', 'gnuplot', 'gnuplot2', 'gnuplot2_r', 'gnuplot_r', 'gray', 'gray_r', 'hot', 'hot_r', 'hsv', 'hsv_r', 'inferno', 'inferno_r', 'jet', 'jet_r', 'magma', 'magma_r', 'nipy_spectral', 'nipy_spectral_r', 'ocean', 'ocean_r', 'pink', 'pink_r', 'plasma', 'plasma_r', 'prism', 'prism_r', 'rainbow', 'rainbow_r', 'seismic', 'seismic_r', 'spring', 'spring_r', 'summer', 'summer_r', 'tab10', 'tab10_r', 'tab20', 'tab20_r', 'tab20b', 'tab20b_r', 'tab20c', 'tab20c_r', 'terrain', 'terrain_r', 'turbo', 'turbo_r', 'twilight', 'twilight_r', 'twilight_shifted', 'twilight_shifted_r', 'viridis', 'viridis_r', 'winter', 'winter_r']
#mycolors = ["darkorange", "gold", "lawngreen", "lightseagreen"]
#nodes = [0.0, 0.4, 0.8, 1.0]
#cmap2 = colors.LinearSegmentedColormap.from_list("mycmap", list(zip(nodes, mycolors)))
#cmap = cm.get_cmap(supported_cmap[-4], 10)
from test.util import mplsetup
mplsetup()

supported_cmap = ['Accent', 'Accent_r', 'Blues', 'Blues_r', 'BrBG', 'BrBG_r', 'BuGn', 'BuGn_r', 'BuPu', 'BuPu_r', 'CMRmap', 'CMRmap_r', 'Dark2', 'Dark2_r', 'GnBu', 'GnBu_r', 'Greens', 'Greens_r', 'Greys', 'Greys_r', 'OrRd', 'OrRd_r', 'Oranges', 'Oranges_r', 'PRGn', 'PRGn_r', 'Paired', 'Paired_r', 'Pastel1', 'Pastel1_r', 'Pastel2', 'Pastel2_r', 'PiYG', 'PiYG_r', 'PuBu', 'PuBuGn', 'PuBuGn_r', 'PuBu_r', 'PuOr', 'PuOr_r', 'PuRd', 'PuRd_r', 'Purples', 'Purples_r', 'RdBu', 'RdBu_r', 'RdGy', 'RdGy_r', 'RdPu', 'RdPu_r', 'RdYlBu', 'RdYlBu_r', 'RdYlGn', 'RdYlGn_r', 'Reds', 'Reds_r', 'Set1', 'Set1_r', 'Set2', 'Set2_r', 'Set3', 'Set3_r', 'Spectral', 'Spectral_r', 'Wistia', 'Wistia_r', 'YlGn', 'YlGnBu', 'YlGnBu_r', 'YlGn_r', 'YlOrBr', 'YlOrBr_r', 'YlOrRd', 'YlOrRd_r', 'afmhot', 'afmhot_r', 'autumn', 'autumn_r', 'binary', 'binary_r', 'bone', 'bone_r', 'brg', 'brg_r', 'bwr', 'bwr_r', 'cividis', 'cividis_r', 'cool', 'cool_r', 'coolwarm', 'coolwarm_r', 'copper', 'copper_r', 'cubehelix', 'cubehelix_r', 'flag', 'flag_r', 'gist_earth', 'gist_earth_r', 'gist_gray', 'gist_gray_r', 'gist_heat', 'gist_heat_r', 'gist_ncar', 'gist_ncar_r', 'gist_rainbow', 'gist_rainbow_r', 'gist_stern', 'gist_stern_r', 'gist_yarg', 'gist_yarg_r', 'gnuplot', 'gnuplot2', 'gnuplot2_r', 'gnuplot_r', 'gray', 'gray_r', 'hot', 'hot_r', 'hsv', 'hsv_r', 'inferno', 'inferno_r', 'jet', 'jet_r', 'magma', 'magma_r', 'nipy_spectral', 'nipy_spectral_r', 'ocean', 'ocean_r', 'pink', 'pink_r', 'plasma', 'plasma_r', 'prism', 'prism_r', 'rainbow', 'rainbow_r', 'seismic', 'seismic_r', 'spring', 'spring_r', 'summer', 'summer_r', 'tab10', 'tab10_r', 'tab20', 'tab20_r', 'tab20b', 'tab20b_r', 'tab20c', 'tab20c_r', 'terrain', 'terrain_r', 'turbo', 'turbo_r', 'twilight', 'twilight_r', 'twilight_shifted', 'twilight_shifted_r', 'viridis', 'viridis_r', 'winter', 'winter_r']


def phys_integral_plot(logAge=False):
    plt.clf()
    '''
    Requires running "phys_integral()" in onezone.py beforehand.
    '''
    phys = np.loadtxt('./output/phys.dat')
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
    fig = plt.figure(figsize =(7, 5))
    ax = fig.add_subplot(111)
    ax2 = ax.twinx()
    time_plot = time_chosen
    if logAge == True:
        time_plot = np.log10(time_chosen)
    ax.hlines(IN.M_inf[IN.morphology], 0, IN.age_Galaxy, label=r'$M_{gal,f}$', linewidth = 1, linestyle = '-.')
    ax.semilogy(time_plot, Mtot, label=r'$M_{tot}$', linewidth=2)
    ax.semilogy(time_plot, Mstar_v, label= r'$M_{star}$', linewidth=2)
    ax.semilogy(time_plot, Mgas_v, label= r'$M_{gas}$', linewidth=2)
    ax.semilogy(time_plot, Mstar_v + Mgas_v, label= r'$M_g + M_s$', linestyle = '--')
    ax.semilogy(time_plot, Mstar_test, label= r'$M_{star,t}$', linewidth=2, linestyle = ':')
    ax2.semilogy(time_plot, Infall_rate, label= r'Infall', color = 'cyan', linestyle='-', linewidth=3)
    ax2.semilogy(time_plot, SFR_v, label= r'SFR', color = 'blue', linestyle='--', linewidth=3)
    ax2.semilogy(time_plot, np.divide(Rate_SNII,1e9), label= r'SNII', color = 'black', linestyle=':', linewidth=3)
    ax2.semilogy(time_plot, np.divide(Rate_SNIa,1e9), label= r'SNIa', color = 'gray', linestyle=':', linewidth=3)
    ax2.semilogy(time_plot, np.divide(Rate_LIMs,1e9), label= r'LIMs', color = 'magenta', linestyle=':', linewidth=3)
    if not logAge:
        ax.set_xlim(0,13.8)
    ax.set_ylim(1e6, 1e11)
    ax2.set_ylim(1e-2, 1e2)
    ax.set_xlabel(r'Age [Gyr]', fontsize = 15)
    ax.set_ylabel(r'Masses [$M_{\odot}$]', fontsize = 15)
    ax2.set_ylabel(r'Rates [$M_{\odot}/yr$]', fontsize = 15)
    ax.set_title(r'$f_{SFR} = $ %.2f' % (IN.SFR_rescaling), fontsize=15)
    ax.legend(fontsize=15, loc='lower left', ncol=2, frameon=True, framealpha=0.8)
    ax2.legend(fontsize=15, loc='lower right', ncol=1, frameon=False)
    plt.tight_layout()
    plt.show(block=False)
    plt.savefig('./figures/total_physical.pdf')
    return None
    

def ZA_sorted_plot(cmap_name='magma_r', cbins=10): # angle = 2 * np.pi / np.arctan(0.4) !!!!!!!
    plt.clf()
    x = ZA_sorted[:,1]#- ZA_sorted[:,0]
    y = ZA_sorted[:,0]
    z = asplund3_percent
    cmap_ = cm.get_cmap(cmap_name, cbins)
    binning = np.digitize(z, np.linspace(0,9.*100/cbins,num=cbins-1))
    percent_colors = [cmap_.colors[c] for c in binning]
    fig, ax = plt.subplots(figsize =(11,5))
    print(f"{type(ax)=}")
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
    plt.savefig('./figures/test/tracked_elements.pdf')
    return None


def iso_evolution(figsiz = (32,10)):
    #plt.clf()
    Mass_i = np.loadtxt('./output/Mass_i.dat')
    Masses = np.log10(Mass_i[:,2:])
    phys = np.loadtxt('./output/phys.dat')
    timex = phys[:,0]
    Z = ZA_sorted[:,0]
    A = ZA_sorted[:,1]
    ncol = aux.find_nearest(np.power(np.arange(20),2), len(Z))
    if len(ZA_sorted) > ncol:
        nrow = ncol
    else:
        nrow = ncol + 1
    fig, axs = plt.subplots(nrow, ncol, figsize =figsiz)#, sharex=True)
    for i, ax in enumerate(axs.flat):
        if i < len(Z):
            ax.plot(timex, Masses[i])
            ax.annotate(f"{ZA_symb_list[i]}({Z[i]},{A[i]})", xy=(0.5, 0.92), xycoords='axes fraction', horizontalalignment='center', verticalalignment='top', fontsize=12, alpha=0.7)
            ax.set_ylim(-7.5, 10.5)
            ax.set_xlim(0,13.8)
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
    plt.tight_layout(rect = [0.02, 0, 1, 1])
    plt.subplots_adjust(wspace=0., hspace=0.)
    plt.show(block=False)
    plt.savefig('./figures/iso_evolution.pdf')
    return None


def iso_abundance(figsiz = (32,10), elem_idx=99): # elem_idx=99 is Fe56, elem_idx=0 is H.
    #plt.clf()
    Mass_i = np.loadtxt('./output/Mass_i.dat')
    #Masses = np.log10(np.divide(Mass_i[:,2:], Mass_i[elem_idx,2:]))
    Fe = np.sum(Mass_i[97:104,7:],axis=0)
    Masses = np.log10(np.divide(Mass_i[:,7:], Fe))
    #XH = np.log10(np.divide(Mass_i[elem_idx,2:], Mass_i[0,2:])) 
    XH = np.log10(np.divide(Fe, Mass_i[0,7:])) 
    Z = ZA_sorted[:,0]
    A = ZA_sorted[:,1]
    ncol = aux.find_nearest(np.power(np.arange(20),2), len(Z))
    if len(ZA_sorted) < ncol:
        nrow = ncol
    else:
        nrow = ncol + 1
    fig, axs = plt.subplots(nrow, ncol, figsize =figsiz)#, sharex=True)
    for i, ax in enumerate(axs.flat):
        if i < len(Z):
            ax.plot(XH, Masses[i])
            ax.annotate(f"{ZA_symb_list[i]}({Z[i]},{A[i]})", xy=(0.5, 0.92), xycoords='axes fraction', horizontalalignment='center', verticalalignment='top', fontsize=12, alpha=0.7)
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
    axs[nrow-1, ncol//2].set_xlabel(f'[{ZA_symb_list[elem_idx]}{A[elem_idx]}/H]', fontsize = 15)
    plt.tight_layout(rect = [0.02, 0, 1, 1])
    plt.subplots_adjust(wspace=0., hspace=0.)
    plt.show(block=False)
    plt.savefig('./figures/iso_abundance.pdf')
    return None

def extract_normalized_abundances(Z_list, Mass_i_loc='output/Mass_i.dat', c=5):
    solar_norm_H = c_class.solarA09_vs_H_bymass[Z_list]
    solar_norm_Fe = c_class.solarA09_vs_Fe_bymass[Z_list]
    Mass_i = np.loadtxt(Mass_i_loc)
    #Fe = np.sum(Mass_i[np.intersect1d(np.where(ZA_sorted[:,0]==26)[0], np.where(ZA_sorted[:,1]==56)[0]), c:], axis=0)
    Fe = np.sum(Mass_i[np.where(ZA_sorted[:,0]==26)[0], c:], axis=0)
    H = np.sum(Mass_i[np.where(ZA_sorted[:,0]==1)[0], c:], axis=0)
    FeH = np.log10(np.divide(Fe, H)) - solar_norm_H[np.where(Z_list==26)[0]]
    abund_i = []
    for i,val in enumerate(Z_list):
        mass = np.sum(Mass_i[np.where(ZA_sorted[:,0]==val)[0], c:], axis=0)
        abund_i.append(np.log10(np.divide(mass,Fe)) - solar_norm_Fe[np.where(Z_list==val)[0]])
    normalized_abundances = np.array(abund_i)
    return normalized_abundances, FeH

def elem_abundance(figsiz = (32,10), c=5, setylim = (-6, 6), setxlim=(-6.5, 0.5)):
    Z_list = np.unique(ZA_sorted[:,0])
    ncol = aux.find_nearest(np.power(np.arange(20),2), len(Z_list))
    if len(Z_list) < ncol:
        nrow = ncol
    else:
        nrow = ncol + 1
    Z_symb_list = IN.periodic['elemSymb'][Z_list] # name of elements for all isotopes
    
    normalized_abundances, FeH = extract_normalized_abundances(Z_list, Mass_i_loc='output/baseline/Mass_i.dat', c=c)
    normalized_abundances_lowZ, FeH_lowZ = extract_normalized_abundances(Z_list, Mass_i_loc='output/lifetimeZ0003/Mass_i.dat', c=c)
    normalized_abundances_highZ, FeH_highZ = extract_normalized_abundances(Z_list, Mass_i_loc='output/lifetimeZ06/Mass_i.dat', c=c)
    normalized_abundances_lowIMF, FeH_lowIMF = extract_normalized_abundances(Z_list, Mass_i_loc='output/Mass_i_IMF1pt2.dat', c=c)
    normalized_abundances_highIMF, FeH_highIMF = extract_normalized_abundances(Z_list, Mass_i_loc='output/Mass_i_IMF1pt7.dat', c=c)
    normalized_abundances_lowSFR, FeH_lowSFR = extract_normalized_abundances(Z_list, Mass_i_loc='output/Mass_i_fiducial.dat', c=c)
    normalized_abundances_highSFR, FeH_highSFR = extract_normalized_abundances(Z_list, Mass_i_loc='output/Mass_i_k_SFR_2.dat', c=c)
    
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
            ax.annotate(f"{Z_list[i]}{Z_symb_list[i]}", xy=(0.5, 0.92), xycoords='axes fraction', horizontalalignment='center', verticalalignment='top', fontsize=12, alpha=0.7)
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
    axs[nrow-1, ncol//2].set_xlabel(f'[Fe/H]', fontsize = 15)
    fig.tight_layout(rect = [0.03, 0, 1, 1])
    fig.subplots_adjust(wspace=0., hspace=0.)
    plt.show(block=False)
    plt.savefig('./figures/elem_abundance.pdf')
    return None


def lifetimeratio_test_plot(colormap='Paired'):
    fig, ax = plt.subplots(1,1, figsize=(7,5))
    divid05 = np.divide(IN.s_lifetimes_p98['Z05'], IN.s_lifetimes_p98['Z0004'])
    divid02 = np.divide(IN.s_lifetimes_p98['Z02'], IN.s_lifetimes_p98['Z0004'])
    divid008 = np.divide(IN.s_lifetimes_p98['Z008'], IN.s_lifetimes_p98['Z0004'])
    ax.semilogx(IN.s_lifetimes_p98['M'], divid05, color='blue', label='Z = 0.05')
    ax.semilogx(IN.s_lifetimes_p98['M'], divid02, color='blue', linestyle='--', label='Z = 0.02')
    ax.semilogx(IN.s_lifetimes_p98['M'], divid008, color='blue', linestyle=':', label='Z = 0.008')
    ax.hlines(1, 0.6,120, color='orange', label='ratio=1')
    ax.vlines(3, 0.75,2.6, color='red', label=r'$3 M_{\odot}$')
    ax.vlines(6, 0.75,2.6, color='red', alpha=0.6, linestyle='--', label=r'$6 M_{\odot}$')
    ax.vlines(9, 0.75,2.6, color='red', alpha=0.3, linestyle = ':', label=r'$9 M_{\odot}$')
    cm = plt.cm.get_cmap(colormap)
    sc=ax.scatter(IN.s_lifetimes_p98['M'], divid05, c=np.log10(IN.s_lifetimes_p98['Z05']), cmap=cm, s=50)
    sc=ax.scatter(IN.s_lifetimes_p98['M'], divid02, c=np.log10(IN.s_lifetimes_p98['Z05']), cmap=cm, s=50)
    sc=ax.scatter(IN.s_lifetimes_p98['M'], divid008, c=np.log10(IN.s_lifetimes_p98['Z05']), cmap=cm, s=50)
    fig.colorbar(sc, label=r'$\tau(M_*)$')
    ax.legend(loc='best', frameon=False, fontsize=13)
    ax.set_ylabel(r'$\tau(X)/\tau(Z=0.0004)$', fontsize=15)
    ax.set_xlabel('Mass', fontsize=15)
    ax.set_ylim(0.81,1.95)
    fig.tight_layout()
    plt.savefig('figures/test/tauratio.pdf')
    