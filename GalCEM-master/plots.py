""" I only achieve simplicity with enormous effort (Clarice Lispector) """
import time
import numpy as np

import prep.inputs as IN
import classes.morphology as morph
import classes.yields as Y
from prep.setup import *

from matplotlib import cm
import matplotlib.pylab as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec
import matplotlib.colors as colors
import matplotlib.ticker as ticker
#cmap = cm.get_cmap('plasma', 100)
supported_cmap = ['Accent', 'Accent_r', 'Blues', 'Blues_r', 'BrBG', 'BrBG_r', 'BuGn', 'BuGn_r', 'BuPu', 'BuPu_r', 'CMRmap', 'CMRmap_r', 'Dark2', 'Dark2_r', 'GnBu', 'GnBu_r', 'Greens', 'Greens_r', 'Greys', 'Greys_r', 'OrRd', 'OrRd_r', 'Oranges', 'Oranges_r', 'PRGn', 'PRGn_r', 'Paired', 'Paired_r', 'Pastel1', 'Pastel1_r', 'Pastel2', 'Pastel2_r', 'PiYG', 'PiYG_r', 'PuBu', 'PuBuGn', 'PuBuGn_r', 'PuBu_r', 'PuOr', 'PuOr_r', 'PuRd', 'PuRd_r', 'Purples', 'Purples_r', 'RdBu', 'RdBu_r', 'RdGy', 'RdGy_r', 'RdPu', 'RdPu_r', 'RdYlBu', 'RdYlBu_r', 'RdYlGn', 'RdYlGn_r', 'Reds', 'Reds_r', 'Set1', 'Set1_r', 'Set2', 'Set2_r', 'Set3', 'Set3_r', 'Spectral', 'Spectral_r', 'Wistia', 'Wistia_r', 'YlGn', 'YlGnBu', 'YlGnBu_r', 'YlGn_r', 'YlOrBr', 'YlOrBr_r', 'YlOrRd', 'YlOrRd_r', 'afmhot', 'afmhot_r', 'autumn', 'autumn_r', 'binary', 'binary_r', 'bone', 'bone_r', 'brg', 'brg_r', 'bwr', 'bwr_r', 'cividis', 'cividis_r', 'cool', 'cool_r', 'coolwarm', 'coolwarm_r', 'copper', 'copper_r', 'cubehelix', 'cubehelix_r', 'flag', 'flag_r', 'gist_earth', 'gist_earth_r', 'gist_gray', 'gist_gray_r', 'gist_heat', 'gist_heat_r', 'gist_ncar', 'gist_ncar_r', 'gist_rainbow', 'gist_rainbow_r', 'gist_stern', 'gist_stern_r', 'gist_yarg', 'gist_yarg_r', 'gnuplot', 'gnuplot2', 'gnuplot2_r', 'gnuplot_r', 'gray', 'gray_r', 'hot', 'hot_r', 'hsv', 'hsv_r', 'inferno', 'inferno_r', 'jet', 'jet_r', 'magma', 'magma_r', 'nipy_spectral', 'nipy_spectral_r', 'ocean', 'ocean_r', 'pink', 'pink_r', 'plasma', 'plasma_r', 'prism', 'prism_r', 'rainbow', 'rainbow_r', 'seismic', 'seismic_r', 'spring', 'spring_r', 'summer', 'summer_r', 'tab10', 'tab10_r', 'tab20', 'tab20_r', 'tab20b', 'tab20b_r', 'tab20c', 'tab20c_r', 'terrain', 'terrain_r', 'turbo', 'turbo_r', 'twilight', 'twilight_r', 'twilight_shifted', 'twilight_shifted_r', 'viridis', 'viridis_r', 'winter', 'winter_r']
#mycolors = ["darkorange", "gold", "lawngreen", "lightseagreen"]
#nodes = [0.0, 0.4, 0.8, 1.0]
#cmap2 = colors.LinearSegmentedColormap.from_list("mycmap", list(zip(nodes, mycolors)))
#cmap = cm.get_cmap(supported_cmap[-4], 10)
plt.rcParams['xtick.major.size'], plt.rcParams['ytick.major.size'] = 10, 10
plt.rcParams['xtick.minor.size'], plt.rcParams['ytick.minor.size'] = 7, 7
plt.rcParams['xtick.major.width'], plt.rcParams['ytick.major.width'] = 2, 2
plt.rcParams['xtick.minor.width'], plt.rcParams['ytick.minor.width'] = 1, 1
plt.rcParams['xtick.labelsize'], plt.rcParams['ytick.labelsize'] = 15, 15
plt.rcParams['axes.linewidth'] = 2

supported_cmap = ['Accent', 'Accent_r', 'Blues', 'Blues_r', 'BrBG', 'BrBG_r', 'BuGn', 'BuGn_r', 'BuPu', 'BuPu_r', 'CMRmap', 'CMRmap_r', 'Dark2', 'Dark2_r', 'GnBu', 'GnBu_r', 'Greens', 'Greens_r', 'Greys', 'Greys_r', 'OrRd', 'OrRd_r', 'Oranges', 'Oranges_r', 'PRGn', 'PRGn_r', 'Paired', 'Paired_r', 'Pastel1', 'Pastel1_r', 'Pastel2', 'Pastel2_r', 'PiYG', 'PiYG_r', 'PuBu', 'PuBuGn', 'PuBuGn_r', 'PuBu_r', 'PuOr', 'PuOr_r', 'PuRd', 'PuRd_r', 'Purples', 'Purples_r', 'RdBu', 'RdBu_r', 'RdGy', 'RdGy_r', 'RdPu', 'RdPu_r', 'RdYlBu', 'RdYlBu_r', 'RdYlGn', 'RdYlGn_r', 'Reds', 'Reds_r', 'Set1', 'Set1_r', 'Set2', 'Set2_r', 'Set3', 'Set3_r', 'Spectral', 'Spectral_r', 'Wistia', 'Wistia_r', 'YlGn', 'YlGnBu', 'YlGnBu_r', 'YlGn_r', 'YlOrBr', 'YlOrBr_r', 'YlOrRd', 'YlOrRd_r', 'afmhot', 'afmhot_r', 'autumn', 'autumn_r', 'binary', 'binary_r', 'bone', 'bone_r', 'brg', 'brg_r', 'bwr', 'bwr_r', 'cividis', 'cividis_r', 'cool', 'cool_r', 'coolwarm', 'coolwarm_r', 'copper', 'copper_r', 'cubehelix', 'cubehelix_r', 'flag', 'flag_r', 'gist_earth', 'gist_earth_r', 'gist_gray', 'gist_gray_r', 'gist_heat', 'gist_heat_r', 'gist_ncar', 'gist_ncar_r', 'gist_rainbow', 'gist_rainbow_r', 'gist_stern', 'gist_stern_r', 'gist_yarg', 'gist_yarg_r', 'gnuplot', 'gnuplot2', 'gnuplot2_r', 'gnuplot_r', 'gray', 'gray_r', 'hot', 'hot_r', 'hsv', 'hsv_r', 'inferno', 'inferno_r', 'jet', 'jet_r', 'magma', 'magma_r', 'nipy_spectral', 'nipy_spectral_r', 'ocean', 'ocean_r', 'pink', 'pink_r', 'plasma', 'plasma_r', 'prism', 'prism_r', 'rainbow', 'rainbow_r', 'seismic', 'seismic_r', 'spring', 'spring_r', 'summer', 'summer_r', 'tab10', 'tab10_r', 'tab20', 'tab20_r', 'tab20b', 'tab20b_r', 'tab20c', 'tab20c_r', 'terrain', 'terrain_r', 'turbo', 'turbo_r', 'twilight', 'twilight_r', 'twilight_shifted', 'twilight_shifted_r', 'viridis', 'viridis_r', 'winter', 'winter_r']


def no_integral_plot():
	'''
	Requires running "no_integral()" in onezone.py beforehand.
	'''
	fig = plt.figure(figsize =(7, 5))
	ax = fig.add_subplot(111)
	ax2 = ax.twinx()
	ax.hlines(IN.M_inf[IN.morphology], 0, IN.age_Galaxy, label=r'$M_{gal,f}$', linewidth = 1, linestyle = '-.')
	ax.semilogy(time_chosen, Mtot, label=r'$M_{tot}$', linewidth=2)
	ax.semilogy(time_chosen, Mgas_v, label= r'$M_{gas}$', linewidth=2)
	ax.semilogy(time_chosen, Mstar_v, label= r'$M_{star}$', linewidth=2)
	ax.semilogy(time_chosen, Mstar_v + Mgas_v, label= r'$M_g + M_s$', linestyle = '--')
	ax.semilogy(time_chosen, Mstar_test, label= r'$M_{star,t}$', linewidth=2, linestyle = ':')
	ax2.semilogy(time_chosen, Infall_rate, label= r'Infall', color = 'cyan', linestyle=':', linewidth=3)
	ax2.semilogy(time_chosen, SFR_v, label= r'SFR', color = 'gray', linestyle=':', linewidth=3)
	ax.set_xlim(0,13.8)
	ax.set_ylim(1e6, 1e11)
	ax2.set_ylim(1e7, 1e11)
	ax.set_xlabel(r'Age [Gyr]', fontsize = 15)
	ax.set_ylabel(r'Masses [$M_{\odot}$]', fontsize = 15)
	ax2.set_ylabel(r'Rates [$M_{\odot}/yr$]', fontsize = 15)
	ax.set_title(r'$f_{SFR} = $ %.2f' % (IN.SFR_rescaling / IN.M_inf[IN.morphology]), fontsize=15)
	ax.legend(fontsize=15, loc='lower left', ncol=2, frameon=False)
	ax2.legend(fontsize=15, loc='lower right', frameon=False)
	plt.tight_layout()
	plt.show(block=False)
	plt.savefig('./figures/total_physical.pdf')
	return None
	

def AZ_sorted_plot(cmap_name='magma_r', cbins=10):
	x = AZ_sorted[:,1]#- AZ_sorted[:,0]
	y = AZ_sorted[:,0]
	z = asplund3_percent
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
	plt.savefig('./figures/test/tracked_elements.pdf')
	return None
