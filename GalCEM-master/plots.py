""" I only achieve simplicity with enormous effort (Clarice Lispector) """
import time
tic = []
tic.append(time.process_time())
import math as m
import numpy as np
import scipy.integrate
import scipy.interpolate as interp
from scipy.integrate import quad
from scipy.misc import derivative

import prep.inputs as IN
import classes.morphology as morph
import classes.yields as Y
from prep.setup import *

import matplotlib.pylab as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as colors
import matplotlib.ticker as ticker
plt.rcParams['xtick.major.size'], plt.rcParams['ytick.major.size'] = 10, 10
plt.rcParams['xtick.minor.size'], plt.rcParams['ytick.minor.size'] = 7, 7
plt.rcParams['xtick.major.width'], plt.rcParams['ytick.major.width'] = 2, 2
plt.rcParams['xtick.minor.width'], plt.rcParams['ytick.minor.width'] = 1, 1
plt.rcParams['xtick.labelsize'], plt.rcParams['ytick.labelsize'] = 15, 15
plt.rcParams['axes.linewidth'] = 2

def no_integral_plot():
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
	ax.set_ylim(1e5, 1e11)
	#ax2.set_ylim(1e5, 1e11)
	ax2.set_ylim(1e8, 1e11)
	ax.set_xlabel(r'Age [Gyr]', fontsize = 15)
	ax.set_ylabel(r'Masses [$M_{\odot}$]', fontsize = 15)
	ax2.set_ylabel(r'Rates [$M_{\odot}/yr$]', fontsize = 15)
	#ax.set_title(r'$\alpha_{SFR} = $ %.1E' % (IN.SFR_rescaling), fontsize = 15)
	ax.set_title(r'$f_{SFR} = $ %.2f' % (IN.SFR_rescaling / IN.M_inf[IN.morphology]), fontsize=15)
	ax.legend(fontsize=15, loc='lower left', frameon=False)
	ax2.legend(fontsize=15, loc='lower right', frameon=False)
	plt.tight_layout()
	plt.show(block=False)
	plt.savefig('./figures/total_physical.pdf')
	
def AZ_sorted_plot():
	fig = plt.figure(figsize =(5, 7))
	ax = fig.add_subplot(111)
	ax.grid(True, which='major', linestyle='--', linewidth=1, color='grey', alpha=0.5)
	ax.grid(True, which='minor', linestyle=':', linewidth=1, color='grey', alpha=0.5)
	ax.scatter(AZ_sorted[:,1]- AZ_sorted[:,0], AZ_sorted[:,0], marker='o', alpha=0.2, color='r', s=20)
	ax.set_ylabel(r'Proton (Atomic) Number Z', fontsize = 15)
	ax.set_xlabel(r'Neutron Number N', fontsize = 15)
	ax.set_title(r'Tracked elements', fontsize=15)
	ax.xaxis.set_major_locator(ticker.MultipleLocator(20))
	ax.yaxis.set_major_locator(ticker.MultipleLocator(20))
	ax.xaxis.set_minor_locator(ticker.MultipleLocator(5))
	ax.yaxis.set_minor_locator(ticker.MultipleLocator(5))
	ax.tick_params(width = 2, length = 10)
	ax.tick_params(width = 1, length = 5, which = 'minor')
	plt.tight_layout()
	plt.show(block=False)
	plt.savefig('./figures/test/tracked_elements.pdf')
	
def run():
	no_integral()
	no_integral_plot()
#run()