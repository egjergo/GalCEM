# Read Orfeo yields specific for 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.colors as mcolors
plt.rcParams['xtick.labelsize'] = 13
plt.rcParams['ytick.labelsize'] = 13
plt.rcParams['axes.linewidth'] = 1
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
plt.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
plt.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'

yieldsT = []
yieldsTable = []
headers = []
folder = 'tab_R'
iso_or_ele = 'iso'
with open(folder+'/tab_yieldstot_'+iso_or_ele+'_exp.dec', 'r') as yieldsMassive:
	for line in yieldsMassive:
		if 'ele' not in line:
			types = np.concatenate([['<U4', 'i4', 'i4'], (len(headers[-1]) - 3) * ['f8']])
			dtypes = list(zip(headers[-1],types))
			#l = np.array(line.split())#, dtype=dtypes)
			l = line.split()
			yieldsT.append(l)
		else:
			headers.append(line.split())
			if yieldsT:
				yieldsTable.append(yieldsT)
			yieldsT = []
	yieldsTable.append(yieldsT) # last entry	

yieldsT = np.array(yieldsTable)

# yieldsT has shape:	[vel-metallicity block, element line, entry]
yieldsLC18 = np.reshape(yieldsT, (4,3,142,13)) # [metallicity, vel, elem line, mass]
#yieldsLC18[0,1,0,:] == yieldsT[1,0,:] 
#yieldsLC18[1,0,0,:] == yieldsT[3,0,:]
#yieldsLC18[1,2,0,:] == yieldsT[5,0,:]
#yieldsLC18[3,2,0,:] == yieldsT[11,0,:]

"""

Done with preamble

"""


#masses = headers[0,4:].astype('<U3')	
#masses = [0.] + np.array(headers[0][4:], dtype='<U3').astype('float')
masses = np.array(headers[0][4:], dtype='<U3').astype('float')
atomicLabel = yieldsLC18[0,0,:,0].astype('str')
atomicNumb = yieldsLC18[0,0,:,1].astype('int')
atomicMass = yieldsLC18[0,0,:,2].astype('int')
initialMass= yieldsLC18[:,:,:,3].astype('float')
FeH_initial = [0., -1., -2., -3.]
Massive_vel = [0., 150., 300.]
yieldGrid = yieldsLC18[:,:,:,4:].astype('float')




yieldsT = []
yieldsTable = []
headers = []
with open(folder+'/tab_yieldstot_ele_exp.dec', 'r') as yieldsMassive:
	for line in yieldsMassive:
		if 'ele' not in line:
			types = np.concatenate([['<U4', 'i4', 'i4'], (len(headers[-1]) - 3) * ['f8']])
			dtypes = list(zip(headers[-1],types))
			#l = np.array(line.split())#, dtype=dtypes)
			l = line.split()
			yieldsT.append(l)
		else:
			headers.append(line.split())
			if yieldsT:
				yieldsTable.append(yieldsT)
			yieldsT = []
	yieldsTable.append(yieldsT) # last entry	

yieldsTele = np.array(yieldsTable)
yieldsLC18ele = np.reshape(yieldsTele, (4,3,53,13)) # [metallicity, vel, elem line, mass]
atomicNumbele = yieldsLC18ele[0,0,:,1].astype('int')
yieldGridele = yieldsLC18ele[:,:,:,4:].astype('float')

	
colors = mcolors.CSS4_COLORS
by_hsv = sorted((tuple(mcolors.rgb_to_hsv(mcolors.to_rgb(color))), name)
                        for name, color in colors.items())
color_names = [name for hsv, name in by_hsv]
color_list = [  0, 18, 24, 
			   79, 69,102, 
			  113,120,130]
#color_listh = np.flip(['#ff0066', '#ff4400', '#ff9900',
#			  '#bbff00', '#66ff00', '#00ffee',
#			  '#00bbff', '#0066ff', '#000000'])
color_listh = np.flip(['#ff0066', '#ff0066', '#ff0066',
			  		   '#0066ff', '#0066ff', '#0066ff',
			  		   '#000000', '#000000', '#000000'])
linestyleh = ['-', '--', ':', '-', '--', ':', '-', '--', ':']
	
def plot_yields(nrow=3, ncol=3, FeH_ini=0, vel=0, figsiz = (10,8)):
	fig, ax = plt.subplots(nrow, ncol, figsize=figsiz)	
	mass_i = 0
	for i in range(nrow):
		for j in range(ncol):
			ax[i,j].scatter(atomicNumb, np.log10(yieldGrid[FeH_ini,vel,:,mass_i]), alpha=0.4,
					label=r'$M_{*}$ = '+str(masses[mass_i]), color=colors[color_names[color_list[mass_i]]])
			ax[i,j].plot(atomicNumbele, np.log10(yieldGridele[FeH_ini,vel,:,mass_i]), alpha=1,
					color=colors[color_names[color_list[mass_i]]])
			ax[i,j].legend(loc='upper right', frameon=True, framealpha=0.6, fontsize=15, markerfirst=False)
			
			if j != 0:
				ax[i,j].set_yticklabels([])
			else:
				if i == nrow // 2:
					if nrow % 2 != 0:
						ax[i,j].set_ylabel(r'Logarithmic mass-weighted yields', fontsize = 20)
					else:
						ax[i,j].set_ylabel(r'Logarithmic mass-weighted yields', fontsize = 20, y = 1.05)
			if i != nrow-1:
				ax[i,j].set_xticklabels([])
			else:
				if j == ncol // 2:
					if ncol % 2 != 0:
						ax[i,j].set_xlabel('Atomic Number', fontsize = 20)
					else:
						ax[i,j].set_xlabel('Atomic Number', fontsize = 20, x = 0.05)
			ax[i,j].xaxis.set_minor_locator(ticker.MultipleLocator(base=5))
			ax[i,j].tick_params(width = 1, length = 3, axis = 'x', which = 'minor', 
												bottom = True, top = True, direction = 'in')
			ax[i,j].yaxis.set_minor_locator(ticker.MultipleLocator(base=1))
			ax[i,j].tick_params(width = 1, length = 3, axis = 'y', which = 'minor', 
												left = True, right = True, direction = 'in')
			ax[i,j].xaxis.set_major_locator(ticker.MultipleLocator(base=25))
			ax[i,j].tick_params(width = 1, length = 5, axis = 'x', which = 'major', 
												bottom = True, top = True, direction = 'in')
			ax[i,j].yaxis.set_major_locator(ticker.MultipleLocator(base=5))
			ax[i,j].tick_params(width = 1, length = 5, axis = 'y', which = 'major', 
												left = True, right = True, direction = 'in')
			mass_i += 1
			ax[i,j].set_xlim(0.5,85.5)
			ax[i,j].set_ylim(-17,2)
			#if (i > i_del or j > j_del): 
			#	fig.delaxes(ax[i,j])
	plt.suptitle(r'Limongi & Chieffi (2018), [Fe/H]$_{ini}$ = '+str(FeH_initial[FeH_ini])+r', $v_{rot}$ = '+str(Massive_vel[vel])+' km/s', fontsize=17)
	plt.tight_layout(rect = [0, 0, 1, .98])
	plt.subplots_adjust(wspace=0., hspace=0.)
	#plt.show(block = False)
	plt.savefig('PLOTS/yields_'+folder+'_'+iso_or_ele+'_FeH'+str(FeH_ini)+'_vel'+str(vel)+'.pdf')
#plot_yields(FeH_ini=0, vel=0)



def plot_yields_vel(nrow=3, ncol=3, FeH_ini=0, figsiz = (10,8)):
	fig, ax = plt.subplots(nrow, ncol, figsize=figsiz)	
	mass_i = 0
	for i in range(nrow):
		for j in range(ncol):
			#ax[i,j].scatter(atomicNumb, np.log10(yieldGrid[FeH_ini,vel,:,mass_i]), alpha=0.4,
			#		label=r'$M_{*}$ = '+str(masses[mass_i]), color=colors[color_names[color_list[mass_i]]])
			ax[i,j].plot(atomicNumbele, np.log10(yieldGridele[FeH_ini,0,:,mass_i]), alpha=1,
					label=r'$M_{*}$ = '+str(masses[mass_i]), color=colors[color_names[color_list[mass_i]]])
			ax[i,j].plot(atomicNumbele, np.log10(yieldGridele[FeH_ini,1,:,mass_i]), alpha=1,
					color=colors[color_names[color_list[mass_i]]], linestyle = ':')
			ax[i,j].plot(atomicNumbele, np.log10(yieldGridele[FeH_ini,2,:,mass_i]), alpha=1,
					color=colors[color_names[color_list[mass_i]]], linestyle = '--')
			ax[i,j].legend(loc='upper right', frameon=True, framealpha=0.6, fontsize=15, markerfirst=False)
			
			if j != 0:
				ax[i,j].set_yticklabels([])
			else:
				if i == nrow // 2:
					if nrow % 2 != 0:
						ax[i,j].set_ylabel(r'Logarithmic mass-weighted yields', fontsize = 20)
					else:
						ax[i,j].set_ylabel(r'Logarithmic mass-weighted yields', fontsize = 20, y = 1.05)
			if i != nrow-1:
				ax[i,j].set_xticklabels([])
			else:
				if j == ncol // 2:
					if ncol % 2 != 0:
						ax[i,j].set_xlabel('Atomic Number', fontsize = 20)
					else:
						ax[i,j].set_xlabel('Atomic Number', fontsize = 20, x = 0.05)
			ax[i,j].xaxis.set_minor_locator(ticker.MultipleLocator(base=5))
			ax[i,j].tick_params(width = 1, length = 3, axis = 'x', which = 'minor', 
												bottom = True, top = True, direction = 'in')
			ax[i,j].yaxis.set_minor_locator(ticker.MultipleLocator(base=1))
			ax[i,j].tick_params(width = 1, length = 3, axis = 'y', which = 'minor', 
												left = True, right = True, direction = 'in')
			ax[i,j].xaxis.set_major_locator(ticker.MultipleLocator(base=25))
			ax[i,j].tick_params(width = 1, length = 5, axis = 'x', which = 'major', 
												bottom = True, top = True, direction = 'in')
			ax[i,j].yaxis.set_major_locator(ticker.MultipleLocator(base=5))
			ax[i,j].tick_params(width = 1, length = 5, axis = 'y', which = 'major', 
												left = True, right = True, direction = 'in')
			mass_i += 1
			ax[i,j].set_xlim(0.5,85.5)
			ax[i,j].set_ylim(-17,2)
			#if (i > i_del or j > j_del): 
			#	fig.delaxes(ax[i,j])
	plt.suptitle(r'Limongi & Chieffi (2018), [Fe/H]$_{ini}$ = '+str(FeH_initial[FeH_ini])+r", $v_{rot}$ = '-' 0 km/s, ':' 150 km/s, '--' 300 km/s", fontsize=17)
	plt.tight_layout(rect = [0, 0, 1, .98])
	plt.subplots_adjust(wspace=0., hspace=0.)
	#plt.show(block = False)
	plt.savefig('PLOTS/yields_'+folder+'_FeH'+str(FeH_ini)+'.pdf')
#plot_yields_vel(FeH_ini=0)




def plot_yields_FeH(nrow=3, ncol=3, vel=0, figsiz = (10,8)):
	fig, ax = plt.subplots(nrow, ncol, figsize=figsiz)	
	mass_i = 0
	for i in range(nrow):
		for j in range(ncol):
			#ax[i,j].scatter(atomicNumb, np.log10(yieldGrid[FeH_ini,vel,:,mass_i]), alpha=0.4,
			#		label=r'$M_{*}$ = '+str(masses[mass_i]), color=colors[color_names[color_list[mass_i]]])
			ax[i,j].plot(atomicNumbele, np.log10(yieldGridele[0,vel,:,mass_i]), alpha=1,
					label=r'$M_{*}$ = '+str(masses[mass_i]), color=colors[color_names[color_list[mass_i]]])
			ax[i,j].plot(atomicNumbele, np.log10(yieldGridele[1,vel,:,mass_i]), alpha=1,
					color=colors[color_names[color_list[mass_i]]], linestyle = ':')
			ax[i,j].plot(atomicNumbele, np.log10(yieldGridele[2,vel,:,mass_i]), alpha=1,
					color=colors[color_names[color_list[mass_i]]], linestyle = '--')
			ax[i,j].plot(atomicNumbele, np.log10(yieldGridele[3,vel,:,mass_i]), alpha=1,
					color=colors[color_names[color_list[mass_i]]], linestyle = '-.')
			ax[i,j].legend(loc='upper right', frameon=True, framealpha=0.6, fontsize=15, markerfirst=False)
			
			if j != 0:
				ax[i,j].set_yticklabels([])
			else:
				if i == nrow // 2:
					if nrow % 2 != 0:
						ax[i,j].set_ylabel(r'Logarithmic mass-weighted yields', fontsize = 20)
					else:
						ax[i,j].set_ylabel(r'Logarithmic mass-weighted yields', fontsize = 20, y = 1.05)
			if i != nrow-1:
				ax[i,j].set_xticklabels([])
			else:
				if j == ncol // 2:
					if ncol % 2 != 0:
						ax[i,j].set_xlabel('Atomic Number', fontsize = 20)
					else:
						ax[i,j].set_xlabel('Atomic Number', fontsize = 20, x = 0.05)
			ax[i,j].xaxis.set_minor_locator(ticker.MultipleLocator(base=5))
			ax[i,j].tick_params(width = 1, length = 3, axis = 'x', which = 'minor', 
												bottom = True, top = True, direction = 'in')
			ax[i,j].yaxis.set_minor_locator(ticker.MultipleLocator(base=1))
			ax[i,j].tick_params(width = 1, length = 3, axis = 'y', which = 'minor', 
												left = True, right = True, direction = 'in')
			ax[i,j].xaxis.set_major_locator(ticker.MultipleLocator(base=25))
			ax[i,j].tick_params(width = 1, length = 5, axis = 'x', which = 'major', 
												bottom = True, top = True, direction = 'in')
			ax[i,j].yaxis.set_major_locator(ticker.MultipleLocator(base=5))
			ax[i,j].tick_params(width = 1, length = 5, axis = 'y', which = 'major', 
												left = True, right = True, direction = 'in')
			mass_i += 1
			ax[i,j].set_xlim(0.5,85.5)
			ax[i,j].set_ylim(-17,2)
			#if (i > i_del or j > j_del): 
			#	fig.delaxes(ax[i,j])
	plt.suptitle(r'Limongi & Chieffi (2018), $v_{rot}$ = '+str(Massive_vel[vel])+r" km/s, [Fe/H] = '-' 0, ':' -1, '--' -2, '-.' -3", fontsize=17)
	plt.tight_layout(rect = [0, 0, 1, .98])
	plt.subplots_adjust(wspace=0., hspace=0.)
	#plt.show(block = False)
	plt.savefig('PLOTS/yields_'+folder+'_vel'+str(vel)+'.pdf')
#plot_yields_FeH(vel=0)
#plot_yields_FeH(vel=1)
#plot_yields_FeH(vel=2)







def plot_yields_mass(nrow=3, ncol=4, figsiz = (11,8)):
	fig, ax = plt.subplots(nrow, ncol, figsize=figsiz)	
	mass_i = 0
	for i in range(nrow):
		for j in range(ncol):
			#ax[i,j].scatter(atomicNumb, np.log10(yieldGrid[FeH_ini,vel,:,mass_i]), alpha=0.4,
			#		label=r'$M_{*}$ = '+str(masses[mass_i]), color=colors[color_names[color_list[mass_i]]])
			for mass_i in range(len(masses)):
				ax[i,j].plot(atomicNumbele, np.log10(yieldGridele[j,i,:,mass_i]), alpha=1,
					label=r'$M_{*}$ = '+str(masses[mass_i]), color=color_listh[mass_i], linestyle=linestyleh[mass_i], linewidth=1)
				ax[i,j].text(40, -15, 'vel='+str(Massive_vel[i])+', [Fe/H]='+str(FeH_initial[j]), fontsize = 10, 
								horizontalalignment='center', verticalalignment='center')
			#ax[i,j].plot(atomicNumbele, np.log10(yieldGridele[1,vel,:,mass_i]), alpha=1,
			#		color=colors[color_names[color_list[mass_i]]], linestyle = ':')
			#ax[i,j].plot(atomicNumbele, np.log10(yieldGridele[2,vel,:,mass_i]), alpha=1,
			#		color=colors[color_names[color_list[mass_i]]], linestyle = '--')
			#ax[i,j].plot(atomicNumbele, np.log10(yieldGridele[3,vel,:,mass_i]), alpha=1,
			#		color=colors[color_names[color_list[mass_i]]], linestyle = '-.')
			if (i == 0 and j == 0):
					ax[i,j].legend(bbox_to_anchor=(0.0, 1.3), loc='upper left', 
								borderaxespad=0., ncol = 5, fontsize = 13, frameon = False)
			#ax[i,j].legend(loc='upper right', frameon=True, framealpha=0.6, fontsize=15, markerfirst=False)
			
			if j != 0:
				ax[i,j].set_yticklabels([])
			else:
				if i == nrow // 2:
					if nrow % 2 != 0:
						ax[i,j].set_ylabel(r'Logarithmic mass-weighted yields', fontsize = 20)
					else:
						ax[i,j].set_ylabel(r'Logarithmic mass-weighted yields', fontsize = 20, y = 1.05)
			if i != nrow-1:
				ax[i,j].set_xticklabels([])
			else:
				if j == ncol // 2:
					if ncol % 2 != 0:
						ax[i,j].set_xlabel('Atomic Number', fontsize = 20)
					else:
						ax[i,j].set_xlabel('Atomic Number', fontsize = 20, x = 0.05)
			ax[i,j].xaxis.set_minor_locator(ticker.MultipleLocator(base=5))
			ax[i,j].tick_params(width = 1, length = 3, axis = 'x', which = 'minor', 
												bottom = True, top = True, direction = 'in')
			ax[i,j].yaxis.set_minor_locator(ticker.MultipleLocator(base=1))
			ax[i,j].tick_params(width = 1, length = 3, axis = 'y', which = 'minor', 
												left = True, right = True, direction = 'in')
			ax[i,j].xaxis.set_major_locator(ticker.MultipleLocator(base=25))
			ax[i,j].tick_params(width = 1, length = 5, axis = 'x', which = 'major', 
												bottom = True, top = True, direction = 'in')
			ax[i,j].yaxis.set_major_locator(ticker.MultipleLocator(base=5))
			ax[i,j].tick_params(width = 1, length = 5, axis = 'y', which = 'major', 
												left = True, right = True, direction = 'in')
			ax[i,j].set_xlim(0.5,85.5)
			ax[i,j].set_ylim(-17,2)
			#if (i > i_del or j > j_del): 
			#	fig.delaxes(ax[i,j])
	plt.suptitle(r'Limongi & Chieffi (2018)', fontsize=17)
	plt.tight_layout(rect = [0, 0, 1, .92])
	plt.subplots_adjust(wspace=0., hspace=0.)
	#plt.show(block = False)
	plt.savefig('PLOTS/yields'+folder+'_mass.pdf')
plot_yields_mass()