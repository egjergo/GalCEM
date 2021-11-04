import numpy as np
import pandas as pd
from pandas.core.common import flatten

import prep.inputs as IN

""""""""""""""""""""""""""""""""""""""""""""""""
"                                              "
"                YIELDS CLASSES                "
" Tracks the contribution by individual events "
"       Suggested upgrade:  superclasses       "
"         for the functions in common          "
"                                              "
" LIST OF CLASSES:                             "
"	__		Isotopes                           "
"	__		Concentrations                     "
"   __      Yields                             "
"	    __		Yields_BBN                     "
"	    __		Yields_SNIa                    "
"	    __		Yields_Massive                 "
"	    __		Yields_LIMs                    "
"                                              "
""""""""""""""""""""""""""""""""""""""""""""""""

class Isotopes:
	'''
	Pick yields
	'''
	def __init__(self):
		self.elemZ = IN.periodic['elemZ']
		self.elemSymb = IN.periodic['elemSymb']
		self.elemName = IN.periodic['elemName']
		self.elemA_characteristic = IN.periodic['elemA']
		
	def pick_by_Symb(self, ndarray_elemZ, elemSymb):
		'''
		Finds the indices 
		
		Input:
			ndarray_elemZ:	the elemZ instance variable in yield classes
			elemSymb:		the periodic symbol of the given element
			
		Returns:
			the indices of the yield classes corresponding to the given element 
		'''
		print(type(self.elemSymb))
		idx = np.where(self.elemSymb == elemSymb)
		return np.where(ndarray_elemZ == self.elemZ[idx])
		
	def pick_by_ZA_sort(self, ZA_sorted, ndarray_elemZ, elemSymb):
		'''
		Finds the indices for class_instance.yields
		
		Input:
			ndarray_elemZ:	the elemZ instance variable in yield classes
			elemSymb:		the periodic symbol of the given element
			
		Returns:
			the indices of the yield classes corresponding to the given element 
		'''
		print(type(self.elemSymb))
		idx = np.where(self.elemSymb == elemSymb)
		return np.where(ndarray_elemZ == self.elemZ[idx])


class Concentrations:
	'''
	Computes the [X,Y] ratios
	and normalizes to solar metallicity (Asplund+09)
	'''
	def __init__(self):
		self.concentration = None
		return None
		
	def log10_avg_elem_vs_X(self, elemZ=1):
		'''
		log10(<M all> / <M elemZ>)
		'''
		return np.log10(np.divide(IN.periodic['elemA'],
				IN.periodic['elemA'][elemZ]))[:IN.asplund1['photospheric'].shape[0]]

	solarA09_vs_H_bynumb = (IN.asplund1['photospheric'] - IN.asplund1['photospheric'][1])
	solarA09_vs_Fe_bynumb = (IN.asplund1['photospheric'] - IN.asplund1['photospheric'][26])

	solarA09_vs_H_bymass = (IN.asplund1['photospheric'] - IN.asplund1['photospheric'][1] + 
							np.log10(np.divide(IN.periodic['elemA'],
							IN.periodic['elemA'][1]))[:IN.asplund1['photospheric'].shape[0]])
	solarA09_vs_Fe_bymass = (IN.asplund1['photospheric'] - IN.asplund1['photospheric'][26] + 
							 np.log10(np.divide(IN.periodic['elemA'],
							 IN.periodic['elemA'][26]))[:IN.asplund1['photospheric'].shape[0]])
	
	asplund3_pd = pd.DataFrame(IN.asplund3, columns=['elemN','elemZ','elemA','percentage'])
	
	def abund_percentage(self, asplund3_pd, ZA_sorted):
		percentages = []
		for ZA_idx in ZA_sorted:
			Z_select = asplund3_pd.loc[asplund3_pd['elemZ'] == ZA_idx[0]]
			ZA_select = Z_select.loc[Z_select['elemA'] == ZA_idx[1]]
			percentages.append(ZA_select.get(['percentage']).to_numpy(dtype=np.float16, na_value=0.).flatten())
		percentages_pd = np.array(percentages, dtype='object')
		for i in range(len(percentages_pd)):
			if percentages_pd[i].size <= 0:
				percentages_pd[i] = np.array([1e-5])
		return np.array(percentages_pd, dtype=np.float16)

	def extract_ZA_pairs_SNIa(self, yields):
		return np.column_stack((yields.elemZ, yields.elemA))
	
	def extract_ZA_pairs_Massive(self, yields):
		return np.column_stack((yields.elemZ, yields.elemA))
		
	def extract_ZA_pairs_LIMs(self, yields):
		return np.column_stack((yields.elemZ_sorting[0][:,0], yields.elemA_sorting[0][:,0]))
	
	def ZA_sorted(self, ZA_all):
		Z_sorted = ZA_all[ZA_all[:,0].argsort()]
		sorting_A = [] # isolating A indices with same Z
		A_sorted = [] # sorted indices for sorting_A
		for i in range(Z_sorted[:,0].max()+1):
			sorting_A.append(np.where(Z_sorted[:,0] == i)[0])
		for i in range(Z_sorted[:,0].max()+1):
			A_sorted.append(sorting_A[i][sorting_A[i].argsort()])
		ZA_sorted = Z_sorted[list(flatten(A_sorted))]
		return np.unique(ZA_sorted, axis=0)
	
	def ZA_Symb(self, ZA_sorted):
		'''
		Returns a complete list of the element symbol associated 
		with each entry in ZA_sorted
		'''
		return [np.where(IN.periodic['elemZ'] == ZA_sorted[i,0])[0][0] for i in range(len(ZA_sorted))]
		
	def R_M_i_idx(self, yields_class, ZA_sorted, Mstar=None, metallicity=None, vel=None):
		'''
		This function is able to associate the entry in isotope-wide tracked quantities
		with the respective yield. e.g.:
		
			for i in range(len(BBN_idx)):
				self.Mass_i_t[BBN_idx[i]] += yields_BBN_class.yields[i] * Minfall_dt
		'''
		yieldchoice_ZA_list = np.array(list(zip(yields_class.elemZ, yields_class.elemA)))
		yieldchoice_idx = [np.where((ZA_sorted[:,0] == yieldchoice_ZA_list[i,0]) 
					& (ZA_sorted[:,1] == yieldchoice_ZA_list[i,1]))[0][0] 
					for i in range(len(yieldchoice_ZA_list))]
		return yieldchoice_idx
		
class Yields:
	''' Parent class for all yields '''
	def __init__(self):
		self.yields = None # mass fraction of a given isotope for a given set of properties
		self.elemZ = None # Atomic number
		self.elemA = None # Mass number
		self.tables = None # variable onto which the input is saved 
		self.ZA_pairs = None # bridge variable to extract ZA_sorting (n.b. not all yields have it!)
		self.elemZ_sorting = None # Atomic number indexed by yield (n.b. not all yields have it!)
		self.elemA_sorting = None # Mass number indexed by yield (n.b. not all yields have it!)
		self.metallicity_bins = None # array of initial metallicity bins for the given authors (n.b. not all yields have it!)  
		self.stellarMass_bins = None # array of initial stellar mass bins for the given authors (n.b. not all yields have it!)
		self.metallicityIni = None # Initial stellar metallicity (n.b. not all yields have it!)
		self.stellarMassIni = None # Initial stellar mass (n.b. not all yields have it!)
		
		
class Yields_BBN(Yields):
	'''
	Yields by Big Bang Nucleosynthesis, from Galli & Palla (2013) by default.
	'''
	def __init__(self, option=IN.yields_BBN_option):
		self.option = option
		self.massCol = None
		
	def import_yields(self):
		if self.option == 'gp13':
			yd = 'input/yields/bbn/'
			
			self.tables = np.genfromtxt(yd + 'galli13.dat', names=['elemZ','elemA', 'numbFrac', 'mass'], 
						 dtype=[('elemZ', '<i8'), ('elemA', '<i8'), ('numbFrac','<f8'), ('mass','<f8')])
			self.elemA = self.tables['elemA']
			self.elemZ = self.tables['elemZ']
			self.massCol = np.multiply(self.tables['numbFrac'], self.tables['mass'])
			self.yields = np.divide(self.massCol, np.sum(self.massCol)) # fraction by mass 


class Yields_SNIa(Yields):
	'''
	Yields by Type Ia Supernova, from Iwamoto+99 by default.
	'''
	def __init__(self, option=IN.yields_SNIa_option):
		self.option = option
		
	def import_yields(self):
		if self.option == 'km20':
			yd = 'input/yields/snia/km20/'
			self.tables = np.genfromtxt(yd + 'yield_nucl.d', names=['elemName','elemA','elemZ','Yield'], 
						 dtype=[('elemName', '<U5'), ('elemA', '<i8'), ('elemZ', '<i8'), ('Yield','<f8')])
			self.yields = self.tables['Yield']
			self.elemA = self.tables['elemA']
			self.elemZ = self.tables['elemZ']
			
		if self.option == 'i99':
			yd = 'input/yields/snia/i99/'
			self.tables = np.genfromtxt(yd + 'table4.dat', 
						 names=['elemName','elemZ','elemA', 'Y_W7', 'Y_W70', 
						 		'Y_WDD1', 'Y_WDD2', 'Y_WDD3', 'Y_CDD1', 'Y_CDD2'], 
						 dtype=[('elemName', '<U5'), ('elemZ', '<i8'), 
						 ('elemA', '<i8'), ('Y_W7','<f8'), ('Y_W70','<f8'), 
						 ('Y_WDD1','<f8'), ('Y_WDD2','<f8'), ('Y_WDD3','<f8'), 
						 ('Y_CDD1','<f8'), ('Y_CDD2','<f8')])
			self.yields = self.tables['Y_CDD1'] 
			self.elemA = self.tables['elemA']
			self.elemZ = self.tables['elemZ']

				
class Yields_Massive(Yields):
	'''
	Yields by massive stars, from Limongi & Chieffi (2018) by default.
	'''
	def __init__(self, option=IN.yields_massive_option):
		self.option = option
		self.rotationalVelocity_bins = None
		
	def import_yields(self):
		if self.option == 'lc18':
			self.metallicity_bins = np.power(10, [0., -1., -2., -3.])
			self.rotationalVelocity_bins = [0., 150., 300.]
			yd = 'input/yields/snii/lc18/tab_R/'
			yieldsT = []
			yieldsTable = []
			headers = []
			
			with open(yd + 'tab_yieldstot_iso_exp.dec', 'r') as yieldsMassive:
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
				yieldsTable.append(yieldsT) 
				
			yieldsT = np.array(yieldsTable)	
			self.tables = np.reshape(yieldsT, (4,3,142,13)) 
			self.stellarMass_bins = np.array(headers[0][4:], dtype='<U3').astype('float')
			self.elemZ = self.tables[0,0,:,1].astype('int')
			self.elemA = self.tables[0,0,:,2].astype('int')
			self.stellarMassIni = self.tables[:,:,:,3].astype('float')
			self.yields = self.tables[:,:,:,4:].astype('float') 
	
	
class Yields_LIMs(Yields):
	'''
	Yields by LIMs, from Karakas et al. (2010) by default.
	'''
	def __init__(self, option=IN.yields_LIMs_option):
		self.option = option
		self.Returned_stellar_mass = None
		
	def is_unique(self, val, split_length):
		it_is = [t[val] for t in self.tables]
		unique = np.array([np.unique(ii) for ii in it_is], dtype=object) 
		it_is = [np.split(it_is[y], split_length[y]) for y in range(len(split_length))]
		it_is[0] = it_is[0][:-1]
		it_is_T = [np.array(ii).T for ii in it_is]
		if (val == 'elemA' or val == 'elemZ'):
			return it_is_T, unique[0]
		elif (val == 'Yield' or val == 'Mfin'):
			return it_is_T
		else:
			return it_is_T, unique.T[0]
		
	def import_yields(self):
		'''
		for option == 'k10': 
			lists of 4 [(77,15)*3, (77,16)]:	yields, *Ini, *_sorting
			arrays:								elemA, elemZ, *_bins
		'''
		if self.option == 'k10':
			self.metallicity_bins = [0.0001, 0.008, 0.004, 0.02]
			split_length = [16, 15, 15, 16] # self.tables length (metallicity arrays)
			yd = 'input/yields/lims/k10/'
			
			self.tables = np.array([np.genfromtxt(yd + 'Z'+str(Z)+'.dat',
				delimiter=',', names=['Mini','Zini','Mfin','elemName','elemZ','elemA',
				'Yield','Mi_windloss','Mi_ini','Xi_avg','Xi_ini','ProdFact'], 
				dtype =[('Mini', '<f8'), ('Zini', '<f8'), ('Mfin', '<f8'), ('elemName', 
				'<U4'), ('elemZ', 'i4'), ('elemA', 'i4'), ('Yield', '<f8'), 
				('Mi_windloss', '<f8'), ('Mi_ini', '<f8'), ('Xi_avg', '<f8'), 
				('Xi_ini', '<f8'), ('ProdFact', '<f8')])
				for Z in self.metallicity_bins], dtype=object)
				
			self.yields = self.is_unique('Yield', split_length)
			self.elemA_sorting, self.elemA = self.is_unique('elemA', split_length)
			self.elemZ_sorting, self.elemZ = self.is_unique('elemZ', split_length)
			self.metallicityIni, self.metallicity_bins = self.is_unique('Zini', split_length)
			self.stellarMassIni, self.stellarMass_bins = self.is_unique('Mini', split_length)
			self.Returned_stellar_mass = self.is_unique('Mfin', split_length)	