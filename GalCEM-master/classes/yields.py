import numpy as np
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
"	__		Yields_BBN                         "
"	__		Yields_SNIa                        "
"	__		Yields_Massive                     "
"	__		Yields_LIMs                        "
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
		
	def pick_by_AZ_sort(self, AZ_sorted, ndarray_elemZ, elemSymb):
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

	def extract_AZ_pairs_SNIa(self, yields):
		return np.column_stack((yields.elemZ, yields.elemA))
	
	def extract_AZ_pairs_Massive(self, yields):
		return np.column_stack((yields.elemZ, yields.elemA))
		
	def extract_AZ_pairs_LIMs(self, yields):
		return np.column_stack((yields.elemZ_sorting[0][:,0], yields.elemA_sorting[0][:,0]))
	
	def AZ_sorted(self, AZ_all):
		Z_sorted = AZ_all[AZ_all[:,0].argsort()]
		sorting_A = [] # isolating A indices with same Z
		A_sorted = [] # sorted indices for sorting_A
		for i in range(Z_sorted[:,0].max()+1):
			sorting_A.append(np.where(Z_sorted[:,0] == i)[0])
		for i in range(Z_sorted[:,0].max()+1):
			A_sorted.append(sorting_A[i][sorting_A[i].argsort()])
		AZ_sorted = Z_sorted[list(flatten(A_sorted))]
		return np.unique(AZ_sorted, axis=0)
	
	def AZ_Symb(self, AZ_sorted):
		'''
		Returns a complete list of the element symbol associated 
		with each entry in AZ_sorted
		'''
		return [np.where(IN.periodic['elemZ'] == AZ_sorted[i,0])[0][0] for i in range(len(AZ_sorted))]
		
	def R_M_i_idx(self, yields_class, AZ_sorted, Mstar=None, metallicity=None, vel=None):
		'''
		This function is able to associate the entry in isotope-wide tracked quantities
		with the respective yield. e.g.:
		
			for i in range(len(BBN_idx)):
				self.Mass_i_t[BBN_idx[i]] += yields_BBN_class.yields[i] * Minfall_dt
		'''
		yieldchoice_AZ_list = np.array(list(zip(yields_class.elemZ, yields_class.elemA)))
		yieldchoice_idx = [np.where((AZ_sorted[:,0] == yieldchoice_AZ_list[i,0]) 
					& (AZ_sorted[:,1] == yieldchoice_AZ_list[i,1]))[0][0] 
					for i in range(len(yieldchoice_AZ_list))]
		return yieldchoice_idx
		
		
class Yields_BBN:
	'''
	Big Bang Nucleosynthesis yields from Galli & Palla (2013)
	'''
	def __init__(self, option = IN.yields_BBN_option):
		self.option = option
		self.tables = None
		self.yields = None
		self.elemA = None
		self.elemZ = None
		self.mass = None
		self.massCol = None
		self.numberFrac = None
		self.totMass = None
		
	def import_yields(self):
		if self.option == 'gp13':
			yd = 'input/yields/bbn/'
			
			self.tables = np.genfromtxt(yd + 'galli13.dat', 
						 names=['elemZ','elemA', 'numbFrac', 'mass'], 
						 
						 dtype=[('elemZ', '<i8'), 
						 ('elemA', '<i8'), ('numbFrac','<f8'), ('mass','<f8')])
						 
			self.numberFrac = self.tables['numbFrac'] # fraction by number
			self.elemA = self.tables['elemA']
			self.elemZ = self.tables['elemZ']
			self.mass = self.tables['mass']
			self.massCol = np.multiply(self.numberFrac, self.mass)
			self.totMass = np.sum(self.massCol)
			self.yields = np.divide(self.massCol, self.totMass) # fraction by mass 


class Yields_SNIa:
	'''
	Kusakabe's yields
	'''
	def __init__(self, option = IN.yields_SNIa_option):
		self.option = option
		self.tables = None
		self.yields = None
		self.elemA = None
		self.elemZ = None
		self.AZ_pairs = None
		
	def import_yields(self):
		if self.option == 'km20':
			yd = 'input/yields/snia/km20/'
			self.tables = np.genfromtxt(yd + 'yield_nucl.d', 
						 names=['elemName','elemA','elemZ','Yield'], 
						 dtype=[('elemName', '<U5'), ('elemA', '<i8'), 
						 ('elemZ', '<i8'), ('Yield','<f8')])
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

				
class Yields_Massive:
	'''
	Limongi & Chieffi (2018) by default
	'''
	def __init__(self, option = IN.yields_massive_option):
		self.option = option
		self.metallicity_bins = None
		self.stellarMass_bins = None
		self.rotationalVelocity_bins = None
		self.tables = None
		self.yields = None
		self.elemA = None
		self.elemZ = None
		self.AZ_pairs = None
		self.metallicityIni = None
		self.stellarMassIni = None
		self.allHeaders = None # All columns in self.tables
		
	def import_yields(self):
		if self.option == 'lc18':
			self.metallicityIni = [0., -1., -2., -3.]
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
			self.allHeaders = np.array(headers[0][4:], dtype='<U3').astype('float')
			self.elemZ = self.tables[0,0,:,1].astype('int')
			self.elemA = self.tables[0,0,:,2].astype('int')
			self.stellarMassIni = self.tables[:,:,:,3].astype('float')
			self.yields = self.tables[:,:,:,4:].astype('float') 
	
	
class Yields_LIMs:
	'''
	Karakas et al. (2010) by default
	'''
	def __init__(self, option = IN.yields_LIMs_option):
		self.option = option
		self.metallicity_bins = None
		self.stellarMass_bins = None
		self.tables = None
		self.yields = None
		self.elemA = None
		self.elemZ = None
		self.Returned_stellar_mass = None
		self.AZ_pairs = None
		self.elemA_sorting = None
		self.elemZ_sorting = None
		self.metallicityIni = None
		self.stellarMassIni = None
		
	def is_unique(self, val, split_length):
		it_is = [t[val] for t in self.tables]
		unique = np.array([np.unique(ii) for ii in it_is], dtype=object) 
		it_is = [np.split(it_is[y], split_length[y]) for y in range(len(split_length))]
		it_is[0] = it_is[0][:-1]
		it_is_T = [np.array(ii).T for ii in it_is]
		if (val == 'elemA' or val == 'elemZ'):
			return it_is_T, unique[0]
		elif val == 'Yield':
			return it_is_T
		else:
			return it_is_T, unique
		
	def import_yields(self):
		if self.option == 'k10':
			self.metallicity_bins = [0.0001, 0.008, 0.004, 0.02]
			split_length = [16, 15, 15, 16]
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
			#self.Returned_stellar_mass =
			