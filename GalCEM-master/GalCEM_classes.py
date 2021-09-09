import numpy as np
import scipy.integrate
import scipy.interpolate as interp
from pandas.core.common import flatten

import input_parameters as IN

'''
LIST OF CLASSES:
	__ 		Stellar_Lifetimes
	__		Infall
	__		Initial_Mass_Function
	__		Star_Formation_Rate
	__		Isotopes
	__		Concentrations
	__		Yields_LIMS
	__		Yields_Massive
	__		Yields_SNIa
'''

class Stellar_Lifetimes:
	'''
	Interpolation of Portinari+98 Table 14
	
	The first column of s_lifetimes_p98 identifies the stellar mass
	All the other columns indicate the respective lifetimes, 
	evaluated at different metallicities.
	
	'''
	def __init__(self):
		self.Z_names = ['Z0004', 'Z008', 'Z02', 'Z05']
		self.Z_binned = [0.0004, 0.008, 0.02, 0.05]
		self.Z_bins = [0.001789, 0.012649, 0.031623] # log-avg'd bins
		self.s_mass = IN.s_lifetimes_p98['M']
	
	def interp_tau(self, idx):
		tau = IN.s_lifetimes_p98[self.Z_names[idx]] / 1e9
		return interp.interp1d(self.s_mass, tau)
	def stellar_lifetimes(self):
		return [self.interp_tau(ids) for ids in range(4)]
		
	def interp_M(self, idx):
		tau = IN.s_lifetimes_p98[self.Z_names[idx]] / 1e9
		return interp.interp1d(tau, self.s_mass)
	def stellar_masses(self):
		return [self.interp_M(idx) for idx in range(4)]
	
	def interp_stellar_lifetimes(self, metallicity):
		'''
		Picks the tau(M) interpolation at the appropriate metallicity
		'''
		Z_idx = np.digitize(metallicity, self.Z_bins)
		return self.stellar_lifetimes()[Z_idx]

	def interp_stellar_masses(self, metallicity):
		'''
		Picks the M(tau) interpolation at the appropriate metallicity
		'''
		Z_idx = np.digitize(metallicity, self.Z_bins)
		return self.stellar_masses()[Z_idx]


class Infall:
	'''
	CLASS
	Computes the infall as an exponential decay
	
	EXAMPLE:
		Infall_class = Infall(morphology)
		infall = Infall_class.inf()
	(infall will contain the infall rate at every time i in time)
	'''
	def __init__(self, morphology='spiral', option=IN.inf_option, time=None):
		self.morphology = IN.morphology
		self.time = time
		self.option = option
	
	def infall_func_simple(self):
		'''
		Analytical implicit function of an exponentially decaying infall.
		Only depends on time (in Gyr)
		'''
		return lambda t: np.exp(-t / IN.tau_inf[self.morphology])
		
	def two_infall(self):
		'''
		My version of Chiappini+01
		'''
		# After radial dependence upgrade
		return None
	
	def infall_func(self):
		'''
		Picks the infall function based on the option
		'''
		if not self.option:
			return self.infall_func_simple()
		elif self.option == 'two-infall':
			return self. two_infall()
		
		return None
	
	def aInf(self):
	    """
	    Computes the infall normalization constant
	    
	    USED IN:
	    	inf() and SFR()
	    """
	    return np.divide(IN.M_inf[self.morphology], 
	    				 scipy.integrate.quad(self.infall_func(),
	                     self.time[0], self.time[-1])[0])

	def inf(self):
	    '''
	    Returns the infall array
	    '''
	    return lambda t: self.aInf() * self.infall_func()(t)
	
	def inf_cumulative(self):
		'''
		Returns the total infall mass
		'''
		return np.sumcum(self.aInf())
		

class Initial_Mass_Function:
	'''
	CLASS
	Instantiates the IMF
	
	You may define custom IMFs, paying attention to the fact that:
	
	integrand = IMF * Mstar
	$\int_{Ml}^Mu integrand dx = 1$
	
	REQUIRES
		Mass (float) as input
	
	RETURNS
		normalized IMF by calling .IMF() onto the instantiated class
	
	Accepts a custom function 'func'
	Defaults options are 'Salpeter55', 'Kroupa03', [...]
	'''
	def __init__(self, Ml, Mu, option=IN.IMF_option, custom_IMF=IN.custom_IMF):
		self.Ml = Ml
		self.Mu = Mu
		self.option = option
		self.custom = custom_IMF
	
	def Salpeter55(self, plaw=1.35):
		return lambda Mstar: Mstar ** (-(1 + plaw))
		
	def Kroupa03(self):#, C = 0.31): #if m <= 0.5: lambda m: 0.58 * (m ** -0.30)/m
		'''
		Kroupa & Weidner (2003)
		'''
		if self.mass <= 1.0:
			plaw = 1.2
		else:
			plaw = 1.7
		return lambda Mstar: Mstar ** (-(1 + plaw))
		
	def IMF_select(self):
		if not self.custom:
			if self.option == 'Salpeter55':
					return self.Salpeter55()
			if self.option == 'Kroupa03':
					return self.Kroupa03()
		if self.custom:
			return self.custom
	
	def integrand(self, Mstar):
		return Mstar * self.IMF_select()(Mstar)
		
	def normalization(self):
		return np.reciprocal(scipy.integrate.quad(self.integrand, self.Ml, self.Mu)[0])
	
	def IMF(self):
		return lambda Mstar: self.IMF_select()(Mstar) * self.normalization()
		#return lambda Mstar: self.integrand(Mstar) * self.normalization()
	
	def IMF_test(self):
		'''
		Returns the normalized integrand integral. If the IMF works, it should return 1.
		'''
		return self.normalization() * scipy.integrate.quad(self.integrand, self.Ml, self.Mu)[0]
		
		
class Star_Formation_Rate:
	'''
	CLASS
	Instantiates the SFR
	
	Accepts a custom function 'func'
	Defaults options are 'SFRgal', 'CSFR', [...]
	'''
	def __init__(self, option=IN.SFR_option, custom=IN.custom_SFR, 
				 option_CSFR=IN.CSFR_option, morphology=IN.morphology):
		self.option = option
		self.custom = custom
		self.option_CSFR = option_CSFR
		self.morphology = morphology

	def SFRgal(self, morphology, k=IN.k_SFR):
		return lambda Mgas: IN.nu[morphology] * Mgas**(-k)	
	
	def CSFR(self):
		'''
		Cosmic Star Formation rate dictionary
			'md14'		Madau & Dickinson (2014)
			'hb06'		Hopkins & Beacom (2006)
			'f07'		Fardal (2007)
			'w08'		Wilken (2008)
			'sh03'		Springel & Hernquist (2003)
		'''
		CSFR = {'md14': (lambda z: (0.015 * np.power(1 + z, 2.7)) 
						 / (1 + np.power((1 + z) / 2.9, 5.6))), 
				'hb06': (lambda z: 0.7 * (0.017 + 0.13 * z) / (1 + (z / 3.3)**5.3)), 
				'f07': (lambda z: (0.0103 + 0.088 * z) / (1 + (z / 2.4)**2.8)), 
				'w08': (lambda z: (0.014 + 0.11 * z) / (1 + (z / 1.4)**2.2)), 
				'sh03': (lambda z: (0.15 * (14. / 15) * np.exp(0.6 * (z - 5.4)) 
						 / (14.0 / 15) - 0.6 + 0.6 * np.exp((14. / 15) * (z - 5.4))))}
		return CSFR.get(self.option_CSFR, "Invalid CSFR option")
		
	def SFR(self):
		if not self.custom:
			if self.option == 'SFRgal':
					return self.SFRgal(self.morphology)
			elif self.option == 'CSFR':
				if self.option_CSFR:
					return CSFR(self.option_CSFR)
				else:
					print('Please define the CSFR option "option_CSFR"')
		if self.custom:
			return self.custom
			
	def outflow(self, Mgas, morphology, SFR=SFRgal, wind_eff=IN.wind_efficiency, k=1):
		return wind_eff * SFR(Mgas, morphology)


""""""""""""""""""""""""""""""""""""""""""""""""
"                                              "
"                YIELDS CLASSES                "
"       Suggested upgrade:  superclasses       "
"         for the functions in common          "
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

	#log10_avg_elem_vs_Fe = self.log10_avg_elem_vs_X(elemZ=26)
	#log10_avg_elem_vs_H = self.log10_avg_elem_vs_X(elemZ=1)
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
		
	def extract_AZ_pairs_LIMS(self, yields):
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
		
	def import_yields(self):
		if self.option == 'gp13':
			yd = 'input/yields/bbn/'
			
			self.tables = np.genfromtxt(yd + 'galli13.dat', 
						 names=['elemZ','elemA', 'Yield'], 
						 
						 dtype=[('elemZ', '<i8'), 
						 ('elemA', '<i8'), ('Yield','<f8')])
						 
			self.yields = self.tables['Yield']
			self.elemA = self.tables['elemA']
			self.elemZ = self.tables['elemZ']


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
						 names=['elemName','elemA','elemZ', 'Y_W7', 'Y_W70', 
						 		'Y_WDD1', 'Y_WDD2', 'Y_WDD3', 'Y_CDD1', 'Y_CDD2'], 
						 dtype=[('elemName', '<U5'), ('elemA', '<i8'), 
						 ('elemZ', '<i8'), ('Y_W7','<f8'), ('Y_W70','<f8'), 
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
			folder = 'tab_R'
			iso_or_ele = 'iso'
			
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
	
	
class Yields_LIMS:
	'''
	Karakas et al. (2010) by default
	'''
	def __init__(self, option = IN.yields_LIMS_option):
		self.option = option
		self.metallicity_bins = None
		self.stellarMass_bins = None
		self.tables = None
		self.yields = None
		self.elemA = None
		self.elemZ = None
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

class Auxiliary:
	def varname( var, dir=locals()):
  		return [ key for key, val in dir.items() if id( val) == id( var)][0]
  
	def is_monotonic(arr):
		print ("for ", varname(arr)) 
		if all(arr[i] <= arr[i + 1] for i in range(len(arr) - 1)): 
			return "monotone increasing" 
		elif all(arr[i] >= arr[i + 1] for i in range(len(arr) - 1)):
			return "monotone decreasing"
		return "not monotonic array"
	
	def find_nearest(array, value):
		array = np.asarray(array)
		idx = (np.abs(array - value)).argmin()
		return idx

	def age_from_z(zf, OmegaLambda0 = 0.7, Omegam0 = 0.3, Omegar0 = 1e-4):
		'''
		Finds the age (or lookback time) given a redshift (or scale factor).
		Assumes flat LCDM. Std cosmology, zero curvature. z0 = 0, a0 = 1.
		Omegam0 = 0.28 # 0.24 DM + 0.04 baryonic
		
		INPUT:	
		(aem = Float. Scale factor.)
		(OmegaLambda0 = 0.72, Omegam0 = 0.28)
		zf  = redshift
		
		OUTPUT
		lookback time.
		'''
		#zf = np.reciprocal(np.float(aem))-1
		H0 = 100 * OmegaLambda0 * 3.24078e-20 * 3.15570e16 
		# [ km s^-1 Mpc^-1 * Mpc km^-1 * s Gyr^-1 ]
		#age0 = integrate.quad(lambda z: 1 / ( (z + 1) *np.sqrt(OmegaLambda0 
		#						+ Omegam0 * (z+1)**3 + Omegar0 * (z+1)**4) ),
		#						 0, np.inf)[0] / H0 # Since BB [Gyr]
		age = integrate.quad(lambda z: 1 / ( (z + 1) *np.sqrt(OmegaLambda0 + 
								Omegam0 * (z+1)**3 + Omegar0 * (z+1)**4) ), 
								zf, np.inf)[0] / H0 # Since BB [Gyr]
		return age #age0 - age # for the lookback time # end fatot()
