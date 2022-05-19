import numpy as np
import pandas as pd
from pandas.core.common import flatten
import os

""""""""""""""""""""""""""""""""""""""""""""""""""""""
"                                                    "
"                YIELDS CLASSES                      "
" Tracks the contribution by individual events       "
"       Suggested upgrade:  superclasses             "
"         for the functions in common                "
"                                                    "
" LIST OF CLASSES:                                   "
"    __        Isotopes                              "
"    __        Concentrations                        "
"    __        Yields (parent class to Yields_*)     "
"    __        Yields_BBN (subclass)                 "
"    __        Yields_SNIa (subclass)                "
"    __        Yields_SNII (subclass)                "
"    __        Yields_LIMs (subclass)                "
"                                                    "
""""""""""""""""""""""""""""""""""""""""""""""""""""""

class Isotopes:
    ''' Handles the isotope and yield selection '''
    def __init__(self, IN):
        self.IN = IN
        self.elemZ = self.IN.periodic['elemZ']
        self.elemSymb = self.IN.periodic['elemSymb']
        self.elemName = self.IN.periodic['elemName']
        self.elemA_characteristic = self.IN.periodic['elemA']
    
    def pick_i_by_iso(self, ZA_sorted, elemZ, elemA):
        '''Finds the isotope entry in the isotope list'''
        idx_Z = np.where(ZA_sorted[:,0] == elemZ)[0]
        idx_A = np.where(ZA_sorted[:,1] == elemA)[0]
        idx =  np.intersect1d(idx_Z, idx_A)[0]
        print("[Z, A] = ", ZA_sorted[idx])
        return idx
        
    def pick_i_by_atomicnumber(self, ZA_sorted, elemZ):
        '''Finds the isotope entry/entries in the isotope list having a certain atomic number'''
        idx_Z = np.where(ZA_sorted[:,0] == elemZ)[0]
        print(ZA_sorted[idx_Z])
        return idx_Z
            
    def pick_i_by_atomicmass(self, ZA_sorted, elemA):
        '''Finds the isotope entry/entries in the isotope list having a certain atomic mass'''
        idx_A = np.where(ZA_sorted[:,1] == elemA)[0]
        print(ZA_sorted[idx_A])
        return idx_A
                
    def pick_i_by_Symbol(self, ZA_sorted, elemSymb):
        '''Finds the isotope entry/entries in the isotope list having a certain atomic mass'''
        id_periodic = np.where(self.elemSymb == elemSymb)[0]
        idx_Z = np.where(ZA_sorted[:,0] == self.elemZ[id_periodic].values)[0]
        print(ZA_sorted[idx_Z])
        return idx_Z


class Concentrations:
    '''
    Computes the [X,Y] ratios
    and normalizes to solar metallicity (Asplund+09)
    '''
    def __init__(self, IN):
        self.IN = IN
        self.concentration = None
        self.solarA09_photospheric = self.IN.asplund1['photospheric']
        self.solarA09_meteoric = self.IN.asplund1['meteoric']
        id_A09nan = np.where(np.isnan(self.solarA09_photospheric)==True)[0]
        self.solarA09 = self.solarA09_photospheric
        idx_Ap_intersect = np.intersect1d(self.IN.periodic['elemZ'].values, 
                                          self.IN.asplund1['elemZ'].values, return_indices=True)[1] # selects idx for periodic
        self.select_periodic = self.IN.periodic.iloc[idx_Ap_intersect, :]
        self.solarA09[id_A09nan] = self.solarA09_meteoric[id_A09nan]
        self.solarA09_vs_H_bynumb = (self.solarA09 - self.solarA09[1])
        self.solarA09_vs_Fe_bynumb = (self.solarA09 - self.solarA09[26])
        self.solarA09_vs_H_bymass = (self.solarA09_vs_H_bynumb + self.log10_avg_elem_vs_X(elemZ=1))
        self.solarA09_vs_Fe_bymass = (self.solarA09_vs_Fe_bynumb + self.log10_avg_elem_vs_X(elemZ=26))
        self.asplund3_pd = self.IN.asplund3 
                
    def log10_avg_elem_vs_X(self, elemZ=1):
        ''' log10(<M all> / <M elemZ>) absolute abundance by mass'''
        return np.log10(np.divide(self.select_periodic['elemA'],
                self.select_periodic['elemA'][elemZ]))
    
    def abund_percentage(self, ZA_sorted):
        ''' Isotopic abundances by number from Asplund et al. (2009)'''
        percentages = []
        for ZA_idx in ZA_sorted:
            Z_select = self.IN.asplund3.loc[self.IN.asplund3['elemZ'].values == ZA_idx[0]]
            ZA_select = Z_select.loc[Z_select['elemA'].values == ZA_idx[1]]
            percentages.append(ZA_select['percentage'].values)
        percentages_pd = np.array(percentages, dtype='object')
        for i, val in enumerate(percentages_pd):
            if val.size <= 0:
                percentages_pd[i] = np.array([1e-5])
        return np.array(percentages_pd, dtype=np.float16)

    def extract_ZA_pairs(self, yields):
        ZA_pairs = np.column_stack((yields.elemZ, yields.elemA))
        return self.ZA_sorted(ZA_pairs)
    
    def ZA_sorted(self, ZA_all):
        Z_sorted = np.array(ZA_all[ZA_all[:,0].argsort()]).astype(int)
        sorting_A = [] # isolating A indices with same Z
        A_sorted = [] # sorted indices for sorting_A
        for i in range(Z_sorted[:,0].max()+1):
            sorting_A.append(np.where(Z_sorted[:,0] == i)[0])
        for i in range(Z_sorted[:,0].max()+1):
            A_sorted.append(sorting_A[i][sorting_A[i].argsort()])
        ZA_sorted = Z_sorted[list(flatten(A_sorted))]
        return np.unique(ZA_sorted, axis=0)
    
        
class Yields:
    ''' Parent class for all Yields_* classes '''
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
        self._dir = os.path.join(os.path.dirname( __file__ ), '..')
        
        
class Yields_BBN(Yields):
    '''
    Yields by Big Bang Nucleosynthesis, from Galli & Palla (2013) by default.
    '''
    def __init__(self, IN, option=None):
        self.IN = IN
        self.option = self.IN.yields_BBN_option if option is None else option
        self.massCol = None
        super().__init__()
        
    def import_yields(self):
        if self.option == 'gp13':
            yd = self._dir + '/input/yields/bbn/'
            
            self.tables = np.genfromtxt(yd + 'galli13.dat', names=['elemZ','elemA', 'numbFrac', 'mass'], 
                         dtype=[('elemZ', '<i8'), ('elemA', '<i8'), ('numbFrac','<f8'), ('mass','<f8')])
            self.elemA = self.tables['elemA']
            self.elemZ = self.tables['elemZ']
            self.massCol = np.multiply(self.tables['numbFrac'], self.tables['mass'])
            self.yields_list = np.divide(self.massCol, np.sum(self.massCol)) # fraction by mass 
            self.yields = np.divide(self.massCol, np.sum(self.massCol)) # fraction by mass 
      
    def construct_yields(self, ZA_sorted):
        yields = []
        for i,val in enumerate(ZA_sorted):
            select_idz = np.where(self.elemZ == val[0])[0]
            select_ida = np.where(self.elemA == val[1])[0]
            select_id = np.intersect1d(select_idz,select_ida)
            if len(select_id) > 0.:
                yields.append(self.yields_list[select_id[0]])
            else:
                yields.append(0.)
        self.yields = np.array(yields)
        
class Yields_SNIa(Yields):
    '''
    Yields by Type Ia Supernova, from Iwamoto+99 by default.
    '''
    def __init__(self, IN, option=None):
        self.IN = IN
        self.option = self.IN.yields_SNIa_option if option is None else option
        super().__init__()
        
    def import_yields(self):
        if self.option == 'k20':
            yd = self._dir + '/input/yields/snia/k20/'
            self.tables = pd.read_fwf(yd + 'yield_nucl.d', comment='#')
            self.yields_list = self.tables['yields']
            self.elemA = self.tables['elemA']
            self.elemZ = self.tables['elemZ']
            
        if self.option == 'i99':
            yd = self._dir + '/input/yields/snia/i99/'
            self.tables = np.genfromtxt(yd + 'table3.csv', skip_header=1, delimiter=',',
                           names=['elemZ','elemA', 'elemSymb', 'TypeII', 
                                'W7', 'W70', 'WDD1', 'WDD2', 'WDD3', 'CDD1', 'CDD2'],
                        dtype=[('elemZ', '<i8'), ('elemA', '<i8'), ('elemSymb', '<U5'),
                         ('TypeII','<f8'), ('W7','<f8'), ('W70','<f8'), 
                         ('WDD1','<f8'), ('WDD2','<f8'), ('WDD3','<f8'), 
                         ('CDD1','<f8'), ('CDD2','<f8')])
            self.yields_list = self.tables['W7'] 
            self.elemA = self.tables['elemA']
            self.elemZ = self.tables['elemZ']
            
    def construct_yields(self, ZA_sorted):
        yields = []
        for i,val in enumerate(ZA_sorted):
            select_idz = np.where(self.elemZ == val[0])[0]
            select_ida = np.where(self.elemA == val[1])[0]
            select_id = np.intersect1d(select_idz,select_ida)
            if len(select_id) > 0.:
                yields.append(self.yields_list[select_id[0]])
            else:
                yields.append(0.)
        self.yields = yields
                
class Yields_SNII(Yields):
    '''
    Yields by SNII stars, from Limongi & Chieffi (2018) by default.
    '''
    def __init__(self, IN, option=None):
        self.IN = IN
        self.option = self.IN.yields_SNII_option if option is None else option
        self.rotationalVelocity_bins = None
        super().__init__()
 
    def import_yields(self):
        if self.option == 'lc18old':
            self.metallicity_bins = np.power(10, [0., -1., -2., -3.])
            self.rotationalVelocity_bins = [0., 150., 300.]
            yd = self._dir + '/input/yields/snii/lc18/tab_R/'
            yieldsT = []
            yieldsTable = []
            headers = []
            
            with open(yd + 'tab_yieldstot_iso_exp.dec', 'r') as yieldsSNII:
                for line in yieldsSNII:
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
    
        if self.option == 'lc18':
            import re
            import glob
            lc18 = pd.read_csv('yield_interpolation/lc18/data.csv')
            lc18_yield_dir = 'yield_interpolation/lc18/models/'
            self.metallicity_bins = np.unique(lc18['metallicity'].values)
            self.elemA = lc18['a'].values #np.unique(lc18['a'].values)
            self.elemZ = lc18['z'].values #np.unique(lc18['z'].values)
            self.yields_list = glob.glob(lc18_yield_dir+'*.pkl')
            patternz = "/z(.*?).a"
            z_list = [re.search(patternz, yl).group(1) for yl in self.yields_list]
            searcha = [".a",".irv0"]
            a_list = [yl[yl.find(searcha[0])+len(searcha[0]):yl.find(searcha[1])] for yl in self.yields_list]
            ZA_list = np.column_stack([z_list, a_list])
            self.ZA_list = ZA_list.astype('int')
            
    def construct_yields(self, ZA_sorted):
        import pickle
        yields = []
        yields_l = pd.Series(self.yields_list, dtype='str')
        for i,val in enumerate(ZA_sorted):
            pattern = '/z'+ str(val[0]) + '.a' + str(val[1])+'.irv0'
            select_id = np.where(yields_l.str.contains(pattern))[0]
            if len(select_id) > 0.:
                yields.append(pickle.load(open(yields_l.iloc[select_id[0]],'rb')))
            else:
                yields.append(pd.DataFrame(columns=['mass', 'metallicity']))
        self.yields = yields
    
    
class Yields_LIMs(Yields):
    '''
    Yields by LIMs, from Karakas et al. (2010) by default.
    '''
    def __init__(self, IN, option=None):
        self.IN = IN
        self.option = self.IN.yields_LIMs_option if option is None else option
        self.Returned_stellar_mass = None
        super().__init__()
 
    def is_unique(self, val, split_length):
        it_is = [t[val] for t in self.tables]
        unique = np.array([np.unique(ii) for ii in it_is], dtype=object) 
        it_is = [np.split(it_is[y], split_length[y]) for y in range(len(split_length))]
        it_is[0] = it_is[0][:-1]
        it_is_T = [np.array(ii).T for ii in it_is]
        if (val == 'elemA' or val == 'elemZ'):
            return it_is_T, unique[0].astype(int)
        elif (val == 'Yield' or val == 'Mfin'):
            return it_is_T
        else:
            return it_is_T, unique.T[0]
        
    def import_yields(self):
        '''
        for option == 'k10': 
            lists of 4 [(77,15)*3, (77,16)]:    yields, *Ini, *_sorting
            arrays:                                elemA, elemZ, *_bins
        '''
        if self.option == 'k10':
            self.metallicity_bins = [0.0001, 0.008, 0.004, 0.02]
            split_length = [16, 15, 15, 16] # self.tables length (metallicity arrays)
            yd = self._dir + '/input/yields/lims/k10/'
            
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
            self.elemZ_sorting, self.elemZ = self.is_unique('elemZ', split_length) # *_sorting associates elemZ to the index. elemZ is unique
            self.metallicityIni, self.metallicity_bins = self.is_unique('Zini', split_length)
            self.stellarMassIni, self.stellarMass_bins = self.is_unique('Mini', split_length)
            self.Returned_stellar_mass = self.is_unique('Mfin', split_length)  
        
        if self.option == 'c15':
            import re
            import glob
            c15 = pd.read_csv('yield_interpolation/c15/data.csv')
            c15_yield_dir = 'yield_interpolation/c15/models/'
            self.metallicity_bins = np.unique(c15['metallicity'].values)
            self.elemA = c15['a'].values #np.unique(c15['a'].values)
            self.elemZ = c15['z'].values #np.unique(c15['z'].values)
            self.yields_list = glob.glob(c15_yield_dir+'*.pkl')
            patternz = "/z(.*?).a"
            z_list = [re.search(patternz, yl).group(1) for yl in self.yields_list]
            searcha = [".a",".irv0"]
            a_list = [yl[yl.find(searcha[0])+len(searcha[0]):yl.find(searcha[1])] for yl in self.yields_list]
            ZA_list = np.column_stack([z_list, a_list])
            self.ZA_list = ZA_list.astype('int')
            
    def construct_yields(self, ZA_sorted):
        import pickle
        yields = []
        yields_l = pd.Series(self.yields_list, dtype='str')
        for i,val in enumerate(ZA_sorted):
            pattern = '/z'+ str(val[0]) + '.a' + str(val[1])+'.irv0'
            select_id = np.where(yields_l.str.contains(pattern))[0]
            if len(select_id) > 0.:
                yields.append(pickle.load(open(yields_l.iloc[select_id[0]],'rb')))
            else:
                yields.append(pd.DataFrame(columns=['mass', 'metallicity']))
        self.yields = yields
        
        
class Yields_MRSN(Yields):
    '''
    Yields by MRSN, from Nishimura et al. (2017) by default.
    '''
    def __init__(self, IN, option=None):
        self.IN = IN
        self.option = self.IN.yields_MRSN_option if option is None else option
        super().__init__()
        self.yd = self._dir + '/input/yields/mrsn/' + self.option
        self.ejectamass = self.ejecta_mass()
        
    def ejecta_mass(self):
        return pd.read_csv(self.yd + '/ejectamass.dat', sep=',', comment='#')

    def import_yields(self):
        #''''''
        if self.option == 'n17':
            import glob
            all_files = glob.glob(self.yd + "/L*")
            all_files = sorted(all_files)
            li = []
            ej_select = []
            for filename in all_files:
                df = pd.read_fwf(filename, skiprows=7, infer_nrows=150)
                li.append(df)
                idlabel = filename.replace(self.yd+'/','').replace('.dat','')
                ej_select.append(self.ejectamass.loc[self.ejectamass['filename']==idlabel]['ejectamass'])
            self.elemA = li[0]['A']
            self.elemZ = li[0]['Z']
            self.massFrac = [i['X'].values for i in li]
            self.yields_list = np.multiply(ej_select, self.massFrac)
                
    def construct_yields(self, ZA_sorted):
        yields = []
        for i,val in enumerate(ZA_sorted):
            select_idz = np.where(self.elemZ == val[0])[0]
            select_ida = np.where(self.elemA == val[1])[0]
            select_id = np.intersect1d(select_idz,select_ida)
            if len(select_id) > 0.:
                yields.append(self.yields_list[select_id[0]])
            else:
                yields.append(0.)
        self.yields = yields

class Yields_NSM(Yields):
    '''
    Yields by NSM, from Rosswog et al. (2014) by default.
    '''
    def __init__(self, IN, option=None):
        self.IN = IN
        self.option = self.IN.yields_NSM_option if option is None else option
        super().__init__()
        self.yd = self._dir + '/input/yields/nsm/' + self.option
        self.ejectamass = 0.04 # Rastinejad+22
        
    def import_yields(self):
        #''''''
        if self.option == 'r14':
            self.tables = pd.read_csv('galcem/input/yields/nsm/r14/Zsolar.dat', sep=',', comment='#')
            self.elemA = self.tables['elemA']
            self.elemZ = self.tables['elemZ']
            self.massFrac = self.tables['massFrac']
            self.yields = np.multiply(self.ejectamass, self.massFrac)  
            
    #def __repr__(self):
    #    self.import_yields()
    #    NSMobject = pd.DataFrame(self.elemZ)
    #    #NSMobject['elemA'] = self.elemA
    #    #NSMobject['massFrac'] = self.massFrac
    #    #NSMobject['yields'] = self.yields
    #    return NSMobject # err: not a string
