
import os
import pickle
import time
import numpy as np
import pandas as pd
import scipy.integrate as integr
#from scipy.interpolate import *
from .classes import morphology as morph
from .classes import yields as yi
from .classes import integration as gcint
from .classes.inputs import Auxiliary

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
        self.morph = morph
        self.yi = yi
        self.gcint = gcint
        self.IN = IN
        self.aux = Auxiliary()
        self.lifetime_class = morph.Stellar_Lifetimes(self.IN)
        
        self.IN.M_inf = self.IN.default_params('M_inf', self.IN.morphology)
        self.IN.Reff = self.IN.default_params('Reff', self.IN.morphology)
        self.IN.tau_inf = self.IN.default_params('tau_inf', self.IN.morphology)
        self.IN.nu = self.IN.default_params('nu', self.IN.morphology)
        self.IN.wind_efficiency = 0 # !!!!!!! override: no outflow
        
        # Setup
        self.Ml = self.IN.Ml_LIMs # Lower limit stellar mass [Msun] 
        self.Mu = self.IN.Mu_collapsars # Upper limit stellar mass [Msun]
        self.time_uniform = np.arange(self.IN.Galaxy_birthtime, 
                            self.IN.Galaxy_age+self.IN.nTimeStep,
                                      self.IN.nTimeStep)
        self.time_logspace = np.logspace(np.log10(self.IN.Galaxy_birthtime),
                    np.log10(self.IN.Galaxy_age), num=self.IN.numTimeStep)
        # For now, leave to time_chosen equal to time_uniform. 
        # Some computations depend on a uniform timestep
        self.time_chosen = self.time_uniform
        self.idx_Galaxy_age = self.aux.find_nearest(self.time_chosen, 
                                                    self.IN.Galaxy_age)
        # Surface density for the disk. 
        # The bulge goes as an inverse square law
        #sigma(t_G) before eq(7). Not used so far !!!!!!!
        #surf_density_Galaxy = self.IN.sd / np.exp(self.IN.solar_radius / self.IN.Reff)
        
        #self.infall_class = morph.Infall(self.IN, time=self.time_chosen)
        #self.infall = self.infall_class.inf()
        self.SFR_class = morph.Star_Formation_Rate(self.IN, self.IN.SFR_option,
                                                   self.IN.custom_SFR)
        self.IMF_class = morph.Initial_Mass_Function(self.Ml, self.Mu, self.IN,
                                    self.IN.IMF_option, self.IN.custom_IMF)
        self.IMF = self.IMF_class.IMF() #() # Function @ input stellar mass
        
        #normalization
        self.Greggio05_SD = np.vectorize(morph.Greggio05) # [func(lifetime)]
        gal_time = np.logspace(-3,1.5, num=1000)
        DTD_SNIa = [D.f_SD_Ia for D in self.Greggio05_SD(gal_time)]
        K = 1/integr.simpson(DTD_SNIa, x=gal_time)
        self.f_SNIa_v = np.array([K * D.f_SD_Ia for D in self.Greggio05_SD(self.time_chosen)])
        
        # Comparison of rates with observations from Mannucci+05
        SNmassfrac = self.IMF_class.IMF_fraction(self.IN.Ml_SNCC, self.IN.Mu_SNCC, massweighted=True)
        SNnfrac = self.IMF_class.IMF_fraction(self.IN.Ml_SNCC, self.IN.Mu_SNCC, massweighted=False)
        N_IMF = integr.quad(self.IMF, self.Ml, self.Mu)[0]
        self.IN.MW_RSNCC = self.IN.Mannucci05_convert_to_SNrate_yr('II', self.IN.morphology, 
                                                           SNmassfrac=SNmassfrac, SNnfrac=SNnfrac, NtotvsMtot=N_IMF)
        N_RSNIa = np.multiply(self.IN.Mannucci05_SN_rate('Ia', self.IN.morphology),
                              1.4 * self.IN.M_inf /1.e10 * 1e-2) # 1.4 Msun for Chandrasekhar's limit (SD scenario) 
        self.IN.MW_RSNIa = np.array([N_RSNIa[0], N_RSNIa[0]+ N_RSNIa[1], N_RSNIa[0] - N_RSNIa[2]])
        
        # Initialize Yields
        self.iso_class = yi.Isotopes(self.IN)
        #self.yields_MRSN_class = yi.Yields_MRSN(self.IN)
        #self.yields_MRSN_class.import_yields()
        #self.yields_NSM_class = yi.Yields_NSM(self.IN)
        #self.yields_NSM_class.import_yields()
        self.yields_LIMs_class = yi.Yields_LIMs(self.IN)
        self.yields_LIMs_class.import_yields()
        self.yields_SNCC_class = yi.Yields_SNCC(self.IN)
        self.yields_SNCC_class.import_yields()
        self.yields_SNIa_class = yi.Yields_SNIa(self.IN)
        self.yields_SNIa_class.import_yields()
        self.yields_BBN_class = yi.Yields_BBN(self.IN)
        self.yields_BBN_class.import_yields()
        
        # Initialize ZA_all
        self.c_class = yi.Concentrations(self.IN)
        self.ZA_BBN = self.c_class.extract_ZA_pairs(self.yields_BBN_class)
        self.ZA_LIMs = self.c_class.extract_ZA_pairs(self.yields_LIMs_class)
        self.yields_LIMs_class.elemZ = self.ZA_LIMs[:,0]
        self.yields_LIMs_class.elemA = self.ZA_LIMs[:,1]
        self.ZA_SNIa = self.c_class.extract_ZA_pairs(self.yields_SNIa_class)
        self.yields_SNIa_class.elemZ = self.ZA_SNIa[:,0]
        self.yields_SNIa_class.elemA = self.ZA_SNIa[:,1]
        self.ZA_SNCC = self.c_class.extract_ZA_pairs(self.yields_SNCC_class)
        self.yields_SNCC_class.elemZ = self.ZA_SNCC[:,0]
        self.yields_SNCC_class.elemA = self.ZA_SNCC[:,1]
        #self.ZA_NSM = self.c_class.extract_ZA_pairs(self.yields_NSM_class)
        #self.yields_NSM_class.elemZ = self.ZA_NSM[:,0]
        #self.yields_NSM_class.elemA = self.ZA_NSM[:,1]
        #self.ZA_MRSN = self.c_class.extract_ZA_pairs(self.yields_MRSN_class)
        #self.yields_MRSN_class.elemZ = self.ZA_MRSN[:,0]
        #self.yields_MRSN_class.elemA = self.ZA_MRSN[:,1]
        ZA_all = np.vstack((self.ZA_LIMs, self.ZA_SNIa,
                            self.ZA_SNCC))#, self.ZA_NSM, self.ZA_MRSN))
        
        #self.Infall_rate = self.infall(self.time_chosen)
        # Sorted list of unique [Z,A] pairs which include all isotopes
        self.ZA_sorted = self.c_class.ZA_sorted(ZA_all) 
        # name of elements for all isotopes
        self.ZA_symb_list = self.IN.periodic['elemSymb'][self.ZA_sorted[:,0]]
        
        # Load Interpolation Models
        self._dir = os.path.dirname(__file__)
        self.yields_BBN_class.construct_yields(self.ZA_sorted)
        self.models_BBN = self.yields_BBN_class.yields
        self.yields_SNCC_class.construct_yields(self.ZA_sorted)
        self.models_SNCC = self.yields_SNCC_class.yields
        self.yields_LIMs_class.construct_yields(self.ZA_sorted)
        self.models_LIMs = self.yields_LIMs_class.yields
        self.yields_SNIa_class.construct_yields(self.ZA_sorted)
        self.models_SNIa = self.yields_SNIa_class.yields
        #self.yields_NSM_class.construct_yields(self.ZA_sorted)
        #self.models_NSM = self.yields_NSM_class.yields
        #self.yields_MRSN_class.construct_yields(self.ZA_sorted)
        #self.models_MRSN = self.yields_MRSN_class.yields
        self.yield_models = {ch: self.__dict__['models_'+ch] 
                            for ch in self.IN.include_channel}
        
        # Initialize Global tracked quantities
        if self.IN.multizone == True:
            self.radii = np.linspace(self.IN.bulge_radius, self.IN.Galaxy_radius, num=self.IN.n_shells)
            self.radii_mean = (self.radii[1:]+self.radii[:-1])/2
            self.surf_func = lambda r: 2 * np.pi * r**2 # Talbot & Arnett (1975)
            surfaces = [self.surf_func(self.radii[0])]
            for i in range(1,len(self.radii)):
                surfaces.append(self.surf_func(self.radii[i])
                                      -self.surf_func(self.radii[i-1]))
            self.surfaces = surfaces
            self.tot_surf = self.surf_func(self.radii[-1])
            self.total_r_dict = {'gal_radius': self.radii, 'gal_surface': self.surfaces}
            self.total_r_df = pd.DataFrame(self.total_r_dict)
            self.total_df = pd.DataFrame(columns=['gal_radius', 'gal_surface', 'gal_age'])
            for _,t in enumerate(self.time_chosen):
                total_dict = {}
                total_dict.update(self.total_r_dict)
                total_dict.update({'gal_age': [t]*len(self.radii)})
                self.total_df = self.total_df.append(pd.DataFrame(total_dict),ignore_index=True) 
            self.infall_f = np.vectorize(self.infall_func)
            self.Infall_rate = self.infall_f(self.total_df['gal_radius'],
                                                            self.total_df['gal_age'])
            self.total_df['infall_rate'] = self.Infall_rate
            self.renorm_infall = integr.nquad(self.infall_func, 
                                [[self.IN.bulge_radius, self.IN.Galaxy_radius], 
                                 [self.time_chosen[0], self.time_chosen[-1]]])[0]
            self.total_df['infall_rate'] *= self.IN.M_inf / self.renorm_infall
            self.renorm_surface = (self.IN.solar_surf_density * 1e6 / 
                                   self.total_df['gal_surface'].iloc[np.where(
                                       self.total_df['gal_radius']==self.IN.solar_radius)[0][-1]])
            self.total_df['gal_surface'] *= self.renorm_surface # [Msun/kpc]
        
        self.asplund3_percent = self.c_class.abund_percentage(self.ZA_sorted)
        # starting idx [int]. Excludes H and He for the metallicity selection
        self.i_Z = np.where(self.ZA_sorted[:,0]>2)[0][0] 
        
        self.Mtot = self.initialize(multizone=self.IN.multizone)
        self.Mstar_v = self.initialize(multizone=self.IN.multizone)
        self.Mgas_v = self.initialize(multizone=self.IN.multizone) 
        self.returned_Mass_v = self.initialize(multizone=self.IN.multizone) 
        self.SFR_v = self.initialize(multizone=self.IN.multizone)
        # Mass_i_v is the gass mass (i,j) where the i rows are the isotopes 
        # and j are the timesteps, [:,j] follows the timesteps
        self.Mass_i_v = self.initialize(matrix=True, multizone=self.IN.multizone)
        self.W_i_comp = {ch: self.initialize(matrix=True, multizone=self.IN.multizone) 
                         for ch in self.IN.include_channel} #dtype=object
        self.W_i_comp['BBN'] = self.initialize(matrix=True, multizone=self.IN.multizone) 
        self.Xi_v = self.initialize(matrix=True, multizone=self.IN.multizone) 
        self.Z_v = self.initialize(multizone=self.IN.multizone) # Metallicity 
        #self.G_v = self.initialize(multizone=self.IN.multizone) # G 
        #self.S_v = self.initialize(multizone=self.IN.multizone) # S = 1 - G 
        self.Rate_SNCC = self.initialize(multizone=self.IN.multizone) 
        self.Rate_LIMs = self.initialize(multizone=self.IN.multizone) 
        self.Rate_SNIa = self.initialize(multizone=self.IN.multizone) 
        self.Rate_NSM = self.initialize(multizone=self.IN.multizone)
        self.Rate_MRSN = self.initialize(multizone=self.IN.multizone) 
    
    def __repr__(self):
        aux = Auxiliary()
        return aux.repr(self)
    
    def infall_func(self, r:float, t:float):
        """Matteucci & Francois (1989)"""
        if r > 3.5:
            tau_D = np.vectorize(lambda radius: 0.464 * radius - 1.59) # Matteucci & Francois (1989)
            expon = np.exp(-t/tau_D(r))
            expon_tG = np.exp(-self.IN.Galaxy_age/tau_D(r))
            denom = tau_D(r) * (1 - expon_tG)
        else:
            tau_D = 0.17
            expon = np.exp(-t/tau_D)
            expon_tG = np.exp(-self.IN.Galaxy_age/tau_D)
            denom = tau_D * (1 - expon_tG)
        return expon / denom

    def initialize(self,matrix=False,multizone=False):
        if multizone==False:
            if matrix==True:
                return self.IN.epsilon * np.ones((len(self.ZA_sorted),
                                            len(self.time_chosen)), dtype=float)
            else:
                return self.IN.epsilon * np.ones(len(self.time_chosen)) 
        else:
            if matrix==True:
                return self.IN.epsilon * np.ones((len(self.ZA_sorted),
                                            len(self.time_chosen), len(self.radii_mean)), dtype=float)
            else:
                return self.IN.epsilon * np.ones((len(self.time_chosen), len(self.radii_mean))) 