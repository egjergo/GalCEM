""""""""""""""""""""""""""""""""""""""""""""""""
"                                              "
"      MAIN CLASSES FOR SINGLE-ZONE RUNS       "
"       Contains classes that solve the        " 
"     integro-differential equations of GCE    "
"                                              "
" LIST OF CLASSES:                             "
"    __        Setup (parent)                  "
"    __        OneZone (subclass)              "
"                                              "
""""""""""""""""""""""""""""""""""""""""""""""""

import pickle
import time
import numpy as np
import pandas as pd
import scipy.integrate as integr
from .gcsetup import Setup
from .classes import integration as gcint
from .classes.inputs import Auxiliary
   
class OneZone(Setup):
    """
    OneZone class
    
    In (Input): an Input configuration instance 
    """
    def __init__(self, IN, outdir = 'runs/mygcrun/'):
        self.tic = []
        self.tic.append(time.process_time())
        super().__init__(IN, outdir=outdir)
        self.tic.append(time.process_time())
        package_loading_time = self.tic[-1]
        print('Package lodaded in %.1e seconds.'%package_loading_time)
    
    def __repr__(self):
        aux = Auxiliary()
        return aux.repr(self)
    
    def main(self):
        ''' Run the OneZone program '''
        self.tic.append(time.process_time())
        self.file1 = open(self._dir_out + "Terminal_output.txt", "w")
        pickle.dump(self.IN,open(self._dir_out + 'inputs.pkl','wb'))
        with open(self._dir_out + 'inputs.txt', 'w') as f: 
            for key, value in self.IN.__dict__.items(): 
                if type(value) is not pd.DataFrame:
                    f.write('%s:\t%s\n' % (key, value))
        with open(self._dir_out + 'inputs.txt', 'a') as f: 
            for key, value in self.IN.__dict__.items(): 
                if type(value) is pd.DataFrame:
                    with open(self._dir_out + 'inputs.txt', 'a') as ff:
                        ff.write('\n %s type %s\n'%(key, str(type(value))))
                    #value.to_csv(self._dir_out + 'inputs.txt', mode='a',
                    #             sep='\t', index=True, header=True)
        self.evolve()
        self.aux.tic_count(string="Computation time", tic=self.tic)
        G_v = np.divide(self.Mgas_v, self.Mtot)
        S_v = 1 - G_v
        print("Saving the output...")
        phys_dat = {
            'time[Gyr]'   : self.time_chosen,
            'Mtot[Msun]'  : self.Mtot, 
            'Mgas[Msun]'  : self.Mgas_v,
            'Mstar[Msun]' : self.Mstar_v, 
            'SFR[Msun/yr]': self.SFR_v/1e9 * self.IN.M_inf, # rescale to Msun/yr from Msun/Gyr/galMass
            'Inf[Msun/yr]': self.Infall_rate/1e9, #/Gyr to /yr conversion
            'Zfrac'       : self.Z_v,
            'Gfrac'       : G_v, 
            'Sfrac'       : S_v, 
            'R_CC[M/yr]'  : self.Rate_SNCC/1e9,
            'R_Ia[M/yr]'  : self.Rate_SNIa/1e9,
            'R_LIMs[M/y]' : self.Rate_LIMs/1e9,
            'DTD_Ia[N/yr]': self.f_SNIa_v
        }
        phys_df = pd.DataFrame(phys_dat)
        phys_df.to_csv(self._dir_out+'phys.dat', index=False, 
                       header="phys = pd.read_csv(run_path+'phys.dat', sep=',', comment='#')")
        np.savetxt(self._dir_out+'Mass_i.dat', np.column_stack((
            self.ZA_sorted, self.Mass_i_v)), fmt=' '.join(['%5.i']*2+['%12.4e']
                                                 *self.Mass_i_v[0,:].shape[0]),
                header = 'elemZ    elemA    masses[Msun] # of every isotope for every timestep')
        np.savetxt(self._dir_out+'X_i.dat', np.column_stack((
            self.ZA_sorted, self.Xi_v)), fmt=' '.join(['%5.i']*2 + ['%12.4e']*
                                                    self.Xi_v[0,:].shape[0]),
                header = 'elemZ    elemA    X_i # abundance mass ratios of every isotope for every timestep (normalized to solar, Asplund et al., 2009)')
        pickle.dump(self.W_i_comp,open(self._dir_out + 'W_i_comp.pkl','wb'))
        self.aux.tic_count(string="Output saved in", tic=self.tic)
        self.file1.close()
    
    def evolve(self):
        '''Evolution routine'''
        #self.file1.write('A list of the proton/isotope number pairs for all the nuclides included in this run.\n ZA_sorted =\n\n')
        #self.file1.write(self.ZA_sorted)
        # First timestep: the galaxy is empty
        self.Mass_i_v[:,0] = np.multiply(self.Mtot[0], self.models_BBN)
        self.Mgas_v[0] = self.Mtot[0]
        # Second timestep: infall only
        self.Mass_i_v[:,1] = np.multiply(self.Mtot[1], self.models_BBN)
        self.Mgas_v[1] = self.Mtot[1]
        for n in range(len(self.time_chosen[:self.idx_Galaxy_age])):
            print('time [Gyr] = %.2f'%self.time_chosen[n])
            self.file1.write('n = %d\n'%n)
            self.total_evolution(n)        
            self.Xi_v[:, n] = np.divide(self.Mass_i_v[:,n], self.Mgas_v[n])
            self.Z_v[n] = np.divide(np.sum(self.Mass_i_v[self.i_Z:,n]),
                                    self.Mgas_v[n])
            self.file1.write(' sum X_i at n %d= %.3f\n'%(n, np.sum(
                             self.Xi_v[:,n])))
            
            if n > 0.: 
                Wi_class = gcint.Wi(n, self.IN, self.lifetime_class, 
                                    self.time_chosen, self.Z_v, self.SFR_v,
                            self.Greggio05_SD, self.IMF, self.ZA_sorted)
                _rates = Wi_class.compute_rates()
                self.Rate_SNCC[n] = _rates['SNCC']
                self.Rate_LIMs[n] = _rates['LIMs']
                self.Rate_SNIa[n] = _rates['SNIa']
                Wi_comp = {ch: Wi_class.compute(ch)
                        for ch in self.IN.include_channel}
                Z_comp = {}
                for ch in self.IN.include_channel:
                    if len(Wi_comp[ch]['birthtime_grid']) > 1.:
                        Z_comp[ch] = pd.DataFrame(
                        Wi_class.Z_component(Wi_comp[ch]['birthtime_grid']),
                                            columns=['metallicity']) 
                    else:
                        Z_comp[ch] = pd.DataFrame(columns=['metallicity']) 
                for i, _ in enumerate(self.ZA_sorted): 
                    self.Mass_i_v[i, n+1] = self.aux.RK4(
                        self.isotopes_evolution,self.time_chosen[n],
                        self.Mass_i_v[i,n], n, self.IN.nTimeStep,
                        i=i, Wi_comp=Wi_comp, Z_comp=Z_comp)
            self.Mass_i_v[:, n] = np.multiply(self.Mass_i_v[:,n], #!!!!!!!
                                              self.Mgas_v[n]/np.sum(self.Mass_i_v[:,n]))
        self.Z_v[-1] = np.divide(np.sum(self.Mass_i_v[self.i_Z:,-1]), 
                                self.Mgas_v[-1])
        self.Xi_v[:,-1] = np.divide(self.Mass_i_v[:,-1], self.Mgas_v[-1]) 

    def Mgas_func(self, t_n, y_n, n, i=None):
        # Explicit general diff eq GCE function
        # Mgas(t)
        #print(f'{self.SFR_tn(n)==self.SFR_v[n]=}')
        return self.Infall_rate[n] - self.SFR_tn(n) * self.IN.M_inf + np.sum([
                self.W_i_comp[ch][:,n-1] for ch in self.IN.include_channel])
    
    def Mstar_func(self, t_n, y_n, n, i=None):
        # Mstar(t)
        return self.SFR_tn(n) * self.IN.M_inf - np.sum([
               self.W_i_comp[ch][:,n-1] for ch in self.IN.include_channel])

    def SFR_tn(self, timestep_n):
        '''
        Actual SFR employed within the integro-differential equation
        Args:
            timestep_n ([int]): [timestep index]
        Returns:
            [function]: [SFR as a function of Mgas] units of [Gyr^-1]
        '''
        return self.SFR_class.SFR(Mgas=self.Mgas_v, Mtot=self.Mtot, 
                                  timestep_n=timestep_n) 
    
    def total_evolution(self, n):
        '''Integral for the total physical quantities'''
        self.SFR_v[n] = self.SFR_tn(n)
        self.Mstar_v[n+1] = self.aux.RK4(self.Mstar_func, self.time_chosen[n],
                                        self.Mstar_v[n], n, self.IN.nTimeStep)
        self.Mgas_v[n+1] = self.aux.RK4(self.Mgas_func, self.time_chosen[n], 
                                        self.Mgas_v[n], n, self.IN.nTimeStep)   

    def isotopes_evolution(self, t_n, y_n, n, **kwargs):
        '''
        Explicit general diff eq GCE function
        INPUT
        t_n        time_chosen[n]
        y_n        dependent variable at n
        n        index of the timestep
        Functions:
        Infall rate: [Msun/Gyr]
        SFR: [Msun/Gyr]
        '''
        i = kwargs['i']
        Wi_comps = kwargs['Wi_comp'] 
        Z_comps = kwargs['Z_comp'] 
        infall_comp = self.Infall_rate[n] * self.models_BBN[i]
        self.W_i_comp['BBN'][i,n] = infall_comp
        sfr_comp = self.SFR_v[n] * self.Xi_v[i,n] # astration
        if n <= 0:
            val = infall_comp - sfr_comp
        else:
            Wi_vals = {}
            for ch in self.IN.include_channel:
                if ch == 'SNIa':
                    Wi_vals[ch] = 0.5 * (self.Rate_SNIa[n] * # Don't count SNIas twice
                                   self.yield_models['SNIa'][i])
                else:
                    if len(Wi_comps[ch]['birthtime_grid']) > 1.:
                        if not self.yield_models[ch][i].empty:
                            yield_grid = Z_comps[ch]
                            yield_grid['mass'] = Wi_comps[ch]['mass_grid']
                            Wi_vals[ch] = self.IN.factor * integr.simps(np.multiply(
                                Wi_comps[ch]['integrand'], 
                                self.yield_models[ch][i](yield_grid)), 
                                        x=Wi_comps[ch]['birthtime_grid'])
                        else:
                            Wi_vals[ch] = 0.
                    else:
                        Wi_vals[ch] = 0.
                self.W_i_comp[ch][i,n] = Wi_vals[ch] 
            val = infall_comp - sfr_comp + np.sum([Wi_vals[ch] 
                                        for ch in self.IN.include_channel])
        return val
