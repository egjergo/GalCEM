def _pick_yields(channel_switch, ZA_Symb, n, stellar_mass_idx=None, metallicity_idx=None, vel_idx=IN.LC18_vel_idx):
    ''' !!!!!!! this function must be edited if you import yields from other authors
    channel_switch    [str] can be 'LIMs', 'Massive', or 'SNIa'
    ZA_Symb            [str] is the element symbol, e.g. 'Na'
    
    'LIMs' requires metallicity_idx and stellar mass_idx
    'Massive' requires metallicity_idx, stellar mass_idx, and vel_idx
    'SNIa' and 'BBN' require None
    '''
    if channel_switch == 'LIMs':
        if (stellar_mass_idx == None or metallicity_idx == None):
            raise Exception('You must import the stellar mass and metallicity grids')
        idx = isotope_class.pick_by_Symb(yields_LIMs_class.elemZ, ZA_Symb)
        return yields_LIMs_class.yields[metallicity_idx][idx, stellar_mass_idx]
    elif channel_switch == 'Massive':
        if (stellar_mass_idx == None or metallicity_idx == None or vel_idx == None):
            raise Exception('You must import the stellar mass, metallicity, and velocity grids')
        metallicity_idx = np.digitize(Z_v[n], yields_Massive_class.metallicity_bins)
        vel_idx = IN.LC18_vel_idx
        idx = isotope_class.pick_by_Symb(yields_Massive_class.elemZ, ZA_Symb)
        return idx#yields_Massive_class.yields[metallicity_idx, vel_idx, idx, stellar_mass_idx]
    elif channel_switch == 'SNIa':
        idx = isotope_class.pick_by_Symb(yields_SNIa_class.elemZ, ZA_Symb)
        return yields_SNIa_class.yields[idx]
        
# Formerly in Isotopes class      
    def pick_by_Symb(self, ndarray_elemZ=0, elemSymb='Symb'):
        '''
        Finds the indices 
        
        Input:
            ndarray_elemZ:    the elemZ instance variable in yield classes
            elemSymb:        the periodic symbol of the given element
            
        Returns:
            the indices of the yield classes corresponding to the given element 
        '''
        print('Symbol: \t %s'%type(self.elemSymb))
        idx = np.where(self.elemSymb == elemSymb)
        return np.where(ndarray_elemZ == self.elemZ[idx])

""" load yield tables """
X_lc18, Y_lc18, models_lc18, averaged_lc18 = yt.load_processed_yields(func_name='lc18', loc='input/yields/snii/lc18/tab_R', df_list=['X', 'Y', 'models', 'avgmassfrac'])
X_k10, Y_k10, models_k10, averaged_k10 = yt.load_processed_yields(func_name='k10', loc='input/yields/lims/k10', df_list=['X', 'Y', 'models', 'avgmassfrac'])
Y_i99 = yt.load_processed_yields_snia(func_name='i99', loc='input/yields/snia/i99', df_list='Y')

def test_fit(x,y,model, y_log10_scaled=False):
    if x.empty:
        print('empty')
    else:
        yhat = model(x)
        yhat_tf = 10**yhat if y_log10_scaled else yhat
        ytf = 10**y if y_log10_scaled else y
        eps_abs = np.abs(yhat_tf-ytf)
        print('\tRMSE: %.1e'%np.sqrt(np.mean(eps_abs**2)))
        print('\tMAE: %.1e'%np.mean(eps_abs))
        print('\tMax Abs Error: %.1e\n'%eps_abs.max())
    return None


 # From setup class
    def load_processed_yields(self,func_name, loc, df_list):
        df_dict = {}
        for df_l in df_list:
            with open('%s/processed/%s.pkl'%(loc,df_l), 'rb') as pickle_file:
                df_dict[df_l] = pickle.load(pickle_file)
        return [df_dict[d] for d in df_list]#df_dict[df_list[0]], df_dict[df_list[1]]#, df_dict[df_list[2]]

    def load_processed_yields_snia(self, func_name, loc, df_list):#, 'models']):
        df_dict = {}
        for df_l in df_list:
            with open('%s/processed/%s.pkl'%(loc,df_l), 'rb') as pickle_file:
                df_dict[df_l] = pickle.load(pickle_file)
        return df_dict[df_list[0]]