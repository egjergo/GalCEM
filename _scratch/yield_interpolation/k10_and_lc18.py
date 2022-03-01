if __name__ == '__main__':
    dfxs = pickle.load(open('input/yields/lims/k10/processed/X.pkl','rb'))
    dfys = pickle.load(open('input/yields/lims/k10/processed/Y.pkl','rb'))
    idxs = np.arange(len(ZA_sorted)) # [11,19,99,103] # carbon12, oxygen, iron56, iron60
    interpolants = []
    for idx in idxs:
        print(f'{idx=}')
        if dfxs[idx].empty:
            interpolants.append(None)
        else:
            df = dfxs[idx].copy()
            df['log10_y'] = np.log10(dfys[idx])
            df['log10_Z_ini'] = np.log10(df['Z_ini'])
            tag = 'k10_%d, Z=%d, A=%d'%(idx, ZA_sorted[idx,0], ZA_sorted[idx,1])
            log10_yield_lims_interpolant = fit_2d_interpolant(
                dfx = df[['log10_Z_ini','mass_ini']],
                dfy = df['log10_y'],
                tag = tag,
                test = True,
                y_log10_scaled = True,
                view_angle = 135)
            interpolants.append(log10_yield_lims_interpolant)
            #pickle.dump(log10_yield_lims_interpolant,open('output/interpolants/log10_yield_lims_interpolant.%d.pkl'%idx,'wb'))
    interpolants = np.array(interpolants, dtype='object') #, header = "dfx = df[['log10_Z_ini','mass_ini']], dfy = df['log10_mass_frac']"
    #pickle.dump(interpolants,open('output/interpolants/log10_yield_lims_interpolants.pkl','wb'))
    
    dfxs = pickle.load(open('input/yields/snii/lc18/tab_R/processed/X.pkl','rb'))
    dfys = pickle.load(open('input/yields/snii/lc18/tab_R/processed/Y.pkl','rb'))
    idxs = np.arange(len(ZA_sorted)) # [11,19,99,103] # carbon12, oxygen, iron56, iron60
    interpolants = []
    for idx in idxs:
        if dfxs[idx].empty:
            interpolants.append(None)
        else:
            df = dfxs[idx].copy()
            df['log10_y'] = np.log10(dfys[idx])
            df['log10_Z_ini'] = np.log10(df['Z_ini'])
            tag = 'lc18_%d, Z=%d, A=%d'%(idx, ZA_sorted[idx,0], ZA_sorted[idx,1])
            log10_yield_snii_interpolant = fit_3d_interpolant(
                dfx = df[['log10_Z_ini','mass_ini','vel_ini']],
                dfy = df['log10_y'],
                tag = tag,
                test = False,
                y_log10_scaled = True,
                view_angle = 135)
            interpolants.append(log10_yield_snii_interpolant)
            #pickle.dump(log10_yield_snii_interpolant,open('output/interpolants/log10_yield_snii_interpolant.%d.pkl'%idx,'wb'))
    interpolants = np.array(interpolants, dtype='object') #, header="dfx = df[['log10_Z_ini','mass_ini','vel_ini']], dfy = df['log10_mass_frac']",
    pickle.dump(interpolants,open('output/interpolants/log10_yield_snii_interpolant.pkl','wb'))