from .yield_interpolation import galcem_interpolant as gcint
import numpy as np
import pandas as pd
import galcem as gc
import os

def parse_lc18_raw():
    yield_eps = 1e-13
    _path = 'galcem/input/yields/sncc/lc18/tab_R/'
    inputs = gc.Inputs()
    Zsun = inputs.solar_metallicity
    remn_loc = 'galcem/input/yields/sncc/lc18/RemnantMass/remnant_masses.txt'
    df_raw = pd.read_fwf(_path+'tab_yieldsnet_iso_exp.dec',
                         dtype={'ID': object},header=None,engine='python')
    df_remnant = pd.read_fwf(remn_loc)
    df_rem = pd.melt(df_remnant,id_vars=df_remnant.columns[:2],
                     var_name='model', value_name='Mrem')
    df_r = df_rem.loc[df_rem['model'].str.contains('R')]
    df_r['M'] = df_r['M'].astype(float)
    df_r['v'] = df_r['v'].astype(float)
    header_idxs = np.argwhere((df_raw[0]=='ele').to_numpy()
                              ).flatten().tolist()+[len(df_raw)]
    zini_map = {'a':Zsun,'b':1e-1*Zsun,'c':1e-2*Zsun,'d':1e-3*Zsun} 
    zini_remmap = {Zsun:'0R',1e-1*Zsun:'-1R',1e-2*Zsun:'-2R',1e-3*Zsun:'-3R'}
    df = pd.DataFrame({'isotope':[],'a':[],'z':[],'yield':[],'mass':[],
            'metallicity':[],'irv':[],'mass_ej':[],'massfrac':[],'ysign':[]})
    for i in range(len(header_idxs)-1):
        start,end = header_idxs[i],header_idxs[i+1]
        _df = df_raw.iloc[start:end]
        cols = _df.iloc[0].tolist()
        cols = [col.strip() for col in cols]
        z_ini_k = cols[4][3]
        z_ini = zini_map[z_ini_k]
        _df.columns = cols
        _df = _df[1:]
        _df = pd.melt(_df,id_vars=cols[:4],value_name='yield')
        _df[['mass','irv']] = _df['variable'].str.split(z_ini_k,1,expand=True)
        _df['mass'] = _df['mass'].astype(float)
        _df['irv'] = _df['irv'].astype(float)
        _df['metallicity'] = z_ini
        df_rem_slice = df_r.loc[df_r['v'].eq(_df['irv'])].loc[
                                df_r['model'].eq(zini_remmap[z_ini])]
        idx_rem = [np.where(df_rem_slice['M']==i)[0][0] for i in _df['mass']]
        remnant_mass = df_rem_slice['Mrem'].iloc[idx_rem]
        # In LC18, remnant -1 means no remnant.
        remnant_mass_pd = remnant_mass.replace(-1.,0.)
        mass_remnant = remnant_mass_pd.to_numpy(float)
        _df['mass_ej'] = _df['mass'] - mass_remnant
        _df['massfrac'] = _df['yield'].to_numpy(float)/_df['mass_ej']
        _df['ysign'] = np.sign(_df['yield'].to_numpy(float))
        _df.drop(['variable','initial'],axis=1,inplace=True)
        _df.columns = ['isotope','z','a','yield','mass','irv','metallicity',
                       'mass_ej','massfrac','ysign']
        num_cols = ['a','z','yield','mass','metallicity','irv','mass_ej',
                    'massfrac','ysign']
        _df = _df[['isotope']+num_cols]
        _df[num_cols] = _df[num_cols].apply(pd.to_numeric)
        df = df.append(_df,ignore_index=True)
    df[['a','z','ysign']] = df[['a','z','ysign']].astype(int)
    #dfy0 = df[df['yield']==0]
    #print('setting %d rows with yield=0 to %.1e'%(len(dfy0),yield_eps))
    #df.loc[df['yield']==0,'yield'] = yield_eps
    return df

if __name__ == '__main__':
    root = os.path.abspath(os.path.dirname(__file__))
    for dirs in ['models', 'figs']:
        if not os.path.exists(root+'/'+dirs):
                os.makedirs(root+'/'+dirs)
    df = parse_lc18_raw()
    df.to_csv(root+'/data.csv',index=False)
    df = df[df['irv']==0]
    gcint.fit_isotope_interpolants(
        df = df,
        root = root,
        tf_funs = {
            'mass':lambda x:np.log10(x), 
            'mass_prime':lambda x:1/(x*np.log(10)),
            'metallicity':lambda x:np.log10(x), 
            'metallicity_prime':lambda x:1/(x*np.log(10)),
            'massfrac':lambda y:np.log10(y), 
            'massfrac_prime':lambda y:1/(y*np.log(10)), 
            'massfrac_inv':lambda y:10**y,
            },
        fit_names = 'all', # 'all', ['lc18_z8.a16.irv0.O16'],
        plot_names = [] # [], 'all', ['lc18_z8.a16.irv0.O16']
        ) 