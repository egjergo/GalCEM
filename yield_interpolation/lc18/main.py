from yield_interpolation.galcem_interpolant import GalCemInterpolant,fit_isotope_interpolants
import numpy as np
import pandas as pd
import os

def parse_lc18_raw():
    yield_eps = 1e-13
    df_raw = pd.read_table('galcem/input/yields/snii/lc18/tab_R/tab_yieldstot_iso_exp_pd.dec',sep=',  ',dtype={'ID': object},header=None,engine='python')
    header_idxs = np.argwhere((df_raw[0]=='ele').to_numpy()).flatten().tolist()+[len(df_raw)]
    zini_map = {'a':.0181,'b':.00181,'c':.000181,'d':.0000181} # Asplund et al. (2009, Table 4)
    df = pd.DataFrame({'isotope':[],'a':[],'z':[],'yield':[],'mass':[],'metallicity':[],'irv':[]})
    for i in range(len(header_idxs)-1):
        start,end = header_idxs[i],header_idxs[i+1]
        _df = df_raw.iloc[start:end]
        cols = _df.iloc[0].tolist()
        cols = [col.strip() for col in cols]
        z_ini_key = cols[4][3]
        z_ini = zini_map[z_ini_key]
        _df.columns = cols
        _df = _df[1:]
        _df = pd.melt(_df,id_vars=cols[:4],value_name='yield')
        _df[['mass','irv']] = _df['variable'].str.split(z_ini_key,1,expand=True)
        _df['metallicity'] = z_ini
        _df.drop(['variable','initial'],axis=1,inplace=True)
        _df.columns = ['isotope','z','a','yield','mass','irv','metallicity']
        num_cols = ['a','z','yield','mass','metallicity','irv']
        _df = _df[['isotope']+num_cols]
        _df[num_cols] = _df[num_cols].apply(pd.to_numeric)
        df = df.append(_df,ignore_index=True)
    df[['a','z']] = df[['a','z']].astype(int)
    dfy0 = df[df['yield']==0]
    print('setting %d rows with yield=0 to %.1e'%(len(dfy0),yield_eps))
    df.loc[df['yield']==0,'yield'] = yield_eps
    return df

if __name__ == '__main__':
    root = os.path.abspath(os.path.dirname(__file__))
    df = parse_lc18_raw()
    df.to_csv(root+'/data.csv',index=False)
    df = df[df['irv']==0]
    fit_isotope_interpolants(
        df = df,
        root = root,
        tf_funs = {
            'mass':lambda x:np.log10(x), 'mass_prime':lambda x:1/(x*np.log(10)),
            'metallicity':lambda x:np.log10(x), 'metallicity_prime':lambda x:1/(x*np.log(10)),
            'yield':lambda y:np.log10(y), 'yield_prime':lambda y:1/(y*np.log(10)), 'yield_inv':lambda y:10**y},
        fit_names = ['z8.a16.irv0.O16'],
        plot_names = ['z8.a16.irv0.O16']) # 'all'
