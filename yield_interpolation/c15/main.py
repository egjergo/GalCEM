from yield_interpolation.galcem_interpolant import GalCemInterpolant,fit_isotope_interpolants
import numpy as np
import pandas as pd
import os

def parse_c15_raw():
    yield_eps = 1e-13
    root_c15_raw_data = 'galcem/input/yields/lims/c15/data/'
    txts = os.listdir(root_c15_raw_data)
    df = pd.DataFrame({col:[] for col in ['Isotope','A','Z','YIELD','MASS','METALLICITY','IRV']})
    for txt_og in txts:
        if '.DS_Store' in txt_og: continue
        df_txt = pd.read_fwf(root_c15_raw_data+txt_og, infer_nrows=400)
        txt = txt_og.replace('zsun','z1.4m2').replace('_20210617_33100.txt','').replace('yields_tot_m','')
        prts = txt.split('_')
        prts = prts[0].split('z')+[prts[1]]
        mass = float(prts[0].replace('p','.'))
        metallicity = float(prts[1].replace('z','').replace('m','e-'))
        irv = int(prts[2])
        if metallicity in [2e-5,5e-5,1e-4,3e-4]: metallicity = metallicity*2.4
        df_txt['MASS'] = mass
        df_txt['METALLICITY'] = metallicity
        df_txt['IRV'] = irv
        df = df.append(df_txt)
        print(txt_og)
        print(df_txt.head())
        print('\n'+'~'*75+'\n')
    df.columns = ['isotope','a','z','yield','mass','metallicity','irv']
    dfy0 = df[df['yield']==0]
    print('setting %d rows with yield=0 to %.1e'%(len(dfy0),yield_eps))
    df.loc[df['yield']==0,'yield'] = yield_eps
    return df

if __name__ == '__main__':
    root = os.path.abspath(os.path.dirname(__file__))
    df = parse_c15_raw()
    df.to_csv(root+'/data.csv',index=False)
    df = df[df['irv']==0]
    fit_isotope_interpolants(
        df = df,
        root = root,
        tf_funs = {
            'mass':lambda x:np.log10(x), 'mass_prime':lambda x:1/(x*np.log(10)),
            'metallicity':lambda x:np.log10(x), 'metallicity_prime':lambda x:1/(x*np.log(10)),
            'yield':lambda y:np.log10(y), 'yield_prime':lambda y:1/(y*np.log(10)), 'yield_inv':lambda y:10**y},
        fit_names = 'all', # 'all', ['c15_z8.a16.irv0.O16'],
        plot_names =  [], # [], 'all', ['c15_z8.a16.irv0.O16'] 
        )