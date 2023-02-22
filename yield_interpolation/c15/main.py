import numpy as np
import pandas as pd
import os
from yield_interpolation.fit_isotope_interpolants import fit_isotope_interpolants

def parse_c15_raw(_path='galcem/input/yields/lims/c15/tot/'):
    yield_eps = 1e-13
    
    c15_phys = pd.read_fwf(_path+'All_All_All_0_last_tp_20221214_21933.txt')
    txts = os.listdir(_path+'data/')
    df = pd.DataFrame({col:[] for col in ['Isotope','A','Z','YIELD','MASS_INI','METALLICITY','IRV','MASS_EJ','MASS_FRACTION']})
    for txt_og in txts:
        print(txt_og)
        if '.DS_Store' in txt_og: continue
        df_txt = pd.read_fwf(_path+'data/'+txt_og, infer_nrows=400)
        txt = txt_og.replace('zsun','z1.4m2').replace('_20210617_33100.txt','').replace('yields_tot_m','')
        prts = txt.split('_')
        prts = prts[0].split('z')+[prts[1]]
        mass = float(prts[0].replace('p','.'))
        metallicity = float(prts[1].replace('z','').replace('m','e-'))
        irv = int(prts[2])
        c15_phys_idx = np.where(np.isclose(c15_phys['MASS'],mass) & 
                       np.isclose(c15_phys['IRV'],irv) & 
                       np.isclose(c15_phys['METALLICITY'],metallicity))
        if metallicity in [2e-5,5e-5,1e-4,3e-4]: metallicity = metallicity*2.4
        df_txt['MASS_INI'] = mass
        df_txt['METALLICITY'] = metallicity
        df_txt['IRV'] = irv
        df_txt['MASS_EJ'] = float(mass - c15_phys['M_H'].iloc[c15_phys_idx])
        df_txt['MASS_FRACTION'] = df_txt['YIELD']/df_txt['MASS_EJ']
        df_txt.loc[np.isclose(df_txt['MASS_EJ'],0),'MASS_FRACTION'] = yield_eps
        df_txt['YSIGN'] = np.sign(df_txt['YIELD'])
        df = df.append(df_txt)
        print(df_txt.head())
        print('\n'+'~'*75+'\n')
    #df.columns = ['isotope','a','z','yield','mass','metallicity','irv']
    #dfy0 = df[df['yield']==0]
    #print('setting %d rows with yield=0 to %.1e'%(len(dfy0),yield_eps))
    #df.loc[df['yield']==0,'yield'] = yield_eps
    print(f'{df.columns=}')
    df.columns = ['isotope','a','z','yield','mass','metallicity','irv','mass_ej','massfrac','ysign']
    return df

if __name__ == '__main__':
    root = os.path.abspath(os.path.dirname(__file__))
    for dirs in ['models', 'figs']:
        if not os.path.exists(root+'/'+dirs):
                os.makedirs(root+'/'+dirs)
    df = parse_c15_raw()
    df.to_csv(root+'/data.csv',index=False)
    df = df[df['irv']==0]
    fit_isotope_interpolants(
        df = df,
        root = root,
        tf_funs = {
            'mass':lambda x:np.log10(x), 'mass.prime':lambda x:1/(x*np.log(10)),
            'metallicity':lambda x:np.log10(x), 'metallicity.prime':lambda x:1/(x*np.log(10)),
            'massfrac':lambda y:np.log10(y), 'massfrac.prime':lambda y:1/(y*np.log(10)), 'massfrac.inv':lambda y:10**y},
        fit_names = 'all', # 'all', ['c15_z8.a16.irv0.O16'],
        plot_names =  'all', # [], 'all', ['c15_z8.a16.irv0.O16'] 
        )