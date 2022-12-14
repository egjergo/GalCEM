from yield_interpolation.galcem_interpolant import GalCemInterpolant,fit_isotope_interpolants
import numpy as np
import pandas as pd
import os

def parse_c15_raw():
    yield_eps = 1e-13
    root_c15_raw_data = 'galcem/input/yields/lims/c15/net/'
    c15_phys = pd.read_fwf(root_c15_raw_data+'All_All_All_0_last_tp_20221214_21933.txt')
    txts = os.listdir(root_c15_raw_data+'data/')
    df = pd.DataFrame({col:[] for col in ['Isotope','A','Z','YIELD','MASS_INI','METALLICITY','IRV','MASS_EJ','MASS_FRACTION']})
    for txt_og in txts:
        if '.DS_Store' in txt_og: continue
        df_txt = pd.read_fwf(root_c15_raw_data+'data/'+txt_og, infer_nrows=400)
        print(df_txt.head())
        txt = txt_og.replace('zsun','z1.4m2').replace('20221214_21534.txt','').replace('yields_net_m','')
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
        print('\n'+'~'*75+'\n')
    df.columns = ['isotope','a','z','yield','mass','metallicity','irv',
                  'mass_ej','massfrac','ysign']
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
            'mass':lambda x:np.log10(x), 
            'mass_prime':lambda x:1/(x*np.log(10)),
            'metallicity':lambda x:np.log10(x), 
            'metallicity_prime':lambda x:1/(x*np.log(10)),
            'massfrac':lambda y: np.sign(y)*np.abs(y)**(1/3),
            'massfrac_prime':lambda y:1/3*np.sign(y)*np.abs(y)**(-2/3),
            'massfrac_inv':lambda y:np.sign(y)*np.abs(y)**3,
            },
        fit_names = 'all', # 'all', ['c15_z8.a16.irv0.O16'],
        plot_names =  'all', # [], 'all', ['c15_z8.a16.irv0.O16'] 
        )