from yield_interpolation.interpolants import fit_3d_interpolant

from numpy import *
import pandas as pd
import os

root_c15 = 'galcem/input/yields/lims/c15/'
root_c15_data = root_c15+'data/'

def c15_to_df(debug=False):
    txts = os.listdir(root_c15_data)
    df = pd.DataFrame({col:[] for col in ['Isotope','A','Z','YIELD','MASS','METALLICITY','IRV']})
    for txt_og in txts:
        if '.DS_Store' in txt_og: continue
        df_txt = pd.read_fwf(root_c15_data+txt_og)
        txt = txt_og.replace('zsun','z1.4m2').replace('_20210617_33100.txt','').replace('yields_tot_m','')
        prts = txt.split('_')
        prts = prts[0].split('z')+[prts[1]]
        mass = float(prts[0].replace('p','.'))
        metallicity = float(prts[1].replace('z','').replace('m','e-'))
        irv = int(prts[2])
        if metallicity in [2e-5, 5e-5, 1e-4, 3e-4]: metallicity = metallicity*2.4
        df_txt['MASS'] = mass
        df_txt['METALLICITY'] = metallicity
        df_txt['IRV'] = irv
        df = df.append(df_txt)
        if debug:
            print(txt_og)
            print(df_txt.head())
            print('\n'+'~'*100+'\n')
    df.to_csv(root_c15+'data.csv',index=False)

def fit_c15_interpolants():
    df = pd.read_csv(root_c15+'data.csv')
    dfy0 = df[df['YIELD']==0]
    print('dropping %d rows with YIELD=0'%len(dfy0))
    df = df[df['YIELD']!=0]
    df['LOG10_METALLICITY'] = log10(df['METALLICITY'])
    df['LOG10_YIELD'] = log10(df['YIELD'])
    dfs = dict(tuple(df.groupby(['Isotope','A','Z'])))
    print(df['YIELD'].min(),df['METALLICITY'].min())
    for ids,_df in dfs.items():
        print('fitting interpolant:',ids)
        fit_3d_interpolant(
            dfx = _df[['LOG10_METALLICITY','MASS','IRV']],
            dfy = _df['LOG10_YIELD'],
            tag = '%s.A%d.Z%d'%ids,
            test = True,
            y_log10_scaled = True,
            view_angle = 135,
            figroot = 'yield_interpolation/figs/c15/')
        print('\n'+'~'*100+'\n')

if __name__ == '__main__':
    c15_to_df(debug=False)
    fit_c15_interpolants()
