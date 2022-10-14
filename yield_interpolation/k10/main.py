from yield_interpolation.galcem_interpolant import GalCemInterpolant,fit_isotope_interpolants
import numpy as np
import pandas as pd
import os

def parse_k10_raw():
    root_k10_raw_data = 'galcem/input/yields/lims/k10/'
    dfs = []
    for metallicity in ['.0001','.02','.004','.008']:
        df = pd.read_csv(root_k10_raw_data+'Z0%s.dat'%metallicity,comment='#',header=None)
        df.columns = ['mass_ini','Z_ini','mass','isotope','z','a','net_yield','Mi_windloss','Mi_ini','Xi_avg','Xi_ini','ProdFact']
        df['metallicity'] = float(metallicity)
        dfs.append(df)
    df = pd.concat(dfs,axis=0)
    df['yield'] = df['net_yield']+df['Xi_ini']*(df['mass_ini']-df['mass'])
    df['isotope'] = df['isotope'].str.strip()
    df.loc[df['isotope']=='al-6','isotope'] = 'al26'
    df = df[df['isotope']!='al*6']
    df = df[['isotope','a','z','yield','mass','metallicity']]
    #df.loc[df['yield']<=0,'yield'] = df.loc[df['yield']>0,'yield'].min()
    df.loc[df['yield']<0,'yield'] = 0
    return df

if __name__ == '__main__':
    root = os.path.abspath(os.path.dirname(__file__))
    df = parse_k10_raw()
    df.to_csv(root+'/data.csv',index=False)
    fit_isotope_interpolants(
        df = df,
        root = root,
        tf_funs = {
            'mass':lambda x:np.log10(x), 'mass_prime':lambda x:1/(x*np.log(10)),
            'metallicity':lambda x:np.log10(x), 'metallicity_prime':lambda x:1/(x*np.log(10)),
            #'yield':lambda y:np.log10(y), 'yield_prime':lambda y:1/(y*np.log(10)), 'yield_inv':lambda y:10**y
        },
        fit_names = 'all', # 'all', ['k10_z8.a16.irv0.o16'],
        plot_names = [], # [], 'all', ['k10_z8.a16.irv0.o16']
        ) 