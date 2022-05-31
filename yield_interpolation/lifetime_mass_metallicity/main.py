from yield_interpolation.galcem_interpolant import GalCemInterpolant
import pandas as pd
import numpy as np
import dill
import os

def parse_lifetime_mass_metallicity_raw():
    # parse input tdata
    df = pd.read_csv('galcem/input/starlifetime/portinari98table14.dat')
    df.columns = [name.replace('#M','mass').replace('Z=','') for name in df.columns]
    df = pd.melt(df,id_vars='mass',value_vars=list(df.columns[1:]),var_name='metallicity',value_name='lifetime_yr')
    df['lifetime_Gyr'] = df['lifetime_yr']/1e9
    df.drop('lifetime_yr',axis=1,inplace=True)
    df['metallicity'] = df['metallicity'].astype(float)
    return df

def fit_lifetime_mass_metallicity_interpolants(df,root):
    # lifetime by mass, metallicity
    #   fit model
    lifetime_by_mass_metallicity = GalCemInterpolant(
        df = df,
        ycol = 'lifetime_Gyr',
        tf_funs = {
            'mass':lambda x:np.log10(x), 'mass_prime':lambda x:1/(x*np.log(10)),
            'metallicity':lambda x:np.sqrt(x), 'metallicity_prime':lambda x:1/(2*np.sqrt(x)),
            'lifetime_Gyr':lambda y:np.log10(y), 'lifetime_Gyr_prime':lambda y:1/(y*np.log(10)), 'lifetime_Gyr_inv':lambda y:10**y},
        name = 'LifetimeInterpolant',
        plot = True,
        fig_root = root+'/figs/',
        fig_view_angle = 45)
    #   print model
    print(lifetime_by_mass_metallicity)
    #   save model
    dill.dump(lifetime_by_mass_metallicity,open(root+'/models/lifetime_by_mass_metallicity.pkl','wb'))
    #   load model
    lifetime_by_mass_metallicity_loaded = dill.load(open(root+'/models/lifetime_by_mass_metallicity.pkl','rb'))
    #   example model use
    yquery = lifetime_by_mass_metallicity_loaded(df)
    dyquery_dmass = lifetime_by_mass_metallicity_loaded(df,dwrt='mass')
    dyquery_dmetallicity = lifetime_by_mass_metallicity_loaded(df,dwrt='metallicity')
    # mass by lifetime, metallicity
    mass_by_lifetime_metallicity = GalCemInterpolant(
        df = df,
        ycol = 'mass',
        tf_funs = {
            'lifetime_Gyr':lambda x:np.log10(x), 'lifetime_Gyr_prime':lambda x:1/(x*np.log(10)),
            'metallicity':lambda x:np.sqrt(x), 'metallicity_prime':lambda x:1/(2*np.sqrt(x)),
            'mass':lambda y:np.log10(y), 'mass_prime':lambda y:1/(y*np.log(10)), 'mass_inv':lambda y:10**y},
        name = 'MassInterpolant',
        plot = True,
        fig_root = root+'/figs/',
        fig_view_angle = 45)
    #       print model
    print(mass_by_lifetime_metallicity)
    #       save model
    dill.dump(mass_by_lifetime_metallicity,open(root+'/models/mass_by_lifetime_metallicity.pkl','wb'))
    #       load model
    mass_by_lifetime_metallicity_loaded = dill.load(open(root+'/models/mass_by_lifetime_metallicity.pkl','rb'))
    #       example model use
    yquery = mass_by_lifetime_metallicity_loaded(df)
    dyquery_dlifetime = mass_by_lifetime_metallicity_loaded(df,dwrt='lifetime_Gyr')
    dyquery_dmetallicity = mass_by_lifetime_metallicity_loaded(df,dwrt='metallicity')

if __name__ == '__main__':
    root = os.path.abspath(os.path.dirname(__file__))
    df = parse_lifetime_mass_metallicity_raw()
    df.to_csv(root+'/data.csv',index=False)
    fit_lifetime_mass_metallicity_interpolants(df,root)
    