from yield_interpolation.FriendlyInterpolants.friendly_interpolants import SmootheSpline2D_FI
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

def plot_mod(fig):
    from matplotlib import cm
    axs = fig.get_axes()
    s_lifetimes_p98 = pd.read_csv('galcem/input/starlifetime/portinari98table14.dat')
    s_lifetimes_p98.columns = [name.replace('#M','M').replace('Z=0.','Z') for name in s_lifetimes_p98.columns]
    colordots = np.array([s_lifetimes_p98[c].to_numpy() for c in s_lifetimes_p98.columns[1:]]).flatten()
    x0label,x1label,ylabel = axs[11].get_xlabel(),axs[11].get_ylabel(),axs[11].get_zlabel()
    x0label,x1label,ylabel = (lab.replace('$\\mathrm{','').replace('}$','') for lab in [x0label,x1label,ylabel])
    axs[9].scatter(df[x0label],df[x1label],c=colordots, cmap=cm.get_cmap('Paired'))
    axs[11].scatter(df[x0label],df[x1label],df[ylabel],c=colordots, cmap=cm.get_cmap('Paired'))

def fit_lifetime_mass_metallicity_interpolants(df,root):
    # lifetime by mass, metallicity
    #   fit model
    lifetime_by_mass_metallicity = SmootheSpline2D_FI(
        df = df,
        ycol = 'lifetime_Gyr',
        tf_funs = {
            'mass':lambda x:x**(1/3.), 'mass.prime':lambda x:1 / (3 * x**(2/3.)),
            'metallicity':lambda x:np.sqrt(x), 'metallicity.prime':lambda x:1/(2*np.sqrt(x)),
            'lifetime_Gyr':lambda y:y**(1/3.), 'lifetime_Gyr.prime':lambda y:1 / (3 * y**(2/3.)), 'lifetime_Gyr.inv':lambda y:y**3},
        name = 'LifetimeInterpolant',
        plot = [None,'mass','metallicity'],#False,
        fig_root = root+'/figs/',
        plot_ops={'scatter':False,'sepfl':0,'sepfr':0,'sepfb':0,'sepft':0},
        plot_mod = plot_mod)
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
    mass_by_lifetime_metallicity = SmootheSpline2D_FI(
        df = df[['lifetime_Gyr','mass','metallicity']],
        ycol = 'mass',
        tf_funs = {
            'lifetime_Gyr':lambda x:x**(1/3.), 'lifetime_Gyr.prime':lambda x:1 / (3 * x**(2/3.)),
            'metallicity':lambda x:np.sqrt(x), 'metallicity.prime':lambda x:1/(2*np.sqrt(x)),
            'mass':lambda y:y**(1/3.), 'mass.prime':lambda y:1/(3 * y**(2/3.)), 'mass.inv':lambda y:y**3},
        name = 'MassInterpolant',
        plot = [None,'lifetime_Gyr','metallicity'], #False,
        fig_root = root+'/figs/',
        plot_ops={'scatter':False,'sepfl':0,'sepfr':0,'sepfb':0,'sepft':0},
        plot_mod = plot_mod)
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
    print(np.multiply(dyquery_dmass, dyquery_dlifetime))
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(np.multiply(dyquery_dmass, dyquery_dlifetime))
    plt.show(block=False)

if __name__ == '__main__':
    root = os.path.abspath(os.path.dirname(__file__)) # 'yield_interpolation/lifetime_mass_metallicity'
    for dirs in ['models', 'figs']:
        if not os.path.exists(root+'/'+dirs):
                os.makedirs(root+'/'+dirs)
    df = parse_lifetime_mass_metallicity_raw()
    df.to_csv(root+'/data.csv',index=False)
    fit_lifetime_mass_metallicity_interpolants(df,root)
    