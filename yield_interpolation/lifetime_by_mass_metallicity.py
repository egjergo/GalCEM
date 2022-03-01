from yield_interpolation.interpolant import GalCemInterpolant
import pandas as pd
import pickle

if __name__ == '__main__':
    # parse input tdata
    df = pd.read_csv('galcem/input/starlifetime/portinari98table14.dat')
    df.columns = [name.replace('#M','mass').replace('Z=','') for name in df.columns]
    df = pd.melt(df,id_vars='mass',value_vars=list(df.columns[1:]),var_name='metallicity',value_name='lifetime_yr')
    df['lifetime_Gyr'] = df['lifetime_yr']/1e9
    df['metallicity'] = df['metallicity'].astype(float)
    # lifetime by mass, metallicity
    #   fit model
    lifetime_by_mass_metallicity = GalCemInterpolant(
        s_y = df['lifetime_Gyr'],
        dfx = df[['mass','metallicity']],
        xlog10cols = ['mass'],
        ylog10col = True)
    #   print model
    print(lifetime_by_mass_metallicity)
    #   plot model
    lifetime_by_mass_metallicity.plot(
        xcols = ['mass','metallicity'],
        xfixed = {},
        figroot = 'yield_interpolation/figs/lifetime_mass_metallicity/lifetime_by_mass_metallicity',
        title = 'Lifetime by Mass, Metallicity',
        view_angle = -45)
    #   save model
    pickle.dump(lifetime_by_mass_metallicity,open('yield_interpolation/interpolants/lifetime_mass_metallicity/lifetime_by_mass_metallicity.pkl','wb'))
    #   load model
    lifetime_by_mass_metallicity_loaded = pickle.load(open('yield_interpolation/interpolants/lifetime_mass_metallicity/lifetime_by_mass_metallicity.pkl','rb'))
    #   example model use
    xquery = pd.DataFrame({'mass':[15],'metallicity':[0.01648]})
    yquery = lifetime_by_mass_metallicity_loaded(xquery)
    # mass by lifetime, metallicity
    mass_by_lifetime_metallicity = GalCemInterpolant(
        s_y = df['mass'],
        dfx = df[['lifetime_Gyr','metallicity']],
        xlog10cols = ['lifetime_Gyr'],
        ylog10col = True)
    #       print model
    print(mass_by_lifetime_metallicity)
    #       plot model
    mass_by_lifetime_metallicity.plot(
        xcols = ['lifetime_Gyr','metallicity'],
        xfixed = {},
        figroot = 'yield_interpolation/figs/lifetime_mass_metallicity/mass_by_lifetime_metallicity',
        title = 'Mass by Lifetime, Metallicity',
        view_angle = -45)
    #       save model
    pickle.dump(mass_by_lifetime_metallicity,open('yield_interpolation/interpolants/lifetime_mass_metallicity/mass_by_lifetime_metallicity.pkl','wb'))
    #       load model
    mass_by_lifetime_metallicity_loaded = pickle.load(open('yield_interpolation/interpolants/lifetime_mass_metallicity/mass_by_lifetime_metallicity.pkl','rb'))
    #       example model use
    xquery = pd.DataFrame({'lifetime_Gyr':[5,4],'metallicity':[0.01648,0.005]})
    yquery = mass_by_lifetime_metallicity_loaded(xquery)