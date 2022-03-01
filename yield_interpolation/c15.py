from yield_interpolation.interpolant import GalCemInterpolant
import pandas as pd
import pickle
import os

if __name__ == '__main__':
    # parameters
    combine_input_data = False
    debug = True
    root_c15 = 'galcem/input/yields/lims/c15/'
    yield_eps = 1e-13
    # combine input data and save
    if combine_input_data:
        root_c15_data = root_c15+'data/'
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
                print('\n'+'~'*75+'\n')
        df.columns = ['isotope','a','z','yield','mass','metallicity','irv']
        df.to_csv(root_c15+'data.csv',index=False)
    # load and parse combined input data
    df = pd.read_csv(root_c15+'data.csv')
    dfy0 = df[df['yield']==0]
    print('setting %d rows with yield=0 to %.1e'%(len(dfy0),yield_eps))
    df.loc[df['yield']==0,'yield'] = yield_eps
    # iterate over a,z pairs and save interpolants based on irv=0
    print('\n'+'~'*75+'\n')
    dfs = dict(tuple(df.groupby(['isotope','a','z'])))
    for ids,_df in dfs.items():
        tag = '%s.a%d.z%d.irv0'%ids
        print('fitting interpolant %s\n'%tag)
        _df = _df[_df['irv']==0]
        # fit model
        interpolant = GalCemInterpolant(
            s_y = _df['yield'],
            dfx = _df[['mass','metallicity']],
            xlog10cols = ['metallicity'],
            ylog10col = True)
        #   print model
        print(interpolant)
        #   plot model
        interpolant.plot(
            xcols = ['mass','metallicity'],
            xfixed = {},
            figroot = 'yield_interpolation/figs/c15/c15_%s'%tag,
            title = 'Lifetime by Mass, Metallicity',
            view_angle = -45)
        #   save model
        pickle.dump(interpolant,open('yield_interpolation/interpolants/c15/c15_%s.pkl'%tag,'wb'))
        #   load model
        interpolant_loaded = pickle.load(open('yield_interpolation/interpolants/c15/c15_%s.pkl'%tag,'rb'))
        #   example model use
        xquery = pd.DataFrame({'mass':[15],'metallicity':[0.01648]})
        yquery = interpolant_loaded(xquery)
        print('~'*75+'\n')
