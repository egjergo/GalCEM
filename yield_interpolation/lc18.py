from yield_interpolation.interpolant import GalCemInterpolant
import pandas as pd
import pickle

if __name__ == '__main__':
    dfxs = pickle.load(open('galcem/input/yields/snii/lc18/tab_R/processed/X.pkl','rb'))
    dfys = pickle.load(open('galcem/input/yields/snii/lc18/tab_R/processed/Y.pkl','rb'))
    for i,(dfx,dfy) in enumerate(zip(dfxs,dfys)):
        if dfx.empty:
            print("dfxs[%d] empty\n\n%s\n"%(i,'~'*75))
            continue
        print('fitting interpolant %s\n'%i)
        v0idxs = dfx['vel_ini']==0
        dfx,dfy = dfx[v0idxs],dfy[v0idxs]
        # fit model
        interpolant = GalCemInterpolant(
            s_y = dfy,
            dfx = dfx[['mass_ini','Z_ini']],
            xlog10cols = ['Z_ini'],
            ylog10col = True)
        #   print model
        print(interpolant)
        #   plot model
        interpolant.plot(
            xcols = ['mass_ini','Z_ini'],
            xfixed = {},
            figroot = 'yield_interpolation/figs/lc18/%s'%i,
            title = 'Lifetime by Mass, Metallicity',
            view_angle = -45)
        #   save model
        pickle.dump(interpolant,open('yield_interpolation/interpolants/lc18/%s.pkl'%i,'wb'))
        #   load model
        interpolant_loaded = pickle.load(open('yield_interpolation/interpolants/lc18/%s.pkl'%i,'rb'))
        #   example model use
        xquery = pd.DataFrame({'mass_ini':[15],'Z_ini':[0.01648]})
        yquery = interpolant_loaded(xquery)
        print('~'*75+'\n')
