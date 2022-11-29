import os
import dill
from ..FriendlyInterpolants import LinearAndNearestNeighbor_GCI

def fit_isotope_interpolants(df,root,tf_funs,fit_names=[],plot_names=[]):
    # iterate over a,z pairs and save interpolants based on irv=0
    print('\n'+'~'*75+'\n')
    dirname = os.path.basename(root)
    dfs = dict(tuple(df.groupby(['isotope','a','z'])))
    itotal = len(dfs)
    for i,(ids,_df) in enumerate(dfs.items()):
        name = '%s_z%d.a%d.irv0.%s'%(dirname,ids[2],ids[1],ids[0])
        if fit_names!='all' and name not in fit_names: continue
        # fit model
        interpolant = LinearAndNearestNeighbor_GCI(
            df = _df[['metallicity','mass','yield']],
            ycol = 'yield',
            tf_funs = tf_funs,
            name = name,
            plot = [[0,0]] if plot_names=='all' or name in plot_names else None,
            fig_root = root+'/figs/',
            fig_view_angle = 135,
            colormap=False)
        #   print model
        print('%d of %d'%(i+1,itotal))
        print(interpolant)
        #   save model
        dill.dump(interpolant,open(root+'/models/%s.pkl'%name,'wb'))
        #   load model
        interpolant_loaded = dill.load(open(root+'/models/%s.pkl'%name,'rb'))
        #   example model use
        yquery = interpolant_loaded(_df)
        print('~'*75+'\n')