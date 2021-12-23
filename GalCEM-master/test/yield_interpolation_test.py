import os
import pickle
import numpy as np
import pandas as pd
import scipy.interpolate as interp
from matplotlib import pyplot as plt
from matplotlib import cm
from test.util import mplsetup
mplsetup()
from prep.setup import *

def interpolation(X,Y, kernel='linear'):
    return interp.RBFInterpolator(X,Y, kernel=kernel)

def pass_meshes(X, Y, model, Xindex=None, nticks=64):
    if Xindex == 0:
        column_bool = [True, False, True]
    if Xindex == 1:
        column_bool = [False, True, True]
    xlim = np.array([X.loc[:, column_bool].values.min(0),X.loc[:, column_bool].values.max(0)])
    ylim = np.array([Y.values.min(),Y.values.max()])
    ticks = [np.linspace(*xlim[:,i],nticks) for i in range(X.shape[1])]
    x1mesh, x2mesh, x3mesh = np.meshgrid(*ticks)
    xquery = np.hstack([x1mesh.reshape(-1,1),x2mesh.reshape(-1,1),x3mesh.reshape(-1,1)])
    yquery = model(xquery)
    ymesh = yquery.reshape(x1mesh.shape)
    return xlim, ylim, x1mesh, x2mesh, ymesh, column_bool

def plot_interpolation(X, Y, model, func, modelname=''):
    cmapc = [cm.plasma, cm.winter]
    color_scatter = ['k', 'r']
    fig,axs = plt.subplots(nrows=1,ncols=2,figsize=(15,7),subplot_kw={"projection": "3d"})
    for i, ax in enumerate(axs):
        xlim, ylim, x1mesh, x2mesh, ymesh, column_bool = pass_meshes(X, Y, model, Xindex=i)
        surf = ax[i].plot_surface(x1mesh, x2mesh, ymesh, cmap=cmapc[i], alpha=.75)
        ax[i].colorbar(surf, shrink=0.5, aspect=5)
        ax[i].scatter(X.loc[:, column_bool].values[:,0],X.loc[:, column_bool].values[:,1],
                      Y.values, color=color_scatter[i])
        ax[i].set_xlim(xlim[:,0])
        ax[i].set_ylim(xlim[:,1])
        ax[i].set_zlim(ylim)
        ax[i].set_zlabel(Y.columns[0], fontsize=20, y=1.1, rotation=90)
        ax[i].set_xlabel(X.loc[:, column_bool].values.columns[0], fontsize=20, y=1.1)
        ax[i].set_ylabel(X.loc[:, column_bool].columns[1], fontsize=20, x=-0.1)
        ax[i].set_title(str(modelname), fontsize=25)
    fig.savefig(f'figures/test/yields_{func.__name__}_{modelname}.pdf', format='pdf', bbox_inches='tight')
    plt.show(block=False)
    return None

def interpolation_test(X,Y, model, func, modelname=' ', nticks=64):
    xlim = np.array([X.min(0),X.max(0)])
    ylim = np.array([Y.min(),Y.max()])
    ticks = [np.linspace(*xlim[:,i],nticks) for i in range(X.shape[1])]
    x1mesh, x2mesh, x3mesh = np.meshgrid(*ticks)
    xquery = np.hstack([x1mesh.reshape(-1,1),x2mesh.reshape(-1,1), x3mesh.reshape(-1,1)])
    yquery = model(xquery)
    Yhat = model(X)
    eps = Y-Yhat
    abs_eps = np.abs(eps)
    print(f'For the isotope {modelname} extracted from {func.__name__} the interpolation performs as follows:')
    print('\nRMSE: %.1e'%np.sqrt(np.mean(abs_eps**2)))
    print('MAE: %.1e'%np.mean(abs_eps))
    print('Max Abs Error: %.1e'%abs_eps.max())
    print(' ')
    ymesh = yquery.reshape(x1mesh.shape)
    return None

def run_test(X,Y, models, func):
    for i, val in enumerate(X):
        if val.shape[0] > 0:
            print(f'{i=}')
            interpolation_test(X[i].values, Y[i].values, models[i], 
            func, modelname=f' Z={ZA_sorted[i,0]}, A={ZA_sorted[i,1]} ')
    return None

def lc18_test(i_elemZ, i_elemA, loc='input/yields/snii/lc18/tab_R', filename='lc18_pandas.csv',
              id_vars=['Z_ini', 'vel_ini'], var_name='mass_ini', value_name='yields',
              col_number=9):
    '''
    !!!!!!! will work only if the columns to melt are appended last
    otherwise, change df.columns

        id_vars     existing columns to include in X
        var_name    new column name for melted headers
        value_name  entry descriptor (will always be yields)
        col_number  number of columns to melt 
    '''
    df = pd.read_csv(f'{loc}/{filename}', comment='#')
    df = df.loc[(df['elemZ'] == i_elemZ) & (df['elemA'] == i_elemA)]
    df = pd.melt(df,id_vars=id_vars, value_vars=list(df.columns[-col_number:]),
                 var_name=var_name, value_name=value_name)
    df = df.apply(pd.to_numeric)
    X = df[id_vars+[var_name]]#.values
    Y = df[value_name]#.values
    Y /= df['mass_ini']
    #print(f'X header: {id_vars+[var_name]}')
    #print(f'Y header: {[value_name]}')
    return X, Y

def k10_test(i_elemZ, i_elemA, loc='input/yields/lims/k10', filename='k10_pandas.csv',
              id_vars=['Z_ini'], var_name='mass_ini', value_name='Xi_avg'):
    df_tot = pd.read_csv(f'{loc}/{filename}', comment='#')
    df_al26 = df_tot.loc[df_tot["elemSymb"].astype(str) == 'al*6 ']
    df = pd.concat([df_tot, df_al26]).drop_duplicates(keep=False)
    df = df.loc[(df['elemZ'] == i_elemZ) & (df['elemA'] == i_elemA)]
    X = df[id_vars+[var_name]]#.values
    Y = df[value_name]#.values
    print(f'X header: {id_vars+[var_name]}')
    print(f'Y header: {[value_name]}')
    return X, Y

def i99_test(i_elemZ, i_elemA, loc='input/yields/snia/i99', filename='table4.dat', value_name='CDD1'):
    df = pd.read_csv(f'{loc}/{filename}', comment='#', sep='\t')
    select = (df['elemZ'] == i_elemZ) & (df['elemA'] == i_elemA)
    df_pick = df.loc[select]
    if df_pick[value_name].empty:
        return 0. 
    else:
        return df_pick[value_name].values[0]

def test_for_ZA_sorted(func):
    X, Y, models = [], [], []
    for i, val in enumerate(ZA_sorted):
        print(f'{i = }')
        Xi, Yi = func(val[0], val[1])
        X.append(Xi)
        Y.append(Yi)
        if Xi.size != 0:
            models.append(interpolation(Xi.values,Yi.values))
        else:
            models.append(None)
            print('X is empty')
    return X, Y, models

def test_for_ZA_sorted_nomodel(func=i99_test):
    '''For yields with no X dependence e.g. i99_test'''
    yields = np.zeros(len(ZA_sorted))
    for i, val in enumerate(ZA_sorted):
        yields[i] = func(val[0], val[1])
    return yields

def save_processed_dataframes(X_or_Y_or_models, ZA_sorted, name='X or Y or models', func_name='lc18', loc='input/yields/snii/lc18/tab_R'):
    with open(f'{loc}/processed/{name}_{func_name}.pkl', 'wb') as f:
        pickle.dump(X_or_Y_or_models, f) 
    return None

def save_processed_yields(X, Y, models, func_name='lc18', loc='input/yields/snii/lc18/tab_R'):
    if not os.path.exists(f'{loc}/processed'):
        os.makedirs(f'{loc}/processed')

    df_list = ['X', 'Y', 'models']
    df_dict = {'X': X, 'Y': Y, 'models': models}
    for df_l in df_list:
        if os.path.exists(f'{loc}/processed/{df_l}_{func_name}.pkl'):
            while True:
                choice = input('Processed X yield file already exists. Overwrite? (y/n)\n')
                if choice.lower() == 'y':
                    save_processed_dataframes(df_dict[df_l], ZA_sorted, name=df_l, func_name=func_name, loc=loc)
                elif choice.lower() == 'n':
                    break
                else:
                    print('Not a valid choice. Pick "y" or "n".')
                    continue
        else:
            save_processed_dataframes(df_dict[df_l], ZA_sorted, name=df_l, func_name=func_name, loc=loc)
    return None

def load_processed_yields(func_name='lc18', loc='input/yields/snii/lc18/tab_R', df_list = ['X', 'Y', 'models']):
    df_dict = {}
    for df_l in df_list:
        #with open(f'{loc}/processed/{df_l}_{func_name}.pkl', 'rb') as pickle_file:
        with open(f'{loc}/processed/{df_l}.pkl', 'rb') as pickle_file:
            df_dict[df_l] = pickle.load(pickle_file)
    return [df_dict[d] for d in df_list]#df_dict[df_list[0]], df_dict[df_list[1]]#, df_dict[df_list[2]]

def load_processed_yields_snia(func_name='lc18', loc='input/yields/snii/lc18/tab_R', df_list = ['X', 'Y']):#, 'models']):
    df_dict = {}
    for df_l in df_list:
        #with open(f'{loc}/processed/{df_l}_{func_name}.pkl', 'rb') as pickle_file:
        with open(f'{loc}/processed/{df_l}.pkl', 'rb') as pickle_file:
            df_dict[df_l] = pickle.load(pickle_file)
    return df_dict[df_list[0]]

#if __name__ == '__main__':
    #X_lc18, Y_lc18, models_lc18 = test_for_ZA_sorted(lc18_test)
    #save_processed_yields(X_lc18, Y_lc18, models_lc18, func_name='lc18', loc='input/yields/snii/lc18/tab_R')
    #X_k10, Y_k10, models_k10 = test_for_ZA_sorted(k10_test)
    #save_processed_yields(X_k10, Y_k10, models_k10, func_name='k10', loc='input/yields/lims/k10')
    ##run_test(X_lc18, Y_lc18, models_lc18, lc18_test)