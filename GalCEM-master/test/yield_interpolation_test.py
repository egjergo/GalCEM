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
    print(f'X header: {id_vars+[var_name]}')
    print(f'Y header: {[value_name]}')
    return X, Y

def k10_test(i_elemZ, i_elemA, loc='input/yields/lims/k10', filename='k10_pandas.csv',
              id_vars=['Z_ini'], var_name='mass_ini', value_name='yields'):
    df = pd.read_csv(f'{loc}/{filename}', comment='#')
    df = df.loc[(df['elemZ'] == i_elemZ) & (df['elemA'] == i_elemA)]
    X = df[id_vars+[var_name]]#.values
    Y = df[value_name]#.values
    print(f'X header: {id_vars+[var_name]}')
    print(f'Y header: {[value_name]}')
    return X, Y

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

def save_processed_dataframes(X_or_Y, ZA_sorted, name='X or Y', func=lc18_test, loc='input/yields/snii/lc18/tab_R'):
    if name != 'models':
        with open(f'{loc}/processed/{name}_{func.__name__}.csv', 'a') as f:
            for i, val in enumerate(ZA_sorted):
                f.write(f'\n\tZ = {val[0]}, \tA = {val[1]}\n')
                X_or_Y[i].to_csv(f, sep='\t') 
    elif name != 'models':
        for i, val in enumerate(ZA_sorted):
            with open(f'{loc}/processed/interp/{name}_Z{val[0]}A{val[1]}_{func.__name__}.pkl', 'wb') as f:
                    pickle.dump(X_or_Y[i], f) 
    return None

def save_processed_yields(X, Y, models, func=lc18_test, loc='input/yields/snii/lc18/tab_R', **kwargs):
    if not os.path.exists(f'{loc}/processed'):
        os.makedirs(f'{loc}/processed')

    if os.path.exists(f'{loc}/processed/X_{func.__name__}.csv'):
        while True:
            choice = input('Processed X yield file already exists. Overwrite? (y/n)\n')
            if choice.lower() == 'y':
                save_processed_dataframes(X, ZA_sorted, name='X', func=func, loc=loc)
            elif choice.lower() == 'n':
                break
            else:
                print('Not a valid choice. Pick "y" or "n".')
                continue
    else:
        save_processed_dataframes(X, ZA_sorted, func=func, name='X', loc=loc)
 
    if os.path.exists(f'{loc}/processed/Y_{func.__name__}.csv'):
        while True:
            choice = input('Processed Y yield file already exists. Overwrite? (y/n)\n')
            if choice.lower() == 'y':
                save_processed_dataframes(Y, ZA_sorted, name='Y', func=func, loc=loc)
            elif choice.lower() == 'n':
                break
            else:
                print('Not a valid choice. Pick "y" or "n".')
                continue
    else:
        save_processed_dataframes(Y, ZA_sorted, func=func, name='Y', loc=loc)

    if not os.path.exists(f'{loc}/processed/interp'):
        os.makedirs(f'{loc}/processed/interp')
    #while True:
    #    choice: input('Processed models file already exists. Overwrite? (y/n)\n')
    #    if choice.lower() == 'y':
    #        save_processed_dataframes(models, ZA_sorted, name='models', func=func, loc=loc)
    #    elif choice.lower() == 'n':
    #        break
    #    else:
    #        print('Not a valid choice. Pick "y" or "n".')
    #        continue     
    #else:
    save_processed_dataframes(models, ZA_sorted, name='models', func=func, loc=loc)
    return None

def read_processed_yields(ZA_sorted, loc='', func=None):
    X, Y, models = [], [], []
    return X, Y, models

X_lc18, Y_lc18, models_lc18 = test_for_ZA_sorted(lc18_test)
#run_test(X_lc18, Y_lc18, models_lc18, lc18_test)
#save_processed_yields(X_lc18, Y_lc18, models_lc18, func=lc18_test, loc='input/yields/snii/lc18/tab_R')
#X_k10, Y_k10, models_k10 = test_for_ZA_sorted(k10_test)