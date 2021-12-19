import numpy as np
import pickle
from matplotlib import pyplot,cm

X = pickle.load(open('input/yields/snii/lc18/tab_R/processed/X_lc18.pkl','rb'))
Y = pickle.load(open('input/yields/snii/lc18/tab_R/processed/Y_lc18.pkl','rb'))
models = pickle.load(open('input/yields/snii/lc18/tab_R/processed/models_lc18.pkl','rb'))

indices = [19,99]#,103] # oxygen, iron56, iron60
ni = len(indices)
velocities = X[0]['vel_ini'].unique().tolist()
nv = len(velocities)

fig,ax = pyplot.subplots(nrows=ni,ncols=nv,figsize=(5*nv,5*ni),subplot_kw={"projection": "3d"})

for i,idx in enumerate(indices):
    print(f'{i=}, {idx=}')
    model = models[idx]
    print(f'{model=}')
    x = X[idx]
    print(f'{x=}')
    data = x.copy()
    print(f'{data=}')
    data['y'] = Y[idx]
    print(f"{data['y']=}")
    data['yhat'] = model(x)
    print(f"{data['yhat']=}")
    data['eps'] = np.abs(data['y']-data['yhat'])
    rmse = np.sqrt(np.mean(data['eps']**2))
    mae = np.mean(data['eps'])
    print('element %i \t\trmse = %-10.1e mae = %-10.1e'%(idx,rmse,mae))
    for j,velocity in enumerate(velocities):
        dataj = data[data['vel_ini']==velocity]
        masses = dataj['mass_ini'].to_numpy()
        logZs = np.log10(dataj['Z_ini'].to_numpy())
        logys = np.log10(dataj['y'].to_numpy())
        logyhats = np.log10(dataj['yhat'].to_numpy())
        n_mass = len(np.unique(masses))
        n_Z = len(np.unique(logZs))
        massgrid = masses.reshape((n_mass,n_Z))
        logZgrid = logZs.reshape((n_mass,n_Z))
        logyhatgrid = logyhats.reshape((n_mass,n_Z))
        ax[i,j].plot_surface(massgrid,logZgrid,logyhatgrid,cmap=cm.Greys,alpha=0.95)
        ax[i,j].scatter(masses,logZs,logys,color='r')
        ax[i,j].set_xlabel('mass_ini')
        ax[i,j].set_ylabel('log(Z_ini)')
        ax[i,j].set_zlabel('log(y)')
        ax[i,j].set_title('element %d, velocity %d'%(idx,velocity))
        rmsej = np.sqrt(np.mean(dataj['eps']**2))
        maej = np.mean(dataj['eps'])
        print('\tvelocity %d\trmse = %-10.1e mae = %-10.1e'%(velocity,rmsej,maej))

fig.savefig('figures/test/velocity_plots.pdf',format='pdf',bbox_inches='tight')

