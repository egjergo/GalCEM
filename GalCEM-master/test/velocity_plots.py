from scipy.interpolate import RBFInterpolator, LinearNDInterpolator, NearestNDInterpolator
import numpy as np
import pandas as pd
import pickle
from matplotlib import pyplot,cm

X = pickle.load(open('input/yields/snii/lc18/tab_R/processed/X_lc18.pkl','rb'))
Y = pickle.load(open('input/yields/snii/lc18/tab_R/processed/Y_lc18.pkl','rb'))
#models = pickle.load(open('input/yields/snii/lc18/tab_R/processed/models_lc18.pkl','rb'))

indices = [19,99] # oxygen, iron56
ni = len(indices)
velocities = X[0]['vel_ini'].unique().tolist()
nv = len(velocities)
nmesh = 64

fig,ax = pyplot.subplots(nrows=ni,ncols=nv,figsize=(5*nv,5*ni),subplot_kw={"projection": "3d"})
for i,idx in enumerate(indices):
    data = X[idx].copy()
    data['y'] = Y[idx].copy()
    data['logy'] = np.log(data['y'])
    data['logZ_ini'] = np.log(data['Z_ini'])
    xdata = data[['logZ_ini','vel_ini','mass_ini']].to_numpy()
    ydata = data['logy'].to_numpy()
    #model = RBFInterpolator(xdata,ydata,kernel='thin_plate_spline')
    model = LinearNDInterpolator(xdata,ydata,rescale=True,fill_value=np.nan)
    data['logyhat'] = model(xdata)
    data['yhat'] = np.exp(data['logyhat'])
    data['eps_abs'] = np.abs(data['y']-data['yhat'])
    rmse = np.sqrt(np.mean(data['eps_abs']**2))
    mae = np.mean(data['eps_abs'])
    print('element %i \t\trmse = %-10.1e mae = %-10.1e'%(idx,rmse,mae))
    for j,velocity in enumerate(velocities):
        dataj = data[data['vel_ini']==velocity]
        rmsej = np.sqrt(np.mean(dataj['eps_abs']**2))
        maej = np.mean(dataj['eps_abs'])
        print('\tvelocity %d\trmse = %-10.1e mae = %-10.1e'%(velocity,rmsej,maej))
        mass_ticks = np.linspace(dataj['mass_ini'].min(),dataj['mass_ini'].max(),nmesh)
        logZ_ticks = np.linspace(dataj['logZ_ini'].min(),dataj['logZ_ini'].max(),nmesh)
        mass_mesh,logZ_mesh = np.meshgrid(mass_ticks,logZ_ticks)
        query_df = pd.DataFrame({
            'logZ_ini': logZ_mesh.flatten(),
            'vel_ini': np.tile(velocity,mass_mesh.size),
            'mass_ini': mass_mesh.flatten()})
        xquery = query_df[['logZ_ini','vel_ini','mass_ini']].to_numpy()
        logyhats = model(xquery)
        logyhats_mesh = logyhats.reshape(mass_mesh.shape)
        ax[i,j].plot_surface(mass_mesh,logZ_mesh,logyhats_mesh,cmap=cm.Greys,alpha=0.9)
        ax[i,j].scatter(dataj['mass_ini'],dataj['logZ_ini'],dataj['logy'],color='r',s=10)
        # metadata
        ax[i,j].set_xlabel('mass_ini')
        ax[i,j].set_ylabel('ln(Z_ini)')
        ax[i,j].set_zlabel('ln(y)')
        ax[i,j].set_title('element %d, velocity %d'%(idx,velocity))
fig.savefig('figures/test/velocity_plots.pdf',format='pdf',bbox_inches='tight')

