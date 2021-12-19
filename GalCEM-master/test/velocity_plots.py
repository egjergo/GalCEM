from scipy.interpolate import RBFInterpolator, LinearNDInterpolator, NearestNDInterpolator
import numpy as np
import pandas as pd
import pickle
from matplotlib import pyplot,cm
#from prep.setup import X_lc18

def yield_interpolation(indices,X,Y,nmesh,name):
    ni = len(indices)
    velocities = X[0]['vel_ini'].unique().tolist()
    nv = len(velocities)
    fig,ax = pyplot.subplots(nrows=ni,ncols=nv,figsize=(5*nv,5*ni),subplot_kw={"projection": "3d"})
    for i,idx in enumerate(indices):
        data = X[idx].copy()
        data['y'] = Y[idx].copy()
        data['logy'] = np.log10(data['y'])
        data['logZ_ini'] = np.log10(data['Z_ini'])
        xdata = data[['logZ_ini','vel_ini','mass_ini']].to_numpy()
        ydata = data['logy'].to_numpy()
        model = LinearNDInterpolator(xdata,ydata,rescale=True,fill_value=np.nan) # https://docs.scipy.org/doc/scipy/reference/interpolate.html
        data['logyhat'] = model(xdata)
        data['yhat'] = 10**data['logyhat']
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
            ax[i,j].set_ylabel('log10(Z_ini)')
            ax[i,j].set_zlabel('log10(yields)')
            ax[i,j].set_title('element %d, velocity %d'%(idx,velocity))
    fig.savefig('figures/test/yield_interpolant.%s.pdf'%name,format='pdf',bbox_inches='tight')
    pyplot.show(block=False)


def yield_interpolation_k10(indices,X,Y,nmesh,name):
    ni = len(indices)
    fig,ax = pyplot.subplots(nrows=1,ncols=ni,figsize=(5*ni,5),subplot_kw={"projection": "3d"})
    for i,idx in enumerate(indices):
        data = X[idx].copy()
        data['y'] = Y[idx].copy()
        data['logy'] = np.log10(data['y'])
        data['logZ_ini'] = np.log10(data['Z_ini'])
        xdata = data[['logZ_ini','mass_ini']].to_numpy() # !!!!!!! pass it automatically the id_vars var_name parameters from the test functions in yield_interpolation_test.py
        ydata = data['logy'].to_numpy()
        model = LinearNDInterpolator(xdata,ydata,rescale=True,fill_value=np.nan) # https://docs.scipy.org/doc/scipy/reference/interpolate.html
        data['logyhat'] = model(xdata)
        data['yhat'] = 10**data['logyhat']
        data['eps_abs'] = np.abs(data['y']-data['yhat'])
        rmse = np.sqrt(np.mean(data['eps_abs']**2))
        mae = np.mean(data['eps_abs'])
        print('element %i \t\trmse = %-10.1e mae = %-10.1e'%(idx,rmse,mae))
        dataj = data
        rmsej = np.sqrt(np.mean(dataj['eps_abs']**2))
        maej = np.mean(dataj['eps_abs'])
        print('\trmse = %-10.1e mae = %-10.1e'%(rmsej,maej))
        mass_ticks = np.linspace(dataj['mass_ini'].min(),dataj['mass_ini'].max(),nmesh)
        logZ_ticks = np.linspace(dataj['logZ_ini'].min(),dataj['logZ_ini'].max(),nmesh)
        mass_mesh,logZ_mesh = np.meshgrid(mass_ticks,logZ_ticks)
        query_df = pd.DataFrame({
            'logZ_ini': logZ_mesh.flatten(),
            'mass_ini': mass_mesh.flatten()})
        xquery = query_df[['logZ_ini','mass_ini']].to_numpy()
        logyhats = model(xquery)
        logyhats_mesh = logyhats.reshape(mass_mesh.shape)
        ax[i].plot_surface(mass_mesh,logZ_mesh,logyhats_mesh,cmap=cm.Greys,alpha=0.9,vmin=np.nanmin(logyhats_mesh), vmax=np.nanmax(logyhats_mesh))
        ax[i].scatter(dataj['mass_ini'],dataj['logZ_ini'],dataj['logy'],color='r',s=10)
        # metadata
        #ax[i].set_xlim(dataj['mass_ini'].min(),dataj['mass_ini'].max())
        #ax[i].set_ylim(dataj['logZ_ini'].min(),dataj['logZ_ini'].max())
        #ax[i].set_zlim(dataj['yhat'].min(),dataj['yhat'].max())
        #print(f"{dataj['yhat'].min()} {dataj['yhat'].max()}")
        ax[i].set_xlabel('mass_ini')
        ax[i].set_ylabel('log10(Z_ini)')
        ax[i].set_zlabel('log10(mass_frac)')
        ax[i].set_title('element %d'%(idx))
    fig.savefig('figures/test/yield_interpolant.%s.pdf'%name,format='pdf',bbox_inches='tight')
    pyplot.show(block=False)

if __name__ == '__main__':
    indices = [19,99] # oxygen, iron56
    nmesh = 64    
    
    #Xlc18 = pickle.load(open('input/yields/snii/lc18/tab_R/processed/X_lc18.pkl','rb'))
    #Ylc18 = pickle.load(open('input/yields/snii/lc18/tab_R/processed/Y_lc18.pkl','rb'))
    #yield_interpolation(indices,Xlc18,Ylc18,nmesh,name='snii.lc18.tab_R')

    Xk10 = pickle.load(open('input/yields/lims/k10/processed/X_k10.pkl','rb'))
    Yk10 = pickle.load(open('input/yields/lims/k10/processed/Y_k10.pkl','rb'))
    yield_interpolation_k10(indices,Xk10,Yk10,nmesh,name='lims.k10')