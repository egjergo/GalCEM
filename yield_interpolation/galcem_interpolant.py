import scipy.interpolate as interp
import numpy as np
import pandas as pd
import pickle

class GalCemInterpolant(object):
    def __init__(self,s_y,dfx,xlog10cols,ylog10col):
        self.empty = False
        self.s_y = s_y.copy()
        self.dfx = dfx.copy()
        self.xlog10cols = xlog10cols
        self.ylog10col = ylog10col
        self.xnames = ['log10_'+xcol if xcol in self.xlog10cols else xcol for xcol in list(dfx.columns)]
        self.dfx = self.tf_dfx(self.dfx)
        self.descrip = self.dfx.describe()
        self.s_ytf = self.s_y.copy()
        if self.ylog10col:
            self.s_ytf = np.log10(self.s_ytf)
            self.s_ytf.name = 'log10_'+self.s_ytf.name
        self.linear_nd_interpolator = interp.LinearNDInterpolator(self.dfx[self.xnames].to_numpy(),self.s_ytf.to_numpy(),rescale=True)
        self.nearest_nd_interpolator = interp.NearestNDInterpolator(self.dfx[self.xnames].to_numpy(),self.s_ytf.to_numpy(),rescale=True)
        self.train_metrics = self.get_train_metrics()
        
    def tf_dfx(self,dfx):
        dfx2 = dfx.copy()
        for xcol in self.xlog10cols:
            if 'log10_'+xcol not in dfx.columns:
                dfx2['log10_'+xcol] = np.log10(dfx2[xcol])
        return dfx2
    
    def __call__(self,dfx,return_yhattf=False):
        dfx2 = self.tf_dfx(dfx)
        x = dfx2[self.xnames].to_numpy()
        yhattf = self.linear_nd_interpolator(x)
        yhattf[np.isnan(yhattf)] = self.nearest_nd_interpolator(x[np.isnan(yhattf)])
        s_yhattf = pd.Series(yhattf,name=self.s_ytf.name)
        if return_yhattf:
            return s_yhattf
        elif self.ylog10col:
            s_yhat = 10**s_yhattf
        else:
            s_yhat = s_yhattf
        s_yhat.name = self.s_y.name
        return s_yhat
    
    def get_train_metrics(self,by={}):
        dfx2 = self.dfx.copy()
        s_y2 = self.s_y.copy()
        for col,val in by.items():
            idxs = dfx2[col]==val
            dfx2 = dfx2[idxs]
            s_y2 = s_y2[idxs]
        yhat = self.__call__(dfx2)
        eps_abs = np.abs(yhat-s_y2.to_numpy())
        train_metrics = {
            'Number of Samples': len(yhat),
            'Root Mean Square Error': np.sqrt(np.mean(eps_abs**2)),
            'Mean Absoute Error': np.mean(eps_abs),
            'Max Abs Error': eps_abs.max()}
        return train_metrics
    
    def __repr__(self):
        s = 'GalChemInterpolan(%s)\n'%','.join(self.xnames)
        s += '\ttrain data description\n\t\t%s'%str(self.descrip).replace('\n','\n\t\t')
        s += '\n\ttrain data metrics\n'
        for metric,val in self.train_metrics.items(): s += '\t\t%25s: %.2e\n'%(metric,val)
        return s
    
    def plot(self,xcols,xfixed={},figroot=None,title=None,view_angle=135):
        xcols = ['log10_'+xcol if xcol in self.xlog10cols else xcol for xcol in xcols]
        nticks = 64
        sepfrac = 0.1
        if len(xcols)!=2: raise Exception('plot_interpolant currently only creates 3d plots, so please ensure len(xcols)==2.')
        x = self.dfx[xcols].to_numpy()
        x0min,x0max = x[:,0].min(),x[:,0].max()
        x0_sep = x0max-x0min
        x0_ticks = np.linspace(x0min-sepfrac*x0_sep,x0max+sepfrac*x0_sep,nticks)
        x1min,x1max = x[:,1].min(),x[:,1].max()
        x1_sep = x1max-x1min
        x1_ticks = np.linspace(x1min-sepfrac*x1_sep,x1max+sepfrac*x1_sep,nticks)
        x0mesh,x1mesh = np.meshgrid(x0_ticks,x1_ticks)
        df_xquery = pd.DataFrame({xcols[0]:x0mesh.flatten(),xcols[1]:x1mesh.flatten()})
        for xcol,val in xfixed.items(): df_xquery[xcol] = val
        s_yhattf_query = self.__call__(df_xquery,return_yhattf=True)
        ymesh = s_yhattf_query.to_numpy().reshape(x1mesh.shape)
        from matplotlib import pyplot,cm
        fig = pyplot.figure(figsize=(15,5))
        ax = fig.add_subplot(1,2,1,projection='3d')
        ax.plot_surface(x0mesh,x1mesh,ymesh,cmap=cm.Greys,alpha=.9,vmin=ymesh.min(),vmax=ymesh.max())
        ax.scatter(x[:,0],x[:,1],self.s_ytf,color='r')
        ax.set_xlabel(xcols[0])
        ax.set_ylabel(xcols[1])
        ax.set_zlabel(self.s_ytf.name)
        ax.view_init(azim=view_angle)
        ax = fig.add_subplot(1,2,2)
        contour = ax.contourf(x0mesh,x1mesh,ymesh,cmap=cm.Greys,alpha=.95,vmin=ymesh.min(),vmax=ymesh.max())
        fig.colorbar(contour,ax=None,shrink=0.5,aspect=5)
        ax.scatter(x[:,0],x[:,1],color='r')
        ax.set_xlabel(xcols[0])
        ax.set_ylabel(xcols[1])
        fig.suptitle(title)
        if figroot: fig.savefig('%s.pdf'%figroot,format='pdf',bbox_inches='tight')
        else: fig.show()

def fit_isotope_interpolants_irv0(df,root):
    # iterate over a,z pairs and save interpolants based on irv=0
    print('\n'+'~'*75+'\n')
    dfs = dict(tuple(df.groupby(['isotope','a','z'])))
    for ids,_df in dfs.items():
        tag = 'a%d.z%d.irv0.%s'%(*ids[1:],ids[0])
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
            figroot = root+'figs/%s'%tag,
            title = 'Yield by Mass, Metallicity',
            view_angle = -45)
        #   save model
        pickle.dump(interpolant,open(root+'models/%s.pkl'%tag,'wb'))
        #   load model
        interpolant_loaded = pickle.load(open(root+'models/%s.pkl'%tag,'rb'))
        #   example model use
        xquery = pd.DataFrame({'mass':[15],'metallicity':[0.01648]})
        yquery = interpolant_loaded(xquery)
        print('~'*75+'\n')