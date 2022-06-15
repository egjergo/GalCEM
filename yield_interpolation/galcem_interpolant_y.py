import scipy.interpolate as interp
import numpy as np
import pandas as pd
import dill

class GalCemInterpolant(object):
    def __init__(self,df,ycol,tf_funs={},name='',plot=None,fig_root='./',fig_view_angle=135,colormap=False):
        
        # initial setup 
        self.ycol = ycol
        self.xcols = [col for col in list(df.columns) if col!=ycol]
        if len(self.xcols)!=2: raise Exception("GalCemInterpolant currently only supports models with 2D domain.")
        self.tf_funs = tf_funs
        self.name = name
        
        # parse y column transforms
        if self.ycol in self.tf_funs:
            assert self.ycol+'_inv' in self.tf_funs
            assert self.ycol+'_prime' in self.tf_funs
        else:
            self.tf_funs[self.ycol] = lambda y:y
            self.tf_funs[self.ycol+'_inv'] = lambda y:y
            self.tf_funs[self.ycol+'_prime'] = lambda y:y
        
        # parse x column transforms
        for col in list(df.columns):
            if col==self.ycol: continue
            if col in self.tf_funs:
                assert col+'_prime' in self.tf_funs                    
            else:
                self.tf_funs[col] = lambda x: x
                self.tf_funs[col+'_prime'] = lambda x: x
        
        # remaining setup
        dftf = pd.DataFrame({col:self.tf_funs[col](df[col].to_numpy()) for col in list(df.columns)})
        self.descrip = df.describe()
        self.descrip_tf = dftf.describe()
        xtf = dftf[self.xcols].to_numpy()
        ytf = dftf[self.ycol].to_numpy()
        self.fit(xtf,ytf)       
        self.train_metrics = self.get_metrics(df)
        # optional plotting in transformed space
        if plot is None: return
        assert plot in ['std','grad']
        dxs = [[0,0]] if plot=='std' else [[0,0],[1,0],[0,1]]
        from matplotlib import pyplot,cm
        cmdot,colordots,colordotstf = 'r','r','r'
        if colormap:
            cmdot = pyplot.cm.get_cmap('Paired')
            s_lifetimes_p98 = pd.read_csv('galcem/input/starlifetime/portinari98table14.dat')
            s_lifetimes_p98.columns = [name.replace('#M','M').replace('Z=0.','Z') for name in s_lifetimes_p98.columns]
            colordots = np.array([s_lifetimes_p98[c].to_numpy() for c in s_lifetimes_p98.columns[1:]]).flatten()
            colordotstf = colordots#self.tf_funs[self.ycol](colordots)
        
        fig = pyplot.figure(figsize=(11,11) if plot=='grad' else (11,11))
        nticks = 64
        sepfrac = 0.1
        for i,dx in enumerate(dxs):
            # transformed domain
            xtf = dftf[self.xcols].to_numpy()
            x0tfmin,x0tfmax = xtf[:,0].min(),xtf[:,0].max()
            x0tf_sep = x0tfmax-x0tfmin
            x0_ticks = np.linspace(x0tfmin-sepfrac*x0tf_sep,x0tfmax+sepfrac*x0tf_sep,nticks)
            x1tfmin,x1tfmax = xtf[:,1].min(),xtf[:,1].max()
            x1tf_sep = x1tfmax-x1tfmin
            x1tf_ticks = np.linspace(x1tfmin-sepfrac*x1tf_sep,x1tfmax+sepfrac*x1tf_sep,nticks)
            x0tfmesh,x1tfmesh = np.meshgrid(x0_ticks,x1tf_ticks)
            x0tfmesh_flat,x1tfmesh_flat = x0tfmesh.flatten(),x1tfmesh.flatten()
            ytf = self.eval_with_grad(x=np.vstack([x0tfmesh_flat,x1tfmesh_flat]).T,dx=dx)
            ytfmesh = ytf.reshape(x0tfmesh.shape)
            
            ax = fig.add_subplot(2,2,1*i+1,projection='3d')
            ax.plot_surface(x0tfmesh,x1tfmesh,ytfmesh,cmap=cm.Greys,alpha=.9,vmin=ytfmesh.min(),vmax=ytfmesh.max())
            if dx==[0,0]: ax.scatter(xtf[:,0],xtf[:,1],dftf[self.ycol].to_numpy(),c=colordotstf, cmap=cmdot)
            ax.set_xlabel(self.xcols[0])
            ax.set_ylabel(self.xcols[1])
            ax.set_zlabel(self.ycol)
            if dx!=[0,0]: ax.set_title('d(%s) / d(%s)'%(self.ycol,self.xcols[dx.index(1)]))
            ax.view_init(azim=fig_view_angle)
            
            ax = fig.add_subplot(2,2,1*i+2)
            contour = ax.contourf(x0tfmesh,x1tfmesh,ytfmesh,cmap=cm.Greys,alpha=.95,vmin=ytfmesh.min(),vmax=ytfmesh.max(),levels=64)
            xlim,ylim = ax.get_xlim(),ax.get_ylim()
            ax.set_aspect((xlim[1]-xlim[0])/(ylim[1]-ylim[0]))
            fig.colorbar(contour,ax=None,shrink=0.5,aspect=5)
            ax.scatter(xtf[:,0],xtf[:,1],c=colordotstf, cmap=cmdot)
            ax.set_xlabel(self.xcols[0])
            ax.set_ylabel(self.xcols[1])
            if dx!=[0,0]: ax.set_title('d(%s) / d(%s)'%(self.ycol,self.xcols[dx.index(1)]))
            # original domain
            x = df[self.xcols].to_numpy()
            x0min,x0max = x[:,0].min(),x[:,0].max()
            x0_ticks = np.linspace(x0min,x0max,nticks)
            x1min,x1max = x[:,1].min(),x[:,1].max()
            x1_ticks = np.linspace(x1min,x1max,nticks)
            x0mesh,x1mesh = np.meshgrid(x0_ticks,x1_ticks)
            x0mesh_flat,x1mesh_flat = x0mesh.flatten(),x1mesh.flatten()
            dfx = pd.DataFrame({self.xcols[0]:x0mesh_flat,self.xcols[1]:x1mesh_flat})
            dfw = None if dx==[0,0] else self.xcols[dx.index(1)]
            y = self.__call__(dfx=dfx,dwrt=dfw)
            ymesh = y.reshape(x0mesh.shape)
            
            ax = fig.add_subplot(2,2,1*i+3,projection='3d')
            ax.plot_surface(x0mesh,x1mesh,ymesh,cmap=cm.Greys,alpha=.9,vmin=ymesh.min(),vmax=ymesh.max())
            if dx==[0,0]: ax.scatter(x[:,0],x[:,1],df[self.ycol].to_numpy(),c=colordots, cmap=cmdot)
            ax.set_xlabel(self.xcols[0])
            ax.set_ylabel(self.xcols[1])
            ax.set_zlabel(self.ycol)
            if dx!=[0,0]: ax.set_title('d(%s) / d(%s)'%(self.ycol,self.xcols[dx.index(1)]))
            ax.view_init(azim=fig_view_angle)
            
            ax = fig.add_subplot(2,2,1*i+4)
            contour = ax.contourf(x0mesh,x1mesh,ymesh,cmap=cm.Greys,alpha=.95,vmin=ymesh.min(),vmax=ymesh.max(),levels=64)
            xlim,ylim = ax.get_xlim(),ax.get_ylim()
            ax.set_aspect((xlim[1]-xlim[0])/(ylim[1]-ylim[0]))
            fig.colorbar(contour,ax=None,shrink=0.5,aspect=5)
            ax.scatter(x[:,0],x[:,1],c=colordots, cmap=cmdot)
            ax.set_xlabel(self.xcols[0])
            ax.set_ylabel(self.xcols[1])
            if dx!=[0,0]: ax.set_title('d(%s) / d(%s)'%(self.ycol,self.xcols[dx.index(1)]))
        fig.suptitle('%s\n%s by %s\nTransformed Domain (top) | Original Domain (bottom)'%(self.name,self.ycol,str(self.xcols)), fontsize=15)
        pyplot.subplots_adjust(left=0,bottom=0.1,right=1,top=0.9,wspace=0.2,hspace=0.2)
        #fig.tight_layout()
        fig.savefig('%s%s.pdf'%(fig_root,name),format='pdf',bbox_inches='tight')
        pyplot.close(fig)
    
    def fit(self, x, y): 
        raise NotImplementedError
    
    def __call__(self,dfx,dwrt=None):
        dftf = pd.DataFrame({col:self.tf_funs[col](dfx[col].to_numpy()) for col in self.xcols})
        xtf = dftf[self.xcols].to_numpy()
        yhattf = self.eval_with_grad(xtf,dx=[0,0])
        yhat = self.tf_funs[self.ycol+'_inv'](yhattf)
        if dwrt is None: return yhat
        # handle derivitives
        x = dfx[self.xcols].to_numpy()
        didx = self.xcols.index(dwrt)
        dx = [0,0]
        dx[didx] = 1
        yhattf_prime = self.eval_with_grad(xtf,dx=dx)
        # chain rule 
        #   y(x0,x1) = T^{-1}( I(g0(x0),g1(x1)) )
        #   dy/dxi = I'(g0(x0),g1(x1)) gi'(xi) / T'( T^{-1}( I(g0(x0),g1(x1)) ) )
        #          = I(x0tf,x1tf), gi'(xi) / T'(yhat)
        dyhat_dxi = yhattf_prime*self.tf_funs[dwrt+'_prime'](x[:,didx])/self.tf_funs[self.ycol+'_prime'](yhat)
        return dyhat_dxi
    
    def eval_with_grad(self, x, dwrt):
        raise NotImplementedError
        
    def get_metrics(self,df):
        y = df[self.ycol].to_numpy()
        yhat = self.__call__(df)
        eps_abs = np.abs(yhat-y)
        eps_rel = np.abs(eps_abs/y)
        metrics = {
            'RMSE Abs': np.sqrt(np.mean(eps_abs**2)),
            'MAE Abs': np.mean(eps_abs),
            'Max Abs': eps_abs.max(),
            'RMSE Rel': np.sqrt(np.mean(eps_rel**2)),
            'MAE Rel:': np.mean(eps_rel),
            'Max Rel': eps_rel.max()}
        return metrics
    
    def __repr__(self):
        s = 'GalCemInterpolant[%s](%s)\n'%(self.name,','.join(self.xcols))
        s += '\ttrain data description\n\t\t%s'%str(self.descrip).replace('\n','\n\t\t')
        s += '\n\ttrain data metrics\n'
        for metric,val in self.train_metrics.items(): s += '\t\t%25s: %.2e\n'%(metric,val)
        return s

class LinearAndNearestNeighbor_GCI(GalCemInterpolant):
    
    def __init__(self, df, ycol, *args, **kwargs):
        super().__init__(df, ycol, *args, **kwargs)
        self.empty = False
    
    def fit(self, x, y):
        assert x.ndim==2 and y.shape==(len(x),)
        self.inhull_model = interp.LinearNDInterpolator(x,y,rescale=True)
        self.outhull_model = interp.NearestNDInterpolator(x,y,rescale=True)

    def eval_with_grad(self, x, dx):
        assert x.ndim==2 and len(dx)==x.shape[1]
        assert (np.atleast_1d(dx)==0).all()
        y = self.inhull_model(x)
        y[np.isnan(y)] = self.outhull_model(x[np.isnan(y)])
        return y

class SmootheSpline2D_GCI(GalCemInterpolant):
    
    def fit(self, x, y):
        assert x.ndim==2 and x.shape[1]==2 and y.shape==(len(x),)
        self.model = interp.SmoothBivariateSpline(x=x[:,0],y=x[:,1],z=y)
    def eval_with_grad(self, x, dx):
        assert x.ndim==2 and x.shape[1]==2 and len(dx)==x.shape[1]
        y = self.model(x=x[:,0],y=x[:,1],dx=dx[0],dy=dx[1],grid=False)
        return y

    
def fit_isotope_interpolants_irv0(df,root):
    # iterate over a,z pairs and save interpolants based on irv=0
    print('\n'+'~'*75+'\n')
    dfs = dict(tuple(df.groupby(['isotope','a','z'])))
    for ids,_df in dfs.items():
        name = 'z%d.a%d.irv0.%s'%(ids[2],ids[1],ids[0])
        _df = _df[_df['irv']==0]
        # fit model
        interpolant = LinearAndNearestNeighbor_GCI(
            df = _df[['mass','metallicity','yield']],
            ycol = 'yield',
            tf_funs = {
                'mass':lambda x:np.log10(x), 'mass_prime':lambda x:1/(x*np.log(10)),
                'metallicity':lambda x:np.log10(x), 'metallicity_prime':lambda x:1/(x*np.log(10)),
                'yield':lambda y:np.log10(y), 'yield_prime':lambda y:1/(y*np.log(10)), 'yield_inv':lambda y:10**y},
            name = name,
            plot = 'std',
            fig_root = root+'/figs/',
            fig_view_angle = -45,
            colormap=False)
        #   print model
        print(interpolant)
        #   save model
        dill.dump(interpolant,open(root+'/models/%s.pkl'%name,'wb'))
        #   load model
        interpolant_loaded = dill.load(open(root+'/models/%s.pkl'%name,'rb'))
        #   example model use
        yquery = interpolant_loaded(_df)
        print('~'*75+'\n')