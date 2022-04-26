import scipy.interpolate as interp
import numpy as np
import pandas as pd
import dill

class GalCemInterpolant(object):
    def __init__(self,df,ycol,tf_funs={},name='',plot=False,fig_root='./',fig_view_angle=135):
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
        self.interpolator = interp.SmoothBivariateSpline(x=xtf[:,0],y=xtf[:,1],z=ytf)
        self.train_metrics = self.get_metrics(df)
        # optional plotting in transformed space
        if not plot: return
        from matplotlib import pyplot,cm
        fig = pyplot.figure(figsize=(15,5*3))
        nticks = 64
        sepfrac = 0.1
        x = dftf[self.xcols].to_numpy()
        x0min,x0max = x[:,0].min(),x[:,0].max()
        x0_sep = x0max-x0min
        x0_ticks = np.linspace(x0min-sepfrac*x0_sep,x0max+sepfrac*x0_sep,nticks)
        x1min,x1max = x[:,1].min(),x[:,1].max()
        x1_sep = x1max-x1min
        x1_ticks = np.linspace(x1min-sepfrac*x1_sep,x1max+sepfrac*x1_sep,nticks)
        x0mesh,x1mesh = np.meshgrid(x0_ticks,x1_ticks)
        x0mesh_flat,x1mesh_flat = x0mesh.flatten(),x1mesh.flatten()
        for i,dx in enumerate([[0,0],[1,0],[0,1]]):
            y = self.interpolator(x=x0mesh_flat,y=x1mesh_flat,dx=dx[0],dy=dx[1],grid=False)
            ymesh = y.reshape(x0mesh.shape)
            # 3d plot on row i
            ax = fig.add_subplot(3,2,(2*i+1),projection='3d')
            ax.plot_surface(x0mesh,x1mesh,ymesh,cmap=cm.Greys,alpha=.9,vmin=ymesh.min(),vmax=ymesh.max())
            if dx==[0,0]: ax.scatter(x[:,0],x[:,1],dftf[self.ycol].to_numpy(),color='r')
            ax.set_xlabel(self.xcols[0])
            ax.set_ylabel(self.xcols[1])
            ax.set_zlabel(self.ycol)
            if dx!=[0,0]: ax.set_title('d(%s) / d(%s)'%(self.ycol,self.xcols[dx.index(1)]))
            ax.view_init(azim=fig_view_angle)
            # 2d contour plot 
            ax = fig.add_subplot(3,2,2*(i+1))
            contour = ax.contourf(x0mesh,x1mesh,ymesh,cmap=cm.Greys,alpha=.95,vmin=ymesh.min(),vmax=ymesh.max(),levels=64)
            fig.colorbar(contour,ax=None,shrink=0.5,aspect=5)
            ax.scatter(x[:,0],x[:,1],color='r')
            ax.set_xlabel(self.xcols[0])
            ax.set_ylabel(self.xcols[1])
            if dx!=[0,0]: ax.set_title('d(%s) / d(%s)'%(self.ycol,self.xcols[dx.index(1)]))
        fig.suptitle('%s\n%s by %s\nTransformed Domain'%(self.name,self.ycol,str(self.xcols)))
        fig.savefig('%s%s.pdf'%(fig_root,name),format='pdf',bbox_inches='tight')
    
    def __call__(self,dfx,dwrt=None):
        dftf = pd.DataFrame({col:self.tf_funs[col](dfx[col].to_numpy()) for col in self.xcols})
        xtf = dftf[self.xcols].to_numpy()
        yhattf = self.interpolator(x=xtf[:,0],y=xtf[:,1],dx=0,dy=0,grid=False)
        yhat = self.tf_funs[self.ycol+'_inv'](yhattf)
        if dwrt is None: return yhat
        # handle derivitives
        x = dfx[self.xcols].to_numpy()
        didx = self.xcols.index(dwrt)
        dx = [0,0]
        dx[didx] = 1
        yhattf_prime = self.interpolator(x=xtf[:,0],y=xtf[:,1],dx=dx[0],dy=dx[1],grid=False)
        dyhat_dxi = yhattf_prime*self.tf_funs[self.xcols[didx]+'_prime'](x[:,didx])/self.tf_funs[self.ycol+'_prime'](yhat)
        return dyhat_dxi
        
    def get_metrics(self,df):
        yhat = self.__call__(df)
        eps_abs = np.abs(yhat-df[self.ycol].to_numpy())
        metrics = {
            'Number of Samples': len(yhat),
            'Root Mean Square Error': np.sqrt(np.mean(eps_abs**2)),
            'Mean Absoute Error': np.mean(eps_abs),
            'Max Abs Error': eps_abs.max()}
        return metrics
    
    def __repr__(self):
        s = 'GalCemInterpolant[%s](%s)\n'%(self.name,','.join(self.xcols))
        s += '\ttrain data description\n\t\t%s'%str(self.descrip).replace('\n','\n\t\t')
        s += '\n\ttrain data metrics\n'
        for metric,val in self.train_metrics.items(): s += '\t\t%25s: %.2e\n'%(metric,val)
        return s
    
def fit_isotope_interpolants_irv0(df,root):
    # iterate over a,z pairs and save interpolants based on irv=0
    print('\n'+'~'*75+'\n')
    dfs = dict(tuple(df.groupby(['isotope','a','z'])))
    for ids,_df in dfs.items():
        name = 'a%d.z%d.irv0.%s'%(*ids[1:],ids[0])
        ##print('fitting interpolant %s\n'%name)
        _df = _df[_df['irv']==0]
        # fit model
        interpolant = GalCemInterpolant(
            df = _df[['mass','metallicity','yield']],
            ycol = 'yield',
            tf_funs = {
                'mass':lambda x:np.log10(x), 'mass_prime':lambda x:1/(x*np.log(10)),
                'metallicity':lambda x:np.log10(x), 'metallicity_prime':lambda x:1/(x*np.log(10)),
                'yield':lambda y:np.log10(y), 'yield_prime':lambda y:1/(y*np.log(10)), 'yield_inv':lambda y:10**y},
            name = name,
            plot = True,
            fig_root = root+'/figs/',
            fig_view_angle = -45)
        #   print model
        print(interpolant)
        #   save model
        dill.dump(interpolant,open(root+'/models/%s.pkl'%name,'wb'))
        #   load model
        interpolant_loaded = dill.load(open(root+'/models/%s.pkl'%name,'rb'))
        #   example model use
        yquery = interpolant_loaded(_df)
        dyquery_dmass = interpolant_loaded(_df,dwrt='mass')
        dquery_dmetallicity = interpolant_loaded(_df,dwrt='metallicity')
        print('~'*75+'\n')