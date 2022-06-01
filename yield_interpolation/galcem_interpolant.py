import numpy as np
import pandas as pd

class GalCemInterpolant(object):
    def __init__(self,df,ycol,tf_funs={},name='',plot=False,fig_root='./',fig_view_angle=135,colormap='Paired'):
        from sklearn.gaussian_process import GaussianProcessRegressor,kernels
        # initial setup 
        self.ycol = ycol
        self.xcols = [col for col in list(df.columns) if col!=ycol]
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
        self.xtf = dftf[self.xcols].to_numpy()
        ytf = dftf[self.ycol].to_numpy()
        self.interpolator = GaussianProcessRegressor(kernel=kernels.RationalQuadratic(alpha_bounds=(1e-8,1e5),length_scale_bounds=(1e-12,1e5)),normalize_y=True).fit(X=self.xtf,y=ytf)
        self.k_alpha = self.interpolator.kernel_.alpha
        self.k_l = self.interpolator.kernel_.length_scale
        self.weights = self.interpolator.alpha_
        self.train_metrics = self.get_metrics(df)
        # optional plotting in transformed space
        if not plot: return
        if len(self.xcols)!=2: raise Exception("Plotting GalCemInterpolant currently only supports 2 dimensional domains.")
        from matplotlib import pyplot,cm
        cmdot = pyplot.cm.get_cmap(colormap)
        s_lifetimes_p98 = pd.read_csv('galcem/input/starlifetime/portinari98table14.dat')
        s_lifetimes_p98.columns = [name.replace('#M','M').replace('Z=0.','Z') for name in s_lifetimes_p98.columns]
        colordots = np.array([s_lifetimes_p98[c].to_numpy() for c in s_lifetimes_p98.columns[1:]]).flatten()
        colordotstf = self.tf_funs[self.ycol](colordots)
        fig = pyplot.figure(figsize=(30,5*3))
        nticks = 64
        sepfrac = 0.1
        self.xtf = dftf[self.xcols].to_numpy()
        x0tfmin,x0tfmax = self.xtf[:,0].min(),self.xtf[:,0].max()
        x0tf_sep = x0tfmax-x0tfmin
        x0_ticks = np.linspace(x0tfmin-sepfrac*x0tf_sep,x0tfmax+sepfrac*x0tf_sep,nticks)
        x1tfmin,x1tfmax = self.xtf[:,1].min(),self.xtf[:,1].max()
        x1tf_sep = x1tfmax-x1tfmin
        x1tf_ticks = np.linspace(x1tfmin-sepfrac*x1tf_sep,x1tfmax+sepfrac*x1tf_sep,nticks)
        x0tfmesh,x1tfmesh = np.meshgrid(x0_ticks,x1tf_ticks)
        ytf,grad_ytf = self.predict_tf(dftf=pd.DataFrame({self.xcols[0]:x0tfmesh.flatten(),self.xcols[1]:x1tfmesh.flatten()}),return_grad=True)
        ytfmesh,grad0_ytf_mesh,grad1_ytf_mesh = ytf.reshape(x0tfmesh.shape),grad_ytf[:,0].reshape(x0tfmesh.shape),grad_ytf[:,1].reshape(x0tfmesh.shape)
        for i,ztfmesh in enumerate([ytfmesh,grad0_ytf_mesh,grad1_ytf_mesh]):
            ax = fig.add_subplot(3,2,2*i+1,projection='3d')
            ax.plot_surface(x0tfmesh,x1tfmesh,ztfmesh,cmap=cm.Greys,alpha=.9,vmin=ztfmesh.min(),vmax=ztfmesh.max())
            if i==0: ax.scatter(self.xtf[:,0],self.xtf[:,1],dftf[self.ycol].to_numpy(),c=colordotstf, cmap=cmdot)
            ax.set_xlabel(self.xcols[0])
            ax.set_ylabel(self.xcols[1])
            ax.set_zlabel(self.ycol)
            if i!=0: ax.set_title('d(%s) / d(%s)'%(self.ycol,self.xcols[i-1]))
            ax.view_init(azim=fig_view_angle)
        fig.suptitle('%s\n%s by %s\nTransformed Domain (left) | Original Domain (right)'%(self.name,self.ycol,str(self.xcols)), fontsize=15)
        fig.savefig('%s%s.pdf'%(fig_root,name),format='pdf',bbox_inches='tight')
        pyplot.close(fig)
    
    def predict_tf(self,dftf,return_grad=False):
        xtf = dftf[self.xcols].to_numpy()
        # https://github.com/scikit-learn/scikit-learn/blob/80598905e/sklearn/gaussian_process/_gpr.py#L327
        dists = xtf[:,None,:]-self.xtf[None,:,:]
        b = 1+np.sum(dists**2,-1)/(2*self.k_alpha*self.k_l**2)
        k = b**(-self.k_alpha)
        yhattf = self.interpolator._y_train_std*k@self.weights+self.interpolator._y_train_mean
        if not return_grad: return yhattf
        # handle derivitives
        grad_k = np.vstack([-self.weights[None,:]@(b[i,:,None]**(-self.k_alpha-1)*(xtf[i]-self.xtf))/(self.k_l**2) for i in range(len(xtf))])
        grad_yhattf = self.interpolator._y_train_std*grad_k
        return yhattf,grad_yhattf
    
    def __call__(self,dfx,return_grad=False):
        dftf = pd.DataFrame({col:self.tf_funs[col](dfx[col].to_numpy()) for col in self.xcols})
        if not return_grad: yhattf = self.predict_tf(dftf,return_grad=return_grad)
        else: yhattf,grad_yhattf = self.predict_tf(dftf,return_grad=return_grad)
        yhat = self.tf_funs[self.ycol+'_inv'](yhattf)
        if not return_grad: return yhat
        x = dfx[self.xcols].to_numpy()
        # chain rule 
        #   y(x0,x1) = T^{-1}( I(g0(x0),g1(x1)) )
        #   dy/dxi = I'(g0(x0),g1(x1)) gi'(xi) / T'( T^{-1}( I(g0(x0),g1(x1)) ) )
        #          = I(x0tf,x1tf) gi'(xi) / T'(yhat)
        giprime = np.vstack([self.tf_funs[self.xcols[i]+'_prime'](x[:,i]) for i in range(len(self.xcols))]).T
        grad_yhat = grad_yhattf*giprime/self.tf_funs[self.ycol+'_prime'](yhat)[:,None]
        return yhat,grad_yhat        
        
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
    
def fit_isotope_interpolants_irv0(df,root):
    import dill
    # iterate over a,z pairs and save interpolants based on irv=0
    print('\n'+'~'*75+'\n')
    dfs = dict(tuple(df.groupby(['isotope','a','z'])))
    for ids,_df in dfs.items():
        name = 'a%d.z%d.irv0.%s'%(*ids[1:],ids[0])
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