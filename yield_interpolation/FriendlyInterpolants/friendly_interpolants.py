import scipy.interpolate
from numpy import *
import pandas as pd

class FriendlyInterpolant(object):
    def __init__(self, df, tf_funs={}, xcols=None, ycol='y', name='model', plot=False, plot_mod=lambda fig: None, fig_root='./', plot_ops={}):
        """
        Args:
            df (pd.DataFrame): dataframe of features and response vector.
            tf_funs (dict): transform functions where keys are df column names potentially with '.inv' or prime appended '.prime'. 
                Response transforms are required to have both col and col+'.inv' keys. 
                If planning to take derivitive with respect to col, supply col+'.prime' key.
            xcols (list): list of feature col names. If None, use all col not equal to ycol. 
            ycol (str): column name of response vector.
            name (str): name of instance for plotting and printing
            plot (pd.DataFrame): columns should be all feature columns. Each row is a vector of derivitives. 
                plot=True is equivalent to having a single row of 0 values.  
                plot=False is equivalent to having no rows. 
            plot_mod (func): function which modifies input fig,ax before saving plot. 
                The plot is closed after __init__ as df is not retained.
            fig_root (str): output directory for figure
            plot_ops (dict): plotting options
        """
        self.empty = False
        for k,v in {'sepfl':0,'sepfr':0,'sepfb':0,'sepft':0,'scatter':True,'view_init_azim':45}.items():
            if k not in plot_ops: plot_ops[k] = v
        if isinstance(df,ndarray):
            assert df.ndim==2
            df = pd.DataFrame(df,columns=['x_{%d}'%i for i in range(df.shape[1]-1)]+['y'])
        self.ycol = ycol
        self.xcols = xcols
        assert self.ycol in df.columns
        if self.xcols is None: self.xcols = [xcol for xcol in list(df.columns) if xcol!=self.ycol]
        assert all(xcol in df.columns for xcol in self.xcols)
        self.d = len(self.xcols)
        assert df.shape[0]>1 and self.d>0
        assert self.ycol in df.columns
        self.xcols = [col for col in list(df.columns) if col!=ycol]
        self.tf_funs = tf_funs
        if self.ycol in self.tf_funs:
            assert self.ycol+'.inv' in self.tf_funs
        else:
            self.tf_funs[self.ycol] = lambda y:y
            self.tf_funs[self.ycol+'.inv'] = lambda y:y
            self.tf_funs[self.ycol+'.prime'] = lambda y:ones(len(y),dtype=float)
        for col in self.xcols:
            if col not in self.tf_funs:
                self.tf_funs[col] = lambda x:x
                self.tf_funs[col+'.prime'] = lambda x:ones(len(x),dtype=float)
        self.name = name        
        dftf = pd.DataFrame({col:self.tf_funs[col](df[col].to_numpy()) for col in list(df.columns)})
        self.descrip = df.describe()
        self.descrip_tf = dftf.describe()
        xtf = dftf[self.xcols].to_numpy()
        ytf = dftf[self.ycol].to_numpy()
        self._fit(xtf,ytf)       
        self.train_metrics = self.get_metrics(df)
        if plot is False or self.d>2: return
        if plot is True or plot is None: plot=[None]
        assert isinstance(plot,list)
        dwrts = [self._parse_dwrt(dwrt) for dwrt in plot]
        from matplotlib import pyplot
        nticks = 257
        if self.d==1:
            xcol = self.xcols[0]
            fig,ax = pyplot.subplots(nrows=len(dwrts),ncols=2,figsize=(4*2,4.5*len(dwrts)))
            ax = atleast_2d(ax)
            for j,dfv in enumerate([dftf,df]):
                xvmin,xvmax = dfv[xcol].min(),dfv[xcol].max()
                xvsep = xvmax-xvmin
                sepfl,sepfr = plot_ops['sepfl'],plot_ops['sepfr']
                xvticks = linspace(xvmin-sepfl*xvsep,xvmax+xvsep*sepfr,nticks)
                for i,dwrt in enumerate(dwrts):
                    if plot_ops['scatter'] and (dwrt==0).all(): ax[i,j].scatter(dfv[xcol],dfv[self.ycol],color='c')
                    yvhat = self.__call__(xvticks[:,None],dwrt) if j==1 else self._eval_with_grad(xvticks[:,None],dwrt)
                    ax[i,j].plot(xvticks,yvhat,color='k')
            for i in range(len(dwrts)):
                ax[i,0].set_xlabel(r'$\mathrm{%s}^\mathrm{tf}$'%xcol)
                ax[i,0].set_ylabel(r'$\mathrm{%s}^\mathrm{tf}$'%self.ycol)
                ax[i,0].set_title(r'$\mathrm{%s}_\mathrm{tf}^{%s}(%s^\mathrm{tf})$'%(self.name,str(tuple(dwrts[i])),xcol))
                ax[i,1].set_xlabel(r'$\mathrm{%s}$'%xcol)
                ax[i,1].set_ylabel(r'$\mathrm{%s}$'%self.ycol)
                ax[i,1].set_title(r'$\mathrm{%s}_{%s}(%s)$'%(self.name,str(tuple(dwrts[i])),xcol))
                for j in range(2):
                    (xmin,xmax),(ymin,ymax) = ax[i,j].get_xlim(),ax[i,j].get_ylim()
                    ax[i,j].set_aspect((xmax-xmin)/(ymax-ymin))
            fig.suptitle('%s = %s(%s)\nTransformed Domain | Original Domain'%(self.ycol,self.name,xcol))
        elif self.d==2:
            from matplotlib import cm
            x0col,x1col = self.xcols
            nrows = len(dwrts)
            fig = pyplot.figure(figsize=(5.5*4,4.5*nrows))
            for j,dfv in enumerate([dftf,df]):
                xv0min,xv0max,xv1min,xv1max = dfv[x0col].min(),dfv[x0col].max(),dfv[x1col].min(),dfv[x1col].max()
                xv0sep,xv1sep = xv0max-xv0min,xv1max-xv1min
                sepfl,sepfr,sepfb,sepft = plot_ops['sepfl'],plot_ops['sepfr'],plot_ops['sepfb'],plot_ops['sepft']
                xv0ticks = linspace(xv0min-sepfl*xv0sep,xv0max+sepfr*xv0sep,nticks)
                xv1ticks = linspace(xv1min-sepfb*xv1sep,xv1max+sepft*xv0sep,nticks)
                xv0mesh,xv1mesh = meshgrid(xv0ticks,xv1ticks)
                xvticks = vstack([xv0mesh.flatten(),xv1mesh.flatten()]).T
                for i,dwrt in enumerate(dwrts):
                    yv = self.__call__(xvticks,dwrt) if j==1 else self._eval_with_grad(xvticks,dwrt)
                    yvmesh = yv.reshape(xv0mesh.shape)
                    ax = fig.add_subplot(nrows,4,4*i+2*j+(1 if j==0 else 2))
                    contour = ax.contourf(xv0mesh,xv1mesh,yvmesh,cmap='gray',alpha=.95,vmin=yvmesh.min(),vmax=yvmesh.max(),levels=64)
                    cbar = fig.colorbar(contour,ax=None,shrink=1,aspect=5)
                    if plot_ops['scatter'] and (dwrt==0).all(): ax.scatter(dfv[x0col],dfv[x1col],color='c')
                    ax.set_xlabel(r'$\mathrm{%s}$'%x0col if j==1 else r'$\mathrm{%s}^\mathrm{tf}$'%x0col)
                    ax.set_ylabel(r'$\mathrm{%s}$'%x1col if j==1 else r'$\mathrm{%s}^\mathrm{tf}$'%x1col)
                    if j==1: ax.set_title(r'$\mathrm{%s}_{%s}(%s,%s)$'%(self.name,str(tuple(dwrts[i])),x0col,x1col))
                    elif j==0: ax.set_title(r'$\mathrm{%s}_\mathrm{tf}^{%s}(%s^\mathrm{tf},%s^\mathrm{tf})$'%(self.name,str(tuple(dwrts[i])),x0col,x1col))
                    (xmin,xmax),(ymin,ymax) = ax.get_xlim(),ax.get_ylim()
                    ax.set_aspect((xmax-xmin)/(ymax-ymin))
                    ax = fig.add_subplot(nrows,4,4*i+2*j+(2 if j==0 else 1),projection='3d')
                    ax.plot_surface(xv0mesh,xv1mesh,yvmesh,cmap='gray',alpha=.95,vmin=yvmesh.min(),vmax=yvmesh.max(),rcount=82,ccount=82)
                    ax.view_init(azim=plot_ops['view_init_azim'])
                    if plot_ops['scatter'] and (dwrt==0).all(): ax.scatter(dfv[x0col],dfv[x1col],dfv[self.ycol],color='c')
                    ax.set_xlabel(r'$\mathrm{%s}$'%x0col if j==1 else r'$\mathrm{%s}^\mathrm{tf}$'%x0col)
                    ax.set_ylabel(r'$\mathrm{%s}$'%x1col if j==1 else r'$\mathrm{%s}^\mathrm{tf}$'%x1col)
                    ax.set_zlabel(r'$\mathrm{%s}$'%self.ycol if j==1 else r'$\mathrm{%s}^\mathrm{tf}$'%self.ycol)
            fig.suptitle('%s = %s(%s,%s)\nTransformed Domain | Original Domain'%(self.ycol,self.name,x0col,x1col))
            #pyplot.subplots_adjust(left=0,bottom=0.1,right=1,top=0.9,wspace=0.2,hspace=0.2)
        plot_mod(fig)
        fig.savefig('%s%s.png'%(fig_root,name),dpi=128,format='png',bbox_inches='tight')
        pyplot.close(fig)
    def _parse_dwrt(self, dwrt):
        if dwrt is None: dwrt = {col:0 for col in self.xcols}
        if isinstance(dwrt,str):
            assert dwrt in self.xcols
            dwrt = {dwrt:1}
        if isinstance(dwrt,list) or isinstance(dwrt,ndarray):
            dwrt = atleast_1d(dwrt)
            assert dwrt.ndim==1 and len(dwrt)==self.d
            dwrt = {col:dwrt[i] for i,col in enumerate(self.xcols)}
        for xcol in self.xcols: dwrt[xcol] = dwrt[xcol] if xcol in dwrt else 0
        dwrt = atleast_1d([dwrt[col] for col in self.xcols]).astype(int)
        assert dwrt.ndim==1 and len(dwrt)==self.d
        return dwrt
    def _fit(self, x, y): 
        raise NotImplementedError
    def __call__(self, dfx, dwrt=None):
        assert dfx.ndim==2
        dwrt = self._parse_dwrt(dwrt)
        if isinstance(dfx,ndarray): dfx = pd.DataFrame(dfx,columns=self.xcols)
        dftf = pd.DataFrame({col:self.tf_funs[col](dfx[col].to_numpy()) for col in self.xcols})
        xtf = dftf[self.xcols].to_numpy()
        yhattf = self._eval_with_grad(xtf,dwrt=zeros(self.d))
        yhat = self.tf_funs[self.ycol+'.inv'](yhattf)
        if (dwrt==0).all(): return yhat
        # handle derivitives
        assert (dwrt>=0).all()
        if sum(dwrt)!=1: raise Exception("currently only supports 1 derivitive at a time")
        yhattfgrad = self._eval_with_grad(xtf,dwrt=dwrt)
        dwrt = self.xcols[argmax(dwrt)]
        # chain rule
        # t_j = T_j(x_j) for j=1,...,d
        # ytf = M(t_1,...,t_d)
        # y = T^{-1}(ytf)
        # dy/dx_i = M'(t_1,...,t_d)*T_j'(x_j)/T'(T^{-1}(ytf))
        dyhat_dxi = yhattfgrad*self.tf_funs[dwrt+'.prime'](dfx[dwrt].to_numpy())/self.tf_funs[self.ycol+'.prime'](yhattf)
        return dyhat_dxi
    def _eval_with_grad(self, x, dwrt):
        raise NotImplementedError
    def get_metrics(self,df):
        y = df[self.ycol].to_numpy()
        yhat = self.__call__(df)
        eps_abs = abs(yhat-y)
        metrics = {
            'RMSE Abs': sqrt(mean(eps_abs**2)),
            'MAE Abs': mean(eps_abs),
            'Max Abs': max(eps_abs)}
        return metrics
    def __repr__(self):
        s = '%s(%s)\n'%(self.name,','.join(self.xcols))
        s += '\ttrain data description\n\t\t%s'%str(self.descrip).replace('\n','\n\t\t')
        s += '\n\ttrain data metrics\n'
        for metric,val in self.train_metrics.items(): s += '\t\t%25s: %.2e\n'%(metric,val)
        return s

class CubicSpline1D_FI(FriendlyInterpolant):
    """ https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.CubicSpline.html#scipy.interpolate.CubicSpline """
    def _fit(self, x, y):
        assert x.shape[1]==1
        x = x.squeeze()
        ord = argsort(x)
        xsort = x[ord]
        ysort = y[ord]
        self.model = scipy.interpolate.CubicSpline(xsort,ysort,extrapolate=True)
    def _eval_with_grad(self, x, dwrt):
        y = self.model(x.squeeze(),dwrt[0])
        return y

class LinearAndNearestNeighbor_FI(FriendlyInterpolant):
    def _fit(self, x, y):
        assert x.shape[1]>1
        self.inhull_model = scipy.interpolate.LinearNDInterpolator(x,y,rescale=True)
        self.outhull_model = scipy.interpolate.NearestNDInterpolator(x,y,rescale=True)
    def _eval_with_grad(self, x, dwrt):
        assert (dwrt==0).all() # derivitives not supported by LinearAndNearestNeighbor_FI
        y = self.inhull_model(x)
        y[isnan(y)] = self.outhull_model(x[isnan(y)])
        return y

class SmootheSpline2D_FI(FriendlyInterpolant):
    def _fit(self, x, y):
        assert x.shape[1]==2 # SmootheSpline2D_FI only supports 2D input features
        self.model = scipy.interpolate.SmoothBivariateSpline(x=x[:,0],y=x[:,1],z=y)
    def _eval_with_grad(self, x, dwrt):
        y = self.model(x=x[:,0],y=x[:,1],dx=dwrt[0],dy=dwrt[1],grid=False)
        return y

if __name__ == '__main__':
    from matplotlib import pyplot
    import pandas as pd
    
    # 1D example
    x = linspace(0,1,16)[1:-1]**2
    y = sin(2*pi*x)/cos(pi*x/2)
    cs1d = CubicSpline1D_FI(
        df = pd.DataFrame({'x':x,'y':y}),
        ycol = 'y',
        tf_funs = {
            'x':lambda x:sqrt(x), 'x.prime':lambda x:1/(2*sqrt(x)),
            'y':lambda y:exp(y), 'y.prime':lambda y:exp(y), 'y.inv':lambda y:log(y)},
        name = 'cs1d',
        plot = [None,'x'])
    print(cs1d)
    yhat = cs1d(pd.DataFrame({'x':random.rand(3)**2}),dwrt={'x':1})
    
    # 2D examples
    x = 10**random.rand(8**2,2)
    y = sin(2*pi*x[:,0])*cos(2*pi*x[:,1])
    df = pd.DataFrame({'x_0':x[:,0],'x_1':x[:,1],'y':y})
    tf_funs = {
        'x_0':lambda x:log10(x), 'x_0.prime':lambda x:1/(x*log(10)),
        'x_1':lambda x:log10(x), 'x_1.prime':lambda x:1/(x*log(10))}
    lnn2d = LinearAndNearestNeighbor_FI(df,tf_funs,name='lnn2d',plot=None)
    print(lnn2d)
    yhat = lnn2d(pd.DataFrame(10**random.rand(3,2),columns=['x_0','x_1']))
    ss2d = SmootheSpline2D_FI(df,tf_funs,name='ss2d',plot=[None,'x_0','x_1'])
    yhat = ss2d(pd.DataFrame(10**random.rand(3,2),columns=['x_0','x_1']),dwrt='x_1')
    print(ss2d)

    # Arbitrary dimension example
    d = 5
    x = random.rand(8**3,d)
    y = prod(sin(2*pi*x),1)
    lnnd = LinearAndNearestNeighbor_FI(hstack([x,y[:,None]]),name='lnn%d'%d)
    print(lnnd)
    yhat = lnnd(random.rand(3,d))


    