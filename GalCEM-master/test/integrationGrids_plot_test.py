import matplotlib.pylab as plt
import numpy as np
import prep.inputs as INp
IN = INp.Inputs()
from onezone import *
from util import mplsetup2
mplsetup2()

class IntegrationGrids_plot_test:
    def __init__(self, age_id):
        self.age_idx = int(age_id)
        self.metallicity = Z_v[self.age_idx] 
        self.Wi_class = Wi(self.age_idx)
        self.channels = ['Massive', 'LIMs', 'SNIa']
        self.grids = ['birthtime', 'lifetime', 'mass']
        self.colors = ['red', 'orange', 'black']

    def grid_picker(self, channel_switch, grid_type):
	    return self.Wi_class.__dict__[channel_switch+'_'+grid_type+'_grid']

    def integration_yields(self, channel_switch):
        fig = plt.figure(figsize=(10,7))
        ax = fig.add_subplot(111)
        ax2 = ax.twinx()
        for i in range(2):
            gr = self.grid_picker(channel_switch, self.grids[i])
            ax.plot(np.arange(len(gr)), gr, color = self.colors[i], label = self.grids[i])
        ax2.semilogy(np.arange(len(self.grid_picker(channel_switch, self.grids[2]))),  
                    self.grid_picker(channel_switch, self.grids[2]), color='black', label = 'mass')
        ax.legend(frameon=False, loc = 'upper left', fontsize = 20)
        ax2.legend(frameon=False, loc = 'upper right', fontsize = 20)
        ax2.set_title(channel_switch+r' integration grids at $Z= %s$ and Age = %.4g Gyr' % (str(self.metallicity), time_chosen[self.age_idx]), fontsize = 20, y=1.05)
        ax.set_ylabel(r'time [Gyr]', fontsize = 20)
        ax2.set_ylabel(r'mass [$M_{\odot}$]', fontsize = 20)
        ax2.set_ylim(IN.Ml_LIMs, IN.Mu_Massive)
        #ax.set_ylim(IN.time_start, IN.time_end)
        ax.set_xlabel(r'grid index', fontsize = 20)
        plt.tight_layout()
        plt.savefig('figures/test/integrationGrids'+channel_switch+'_fit_test_n'+str(self.age_idx)+'.pdf')
        plt.show(block=False)
        return None

    def run(self):
        for channel in self.channels:
            self.integration_yields(channel)

integr = IntegrationGrids_plot_test(input('Pick age_id: '))
integr.run()