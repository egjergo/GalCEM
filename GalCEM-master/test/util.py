from matplotlib import pyplot as plt
def mplsetup():    
    plt.rcParams['xtick.major.size'], plt.rcParams['ytick.major.size'] = 10, 10
    plt.rcParams['xtick.minor.size'], plt.rcParams['ytick.minor.size'] = 7, 7
    plt.rcParams['xtick.major.width'], plt.rcParams['ytick.major.width'] = 2, 2
    plt.rcParams['xtick.minor.width'], plt.rcParams['ytick.minor.width'] = 1, 1
    plt.rcParams['xtick.labelsize'], plt.rcParams['ytick.labelsize'] = 15, 15
    plt.rcParams['axes.linewidth'] = 2

def mplsetup2():
    plt.rcParams['xtick.major.size'] = 15
    plt.rcParams['xtick.major.width'] = 2
    plt.rcParams['xtick.minor.size'] = 10
    plt.rcParams['xtick.minor.width'] = 1
    plt.rcParams['ytick.major.size'] = 15
    plt.rcParams['ytick.major.width'] = 2
    plt.rcParams['ytick.minor.size'] = 10
    plt.rcParams['ytick.minor.width'] = 1
    plt.rcParams['xtick.labelsize'] = 20
    plt.rcParams['ytick.labelsize'] = 20
    plt.rcParams['axes.linewidth'] = 2