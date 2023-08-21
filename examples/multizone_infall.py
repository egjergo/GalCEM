import galcem as glc 
from matplotlib import pyplot as plt

inputs = glc.Inputs()
setup = glc.Setup(inputs)

# everything is seved in setup.total_df


fig = plt.figure()
threed = fig.add_subplot(projection='3d')

X = np.log10(setup.total_df['gal_radius'])
Y = np.log10(setup.total_df['gal_age'])
Z = np.log10(setup.total_df['infall_rate'])
threed.plot_trisurf(X,Y,Z) # play with azimuthal angle, etc.

threed.set_xlabel('log(Galactic radius) [kpc]')
threed.set_ylabel('log(Galactic age) [Gyr]')
threed.set_zlabel('log(infall rate)')
plt.show(block=False)
plt.savefig('multizone_infall.pdf', bbox_inches='tight')
