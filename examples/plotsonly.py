import galcem as gc
inputs = gc.Inputs()
#inputs.nTimeStep = .25
directory_name = str(input("Enter the run's folder name (default is 'base'): ") or "base")
#oz = gc.OneZone(inputs,outdir='runs/'+directory_name+'/')
#print('Loaded an instance of the OneZone class')
#oz.main()
pl = gc.Plots(outdir='runs/'+directory_name+'/')
print('Loaded an instance of the Plots class')
pl.plots()
#pl.observational_lelemZ()
#pl.obs_lelemZ()
#pl.observational()
#pl.DTD_plot()
#pl.iso_evolution()
#pl.OH_evolution()
#pl.FeH_evolution()