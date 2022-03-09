import galcem as gc
inputs = gc.Inputs()
#inputs.nTimeStep = .25
directory_name = str(input("Enter the run's folder name (default is 'base'): ") or "base")
oz = gc.OneZone(inputs,outdir='runs/'+directory_name+'/')
print('Loaded an instance of the OneZone class')
#oz.main()
pl = gc.Plots(outdir='runs/'+directory_name+'/')
print('Loaded an instance of the Plots class')
#pl.plots()
pl.phys_integral_plot()
pl.phys_integral_plot(logAge=True)