import galcem as gc
inputs = gc.Inputs()
#inputs.nTimeStep = .25
directory_name = str(input("Enter the run's folder name (default is 'base'): ") or "base")
oz = gc.OneZone(inputs,outdir='runs/'+directory_name+'/')
#oz.main()
#oz.plots()
oz.iso_evolution_comp()
#oz_plts = gc.OneZone_Plots(inputs,outdir='runs/'+directory_name+'/')
#oz_plts.plots()
