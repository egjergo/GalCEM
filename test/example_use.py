import galcem as gc

inputs = gc.Inputs()
#inputs.nTimeStep = .1
oz = gc.OneZone(inputs,outdir='runs/ags2/')
oz.main()
oz.plots()
