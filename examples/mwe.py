import galcem as gc
inputs = gc.Inputs()
inputs.nTimeStep = .25
oz = gc.OneZone(inputs,outdir='runs/ags1/')
oz.main()
oz.plots()
