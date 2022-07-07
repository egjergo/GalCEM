import datetime
import galcem as gc

inputs = gc.Inputs()
inputs.nTimeStep = .1 #.002 #.25

directory_name = str(input("Enter the run's folder name (default is 'base'): ") or "base")
dir_name = 'runs/'+f"{datetime.datetime.now():%Y%m%d}"+'_'+directory_name+'_'+str(int(inputs.nTimeStep*1000))+'Myr/'

oz = gc.OneZone(inputs,outdir=dir_name)
print('Loaded an instance of the OneZone class')
oz.main()

pl = gc.Plots(outdir=dir_name)
print('Loaded an instance of the Plots class')
pl.plots()