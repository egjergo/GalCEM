import galcem as gc
import pickle

directory_name = str(input("Enter the run's folder name (default is 'YYYYMMDD_base_deltatimeMyr'): ") or "base")
inputs = pickle.load(open('runs/'+directory_name+'/inputs.pkl','rb'))

pl = gc.Plots(outdir='runs/'+directory_name+'/')
print('Loaded an instance of the Plots class')
#pl.plots()
#pl.observational_lelemZ()
#pl.obs_lelemZ()
pl.observational()
#pl.DTD_plot()
#pl.iso_evolution()
#pl.OH_evolution()
#pl.FeH_evolution()
#pl.phys_integral_plot()
#pl.phys_integral_plot(logAge=True)
#pl.iso_evolution_comp()