import galcem as gc
import pickle

directory_name = str(input("Enter the run's folder name (default format is 'YYYYMMDD_base_deltatimeMyr'): "))
inputs = pickle.load(open('runs/'+directory_name+'/inputs.pkl','rb'))

pl = gc.Plots(outdir='runs/'+directory_name+'/')
print('Loaded an instance of the Plots class')
#pl.plots()
#pl.observational_lelemZ()
#pl.obs_lelemZ()
#pl.observational()
#pl.DTD_plot()
#pl.iso_evolution()
#pl.OH_evolution()
#pl.FeH_evolution()
#pl.phys_integral_plot()
#pl.phys_integral_plot(logAge=True)
pl.iso_evolution_comp(logAge=False)
pl.iso_evolution_comp(logAge=True)
#pl.ind_evolution()
#pl.ind_evolution(logAge=True)
#pl.tracked_elements_3D_plot()
