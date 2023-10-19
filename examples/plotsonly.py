import galcem as glc
import pickle

directory_name = str(input("Enter the run's folder name (default format is 'YYYYMMDD_base_deltatimeMyr'): "))
inputs = pickle.load(open('runs/'+directory_name+'/inputs.pkl','rb'))

pl = glc.Plots(outdir='runs/'+directory_name+'/')
print('Loaded an instance of the Plots class')
#pl.plots()
pl.total_evolution_plot(logAge=False)
pl.total_evolution_plot(logAge=True)
#pl.observational_plot()
#pl.observational_lelemZ_plot()
#pl.obs_lZ_plot()
#pl.observational_lelemZ()
#pl.obs_lelemZ()
#pl.observational()
#pl.DTD_plot()
#pl.iso_evolution()
#pl.FeH_evolution_plot(logAge=True)
#pl.Z_evolution_plot(logAge=True)
#pl.FeH_evolution_plot(logAge=False)
#pl.Z_evolution_plot(logAge=False)
#pl.phys_integral_plot()
#pl.phys_integral_plot(logAge=True)
#pl.iso_evolution_comp_plot(logAge=False)
#pl.iso_evolution_comp_plot(logAge=True)
#pl.iso_evolution_comp_lelemz_plot()
#pl.ind_evolution()
#pl.ind_evolution(logAge=True)
#pl.tracked_elements_3D_plot()
