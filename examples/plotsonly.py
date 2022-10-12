import galcem as gc
import pickle

directory_name = str(input("Enter the run's folder name (default format is 'YYYYMMDD_base_deltatimeMyr'): "))
inputs = pickle.load(open('runs/'+directory_name+'/inputs.pkl','rb'))

pl = gc.Plots(outdir='runs/'+directory_name+'/')
print('Loaded an instance of the Plots class')
pl.plots()
#pl.ind_evolution()