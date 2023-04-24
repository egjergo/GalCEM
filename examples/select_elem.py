import pickle
import galcem as glc


directory_name = str(input("Enter the run's folder name (default format is 'YYYYMMDD_base_deltatimeMyr'): "))
inputs = pickle.load(open('runs/'+directory_name+'/inputs.pkl','rb'))
setup_glc = glc.Setup(inputs)

# List of atomic number and atomic mass for the isotopes included in this run
run_iso = setup_glc.ZA_sorted

# The iso_class instance of the Isotopes class (in yields.py) lets you select the index (in run_iso) of various isotopes
# For example, for C12 (Z=6)
C12_idx = setup_glc.iso_class.pick_i_by_iso(setup_glc.ZA_sorted, 6, 12)
# This is the Carbon 12 index in all the outputs (Mass_i.dat, X_i.dat, W_i_comp.pkl)