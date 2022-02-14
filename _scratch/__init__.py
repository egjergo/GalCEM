# init file for the GalCEM package
print ('Executing GalCEM')

if __name__ == "__main__": 
	print ("GalCEM Module invoked directly")
	import onezone as o
	o.main()
    #elem = 'Na' e.g.
    #o.pick_yields('LIMS', 'Na', stellar_mass_idx=0, metallicity_idx=0)
	#o.pick_yields('Massive', 'Na', stellar_mass_idx=0, metallicity_idx=0, vel_idx=0)
	#o.pick_yields('SNIa', 'Na')
	
	# Concentrations:
	#log10_avg_elem_vs_Fe = self.log10_avg_elem_vs_X(elemZ=26)
	#log10_avg_elem_vs_H = self.log10_avg_elem_vs_X(elemZ=1)
else: 
    print ("GalCEM Module imported")