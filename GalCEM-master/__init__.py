# init file for the GalCEM package
print ('Executing module the Galactic Chemical Evolution Model')
print ('GalCEM')

if __name__ == "__main__": 
    print ("GalCEM Module invoked directly")
	import onezone as o
    o.main()
    #elem = 'Na' e.g.
    #o.pick_yields('LIMS', 'Na', stellar_mass_idx=0, metallicity_idx=0)
	#o.pick_yields('Massive', 'Na', stellar_mass_idx=0, metallicity_idx=0, vel_idx=0)
	#o.pick_yields('SNIa', 'Na')
else: 
    print ("GalCEM Module imported")