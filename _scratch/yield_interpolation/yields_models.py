import onezone as o


#   lims
# self.yields = self.is_unique('Yield', split_length)
# self.elemA_sorting, self.elemA = self.is_unique('elemA', split_length)
# self.elemZ_sorting, self.elemZ = self.is_unique('elemZ', split_length)
# self.metallicityIni, self.metallicity_bins = self.is_unique('Zini', split_length)
# self.stellarMassIni, self.stellarMass_bins = self.is_unique('Mini', split_length)
# self.Returned_stellar_mass = self.is_unique('Mfin', split_length)
# 
# o.yields_LIMs_class.yields
# TODO
za_sorted = o.ZA_sorted
print(za_sorted[:5],'\n\n')

# snii
'''
self.elemZ = self.tables[0,0,:,1].astype('int')
self.elemA = self.tables[0,0,:,2].astype('int')
self.stellarMassIni = self.tables[:,:,:,3].astype('float')
self.yields = self.tables[:,:,:,4:].astype('float') 
'''
elemZ = o.yields_Massive_class.elemZ
elemA = o.yields_Massive_class.elemA
stellarMassIni = o.yields_Massive_class.stellarMassIni
yields = o.yields_Massive_class.yields
print(elemZ.shape, elemA.shape, stellarMassIni.shape, yields.shape,'\n\n')
print(elemZ[:5],'\n\n')
print(elemA[:5])
print(o.yields_Massive_class.tables[0,0,:,1].astype('int'))
isotopes = yields.shape[2]
for i in range(isotopes):
    isotope = yields[:,:,i,:]
    break