# init file for the GalCEM package
import onezone as o
print ('Executing module the Galactic Chemical Evolution Model')
print ('GalCEM')

if __name__ == "__main__": 
    print ("GalCEM Module invoked directly")
    o.main()
else: 
    print ("GalCEM Module imported")