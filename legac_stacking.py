import numpy as np
import pandas as pd
from astropy.table import Table
from astropy.io import fits
import spectres
import load_data as ld
import matplotlib.pyplot as plt


lega_c = pd.read_table("/Users/massissiliahamadouche/Downloads/legac_team2018-02-16.cat", delimiter = ' ', skiprows = [i for i in range(1,129)])
#print(len(lega_c))
df = pd.DataFrame(lega_c)
ind_list = df.set_index(df['id'])
#print(ind_list)
IDs = df['id'].values
redshifts = df['z_spec'].values[0]
#print(IDs)

#input()

#index_list = df.set_index('M'+df['mask'].str.decode("utf-8").str.rstrip() +'_'+ df['id'].str.decode("utf-8").str.rstrip() )
#df['SPECT_ID'] = df['SPECT_ID'].str.decode('utf-8')
#print(index_list)

def legac_stacks(objects_list): #input array of redshifts for given list of objects
    new_wavs = np.arange(3300, 5400, 2.0)
    new_spec = np.zeros(len(new_wavs))
    new_errs = np.zeros(len(new_wavs))

    old_spec =[]
    old_errs = []

    spec = []
    spec_err = []
    med_norm = []
    med_spec_units  = []

    median_spec = np.zeros(len(new_wavs))
    errs = np.zeros(len(new_wavs))
    spec_ = np.zeros(len(new_wavs))
    spec_errs = np.zeros(len(new_wavs))
    med_spec_units=[]
    med_spectrum =np.zeros(len(new_wavs))
    #print(objects_list)
    for ID in objects_list:
        #print(ID)
        #objects_list = list(objects_list)
        index = list(IDs).index(ID)
        #print(index)
        #input()
        z = df['z_spec'].values[index]
        #z = ind_list.loc[ID, 'z_spec']
        #print(z)
        #input()
        spectrum = ld.load_legac_spec(ID)

        #print(spectrum)
        #wav_mask = (spectrum[:,0]>5900) & (spectrum[:,0]<8500) #cut out the noisy parts of the spectra
        np.set_printoptions(threshold=np.inf)
        flux = np.array(spectrum[:,1])/(10**19)#[wav_mask]) #10^19 erg/s/cm^2/Å
        #print(flux)
        flux_errs = np.array(spectrum[:,2])/(10**19)#[wav_mask]) 10^19 erg/s/cm^2/Å
        #print(flux_errs)
        #input()
        wavs = np.array(spectrum[:,0])#[wav_mask])

        #print(wavs)
        zeros_mask = (flux == 0.)|(flux_errs == 0.)
        flux[zeros_mask] = np.nan
        flux_errs[zeros_mask] = np.nan
        #fig, ax = plt.subplots(figsize=[20,4])
        #ax.plot(wavs, flux)
        #ax.set_title('LEGA-C spectra test')
        #plt.show()

        rest_wavs = wavs/(1.0+z)
        #print(rest_wavs)
        mask = (rest_wavs > 3850.) & (rest_wavs < 3950.) #kinda featureless - see other spectra
        #mask =  (rest_wavs > 4000.) & (rest_wavs < 4300.) #blue cont for Hdelta
        # lower window of the d4000 break #  usually a fairly featureless region of spectrum
        #print(flux[mask])
        #input()
        old_spec = flux/np.nanmedian(flux[mask])#[mask]) #normalisation median from that region
        #np.set_printoptions(threshold=np.inf)

        old_errs = flux_errs/np.nanmedian(flux[mask])

        med_norm.append(np.nanmedian(flux[mask]))#flux[mask]
        new_spec, new_errs = spectres.spectres(new_wavs, rest_wavs, old_spec, spec_errs=old_errs)
        spec.append(new_spec)
        #print(spec)
        spec_err.append(new_errs)

    spec = np.transpose(spec)
    spec_err = np.transpose(spec_err)
    standev_err = np.zeros(len(new_wavs))
    med_new = np.nanmedian(med_norm)
    #print(med_new)

    for m in range(len(new_wavs)):
        spec_ = spec[m,:]
        spec_errs = spec_err[m,:]
        standev_err[m] = (np.nanstd(spec_, axis=0)*1.25) #changed error calculation from mad to stand dev * 1.25
        #stack_error[m] = standev_err[m]/np.sqrt(len(spec))
        #print(standev_err[m])
        median_spec[m]=np.nanmedian(spec_)

    med_spec_units = median_spec*med_new #test removing normalisations to see fluxes nh
    stack_error = (standev_err/np.sqrt(len(objects_list)))*med_new#standard error on the stacks
    #print(stack_error)
    return med_spec_units, stack_error #returns an array of the new median stacked fluxes


#objects = [126153, 126275, 126578, 206616, 210716, 216233, 28341]

#stack_legac = legac_stacks(objects)

#print(stack_legac)
#new_wavs = np.arange(3700, 5300, 2.0)
#import matplotlib.pyplot as plt

#fig, ax = plt.subplots(figsize=[20,4])
#ax.plot(new_wavs, stack_legac[0])
#ax.set_title('Stack LEGA-C test')
#plt.savefig('stack_legac_test.pdf')
