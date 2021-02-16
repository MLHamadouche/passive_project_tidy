import numpy as np
import pandas as pd
from astropy.table import Table
from astropy.io import fits
import spectres
import load_data as ld
from astropy.stats import median_absolute_deviation
import matplotlib.pyplot as plt

"""
Outside of the function, define the pandas list of IDs that is needed to index the redshifts on line 49 (after for loop).
The function takes the argument 'objects_list' whcih is just a list of strings of the IDs of spectra to be loaded
by the LoadData function (imported as ld.load_vandels_spectra), or whichever loading data function as long as it takes the same
ID format (e.g. 'CDFS-HST034930SELECT') and outputs a 3d array, where spectrum[:,0]= wavelengths, spectrum[:,1] = fluxes,
spectrum[:,2] = flux errors. new_wavs is defined inside the function but can be modified.
"""

concat_3dhst = Table.read('../passive_project/FirstProjectCatalogs/concat_3dhst_passive_match.fits').to_pandas() #this is the crossmatched size catalog
#for our objects. Theres only 115 i think, but this table can be changed easily to another table of objects and everything else kept the same.
df = pd.DataFrame(concat_3dhst)

ID_ = df.set_index(concat_3dhst['FIELD'].str.decode("utf-8").str.rstrip() + concat_3dhst['ID_1'].astype(str).str.pad(6, side='left', fillchar='0') + concat_3dhst['CAT'].str.decode("utf-8"))

#For the error spectrum, what we really want is the standard deviation as measured by 1.483 times the median absolute deviation
#(https://en.wikipedia.org/wiki/Median_absolute_deviation). You then need to divide by the square root of the number of spectra
#in the stack to get the standard error.

def stacks(objects_list): #input array of redshifts for given list of objects
    new_wavs = np.arange(2400, 4200, 2.5)
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
    print(objects_list)
    for ID in objects_list:
        #print(id)
        z = ID_.loc[ID, 'zspec']
        #print(z)
        spectrum = ld.load_vandels_spectra(ID)
        #print(spectrum)
        wav_mask = (spectrum[:,0]>5200) & (spectrum[:,0]<9250) #cut out the noisy parts of the spectra

        flux = spectrum[:,1][wav_mask]
        flux_errs = spectrum[:,2][wav_mask]
        wavs = spectrum[:,0][wav_mask]
        zeros_mask = (flux == 0.)|(flux_errs == 0.)
        flux[zeros_mask] = np.nan
        flux_errs[zeros_mask] = np.nan

        rest_wavs = wavs/(1.0 + z)
        mask =  (rest_wavs > 3000) & (rest_wavs < 3500) # fairly featureless region of spectrum
        old_spec = flux/np.nanmedian(flux[mask]) #normalisation median from that region
        old_errs = flux_errs/np.nanmedian(flux[mask])

        med_norm.append(np.nanmedian(flux[mask]))
        new_spec, new_errs = spectres.spectres(new_wavs, rest_wavs, old_spec, spec_errs=old_errs)
        spec.append(new_spec)
        spec_err.append(new_errs)

    spec = np.transpose(spec)
    spec_err = np.transpose(spec_err)
    standev_err = np.zeros(len(new_wavs))
    med_new = np.nanmedian(med_norm)

    for m in range(len(new_wavs)):
        spec_ = spec[m,:]
        spec_errs = spec_err[m,:]
        standev_err[m] = (np.nanstd(spec_, axis=0)*1.25) #changed error calculation from mad to stand dev * 1.25
        #stack_error[m] = standev_err[m]/np.sqrt(len(spec))
        #print(standev_err[m])
        median_spec[m]=np.nanmedian(spec_)

    med_spec_units = median_spec*med_new #test removing normalisations to see fluxes nh
    stack_error = standev_err/np.sqrt(len(objects_list)) #standard error on the stacks 
    #print(stack_error)
    return med_spec_units, stack_error*med_new #returns an array of the new median stacked fluxes
