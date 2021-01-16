import numpy as np
import pandas as pd
from astropy.table import Table
from astropy.io import fits
import spectres
import load_data as ld



uds_hst_filt =  np.loadtxt("../passive_project/catalogs/UDS_HST_filt_list.txt", dtype="str")
uds_ground_filt = np.loadtxt("../passive_project/catalogs/UDS_GROUND_filt_list.txt", dtype = "str")
cdfs_ground_filt = np.loadtxt("../passive_project/catalogs/CDFS_GROUND_filt_list.txt", dtype="str")
cdfs_hst_filt= np.loadtxt("../passive_project/catalogs/CDFS_HST_filt_list.txt", dtype="str")

#passive_cut = Table.read('FirstProjectCatalogs/xmatch_spec_derived237objs.fits').to_pandas()
#passive_cut = Table.read('FirstProjectCatalogs/x_match_final_passive_sample_edit.fits').to_pandas()
concat_3dhst = Table.read('../passive_project/FirstProjectCatalogs/concat_3dhst_passive_match.fits').to_pandas()
redshifts = concat_3dhst['zspec']
#print(redshifts)
ID_list = np.array(concat_3dhst['FIELD'].str.decode("utf-8").str.rstrip() + concat_3dhst['ID_1'].astype(str).str.pad(6, side='left', fillchar='0')+ concat_3dhst['CAT'].str.decode("utf-8"))
df = pd.DataFrame(concat_3dhst)

ID_ = df.set_index(concat_3dhst['FIELD'].str.decode("utf-8").str.rstrip() + concat_3dhst['ID_1'].astype(str).str.pad(6, side='left', fillchar='0') + concat_3dhst['CAT'].str.decode("utf-8"))


#for photometrt, need to use only the 12 filters UVBRizYJHK+CH1CH2

def stacks_phot(objects_list):

    new_wavs = np.arange(2400, 4200, 2.5)

    #for photometry stacking

    phot = []#np.zeros((12,len(objects_list)))
    phot_err = []#np.zeros((12,len(objects_list)))
    med_norm = []
    med_phot_units  = []

    median_phot = np.zeros(12)
    errs = np.zeros(12)
    phot_ = np.zeros(12)
    phot_errs = np.zeros(12)

    #med_photometry=np.zeros(len(new_wavs))

    #for spectrum stacking
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
        #print(id)
        z = ID_.loc[ID, 'zspec']
        #spectrum stuff
        spectrum = ld.load_vandels_spectra(ID)
        #print(spectrum)
        #ißnput()
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
        #print(old_spec, old_errs)
        #photometry stuff
        photometry = ld.load_vandels_stacks(ID)
        #print(ID)
        phot_flux = photometry[:,0]
        #input()
        #print('phot_flux', phot_flux)
        #input()
        phot_flux_errors = photometry[:,1]
        zeros_mask = (phot_flux == 0.)|(phot_flux_errors == 0.)
        phot_flux[zeros_mask] = np.nan
        phot_flux_errors[zeros_mask] = np.nan
        old_phot_flux = phot_flux#/np.nanmedian(flux[mask])
        old_phot_errs = phot_flux_errors#/np.nanmedian(flux[mask])
        #print('len(old_phot_flux)', len(old_phot_flux))

        #print('len(new_phot_flux)', len(new_phot))
        med_norm.append(np.nanmedian(flux[mask]))
        new_spec, new_errs = spectres.spectres(new_wavs, rest_wavs, old_spec, spec_errs=old_errs)
        #print(new_spec.dtype)
        #print(new_spec.shape)
        phot.append(old_phot_flux)
        phot_err.append(old_phot_errs)

        spec.append(new_spec)
        spec_err.append(new_errs)

    #tranpose the photometry fluxes so theyre the same shape as the spectrum ones
    phot = np.transpose(phot)
    phot_err = np.transpose(phot_err)
    standev_phot_errs = np.zeros(len(photometry[:,0]))

    spec = np.transpose(spec)
    spec_err = np.transpose(spec_err)
    standev_err = np.zeros(len(new_wavs))
    med_new = np.nanmedian(med_norm)
    #print(spec.shape)
    #print(f'med_new = {med_new}')
    #print('stacking median spectra')
    for m in range(len(new_wavs)):
        spec_ = spec[m,:]
        spec_errs = spec_err[m,:]
        standev_err[m] = (np.nanstd(spec_, axis=0)*1.25) #changed error calculation from mad to stand dev * 1.25
        #stack_error[m] = standev_err[m]/np.sqrt(len(spec))
        #print(standev_err[m])
        median_spec[m]=np.nanmedian(spec_)

    print('stacking median photometry')
    for n in range(len(photometry[:,0])): #number of photometry points = number of filters
        phot_ = phot[n,:]
        #print(phot_)
        standev_phot_errs[n] = (np.nanstd(phot_, axis=0)*1.25)
        #print(phot_errs[n])
        median_phot[n] = np.nanmedian(phot_)

    med_spec_units = median_spec*med_new #normalisations both the same
    med_phot_units = median_phot*med_new

    phot_stack_error = standev_phot_errs/np.sqrt(len(objects_list))
    spec_stack_error = standev_err/np.sqrt(len(objects_list))

    stacked_spectrum = med_spec_units, spec_stack_error*med_new
    stacked_photometry = med_phot_units, phot_stack_error*med_new

    return stacked_spectrum, stacked_photometry, med_new
"""
print(ID_list[0:10])
stacked_spec, stack_phot, med_new = stacks_phot(ID_list[0:10])

print(stacked_spec,'\n', stack_phot)

eff_wavs = [ 3731.62130113,  4328.7247591,   5959.65935111,  7704.8374697,
  8084.3513428,   9049.08564392,  10585.0870973,
 12516.25874447, 15391.40826974,  21576.74743872,
 35572.60461226, 45048.50974132] #from the SED code cdfs hst filters minus two but i cant remember which ones - only testing tho
import matplotlib.pyplot as plt
fig,ax = plt.subplots(figsize = [12,5])
for i in ID_list[0:10]:
    photometry = ld.load_vandels_stacks(i)
    ax.scatter(eff_wavs, photometry[:,0]*med_new, color = 'black')
ax.fill_between(eff_wavs, y1 = stack_phot[0]- stack_phot[1], y2 = stack_phot[0]+ stack_phot[1], facecolor='lightblue', label = 'median photometry stack')
ax.scatter(eff_wavs, stack_phot[0])
ax.set_ylabel('Flux')
ax.set_xlabel('Wavelength ($\\AA$)')
#ax.set_ylim(-2,30)
plt.legend()
#plt.show()
plt.savefig('test2_phot_stack_errors_dispersion.pdf')
"""
