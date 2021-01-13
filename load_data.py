import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from astropy.table import Table
from astropy.io import fits
import os
import re
from collections import OrderedDict
from glob import glob
"""
This file contains functions for use in bagpipes, and my own codes to load photometry and spectra for list of IDs.
load_vandels is the function which loads the photometry as a 2D array of 2 columns of fluxes and flux errors.
load_vandels_spectra does the same but for the VANDELS spectra, outputting the spectrum as a 2D array of wavelengths,
fluxes and flux errors. load_vandels_both simply returns the 'spectrum' and 'photometry' from the other two functions.
"""

#fluxes and flux errors columns are in wavelength order same as filters
#must input ID as e.g. CDFS_GROUND0000xxSELECT where chars are filled to 6 padded by leading zeroes.

def find_file(ID, extension):
    new_ID = re.search('\d+', ID).group()
    new_ID = new_ID.lstrip('0')
    #print(new_ID)
    for root, dirs, files in os.walk('/Users/PhDStuff/passive_project/catalogs'):
        if "CDFS" in ID:
            if 'HST' in ID:
                files = "VANDELS_CDFS_HST_PHOT_v1.0.fits"
            else:
                files = "VANDELS_CDFS_GROUND_PHOT_v1.0.fits"
        else:
            if 'HST' in ID:
                files = "VANDELS_UDS_HST_PHOT_v1.0.fits"
            else:
                files = "VANDELS_UDS_GROUND_PHOT_v1.0.fits"
        #print(files)
        path = os.path.join(root, files)
        data_file = Table.read(path).to_pandas()
        catalog = pd.DataFrame(data_file)
        ID_list = catalog['ID'].astype(str)

        if new_ID in ID_list.values:
            new_path = path
            both = os.path.split(path)
            prefix_for_load = both[1]
            prefix = prefix_for_load.split('VANDELS_')[1]
            pre = prefix.split('_PHOT')[0]
            pre1 = pre.split('_')[0]
            pre2 = pre.split('_')[1]
            #print(pre1, pre2)
            new_pre = pre1+'-'+pre2
            #print(new_pre)
            cols = catalog.columns.str.rstrip('')
            all_cols = cols.to_list()
            fluxcols = all_cols[5:]
            flux_errs = []
            flux = []

            if 'isofactor' in catalog.columns:
                for f in fluxcols:
                    if f.endswith('_2as_err') or f.startswith('CH1') and f.endswith('tphot_err') or f.startswith('CH2') and f.endswith('tphot_err'): #f == 'isofactor' and
                        flux_errs.append(f)
                    flux_errs = list(OrderedDict.fromkeys(flux_errs))

                    if f.endswith('_2as') or f.startswith('CH1') and f.endswith('tphot') or f.startswith('CH2') and f.endswith('tphot'):#f == 'isofactor'
                        flux.append(f)
                    flux = list(OrderedDict.fromkeys(flux))

            else:
                for fe in fluxcols:
                    if fe.endswith('_err'):
                        flux_errs.append(fe)
                    else:
                        flux.append(fe)

        return new_path, new_pre, flux_errs, flux, new_ID

#'CDFS_GROUND000013'
#path, prefix, flux_errs, flux, new_ID = find_file('CDFS-HST034930SELECT', 'fits')
#'CDFS-HST034930SELECT'
#'UDS-HST021385SELECT'
#print(f'path:{ path}\nprefix:{prefix}\n flux: {flux}\n flux_errs: {flux_errs}')

def load_vandels(object):
    path, prefix, flux_errs, flux_cols, new_ID = find_file(object, 'fits')
    #print(new_ID)
    #print('path', path)
    #print(flux_cols.type)
    cat_file = Table.read(path).to_pandas()
    catalog = pd.DataFrame(cat_file)

    ind = catalog.set_index(str(prefix) + catalog['ID'].astype(str).str.pad(6, side="left", fillchar="0") + catalog['CAT'].str.decode("utf-8"))
    #print(iso)

    if 'isofactor' in catalog.columns:
        iso = ind.loc[object, 'isofactor']
        #print('GROUND CATALOGUES')
        if 'CDFS' in object:
            offset = np.loadtxt("/Users/PhDStuff/passive_project/vandels/offsets_cdfs_ground.txt")


        else:
            offset= np.loadtxt("/Users/PhDStuff/passive_project/vandels/offsets_uds_ground.txt")


        fluxes = (ind.loc[object,flux_cols]*iso).astype(float)
        fluxerrs=ind.loc[object, flux_errs].astype(float)

        photometry = np.c_[fluxes,fluxerrs]
        photometry[:,0] *= offset
    else:
        #print('HST CATALOGUES')
        if 'UDS' in object:
            offset = np.loadtxt("/Users/PhDStuff/passive_project/vandels/offsets_uds_hst.txt")

        else:
            offset= np.loadtxt("/Users/PhDStuff/passive_project/vandels/offsets_cdfs_hst.txt")


        fluxes = ind.loc[object, flux_cols].values.astype(float)

        fluxerrs=ind.loc[object, flux_errs].values.astype(float)
        photometry = np.c_[fluxes,fluxerrs]
        photometry[:,0] *= offset

    for i in range(len(photometry)):
        if (photometry[i, 0] == 0.) or (photometry[i, 1] <= 0):
            photometry[i,:] = [0., 9.9*10**99.]

    for i in range(len(photometry)):
        if i < 10:
            max_snr = 20.

        else:
            max_snr = 10.

        if photometry[i, 0]/photometry[i, 1] > max_snr:
            photometry[i, 1] = photometry[i, 0]/max_snr

    return photometry


def load_vandels_stacks(object):
    path, prefix, flux_errs, flux_cols, new_ID = find_file(object, 'fits')
    #print(new_ID)
    #print('path', path)
    #print(flux_cols.type)
    cat_file = Table.read(path).to_pandas()
    catalog = pd.DataFrame(cat_file)

    ind = catalog.set_index(str(prefix) + catalog['ID'].astype(str).str.pad(6, side="left", fillchar="0") + catalog['CAT'].str.decode("utf-8"))
    #print(iso)
    #print(flux_cols)
    if 'CDFS' in object:

        if 'isofactor' in catalog.columns:
            cdfs_cols = [0,1,5,8,11,12,14,15,16,17,18,19]
            iso = ind.loc[object, 'isofactor']

            offset = np.loadtxt("/Users/PhDStuff/passive_project/vandels/offsets_cdfs_ground.txt")
        else:
            cdfs_cols = [0,1,2,2,3,5,7,8,9,11,12,13]
            offset = np.loadtxt("/Users/PhDStuff/passive_project/vandels/offsets_cdfs_hst.txt")
            iso = 1.
        #if flux_cols[cd] == flux_cols[cd-1], flux_cols[cd]=0.5*(flux_cols[2]+flux_cols[3])


        stack_flux = [flux_cols[cd] for cd in cdfs_cols]
        stack_errs = [flux_errs[cdfs] for cdfs in cdfs_cols]
        print(stack_flux, len(stack_flux))


        fluxes = (ind.loc[object,stack_flux]*iso).astype(float)

        fluxes[3] = 0.5*(fluxes[2]+fluxes[4]) #for R flux cdfs cols - column name is still F606W so need to change it doesnt matter tho
        fluxerrs = ind.loc[object, stack_errs].astype(float)

        photometry = np.c_[fluxes,fluxerrs]
        photometry[:,0] *= [offset[s] for s in cdfs_cols]

    elif 'UDS' in object:
        #print(flux_cols)

        if 'isofactor' in catalog.columns:
            cols = [0,1,2,3,4,6,8,9,10,11,12,13]
            iso = ind.loc[object, 'isofactor']
            offset = np.loadtxt("/Users/PhDStuff/passive_project/vandels/offsets_uds_ground.txt")

        else:
            cols = [0,1,3,4,5,7,8,10,12,13,15,16]
            offset = np.loadtxt("/Users/PhDStuff/passive_project/vandels/offsets_uds_hst.txt")
            iso = 1.
        stack_flux = [flux_cols[con] for con in cols]
        stack_errs = [flux_errs[co] for co in cols]
        #print(offset)
        fluxes = (ind.loc[object,stack_flux]*iso).values.astype(float)
        fluxerrs = ind.loc[object, stack_errs].values.astype(float)

        photometry = np.c_[fluxes,fluxerrs]
        photometry[:,0] *= [offset[o] for o in cols]


    for i in range(len(photometry)):
        if (photometry[i, 0] == 0.) or (photometry[i, 1] <= 0):
            photometry[i,:] = [0., 9.9*10**99.]

    for i in range(len(photometry)):
        if i < 10:
            max_snr = 20.

        else:
            max_snr = 10.

        if photometry[i, 0]/photometry[i, 1] > max_snr:
            photometry[i, 1] = photometry[i, 0]/max_snr

    return photometry

print(load_vandels_stacks('CDFS-HST034908SELECT'))
#print(load_vandels_stacks('UDS-HST021385SELECT'))
#print(load_vandels_stacks('UDS-GROUND199858SELECT'))
#print(load_vandels_stacks('CDFS-GROUND247244SELECT'))

def load_vandels_spectra(ID_no):
    pre = ID_no.split('-')[0]
    #print(pre)
    new_ID = re.search('\d+', ID_no).group()
    #ID = ID.lstrip('0')
    ID_new = str(pre) + str(new_ID)
    globpath = os.path.join('../passive_project/new_vandels_spec/', '*.fits')
    filelist = glob(globpath)
    #print(filelist)
    for i in range(len(filelist)):
        if ID_new in str(filelist[i]):
            hdulist = fits.open(filelist[i])
            flux = hdulist[0].data
            flux_err = hdulist[3].data
            redshift = hdulist[0].header['HIERARCH PND Z']
            wav_first_pixel = hdulist[0].header['CRVAL1']
            delt_wav = hdulist[0].header['CDELT1']
            wa_end = wav_first_pixel + (2154*delt_wav)
            wave = np.arange(wav_first_pixel, wa_end, delt_wav)
            spectrum=np.c_[wave, flux, flux_err]

    return spectrum

#test:
#print(load_vandels_spectra('UDS-HST004674SELECT'))


def load_vandels_both(ID_):
    photometry = load_vandels(ID_)
    spectrum = load_vandels_spectra(ID_)

    return  spectrum, photometry


#test:
#print(load_vandels_both('UDS-GROUND154207SELECT'))
