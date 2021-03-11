import numpy as np
import pandas as pd
from astropy.table import Table
from astropy.io import fits
import load_data as ld
import spectres

#input median stacked fluxes,
wavs = np.arange(2400, 4200, 2.5)

def Dn4000(fluxes):
    #n is for narrow - less sensitive to reddening effects, used more in literature - 4000A to 4100A instead of 4050 - 4250
    #indices for Dn4000 for rest frame wavelenghts of spectra
    #measure the strength of the 4000A break (Dn4000)
    mask_3850 = (wavs <= 3950) & (wavs >= 3850) #blue continuum
    flux_3850 = np.nanmean(fluxes[mask_3850])
    #print(flux_3850)
    mask_4000 = (wavs >= 4000) & (wavs <= 4100) #red continuum
    flux_4000 = np.nanmean(fluxes[mask_4000])
    #print(flux_4000)
    Dn4000_index = flux_4000/flux_3850


    return Dn4000_index


def C29_33(fluxes):
    c = 3*10**8
    F_nu = fluxes*(wavs**2)/c
    #indices for the C(29-33) line for rest frame wavelengths of spectra
    mask_2900 = (wavs <= 3100) & (wavs >= 2700)
    flux_2900 = np.nanmean(F_nu[mask_2900])
    mask_3300 = (wavs <= 3500) & (wavs >= 3100)
    flux_3300 = np.nanmean(F_nu[mask_3300])

    flux_ratio = flux_2900/flux_3300
    C_ind = -2.5*np.log10(flux_ratio)

    return C_ind


def H_delta(fluxes, wavs):
    #continuum & line edges from Bolagh.M, 1999
    #changed to make either side 40 Angstroms so no weighting change needed
    #======== bagpipes =====================
    blue_cont = [4042., 4082.] #bagpipes
    red_cont = [4122., 4162.]
    line_edges = [4082., 4122.]
    #======== Worthey & Ottaviani (1997) ============================
    #blue_cont = [4041.6, 4079.75] # used in Wu paper
    #red_cont = [4128.5, 4161.0]
    #line_edges = [4083.5, 4122.25]
    #=========================================================
    mask_blue = (wavs > blue_cont[0]) & (wavs < blue_cont[1]) #blue continuum
    flux_blue = np.array(fluxes[mask_blue])
    mask_red = (wavs > red_cont[0]) & (wavs < red_cont[1]) #red continuum
    flux_red = np.array(fluxes[mask_red])
    #flux_red = fluxes[mask_red]
    line = (wavs > line_edges[0]) & (wavs < line_edges[1])

    flux_feature = np.nanmean(np.array(fluxes[line]))

    line_width  = line_edges[1] - line_edges[0] #angstroms
    flux_continuum = np.sum(np.nanmean(flux_blue)+np.nanmean(flux_red))/2 #one number
    H_delta_EW = line_width*(flux_continuum - flux_feature)
    H_delta_EW /=flux_continuum
    #multiply by 40 angstroms / continuum flux
    # EW found by summing fluxes in region, and fitting with gaussian

    return H_delta_EW #negative if emission, postive if absorption



def Mg_UV(flux):
    mask_2625 = (wavs < 2625)& (wavs > 2525)
    mask_2725 = (wavs > 2625) & (wavs < 2725)
    mask_2825 = (wavs > 2725) & (wavs < 2825)
    int_flux_2725 = np.trapz(flux[mask_2725], x = wavs[mask_2725])
    int_flux_2625 = np.trapz(flux[mask_2625], x = wavs[mask_2625])
    int_flux_2825 = np.trapz(flux[mask_2825], x = wavs[mask_2825])

    Mg_UV_index = (2 * int_flux_2725)/(int_flux_2625 + int_flux_2825)

    return Mg_UV_index

#concat_3dhst = Table.read('../passive_project/FirstProjectCatalogs/concat_3dhst_passive_match.fits').to_pandas()
#redshifts = concat_3dhst['zspec']
#print(redshifts)
#ID_list = np.array(concat_3dhst['FIELD'].str.decode("utf-8").str.rstrip() + concat_3dhst['ID_1'].astype(str).str.pad(6, side='left', fillchar='0')+ concat_3dhst['CAT'].str.decode("utf-8"))
#df = pd.DataFrame(concat_3dhst)

#ID_ = df.set_index(concat_3dhst['FIELD'].str.decode("utf-8").str.rstrip() + concat_3dhst['ID_1'].astype(str).str.pad(6, side='left', fillchar='0') + concat_3dhst['CAT'].str.decode("utf-8"))
lega_c = pd.read_table("merged_legac_cat.cat", delimiter = ',')

df = pd.DataFrame(lega_c)
#print(df)
ind_list = df.set_index(df['id'])
#print(ind_list)
IDs = df['id'].values
redshifts = df['z_spec']


def calc_indices(objects_list, index_list):
    new_wavs = np.arange(2400, 4200, 1.25)
    new_spec = np.zeros(len(new_wavs))
    new_errs = np.zeros(len(new_wavs))

    old_spec =[]
    old_errs = []

    spec = []
    spec_err = []
    med_norm = []
    med_spec_units  = []

    colour_index = []
    D4000_index =[]
    Mg_UV_index = []
    H_delta_EW = []
    for ID in objects_list:
        #ID = ID.decode("utf-8")
        #print(ID)
        #input()

        index = list(IDs).index(ID)
        #print(index)
        #input()
        z = df['z_spec'].values[index]
        #z = ind_list.loc[ID, 'z_spec']
        #print(z)
        #input()
        spectrum = ld.load_legac_spec(ID)

        #vandels
        #z = index_list.loc[ID, 'redshifts']
        #spectrum = ld.load_vandels_spectra(ID)
        #wav_mask = (spectrum[:,0]>5200) #& (spectrum[:,0]<970)

        flux = spectrum[:,1]/(10**19)
        flux_errs = spectrum[:,2]/(10**19)
        wavs = spectrum[:,0]

        zeros_mask = (flux == 0.)|(flux_errs == 0.)
        flux[zeros_mask] = np.nan
        flux_errs[zeros_mask] = np.nan

        rest_wavs = wavs/(1.0+z)
        #print(rest_wavs)
        mask = (rest_wavs > 3850.) & (rest_wavs < 3950.)
        #rest_wavs = wavs/(1.0 + z)
        #mask =  (rest_wavs > 3000) & (rest_wavs < 3500) # fairly featureless region of spectrum
        old_spec = flux/np.nanmedian(flux[mask]) #normalisation median from that region
        old_errs = flux_errs/np.nanmedian(flux[mask])

        new_spec, new_errs = spectres.spectres(new_wavs, rest_wavs, old_spec, spec_errs=old_errs)
        #colour_index.append(C29_33(new_spec))
        D4000_index.append(Dn4000(new_spec))
        #Mg_UV_index.append(Mg_UV(new_spec))
        H_delta_EW.append(H_delta(new_spec, new_wavs))

        #create a table so its easier to see the indices
    col1 = fits.Column(name = 'ID', format='30A', array=objects_list)
    col4 = fits.Column(name = 'RA', format = 'E', array = df['ra'].values)
    col5 = fits.Column(name = 'DEC', format = 'E', array = df['dec'].values)
    #col5 = fits.Column(name='C(29-33)', format='E', array=colour_index)
    #col6 = fits.Column(name='MgUV', format='E', array=Mg_UV_index)
    col7 = fits.Column(name ='Dn4000', format = 'E', array = D4000_index)
    col8 = fits.Column(name = 'new_EW(Hd)', format = 'E', array = H_delta_EW)
    hdu = fits.BinTableHDU.from_columns([col1, col4, col5, col7, col8])
    file =  "legac_massi_indices.fits"
    hdu.writeto(file)

    spec_inds_file = Table.read(file).to_pandas()
    spec_ind_df = pd.DataFrame(spec_inds_file)
    return spec_ind_df

#spec_ind_df = calc_indices(IDs, ind_list)

"""
cat3 = Table.read("../passive_project/Re_cat_strict_UVJ_cut.fits").to_pandas()
df = pd.DataFrame(cat3)
IDs = np.array(df['IDs'])
print(len(IDs))

IDs = [i.decode('utf-8') for i in df['IDs']]

print(IDs)

index_list = df.set_index(i.decode('utf-8') for i in df['IDs'])
spec_ind_df = calc_indices(IDs, index_list)

print(spec_ind_df)
"""
