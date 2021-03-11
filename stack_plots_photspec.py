import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from astropy.table import Table
from astropy.io import fits
import spectres
import load_data as ld
from astropy.stats import sigma_clip
import scipy
import photometry_stacking as ps
plt.rc('text', usetex=True)



def vdw_relation(logA, alpha, x_values):
    logR_eff = logA + alpha*(np.log10((10**x_values)/(5*10**10)))
    return logR_eff
alpha = 0.76
log_A = 0.29

new_wavs = np.arange(2400, 4200, 2.5)
from astropy.cosmology import FlatLambdaCDM
from astropy import units as u
import scipy
cosmo  = FlatLambdaCDM(H0=70, Om0=0.3)
Mpc_to_kpc = 1000

def stack_ids(lower_lim, higher_lim, cat):
    df4 = pd.DataFrame(cat)
    df4 = df4.groupby((df4['stellar_mass_50']>lower_lim)&(df4['stellar_mass_50']<=higher_lim)).get_group(True)
    stricter_masses = df4["stellar_mass_50"]
    IDs = [s.rstrip().decode('utf-8') for s in df4['new_id']]
    redshifts_4 = df4['zspec'].values
    masses2 = df4['stellar_mass_50']
    #print(len(df4))

    #masses2 = np.array(masses2)
    arcsec_per_kpc = cosmo.arcsec_per_kpc_proper(redshifts_4)

    Re_kpc_vandels = (df4['re_arcsecs'].values*u.arcsec)/arcsec_per_kpc
    Re_kpc_errs_vandels = (df4['re_err_arcsecs'].values*u.arcsec)/arcsec_per_kpc

    Re_kpc_vandels = Re_kpc_vandels.value
    Re_kpc_errs_vandels = Re_kpc_errs_vandels.value
    stricter_masses = np.array(masses2)
    stricter_sizes= Re_kpc_vandels
    R_e_errs_4 = Re_kpc_errs_vandels
    #R_e_errs_4 = df4["Re_kpc_errs"]
    #redshifts_4 = df4['redshifts']
    size_4 = np.array(np.log10(stricter_sizes))
    #index_4 = np.log10(stricter_sizes).index.to_list()
    x_array_4 = np.linspace(lower_lim, higher_lim, len(size_4))
    vdw_norm_model_4 = log_A + alpha*np.log10((10**x_array_4)/(5*10**10))
    a, b = 0.51, 0.44900000000000106
    vandels_wu_relation = a * (x_array_4 - 11) + b
    #mask_4 = (size_4>vdw_norm_model_4)
    #mask1_4 = (size_4<vdw_norm_model_4)


    mask_4 = (size_4>vandels_wu_relation)
    mask1_4 = (size_4<vandels_wu_relation)

    #index_masked_mass_v = np.log10(stricter_sizes)[mask_v].index.tolist()
    index_masked_mass_v = np.where(mask_4 ==True)
    index_masked_mass_v = np.array(index_masked_mass_v).tolist()
    #print(index_masked_mass_v[0])
    #index_masked2_mass_v = np.log10(stricter_sizes)[mask1_v].index.tolist()
    index_masked2_mass_v = np.where(mask1_4 == True)
    index_masked2_mass_v = np.array(index_masked2_mass_v).tolist()
    IDs_above_vandels = []

    for ind in index_masked_mass_v[0]:
        IDs_above_vandels.append(IDs[ind])
    #IDs_above_vandels = IDs[index_masked_mass_v[0]].values
    IDs_below_vandels = []
    for ind2 in index_masked2_mass_v[0]:
        IDs_below_vandels.append(IDs[ind2])

    #index_masked_mass_4 = np.log10(stricter_sizes)[mask_4].index.to_list()
    #index_masked2_mass_4 = np.log10(stricter_sizes)[mask1_4].index.to_list()
    #IDs_above = IDs[index_masked_mass_4].str.decode("utf-8").str.rstrip().values
    #IDs_below = IDs[index_masked2_mass_4].str.decode("utf-8").str.rstrip().values
    #len_above = len(IDs_above)
    #len_below = len(IDs_below)
    #stacking_4_above, stacking_errors_above = stacks(IDs_above)
    #stacking_4_below, stacking_errors_below = stacks(IDs_below)
    return IDs_above_vandels, IDs_below_vandels
filt_list = ['/Users/PhDStuff/passive_project/vandels/cfht_U',
       '/Users/PhDStuff/passive_project/vandels/subaru_B',
       '/Users/PhDStuff/passive_project/vandels/subaru_V',
       '/Users/PhDStuff/passive_project/vandels/subaru_R',
       '/Users/PhDStuff/passive_project/vandels/subaru_i',
       '/Users/PhDStuff/passive_project/vandels/subaru_newz',
       '/Users/PhDStuff/passive_project/vandels/vista_Y',
       '/Users/PhDStuff/passive_project/vandels/wfcam_J',
       '/Users/PhDStuff/passive_project/vandels/wfcam_H',
       '/Users/PhDStuff/passive_project/vandels/wfcam_K',
       '/Users/PhDStuff/passive_project/vandels/IRAC1',
       '/Users/PhDStuff/passive_project/vandels/IRAC2']

def load_filter_files(filter_list):
    filter_curves = []

    for filter in filter_list:
        filter_curves.append(np.loadtxt(str(filter)))

    return filter_curves
filt_curves = load_filter_files(filt_list)
#print(self.filter_curves)
def calc_eff_wavs(filter_curves):
    eff_wavs = []

    for f in filter_curves:
        flux_filter = f[:,1]/np.max(f[:,1])
        wav_filter = f[:,0]
        eff_wavs.append(np.sum(wav_filter*flux_filter)/np.sum(flux_filter))

    return eff_wavs

eff_wavs = calc_eff_wavs(filt_curves)
#eff_wavs = [ 3731.62130113,  4328.7247591,   5959.65935111,  7704.8374697,
 # 8084.3513428,   9049.08564392,  10585.0870973,
 #12516.25874447, 15391.40826974,  21576.74743872,
 #35572.60461226, 45048.50974132]

def plot_stacks_single(spec_stack_above, spec_stack_below, len_ID_sa, len_ID_sb, phot_stack_above, phot_stack_below, phot_err_a, phot_err_b, len_ID_pa, len_ID_pb, name):
    plt.rc('font', family='serif')
    plt.rc('xtick', labelsize='medium')
    plt.rc('ytick', labelsize='medium')
    fig, (ax1, ax2) = plt.subplots(2, figsize = [20,12], gridspec_kw={'hspace': 0.25})
    fig.suptitle('Photometry and spectroscopy median stacks: 10.75 $<$ log$_{10}$(M*) $\leq$ 11.3', size = 20)#plt.figure(figsize=(20,8))
    ax1.plot(new_wavs, spec_stack_above[0]*10**18, color='r', lw=1., ls ='-', label = f' Above van der Wel relation (N = {len_ID_sa})' )
    ax1.plot(new_wavs, spec_stack_below[0]*10**18, color='k', lw=1., label = f' Below van der Wel relation (N = {len_ID_sb})')
    ax1.fill_between(new_wavs, y1 = spec_stack_above[0]*10**18 - spec_stack_above[1]*10**18, y2 = spec_stack_above[0]*10**18 + spec_stack_above[1]*10**18, facecolor='r', alpha = 0.5)
    ax1.fill_between(new_wavs, y1 = spec_stack_below[0]*10**18 - spec_stack_below[1]*10**18, y2 = spec_stack_below[0]*10**18 + spec_stack_below[1]*10**18, facecolor='k', alpha = 0.5)
    ax1.plot(new_wavs, np.zeros(len(new_wavs)), color = 'grey', ls = '--', lw = 1.)
    ax1.set_xlabel("Wavelength ($\mathrm{\AA}$)", size=12)
    ax1.set_ylabel("Flux $(10^{-18}\ \mathrm{erg\ s^{-1}\ cm^{-2}\ \\AA{^{-1}})}$", size=12)#("Flux (normalised units)", size =12)#
    ax1.set_xlim(2400, 4200)
    ax1.legend(fontsize=10, loc = 'upper left')
    ax1.set_title(r'Median spectroscopy stacks', size = 15)# excluding possible AGN (CDFS + UDS)')

    ax2.errorbar(np.log10(eff_wavs), phot_stack_above*10**18, phot_err_a*10**18, color="k", mfc = 'r', linestyle='none', markersize = 10,  capsize = 8,  marker = 'o', mec = 'k', label = f' Above van der Wel relation (N = {len_ID_pa})') #change to ax.errorbar after test
    ax2.errorbar(np.log10(eff_wavs), phot_stack_below*10**18,phot_err_b*10**18, color="k", mfc = 'k', linestyle='none', markersize = 10, capsize = 8,  marker = 'o',  mec = 'k', label = f' Below van der Wel relation (N = {len_ID_pb})')
    ax2.set_xlabel("log$_{10}(\lambda$/ $\mathrm{\AA}$)", size=14)
    ax2.set_ylabel("Flux $(10^{-18}\ \mathrm{erg\ s^{-1}\ cm^{-2}\ \\AA{^{-1}})}$", size=12)#("Flux (normalised units)", size =12)#
    ax2.plot(np.linspace(3,5, 100), np.zeros(100), color = 'grey', ls = '--', lw = 1.)

    ax2.set_xlim(3.5, 4.75)
    #plt.ylim(0. ,1.75)
    ax2.legend(fontsize=10, loc = 'upper left')
    #ax2.set_xscale('log')
    ax2.set_title(r'Median photometry stacks', size = 15)# excluding possible AGN (CDFS + UDS)')
    plt.subplots_adjust(top=0.92)
    plt.savefig(str(name)+'.pdf')
    plt.close()


def plot_stacks(spec_stack_above, spec_stack_below, len_ID_sa, len_ID_sb):
    plt.rc('font', family='serif')
    plt.rc('xtick', labelsize='medium')
    plt.rc('ytick', labelsize='medium')
    fig, (ax1) = plt.subplots(figsize = [20,6])#, gridspec_kw={'hspace': 0.25})
    fig.suptitle('Spectroscopy median stacks: 10.7 $<$ log$_{10}$(M*) $\leq$ 11.2', size = 20)#plt.figure(figsize=(20,8))
    ax1.plot(new_wavs, spec_stack_above[0]*10**18, color='r', lw=1., ls ='-', label = f' Above Wu et al. relation (N = {len_ID_sa})' )
    ax1.plot(new_wavs, spec_stack_below[0]*10**18, color='k', lw=1., label = f' Below Wu et al. relation (N = {len_ID_sb})')
    ax1.fill_between(new_wavs, y1 = spec_stack_above[0]*10**18 - spec_stack_above[1]*10**18, y2 = spec_stack_above[0]*10**18 + spec_stack_above[1]*10**18, facecolor='r', alpha = 0.5)
    ax1.fill_between(new_wavs, y1 = spec_stack_below[0]*10**18 - spec_stack_below[1]*10**18, y2 = spec_stack_below[0]*10**18 + spec_stack_below[1]*10**18, facecolor='k', alpha = 0.5)
    ax1.set_xlabel("Wavelength ($\mathrm{\AA}$)", size=12)
    ax1.set_ylabel("Flux $(10^{-18}\ \mathrm{erg\ s^{-1}\ cm^{-2}\ \\AA{^{-1}})}$" , size=12)#$(10^{-18}\ \mathrm{erg\ s^{-1}\ cm^{-2}\ \\AA{^{-1}})}$", size=12)
    ax1.set_xlim(2350, 4240)
    ax1.legend(fontsize=10, loc = 'upper left')
    ax1.set_title(r'Median spectroscopy stacks', size = 15)# excluding possible AGN (CDFS + UDS)')
    plt.savefig('NEW84_spec_stacks_WUrel_10_7_11_2_real_units_check.pdf')

#cat3 = Table.read("/Users/Important_Tables_PhD_Project/NEW_passive_uvj_sizes_d4000.fits").to_pandas()
cat3 = Table.read("/Users/Important_Tables_PhD_Project/NEW_uvj_84_sizes_d4000.fits").to_pandas()
IDs = cat3['new_id']
#DA5A86 pink
#5A86DA blue
#'#44754F'
#'#75446A'
#30504C
#4C3050
#006666 green
#13662B
#660066 purple
#109966
#991043
#orange = #F44336
IDs_above, IDs_below = stack_ids(10.7, 11.2, cat3)

print(IDs_above, IDs_below)

spec_ab, phot_ab, med_new, med_norm = ps.stacks_phot(IDs_above)
spec_be, phot_be, med_new_b, med_norm_b = ps.stacks_phot(IDs_below)
#36C0C0
#C03636

#print(len(med_norm))


import cmasher as cmr

colors = cmr.take_cmap_colors('viridis', 10, return_fmt='hex')
print(colors)
#input()'#1B0C41', '#FB9906' '#440154' '#35B779'
cols = ['#000004', '#1B0C41', '#4C0C6B', '#781C6D', '#A52C60', '#CF4446', '#ED6925', '#FB9906', '#F7D13D', '#FCFFA4']
#colors = ['#000004', '#180F3D', '#451077', '#721F81', '#9E2F7F', '#CD4071', '#F1605D', '#FD9467', '#FECA8D', '#FCFDBF']
#for i in colors:
#    plt.plot([1,2,3],[2,4,6], color = i, lw = 15)
#    plt.show()
"""
conversion = ((10**-29)*(2.9979*10**18))/(np.array(eff_wavs)**2)
plt.rc('font', family='serif')
plt.rc('xtick', labelsize='medium')
plt.rc('ytick', labelsize='medium')

fig,(ax1, ax2) = plt.subplots(2, figsize = [12,4], sharex = True)

for i in IDs_above:
    photometry = ld.load_vandels_stacks(i)
    index = list(IDs_above).index(str(i))
    #print(index)
    #print(med_norm[index])
    flux = (photometry[:,0]*conversion)
    flux/=(med_norm[index])
    #print(med_norm[i])
    #print(flux)

    ax1.scatter(np.log10(eff_wavs), (flux*med_new)*10**18, color = 'lightblue',)

for j in IDs_below:
    photometry = ld.load_vandels_stacks(j)
    index2 = list(IDs_below).index(str(j))
    flux = (photometry[:,0]*conversion)
    flux /= (med_norm_b[index2])

    ax2.scatter(np.log10(eff_wavs), flux*med_new_b*10**18, color = 'bisque',)
#ax.scatter(eff_wavs, stack_phot[0])
#ax.fill_between(eff_wavs, y1 = phot_ab[0]- phot_ab[1], y2 = phot_ab[0]+ phot_ab[1], facecolor='lightblue', label = 'median photometry stack')
ax1.scatter(np.log10(eff_wavs), phot_ab[0]*10**18, color = 'darkblue', edgecolors = 'k', label = 'above')
#ax.fill_between(eff_wavs, y1 = phot_be[0]- phot_be[1], y2 = phot_be[0]+ phot_be[1], facecolor='orange', label = 'median photometry stack')
ax2.scatter(np.log10(eff_wavs), phot_be[0]*10**18, color = 'darkorange', edgecolors = 'k', label = 'below')
#ax1.set_ylabel('Flux $(10^{-18}\ \mathrm{erg\ s^{-1}\ cm^{-2}\ \\AA{^{-1}})}$')
#plt.ylabel('Flux $(10^{-18}\ \mathrm{erg\ s^{-1}\ cm^{-2}\ \\AA{^{-1}})}$')
ax2.set_xlabel('log$_{10}(\lambda$/ $\mathrm{\AA}$)')

fig.text(0.08, 0.5, 'Flux $(10^{-18}\ \mathrm{erg\ s^{-1}\ cm^{-2}\ \\AA{^{-1}})}$', va='center', rotation='vertical')
ax1.legend(loc = 'upper left')
ax2.legend(loc = 'upper left')
ax2.set_ylim(0,3)
plt.close()
#plt.savefig('test_phot_stack_errors_dispersion_10_5_11_norms_2.pdf')
"""
plot_stacks(spec_ab, spec_be, len(IDs_above), len(IDs_below))
#plot_stacks_single(spec_ab, spec_be, len(IDs_above), len(IDs_below), phot_ab[0], phot_be[0],phot_ab[1], phot_be[1], len(IDs_above), len(IDs_below), 'phot_spec_stacks_10_75_11_3_redblack_realunits')
