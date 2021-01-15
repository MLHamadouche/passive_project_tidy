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

def stack_ids(lower_lim, higher_lim, cat):
    df4 = pd.DataFrame(cat)
    df4 = df4.groupby((df4['log10(M*/Msun)']>lower_lim)&(df4['log10(M*/Msun)']<=higher_lim)).get_group(True)
    stricter_masses = df4["log10(M*/Msun)"]
    stricter_sizes= df4["Re_kpc"]
    R_e_errs_4 = df4["Re_kpc_errs"]
    redshifts_4 = df4['redshifts']
    size_4 = np.array(np.log10(stricter_sizes))
    index_4 = np.log10(stricter_sizes).index.to_list()
    x_array_4 = np.linspace(lower_lim, higher_lim, len(size_4))
    vdw_norm_model_4 = log_A + alpha*np.log10((10**x_array_4)/(5*10**10))
    mask_4 = (size_4>vdw_norm_model_4)
    mask1_4 = (size_4<vdw_norm_model_4)
    index_masked_mass_4 = np.log10(stricter_sizes)[mask_4].index.to_list()
    index_masked2_mass_4 = np.log10(stricter_sizes)[mask1_4].index.to_list()
    IDs_above = IDs[index_masked_mass_4].str.decode("utf-8").str.rstrip().values
    IDs_below = IDs[index_masked2_mass_4].str.decode("utf-8").str.rstrip().values
    #len_above = len(IDs_above)
    #len_below = len(IDs_below)

    #stacking_4_above, stacking_errors_above = stacks(IDs_above)
    #stacking_4_below, stacking_errors_below = stacks(IDs_below)
    return IDs_above, IDs_below
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
    fig, (ax1, ax2) = plt.subplots(2, figsize = [16,10], gridspec_kw={'hspace': 0.3})
    fig.suptitle('Photometry and spectroscopy median stacks: 10.3 $<$ log$_{10}$(M*) $<$ 11.3', size = 20)#plt.figure(figsize=(20,8))
    ax1.plot(new_wavs, spec_stack_above*10**18, color="r", lw=2., ls ='-', label = f' Above van der Wel relation (N = {len_ID_sa})' )
    ax1.plot(new_wavs, spec_stack_below*10**18, color="k", lw=2., label = f' Below van der Wel relation (N = {len_ID_sb})')

    ax1.set_xlabel("Wavelength ($\mathrm{\AA}$)", size=12)
    ax1.set_ylabel("Flux $(10^{-18}\ \mathrm{erg\ s^{-1}\ cm^{-2}\ \\AA{^{-1}})}$", size=12)
    ax1.set_xlim(2350, 4240)
    ax1.legend(fontsize=10)
    ax1.set_title(r'Median spectroscopy stacks', size = 15)# excluding possible AGN (CDFS + UDS)')

    ax2.errorbar(np.log10(eff_wavs), phot_stack_above, phot_err_a, color="k", mfc = 'r', linestyle='none', markersize = 10,  capsize = 8,  marker = 'o', mec = 'k', label = f' Above van der Wel relation (N = {len_ID_pa})') #change to ax.errorbar after test
    ax2.errorbar(np.log10(eff_wavs), phot_stack_below,phot_err_b, color="k", mfc = 'k', linestyle='none', markersize = 10, capsize = 8,  marker = 'o',  mec = 'k', label = f' Below van der Wel relation (N = {len_ID_pb})')
    ax2.set_xlabel("log$_{10}(\lambda$/ $\mathrm{\AA}$)", size=14)
    ax2.set_ylabel("Flux $(10^{-18}\ \mathrm{erg\ s^{-1}\ cm^{-2}\ \\AA{^{-1}})}$", size=12)
    #plt.xlim(2350, 4240)
    #plt.ylim(0. ,1.75)
    ax2.legend(fontsize=10, loc = 'upper left')
    ax2.set_title(r'Median photometry stacks', size = 15)# excluding possible AGN (CDFS + UDS)')

    plt.savefig(str(name)+'.pdf')
    plt.close()

cat3 = Table.read("../passive_project/Re_cat_strict_UVJ_cut.fits").to_pandas()
IDs = cat3['IDs']

IDs_above, IDs_below = stack_ids(10.3, 11.3, cat3)

print(IDs_above, IDs_below)

spec_ab, phot_ab = ps.stacks_phot(IDs_above)
spec_be, phot_be = ps.stacks_phot(IDs_below)

plot_stacks_single(spec_ab[0], spec_be[0], len(IDs_above), len(IDs_below), phot_ab[0], phot_be[0],phot_ab[1], phot_be[1], len(IDs_above), len(IDs_below), 'phot_spec_test_stacks_10_3_11_3')
