import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from astropy.table import Table
from astropy.io import fits
from stacking_code import stacks
import load_data as ld
import bagpipes as pipes


dblplaw = {}
dblplaw["tau"] = (0., 15.)
dblplaw["alpha"] = (0.01, 1000.)
dblplaw["beta"] = (0.01, 1000.)
dblplaw["alpha_prior"] = "log_10"
dblplaw["beta_prior"] = "log_10"
dblplaw["massformed"] = (1., 15.)
dblplaw["metallicity"] = (0.1, 2.5)
dblplaw["metallicity_prior"] = "log_10"

nebular = {}
nebular["logU"] = -3.

dust = {}
dust["type"] = "Salim"
dust["eta"] = 2.
dust["Av"] = (0., 4.)
dust["delta"] = (-0.3, 0.3)
dust["delta_prior"] = "Gaussian"
dust["delta_prior_mu"] = 0.
dust["delta_prior_sigma"] = 0.1
dust["B"] = (0., 5.)

fit_instructions = {}
fit_instructions["redshift"] = 1.
fit_instructions["t_bc"] = 0.01

fit_instructions["dblplaw"] = dblplaw
fit_instructions["nebular"] = nebular
fit_instructions["dust"] = dust

fit_instructions["veldisp"] = (100., 500.)   #km/s
fit_instructions["veldisp_prior"] = "log_10"


calib = {}
calib["type"] = "polynomial_bayesian"

calib["0"] = (0.5, 1.5) # Zero order is centred on 1, at which point there is no change to the spectrum.
calib["0_prior"] = "Gaussian"
calib["0_prior_mu"] = 1.0
calib["0_prior_sigma"] = 0.25

calib["1"] = (-0.5, 0.5) # Subsequent orders are centred on zero.
calib["1_prior"] = "Gaussian"
calib["1_prior_mu"] = 0.
calib["1_prior_sigma"] = 0.25

calib["2"] = (-0.5, 0.5)
calib["2_prior"] = "Gaussian"
calib["2_prior_mu"] = 0.
calib["2_prior_sigma"] = 0.25

fit_instructions["calib"] = calib

noise = {}
noise["type"] = "white_scaled"
noise["scaling"] = (1., 10.)
noise["scaling_prior"] = "log_10"
fit_instructions["noise"] = noise

#list_of_IDs = ['UDS-HST013753SELECT' ,'UDS-HST019329SELECT' ,'UDS-HST021218SELECT','UDS-HST024977SELECT' ,'CDFS-HST013637SELECT', 'CDFS-HST016360SELECT', 'CDFS-GROUND140661SELECT' ,'CDFS-HST026535SELECT', 'CDFS-HST001851SELECT']


cat3 = Table.read("../passive_project/Re_cat_strict_UVJ_cut.fits").to_pandas()

IDs = cat3['IDs']

def vdw_relation(logA, alpha, x_values):
    logR_eff = logA + alpha*(np.log10((10**x_values)/(5*10**10)))
    return logR_eff
alpha = 0.76
log_A = 0.29
    #x = np.linspace(9.5, 11.5, len(strict_sizes))
    #vdw_norm_model = vdw_relation(0.29, alpha, x)

new_wavs = np.arange(2400, 4200, 2.5)

def stack_lims(lower_lim, higher_lim):
    df4 = pd.DataFrame(cat3)
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
    len_above = len(IDs_above)
    len_below = len(IDs_below)

    stacking_4_above, stacking_errors_above = stacks(IDs_above)
    stacking_4_below, stacking_errors_below = stacks(IDs_below)
    #stacking_both_4_above = stacking_4_above, stacking_errors_above
    #stacking_both_4_below = stacking_4_below, stacking_errors_below
    #stacking_all = stacking_both_4_above, stacking_both_4_below
    #len_IDs = len_above, len_below

    return stacking_4_above, stacking_errors_above, IDs_above, stacking_4_below, stacking_errors_below, IDs_below, np.median(stricter_masses)

#stacking_4_above, stacking_errors_above, IDs_above, stacking_4_below, stacking_errors_below, IDs_below, med_mass = stack_lims(10.5, 10.75)

#print(IDs_above)
flux, error = stacking_4_above, stacking_errors_above
#flux, error = stacking_4_below, stacking_errors_below
"""
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('xtick', labelsize='x-small')
plt.rc('ytick', labelsize='x-small')

fig, ax = plt.subplots(figsize = [20, 5] )
#plt.plot(new_wavs, stacking_4_above*10**18, linewidth  = 0.8, color = 'black')
ax.plot(new_wavs, flux*10**18, lw = 1.7, color='darkcyan')
ax.fill_between(new_wavs, y1 = flux*10**18 - error*10**18, y2 = flux*10**18 + error*10**18, facecolor='lightblue')
ax.set_xlabel(r'Wavelength $\\AA$',size = 18 )
ax.set_ylabel('Flux $(10^{-18}\ \mathrm{erg\ s^{-1}\ cm^{-2}\ \\AA{^{-1}})}$', size=18)
ax.set_xlim(2400, 4200)
ax.set_ylim(0.1, 2.4)
plt.savefig('test_fill_between.pdf')
plt.close()
"""

list_of_IDs = IDs_above

def load_stacked_spectrum(id):

    stacked_spectrum, stack_errors = stacks(np.array(list_of_IDs))
    wavs = np.arange(2400, 4200, 2.5)*2

    flux_errs = stack_errors #need to ask about how to get median error spectrum out of the stacks
    #print(flux_errs)

    spectrum = np.c_[wavs, stacked_spectrum, flux_errs]

    return spectrum

"""
galaxy = pipes.galaxy('0001', load_data =load_stacked_spectrum, photometry_exists=False, spectrum_exists = True)
#galaxy.plot()

fit = pipes.fit(galaxy, fit_instructions, run="spectroscopy_test_stack_above_10.5_10.75")
fit.fit(verbose=False)

#fit._print_results()
mwa_array = []
#fit.posterior.get_advanced_quantities()
mwa_array.append(np.median(fit.posterior.samples["mass_weighted_age"]))
#fit1.posterior.get_advanced_quantities()
#print(np.median(fit1.posterior.samples["mass_weighted_age"]))
#fit2.posterior.get_advanced_quantities()
#print(np.median(fit2.posterior.samples["mass_weighted_age"]))

fig = fit.plot_spectrum_posterior(save=True, show=False)
fig = fit.plot_1d_posterior()
fig = fit.plot_calibration(save=True, show=False)
fig = fit.plot_sfh_posterior(save=True, show=False)
fig = fit.plot_corner(save=True, show=False)

"""
"""
stacking_4_above2, stacking_errors_above2, IDs_above2, stacking_4_below2, stacking_errors_below2, IDs_below2, med_mass2 = stack_lims(10.75, 11.0)

#print(IDs_above)
flux2, error2 = stacking_4_above2, stacking_errors_above2
list_of_IDs2 = IDs_above2

def load_stacked_spectrum(id):

    stacked_spectrum, stack_errors = stacks(np.array(list_of_IDs2))
    wavs = np.arange(2400, 4200, 2.5)*2

    flux_errs = stack_errors #need to ask about how to get median error spectrum out of the stacks
    #print(flux_errs)

    spectrum = np.c_[wavs, stacked_spectrum, flux_errs]

    return spectrum


galaxy2 = pipes.galaxy('0001', load_data =load_stacked_spectrum, photometry_exists=False, spectrum_exists = True)
#galaxy.plot()

fit2 = pipes.fit(galaxy2, fit_instructions, run="spectroscopy_test_stack_above_10.75_11.0")
fit2.fit(verbose=False)

mwa_array.append(np.median(fit2.posterior.samples["mass_weighted_age"]))

stacking_4_above3, stacking_errors_above3, IDs_above3, stacking_4_below3, stacking_errors_below3, IDs_below3, med_mass3 = stack_lims(11.0, 11.3)

#print(IDs_above)
flux3, error3 = stacking_4_above3, stacking_errors_above3
list_of_IDs3 = IDs_above3

def load_stacked_spectrum(id):

    stacked_spectrum, stack_errors = stacks(np.array(list_of_IDs3))
    wavs = np.arange(2400, 4200, 2.5)*2

    flux_errs = stack_errors #need to ask about how to get median error spectrum out of the stacks
    #print(flux_errs)

    spectrum = np.c_[wavs, stacked_spectrum, flux_errs]

    return spectrum


galaxy3 = pipes.galaxy('0001', load_data =load_stacked_spectrum, photometry_exists=False, spectrum_exists = True)
#galaxy.plot()

fit3 = pipes.fit(galaxy3, fit_instructions, run="spectroscopy_test_stack_above_11.0")
fit3.fit(verbose=False)

mwa_array.append(np.median(fit3.posterior.samples["mass_weighted_age"]))


print(mwa_array)



####### below ############
mwa_array_below = []

stacking_4_above4, stacking_errors_above4, IDs_above4, stacking_4_below4, stacking_errors_below4, IDs_below4, med_mass4 = stack_lims(10.5, 10.75)

#print(IDs_above)
flux4, error4 = stacking_4_below4, stacking_errors_below4
list_of_IDs4 = IDs_below4

def load_stacked_spectrum(id):

    stacked_spectrum, stack_errors = stacks(np.array(list_of_IDs4))
    wavs = np.arange(2400, 4200, 2.5)*2

    flux_errs = stack_errors #need to ask about how to get median error spectrum out of the stacks
    #print(flux_errs)

    spectrum = np.c_[wavs, stacked_spectrum, flux_errs]

    return spectrum


galaxy4 = pipes.galaxy('0001', load_data =load_stacked_spectrum, photometry_exists=False, spectrum_exists = True)
#galaxy.plot()

fit4 = pipes.fit(galaxy4, fit_instructions, run="spectroscopy_test_stack_below_10.5_10.75")
fit4.fit(verbose=False)

mwa_array_below.append(np.median(fit4.posterior.samples["mass_weighted_age"]))

stacking_4_above5, stacking_errors_above5, IDs_above5, stacking_4_below5, stacking_errors_below5, IDs_below5, med_mass5 = stack_lims(10.75, 11.0)

#print(IDs_above)
flux5, error5 = stacking_4_below5, stacking_errors_below5
list_of_IDs5 = IDs_below5

def load_stacked_spectrum(id):

    stacked_spectrum, stack_errors = stacks(np.array(list_of_IDs5))
    wavs = np.arange(2400, 4200, 2.5)*2

    flux_errs = stack_errors #need to ask about how to get median error spectrum out of the stacks
    #print(flux_errs)

    spectrum = np.c_[wavs, stacked_spectrum, flux_errs]

    return spectrum


galaxy5 = pipes.galaxy('0001', load_data =load_stacked_spectrum, photometry_exists=False, spectrum_exists = True)
#galaxy.plot()

fit5 = pipes.fit(galaxy5, fit_instructions, run="spectroscopy_test_stack_below_10.75_11.0")
fit5.fit(verbose=False)

mwa_array_below.append(np.median(fit5.posterior.samples["mass_weighted_age"]))


stacking_4_above6, stacking_errors_above6, IDs_above6, stacking_4_below6, stacking_errors_below6, IDs_below6, med_mass6 = stack_lims(11.0, 11.3)

#print(IDs_above)
flux6, error6 = stacking_4_below6, stacking_errors_below6
list_of_IDs6 = IDs_below6

def load_stacked_spectrum(id):

    stacked_spectrum, stack_errors = stacks(np.array(list_of_IDs6))
    wavs = np.arange(2400, 4200, 2.5)*2

    flux_errs = stack_errors #need to ask about how to get median error spectrum out of the stacks
    #print(flux_errs)

    spectrum = np.c_[wavs, stacked_spectrum, flux_errs]

    return spectrum


galaxy6 = pipes.galaxy('0001', load_data =load_stacked_spectrum, photometry_exists=False, spectrum_exists = True)
#galaxy.plot()

fit6 = pipes.fit(galaxy6, fit_instructions, run="spectroscopy_test_stack_below_11.0")
fit6.fit(verbose=False)

mwa_array_below.append(np.median(fit6.posterior.samples["mass_weighted_age"]))
mass_array1 = []
mass_array2 = []
mass_array1.append(med_mass)
mass_array1.append(med_mass2)
mass_array1.append(med_mass3)
mass_array2.append(med_mass4)
mass_array2.append(med_mass5)
mass_array2.append(med_mass6)
print(mwa_array_below, mass_array1, mass_array2)


plt.rc('font', family='serif')
plt.rc('xtick', labelsize='small')
plt.rc('ytick', labelsize='small')
fig2, ax2 = plt.subplots(figsize=[12,8.5])
ax2.plot(mass_array1, mwa_array_below,'D-',color = 'teal', label = 'below')
ax2.plot(mass_array1, mwa_array, 'D-', color = 'mediumorchid', label = 'above')
ax2.set_xlabel('median mass in stack')
ax2.set_ylabel('mass weighted age (from fit)')
plt.legend()
plt.savefig('test_agevmass.pdf')
plt.close()
"""
