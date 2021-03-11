import numpy as np
import pandas as pd
from astropy.table import Table
from astropy.io import fits
import matplotlib.pyplot as plt
import matplotlib
import spectres
import load_data as ld
from glob import glob
import os
from astropy.stats import sigma_clip
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from astropy.cosmology import FlatLambdaCDM
from astropy import units as u
import scipy
from astropy.io import ascii


#print(lega_c.shape)
#### reducing sample #####
#lega_c = lega_c.groupby((lega_c['use']==1)).get_group(True) #f_use = 1

plt.rc('text', usetex=True)
lega_c = pd.read_table("/Users/massissiliahamadouche/Downloads/legac_team2018-02-16.cat", delimiter = ' ', skiprows = [i for i in range(1,129)])
#print(len(lega_c))

df = pd.DataFrame(lega_c)
#### reducing sample #####
df = df.groupby((df['use']==1)).get_group(True) #f_use = 1
#print(len(df)) #1257
df = df.groupby( (df['z_spec']>=0.6) &(df['z_spec']<=1.)).get_group(True) #0.6<z<1.0

df = df.groupby((df['f_sfr']==0) & (df['f_galfit']==0) & (df['f_phot']==0) & (df['f_ppxf']==0)&(df['f_spec']==0) &(df['f_z']==0)&(df['f_int']==0)).get_group(True)
print(len(df))
#df = df.groupby((df['SN']>=20.)).get_group(True) #s/n >10
#df = df.groupby((df['SN_rf_4000']>=10.)).get_group(True) #S/N SN_rf_4000 > 10
#df = df.groupby((df['eqw_obs_Hb']>=-1)).get_group(True)
df =df.groupby((df['uvj_sfq']==1)).get_group(True)
df =df.groupby((df['fast_lmass']>10.3)).get_group(True)

#df = df.groupby((np.log(df['lsfr_UV_IR'])<-10)).get_group(True) #S/N SN_rf_4000 > 10
print(len(df))
df.set_index('id')

import astropy
from astropy.cosmology import FlatLambdaCDM
from scipy.integrate import simps
from astropy import units as u

cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
Mpc_m = 3.086*10**22 #Mpc in m
Mpc_cm = Mpc_m*10**2 #in centimetres

legac_Vc = cosmo.comoving_volume(0.7)
vandels_Vc = ((0.2/41253))*((cosmo.comoving_volume(1.3).value) - (cosmo.comoving_volume(1).value))
n_density_vandels = 18/vandels_Vc

legac_Vc = ((1.3/41253))*((cosmo.comoving_volume(1.0).value) - (cosmo.comoving_volume(0.6).value))
n_density_legac = 152/legac_Vc
print(n_density_legac, n_density_vandels)

degsq =  (6.1 *u.arcmin).to(u.deg)
#print(degsq)
threedhst = ((degsq.value/41253))*((cosmo.comoving_volume(4).value) - (cosmo.comoving_volume(3).value))
n_density_3dhst = 20/threedhst
#n = 20/(0.2*(np.pi/180)**2*dist)
#print(n_density_3dhst)
print('Vc = ', str("{:.2E}".format((vandels_Vc))))
print('Vc = ', str("{:.2E}".format((legac_Vc))))
print('Nc = ', str("{:.2E}".format((n_density_legac))))

#############
input()

UV = df['UV'].values
VJ = df['VJ'].values
age = df['fast_lage'].values
Re_kpc = df['galfit_re_kpc']
ssfr = df['fast_lsfr'].values
mass = df['fast_lmass'].values
objID = np.array(df['id'].values)

lega_c_indices = Table.read("/Users/massissiliahamadouche/Downloads/legac_dr2_cat.fits").to_pandas()
df2 = pd.DataFrame(lega_c_indices)
objID_2 = lega_c_indices['OBJECT']

df2.set_index('OBJECT')

df2 = df2.rename(columns={'OBJECT':'id'})
left = df
right = df2

merged_df = left.merge(right, how = 'inner', left_on = 'id',right_on= 'id')
merged_df = merged_df.drop_duplicates('id')
merged_df = merged_df.dropna(subset=['id', 'LICK_D4000_N']) #remove rows with d4000 = NaN
merged_df= merged_df[merged_df['LICK_D4000_N'] != 0] #remove rows with d4000 = 0.
print(len(merged_df))
#print(merged_df['LICK_D4000_N'].values)
index_list_wu  = merged_df.set_index(merged_df['id'])
my_best_c = 0.29000000000000054
a, b = 0.51, 0.63
a1, b1 = 0.55, 0.63
x = np.linspace(9., 12.5, len(Re_kpc))
wu_relation = a * (x - 11) + b
wu_relation2 = a1*(x - 11) + b1
alpha = 0.76
log_A = 0.22

log_Reff = log_A + alpha*np.log10((10**x)/(5*10**10))

cat3 = Table.read("/Users/Important_Tables_PhD_Project/NEW_passive_uvj_sizes_d4000.fits").to_pandas()
df4 = pd.DataFrame(cat3)
mass_max = np.max(df4['stellar_mass_50'].values)

print('MAX VANDELS MASS:', mass_max)

df4 = df4.groupby((df4['stellar_mass_50']>10.3) & (df4['stellar_mass_50']<=mass_max)).get_group(True)#10.3<m<11.5 = 61 objects, 10.4<m<11.5 =55 &(df4['log10(M*/Msun)']<=11.5)

IDs = [s.rstrip().decode('utf-8') for s in df4['new_id']]
redshifts_4 = df4['zspec'].values
masses2 = df4['stellar_mass_50']
print(len(df4))

from astropy.cosmology import FlatLambdaCDM
from astropy import units as u
import scipy

cosmo  = FlatLambdaCDM(H0=70, Om0=0.3)
Mpc_to_kpc = 1000
#masses2 = np.array(masses2)
arcsec_per_kpc = cosmo.arcsec_per_kpc_proper(redshifts_4)

Re_kpc_vandels = (df4['re'].values*u.arcsec)/arcsec_per_kpc
Re_kpc_errs_vandels = (df4['dre'].values*u.arcsec)/arcsec_per_kpc

Re_kpc_vandels = Re_kpc_vandels.value
Re_kpc_errs_vandels = Re_kpc_errs_vandels.value
print(Re_kpc_vandels)


#df4  = df4.groupby((df4['z_spec']>=0.96)&(df4['z_spec']<=1.3)).get_group(True) #for mass completeness (see Adams paper)


d4000_vandels = df4['Dn4000'].values
for i in range(len(d4000_vandels)):
    if d4000_vandels[i] == 0.:
        d4000_vandels[i] = np.nan
print(d4000_vandels)

stricter_masses = np.array(masses2)
stricter_sizes= Re_kpc_vandels
R_e_errs_4 = Re_kpc_errs_vandels

index_list_size_vandels = df4.set_index(k.rstrip().decode('utf-8') for k in df4['new_id'])
print(index_list_size_vandels)


"""
#from spectral_indices import calc_indices
IDs = df4['IDs']
#print('no. of vandels passive gals in this mass range: ', len(IDs))
IDs = [i.decode('utf-8') for i in df4['IDs']]
#print('IDs of vandels passive gals in this mass range: ', IDs)
index_list = df4.set_index(i.decode('utf-8') for i in df4['IDs'])
spec_ind_df = calc_indices(IDs, index_list)
#print(spec_ind_df)

spec_inds = Table.read("spectral_indices"+str(len(stricter_sizes))+".fits").to_pandas()
#IDs = cat3['IDs']
df_si = pd.DataFrame(spec_inds)
index_list_ind = df_si.set_index(s.decode('utf-8') for s in df_si['ID'])
D4000_vandels = df_si['Dn4000']
H_delta = df_si['EW(Hd)']
"""

size_4 = np.log10(stricter_sizes)
#index_4 = np.log10(stricter_sizes).index.to_list()
print(len(size_4))
#merged LEGA-C catalog with D4000 and sizes
merged_df = merged_df.groupby((merged_df['fast_lmass']>10.3)&(merged_df['fast_lmass']<=11.62)).get_group(True)#& (merged_df['fast_lmass']<=11.62)
print(len(merged_df))
Re_kpc = merged_df['galfit_re_kpc']
ssfr = merged_df['fast_lsfr'].values
mass = merged_df['fast_lmass'].values
d4000 = merged_df['LICK_D4000_N'].values
ID_wu = merged_df['id']
print(np.max(mass))

#merged_df.to_csv('merged_legac_cat.cat')

#print(len(Re_kpc))
#stats of d4000 for the whole reduced lega-c sample
legac_std = np.nanstd(d4000)
legac_mean = np.nanmean(d4000)
legac_median = np.nanmedian(d4000)

vandels_std  = np.nanstd(d4000_vandels)
vandels_mean = np.nanmean(d4000_vandels)
vandels_median = np.nanmedian(d4000_vandels)


print('ALL OBJECTS: \n LEGA-C: \n std = ', legac_std , '\n mean = ', legac_mean, '\n median = ', legac_median,
'\n VANDELS: \n std = ', vandels_std , '\n mean = ', vandels_mean, '\n median = ', vandels_median)

#finding the best fit c for the z~0.75 van Der Wel ETG size-mass relation
al, A = 0.71, 0.42

def vdw_relation(logA, alpha, x_values):
    logR_eff = logA + alpha*(np.log10((10**x_values)/(5*10**10)))
    return logR_eff

#vdw_0_75 = vdw_relation(0.42, 0.71, x)
log_A_model = np.arange(-0.3, 0.3, 0.01)
best_chi2 = np.inf

for cvals2 in range(len(log_A_model)):
    c_vals2 = log_A_model[cvals2]

    vdw_model = vdw_relation(c_vals2, 0.71, stricter_masses) #for the 0.75 relation
    diffs2 = vdw_model - size_4
    #print(diffs)
    #print(y_model, '\n', np.log10(R_c))#(0.434*(Rc_errs/R_c)
    chisq2 = np.sum((diffs2**2)/((4.34*(R_e_errs_4/stricter_sizes))))

    if chisq2 < best_chi2:
        best_chi_vdw = chisq2
        best_c_vdw = c_vals2

vandels_c_0_75 = best_c_vdw
x = np.linspace(10.0, 11.8, len(Re_kpc))
wu_new_relation = a * (x - 11) + b
vdw_0_75 = vdw_relation(0.42, 0.71, x)
x_v = np.linspace(10.0, 11.8, len(size_4))
print(f'best c: {vandels_c_0_75}')
vandels_0_75_relation = vdw_relation(vandels_c_0_75, 0.71, x)
print(f'offset between vdw and vandels for z = 0.75 is:{np.round((0.42 - vandels_c_0_75),3)}')
vandels_relation = vdw_relation(vandels_c_0_75, 0.71, x_v)

a, b = 0.51, 0.63
a1, b1 = 0.55, 0.63

b_model = np.arange(-0.61, 0.45, 0.001)
best_chi2 = np.inf
x_arr = np.array(stricter_masses)
for bvals in range(len(b_model)):
    b_vals = b_model[bvals]

    wu_model = a*(x_arr - 11) + (b_vals)
    diffs2 = wu_model - size_4
    #print(diffs)
    #print(y_model, '\n', np.log10(R_c))#(0.434*(Rc_errs/R_c)
    chisq2 = np.sum((diffs2**2)/((4.34*(R_e_errs_4/stricter_sizes))))

    if chisq2 < best_chi2:
        best_chi_wu = chisq2
        best_b_wu = b_vals

vandels_wu_b = best_b_wu

print('vandels best b for wu ', vandels_wu_b)
print('offset wu ', vandels_wu_b - b)

x_van = np.linspace(10.0, 11.8, len(size_4))
vandels_wu_relation = a*(x_van - 11) + vandels_wu_b

"""
equation_vdw = '$\mathrm{log_{10}{(R_{e}/kpc)}}$ = ' + str(round(al,4)) + ' $\mathrm{log_{10}{(M*/((5 x 10^{10})M_{\odot}))}}$' ' + ' + str(round(vandels_c_0_75,4))
equation_wu = '$\mathrm{log_{10}{(R_{e}/kpc)}}$ = ' + str(round(a,2)) + ' $\mathrm{log_{10}{(M*/M_{\odot})}}$' ' + ' + str(round(vandels_wu_b,2))
plt.rc('font', family='serif')
plt.rc('xtick', labelsize='small')
plt.rc('ytick', labelsize='small')
fig1, ax1 = plt.subplots(figsize=[16,10.5])
im1 = ax1.scatter(mass, np.log10(Re_kpc), s=300, c=d4000, cmap=plt.cm.bone, alpha = 0.85,  marker='o', linewidth=0.5)
#ax1.scatter(np.array(stricter_masses), size_4, s = 100, c = 'purple', marker = 'o', ec = 'k', label  = f'VANDELS PASSIVE GALAXIES (N = {len(stricter_sizes)})')
#im2 = ax1.scatter(np.array(stricter_masses), size_4, s=300, c=d4000_vandels,  vmin=1., vmax=1.9,  cmap=plt.cm.magma, marker='o', linewidth=0.5)#, label  = f'VANDELS PASSIVE GALAXIES (N = {len(stricter_sizes)})')
cbar1 = fig1.colorbar(im1, ax=ax1)

cbar1.set_label('D$_{n}$4000', size=18)
ax1.plot(x, wu_new_relation, 'k', lw = 2.5, label= 'Wu et al. 2018 $z \sim$ 0.7')
#ax1.plot(x, wu_relation2, 'r', lw = 2.5, label= 'Wu et al. 2018 $z \sim$ 0.7 (2)')
ax1.scatter(np.array(stricter_masses), size_4, s = 100, c = 'red', marker = 'o', ec = 'k', label  = f'VANDELS PASSIVE GALAXIES (N = {len(stricter_sizes)})')
#ax1.plot(x_v, vandels_relation, 'k', lw = 2.5, label=f'van der Wel et al. 2014 $z$ = 0.75 ETG \n relation')
#ax1.plot(x_v, vandels_relation, 'k', lw = 2.5, label=equation_vdw)
ax1.plot(x, vdw_0_75, 'k', ls = '--', lw = 2.5, label = 'van der Wel et al. 2014 $z$ = 0.75')
#ax1.plot(x_van, vandels_wu_relation, 'r' ,lw = 2.5,  label= f'VANDELS offset ({np.round(vandels_wu_b -b,3)} dex) \n Wu et al. 2018 relation ($z$ = 0.75)')
#ax1.plot(x_van, vandels_wu_relation, 'r' ,lw = 2.5,  label= equation_wu)

ax1.set_xlabel('$\mathrm{log_{10}{(M*/M_{\odot})}}$', size = 18)
ax1.set_ylabel('$\mathrm{log_{10}{(R_{e}/kpc)}}$', size = 18)
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.legend(prop={'size': 12}, loc='lower right')
plt.title('LEGA-C log$_{10}$(Re) v log$_{10}$(M*/M$_{\odot}$)', size = 20)
plt.xlim(10.2, 11.7)
plt.ylim(-0.5, 1.35)
#ax1.axvline(x=11.4, color = 'k', ls = '--', lw = 0.8)
#ax1.axvline(x=10.65, color = 'k', ls = '--', lw = 0.8)
#plt.clim(0., 2.0)
#plt.show()
plt.savefig('NEW_d4000_legac+vandels_bone.pdf')
#plt.savefig('vandels_0_75offset_EWHd_cbar.pdf')
plt.close()
input()
"""



print("\n Finding stats for D4000 above + below (offset) WU RELATION z ~ 0.75 line for VANDELS galaxies:")
print(np.max(size_4), np.max(stricter_masses))
#mask_v = (size_4>vandels_wu_relation)#
mask_v = (size_4>vandels_relation)
#mask1_v = (size_4<=vandels_wu_relation)
mask1_v = (size_4<=vandels_relation)
print(mask_v)
#index_masked_mass_v = np.log10(stricter_sizes)[mask_v].index.tolist()
index_masked_mass_v = np.where(mask_v ==True)
index_masked_mass_v = np.array(index_masked_mass_v).tolist()
print(index_masked_mass_v[0])

#index_masked2_mass_v = np.log10(stricter_sizes)[mask1_v].index.tolist()
index_masked2_mass_v = np.where(mask1_v == True)
index_masked2_mass_v = np.array(index_masked2_mass_v).tolist()
IDs_above_vandels = []

for ind in index_masked_mass_v[0]:
    IDs_above_vandels.append(IDs[ind])

#IDs_above_vandels = IDs[index_masked_mass_v[0]].values

IDs_below_vandels = []
for ind2 in index_masked2_mass_v[0]:
    IDs_below_vandels.append(IDs[ind2])

#IDs_above_vandels = IDs[index_masked_mass_v[0]].str.decode("utf-8").str.rstrip().values
#IDs_below_vandels = IDs[index_masked2_mass_v[0]].str.decode("utf-8").str.rstrip().values
print('IDS', IDs_above_vandels, IDs_below_vandels)

#print(index_list_ind)

d4000_vandels_above = []
mass_v_above = []
for id1 in IDs_above_vandels:
    print(id1)
    d4000_vandels_above.append(index_list_size_vandels.loc[id1, 'Dn4000'])
    mass_v_above.append(index_list_size_vandels.loc[str(id1), 'stellar_mass_50'])

print(d4000_vandels_above)
input()
d4000_vandels_below = []
mass_v_below = []

"""
for id2 in index_masked2_mass_v:
    print(id2)
    d4000_vandels_below.append(index_list_size_vandels.iloc[id2, 'Dn4000'])
print(d4000_vandels_below)
input()
"""

for id2 in IDs_below_vandels:
    #print(id2)
    d4000_vandels_below.append(index_list_size_vandels.loc[id2, 'Dn4000'])
    mass_v_below.append(index_list_size_vandels.loc[str(id2), 'stellar_mass_50'])

vandels_std_ab = np.nanstd(d4000_vandels_above)
vandels_mean_ab = np.nanmean(d4000_vandels_above)
vandels_median_ab = np.nanmedian(d4000_vandels_above)

vandels_std_be = np.nanstd(d4000_vandels_below)
vandels_mean_be = np.nanmean(d4000_vandels_below)
vandels_median_be = np.nanmedian(d4000_vandels_below)

print('for 10.5 - 11.3 mass range', len(d4000_vandels_above), len(d4000_vandels_below), 'mass_above', np.nanmedian(mass_v_above), 'mass_below', np.nanmedian(mass_v_below))
print('VANDELS: \n ABOVE van der Wel z~0.75 relation: \n std = ', vandels_std_ab , '\n mean = ', vandels_mean_ab, '\n median = ', vandels_median_ab,
'\n BELOW offset van der Wel z~0.75 relation: \n std = ', vandels_std_be , '\n mean = ', vandels_mean_be, '\n median = ', vandels_median_be)

print('END')
input()

a, b = 0.51, 0.63
x = np.linspace(10.75, 11.62, len(Re_kpc))
wu_new_relation = a * (x - 11) + b
vdw_0_75 = vdw_relation(0.42, 0.71, x)

print("\n Finding stats for D4000 above + below Wu z~ 0.75 relation for LEGA-C galaxies:")
#mask_wu = (np.log10(Re_kpc)>wu_new_relation)#
mask_wu = (np.log10(Re_kpc)>vdw_0_75)#
#mask1_wu = (np.log10(Re_kpc)<wu_new_relation)#
mask1_wu = (np.log10(Re_kpc)<vdw_0_75)
index_masked_mass_wu = np.log10(Re_kpc)[mask_wu].index.to_list()
index_masked2_mass_wu = np.log10(Re_kpc)[mask1_wu].index.to_list()
IDs_above_wu = ID_wu[index_masked_mass_wu].values#.str.decode("utf-8").str.rstrip().values
IDs_below_wu  = ID_wu[index_masked2_mass_wu].values#.str.decode("utf-8").str.rstrip().values

d4000_massi_calc = Table.read("legac_massi_indices_xmatch.fits").to_pandas()
legac_massi = pd.DataFrame(d4000_massi_calc)

index_list_mas_legac = legac_massi.set_index(k.rstrip().decode('utf-8') for k in legac_massi['ID'])
print(len(index_list_wu), len(index_list_mas_legac))

#print(index_list_mas_legac)
#input()
#print(IDs_above_wu, IDs_below_wu)
d4000_i_above = []
mass_above = []
for id in IDs_above_wu:
    d4000_i_above.append(index_list_wu.loc[id, 'LICK_D4000_N'])#index_list_mas_legac str(id)
    mass_above.append(index_list_wu.loc[id, 'fast_lmass'])


d4000_i_below = []
mass_below = []
for idw in IDs_below_wu:
    d4000_i_below.append(index_list_wu.loc[idw, 'LICK_D4000_N'])
    mass_below.append(index_list_wu.loc[idw, 'fast_lmass'])
print('above', len(d4000_i_above),'below', len(d4000_i_below))
print(np.nanmedian(mass_above), np.nanmedian(mass_below))

legac_std_ab = np.nanstd(d4000_i_above)
legac_mean_ab = np.nanmean(d4000_i_above)
legac_median_ab = np.nanmedian(d4000_i_above)

legac_std_be = np.nanstd(d4000_i_below)
legac_mean_be = np.nanmean(d4000_i_below)
legac_median_be = np.nanmedian(d4000_i_below)


print('LEGA-C : \n ABOVE vdw z~0.75 relation: \n std = ', legac_std_ab , '\n mean = ', legac_mean_ab, '\n median = ', legac_median_ab,
'\n BELOW vdw z~0.75 relation: \n std = ', legac_std_be , '\n mean = ', legac_mean_be, '\n median = ', legac_median_be)

input()
print('stacking above and below for log10(M*) > 10.6:')

new_wavs = np.arange(3300, 5400, 2.0)
def plot_stacks(spec_stack_above, spec_stack_below, len_ID_sa, len_ID_sb):
    plt.rc('font', family='serif')
    plt.rc('xtick', labelsize='medium')
    plt.rc('ytick', labelsize='medium')
    fig, (ax1) = plt.subplots(figsize = [15,5])#, gridspec_kw={'hspace': 0.25})
    fig.suptitle('LEGA-C stacked spectra: 10.75 $<$ log$_{10}$(M*) $<$ 11.62', size = 20)#plt.figure(figsize=(20,8))
    ax1.plot(new_wavs, spec_stack_above[0], color='r', lw=1., ls ='-', label = f' Above van Der Wel relation (N = {len_ID_sa})' )
    ax1.plot(new_wavs, spec_stack_below[0], color='k', lw=1., label = f' Below van Der Wel relation (N = {len_ID_sb})')
    ax1.fill_between(new_wavs, y1 = spec_stack_above[0] - spec_stack_above[1], y2 = spec_stack_above[0] + spec_stack_above[1], facecolor='r', alpha = 0.5)
    ax1.fill_between(new_wavs, y1 = spec_stack_below[0] - spec_stack_below[1], y2 = spec_stack_below[0] + spec_stack_below[1], facecolor='k', alpha = 0.5)
    ax1.set_xlabel("Wavelength ($\mathrm{\AA}$)", size=12)
    ax1.set_ylabel("Flux $(10^{-18}\ \mathrm{erg\ s^{-1}\ cm^{-2}\ \\AA{^{-1}})}$" , size=12)#$(10^{-18}\ \mathrm{erg\ s^{-1}\ cm^{-2}\ \\AA{^{-1}})}$", size=12)
    ax1.set_xlim(3350, 5300)
    ax1.legend(fontsize=10, loc = 'upper left')
    #ax1.set_title(r'Median spectroscopy stacks', size = 15)# excluding possible AGN (CDFS + UDS)')
    plt.savefig('legac_stacks_10_6_11_6_test_massbias.pdf')

import legac_stacking as lg

legac_stacks_above = lg.legac_stacks(IDs_above_wu)
legac_stacks_below = lg.legac_stacks(IDs_below_wu)
print(legac_stacks_above)
plot_stacks(legac_stacks_above, legac_stacks_below, len(IDs_above_wu), len(IDs_below_wu))





input()

"""

plt.rc('font', family='serif')
plt.rc('xtick', labelsize='small')
plt.rc('ytick', labelsize='small')

fig1, ax1 = plt.subplots(figsize=[16,10.5])
im1 = ax1.scatter(mass, np.log10(Re_kpc), s=300, c=d4000, cmap=plt.cm.magma, marker='o', linewidth=0.5, alpha = 0.8)
#im2 = ax1.scatter(np.array(stricter_masses), size_4, s=300, c=H_delta, cmap=plt.cm.magma, marker='o', linewidth=0.5, )# label  = f'VANDELS PASSIVE GALAXIES (N = {len(stricter_sizes)})')
cbar1 = fig1.colorbar(im1, ax=ax1)
cbar1.set_label('D$_{n}$4000', size=18)
ax1.plot(x, wu_new_relation, 'midnightblue', lw = 2.5, label= 'Wu et al. 2018 $z \sim$ 0.7')
#ax1.plot(x, wu_relation2, 'r', lw = 2.5, label= 'Wu et al. 2018 $z \sim$ 0.7 (2)')
ax1.scatter(np.array(stricter_masses), size_4, s = 200, c = 'r', marker = 'o', ec = 'k', label  = f'VANDELS PASSIVE GALAXIES (N = {len(stricter_sizes)})')
ax1.plot(x, vdw_0_75, 'grey', lw = 2.5, label=f'van der Wel et al. 2014 $z$ = 0.75 ETG \n relation')
ax1.plot(x, vandels_0_75_relation, 'r' ,lw = 2.5,  label= 'VANDELS shifted van der Wel. 2014 relation ($z$ = 0.75)')
ax1.set_xlabel('$\mathrm{log_{10}{(M*/M_{\odot})}}$', size = 18)
ax1.set_ylabel('$\mathrm{log_{10}{(R_{e}/kpc)}}$', size = 18)
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.legend(prop={'size': 12}, loc='lower right')
plt.title('LEGA-C log$_{10}$(Re) v log$_{10}$(M*/M$_{\odot}$)', size = 20)
plt.xlim(10.0, 11.8)
plt.ylim(-0.7, 1.5)
plt.savefig('vdw_0_75_legac_wu_vandels_d4000_masscomplete.pdf')
#plt.savefig('vandels_0_75offset_EWHd_cbar.pdf')
plt.close()

"""


vandels_size_cat = Table.read('VANDELS_size_d4000.fits').to_pandas()

v_sizes = vandels_size_cat['Re_kpc']
v_masses = vandels_size_cat['log10(M*/Msun)']

v_errors = vandels_size_cat['Re_kpc_errs']

v_model = np.arange(-8., -5.7, 0.001)
print(v_model)
print('vandels sizes', np.log10(np.array(v_sizes)), '\n', 'masses', np.array(v_masses))

input()

best_chiv = np.inf
x_arr_v = np.array(v_masses)
x = np.linspace(10.0, 11.3, len(v_sizes))
from scipy.optimize import curve_fit

def f(x, A, B): # this is your 'straight line' y=f(x)
    y = A*x + B
    return y

popt, pcov = curve_fit(f, x_arr_v, v_sizes ) # your data x, y to fit
print('scipy.optimise a', popt[0],'b',  popt[1])

def best_fit_slope_and_intercept(xs,ys):
    m = (((np.mean(xs)*np.mean(ys)) - np.mean(xs*ys)) /
         ((np.mean(xs)*np.mean(xs)) - np.mean(xs*xs)))

    b = np.mean(ys) - m*np.mean(xs)

    return m, b

m, b = best_fit_slope_and_intercept(x_arr_v,v_sizes)

print('regression line:', 'm = ', np.log10(m),'b = ', np.log10(-b))

model  = np.log10(m)*(x) - np.log10(-b)

print(model)

for bs in range(len(v_model)):
    b_s = v_model[bs]
    #new_v_model =
    new_v_model = 0.56*(x_arr_v) + b_s
    print(new_v_model)
    #input()
    diffs_v = new_v_model - np.log10(v_sizes)
    print(diffs_v)
    #print(y_model, , np.log10(R_c))#(0.434*(Rc_errs/R_c)
    chisqv = np.sum((diffs_v**2)/(4.34*(np.log10(v_errors)/np.log10(v_sizes))))

    if chisqv < best_chiv:
        best_chi_v = chisqv
        best_b_v = b_s

vandels_b_v = best_b_v
print(vandels_b_v, best_chi_v)
print(np.sort(v_sizes))
x = np.linspace(10.0, 11.5, len(v_sizes))
new_v_model = 0.56*(x) + (vandels_b_v)
print(new_v_model)

l_model = np.arange(-7.55, -5.55, 0.01)
best_chil = np.inf
for ls in range(len(l_model)):
    l_s = l_model[ls]
    #new_v_model =
    new_l_model = 0.56*(mass) + l_s
    #print(new_l_model)
    #input()
    diffs_l = new_l_model - np.log10(Re_kpc)
    #print(diffs_l)
    #print(y_model,, np.log10(R_c))#(0.434*(Rc_errs/R_c)
    chisql = np.sum((diffs_l**2)/(np.nanstd(Re_kpc)*1.25/np.sqrt(len(Re_kpc))))
    #(4.34*(np.log10(v_errors)/np.log10(v_sizes))))

    if chisql < best_chil:
        best_chi_l = chisql
        best_b_l = l_s

legac_b = best_b_l
print(legac_b, best_chi_l)
print(np.sort(Re_kpc))
legac_x = np.linspace(10.0, 11.8, len(Re_kpc))
new_l_model = 0.56*(legac_x) + (legac_b)
print(new_l_model)



input()

plt.rc('font', family='serif')
plt.rc('xtick', labelsize='small')
plt.rc('ytick', labelsize='small')
fig1, ax1 = plt.subplots(figsize=[16,10.5])
#ax1.scatter(mass, np.log10(Re_kpc), s=300, cmap=plt.cm.magma, marker='o', linewidth=0.5, alpha = 0.8)
#im2 = ax1.scatter(np.array(stricter_masses), size_4, s=300, c=H_delta, cmap=plt.cm.magma, marker='o', linewidth=0.5, )# label  = f'VANDELS PASSIVE GALAXIES (N = {len(stricter_sizes)})')
#cbar1 = fig1.colorbar(im1, ax=ax1)
#cbar1.set_label('D$_{n}$4000', size=18)
#ax1.plot(x, wu_new_relation, 'midnightblue', lw = 2.5, label= 'Wu et al. 2018 $z \sim$ 0.7')
#ax1.plot(x, wu_relation2, 'r', lw = 2.5, label= 'Wu et al. 2018 $z \sim$ 0.7 (2)')
ax1.scatter(np.array(mass), np.log10(Re_kpc), s = 200, c = 'r', marker = 'o', ec = 'k', label  = f'LEGA-C GALAXIES (N = {len(Re_kpc)})')
ax1.plot(legac_x, new_l_model, 'grey', lw = 2.5, label = ('Model R$_e$(kpc) ='+ str("{:.2E}".format(10**(legac_b)))+'$\mathrm(M/M_{\odot})^{0.56}$'))
#ax1.plot(x, model, 'r' ,lw = 2.5,  label= 'model scipy.optimise')
ax1.set_xlabel('$\mathrm{log_{10}{(M*/M_{\odot})}}$', size = 18)
ax1.set_ylabel('$\mathrm{log_{10}{(R_{e}/kpc)}}$', size = 18)
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.legend(prop={'size': 12}, loc='lower right')
plt.title('LEGA-C log$_{10}$(Re) v log$_{10}$(M*/M$_{\odot}$)', size = 20)
#plt.xlim(10.0, 11.8)
#plt.ylim(-0.7, 3.5)
plt.savefig('legac_0.56_relation.pdf')
#plt.savefig('vandels_0_75offset_EWHd_cbar.pdf')
#plt.close()


#732 objects classified as passive from the uvj_sfq =1 in table
