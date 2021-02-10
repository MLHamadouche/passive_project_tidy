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

cat3 = Table.read("../passive_project/Re_cat_strict_UVJ_cut.fits").to_pandas()
#IDs = cat3['IDs']
df4 = pd.DataFrame(cat3)
df4 = df4.groupby((df4['log10(M*/Msun)']>10.3)& (df4['log10(M*/Msun)']<=10.6)).get_group(True) #10.3<m<11.5 = 61 objects, 10.4<m<11.5 =55 &(df4['log10(M*/Msun)']<=11.5)
df4  = df4.groupby((df4['redshifts']>=1.0)&(df4['redshifts']<=1.3)).get_group(True) #for mass completeness (see Adams paper)
print(len(df4))

input()
stricter_masses = df4["log10(M*/Msun)"]
stricter_sizes= df4["Re_kpc"]
R_e_errs_4 = df4["Re_kpc_errs"]
redshifts_4 = df4['redshifts']
index_list_size_vandels = df4.set_index(k.decode('utf-8') for k in df4['IDs'])

from spectral_indices import calc_indices
IDs = df4['IDs']
#print('no. of vandels passive gals in this mass range: ', len(IDs))
#IDs = [i.decode('utf-8') for i in df4['IDs']]
#print('IDs of vandels passive gals in this mass range: ', IDs)
#index_list = df4.set_index(i.decode('utf-8') for i in df4['IDs'])
#spec_ind_df = calc_indices(IDs, index_list)
#print(spec_ind_df)

spec_inds = Table.read("spectral_indices"+str(len(stricter_sizes))+".fits").to_pandas()
#IDs = cat3['IDs']
df_si = pd.DataFrame(spec_inds)
index_list_ind = df_si.set_index(s.decode('utf-8') for s in df_si['ID'])
D4000_vandels = df_si['Dn4000']
H_delta = df_si['EW(Hd)']


size_4 = np.array(np.log10(stricter_sizes))
index_4 = np.log10(stricter_sizes).index.to_list()
merged_df = merged_df.groupby((merged_df['fast_lmass']>10.9) & (merged_df['fast_lmass']<=11.62) ).get_group(True)
#merged LEGA-C catalog with D4000 and sizes
print(len(merged_df))
Re_kpc = merged_df['galfit_re_kpc']
ssfr = merged_df['fast_lsfr'].values
mass = merged_df['fast_lmass'].values
d4000 = merged_df['LICK_D4000_N'].values
ID_wu = merged_df['id']
print(np.max(mass))
input()
merged_df.to_csv('merged_legac_cat.cat')

#print(len(Re_kpc))
#stats of d4000 for the whole reduced lega-c sample
legac_std = np.nanstd(d4000)
legac_mean = np.nanmean(d4000)
legac_median = np.nanmedian(d4000)

vandels_std  = np.nanstd(D4000_vandels)
vandels_mean = np.nanmean(D4000_vandels)
vandels_median = np.nanmedian(D4000_vandels)


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
x = np.linspace(10.9, 11.62, len(Re_kpc))
wu_new_relation = a * (x - 11) + b
vdw_0_75 = vdw_relation(0.42, 0.71, x)

print(f'best c: {vandels_c_0_75}')
vandels_0_75_relation = vdw_relation(vandels_c_0_75, 0.71, x)
print(f'offset between vdw and vandels for z = 0.75 is:{np.round((0.42 - vandels_c_0_75),3)}')
vandels_relation = vdw_relation(vandels_c_0_75, 0.71, np.linspace(10.3, 10.6, len(size_4)))
"""
fig1, ax1 = plt.subplots(figsize=[16,10.5])
#im1 = ax1.scatter(mass, np.log10(Re_kpc), s=300, c=d4000, cmap=plt.cm.magma, marker='o', linewidth=0.5, alpha = 0.8)
im2 = ax1.scatter(np.array(stricter_masses), size_4, s=300, c=D4000_vandels, cmap=plt.cm.magma, marker='o', linewidth=0.5, label  = f'VANDELS PASSIVE GALAXIES (N = {len(stricter_sizes)})')
cbar1 = fig1.colorbar(im2, ax=ax1)
cbar1.set_label('D$_{n}$4000', size=18)
#ax1.plot(x, wu_new_relation, 'midnightblue', lw = 2.5, label= 'Wu et al. 2018 $z \sim$ 0.7')
#ax1.plot(x, wu_relation2, 'r', lw = 2.5, label= 'Wu et al. 2018 $z \sim$ 0.7 (2)')
#ax1.scatter(np.array(stricter_masses), size_4, s = 200, c = 'r', marker = 'o', ec = 'k', label  = f'VANDELS PASSIVE GALAXIES (N = {len(stricter_sizes)})')
#ax1.plot(x, vdw_0_75, 'grey', lw = 2.5, label=f'van der Wel et al. 2014 $z$ = 0.75 ETG \n relation')
ax1.plot(x, vandels_0_75_relation, 'r' ,lw = 2.5,  label= 'VANDELS offset van der Wel. 2014 relation ($z$ = 0.75)')
ax1.set_xlabel('$\mathrm{log_{10}{(M*/M_{\odot})}}$', size = 18)
ax1.set_ylabel('$\mathrm{log_{10}{(R_{e}/kpc)}}$', size = 18)
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.legend(prop={'size': 12}, loc='lower right')
plt.title('VANDELS log$_{10}$(Re) v log$_{10}$(M*/M$_{\odot}$)', size = 20)
plt.xlim(10.1, 11.3)
plt.ylim(-0.7, 1.5)
plt.savefig('vandels_shiftedvdw0_75_d4000.pdf')
#plt.savefig('vandels_0_75offset_EWHd_cbar.pdf')
plt.close()
"""

print("\n Finding stats for D4000 above + below (offset) z ~ 0.75 line for VANDELS galaxies:")
print(np.max(size_4), np.max(stricter_masses))
mask_v = (size_4>vandels_relation)
mask1_v = (size_4<=vandels_relation)
print(mask_v)
index_masked_mass_v = np.log10(stricter_sizes)[mask_v].index.to_list()
print(index_masked_mass_v)
index_masked2_mass_v = np.log10(stricter_sizes)[mask1_v].index.to_list()
IDs_above_vandels = IDs[index_masked_mass_v].str.decode("utf-8").str.rstrip().values
IDs_below_vandels  = IDs[index_masked2_mass_v].str.decode("utf-8").str.rstrip().values
#print(IDs_above_vandels, IDs_below_vandels)

#print(index_list_ind)

d4000_vandels_above = []
mass_v_above = []
for id1 in IDs_above_vandels:
    d4000_vandels_above.append(index_list_ind.loc[id1, 'Dn4000'])
    mass_v_above.append(index_list_size_vandels.loc[str(id1), 'log10(M*/Msun)'])

d4000_vandels_below = []
mass_v_below = []
for id2 in IDs_below_vandels:
    d4000_vandels_below.append(index_list_ind.loc[str(id2), 'Dn4000'])
    mass_v_below.append(index_list_size_vandels.loc[str(id2), 'log10(M*/Msun)'])

vandels_std_ab = np.nanstd(d4000_vandels_above)
vandels_mean_ab = np.nanmean(d4000_vandels_above)
vandels_median_ab = np.nanmedian(d4000_vandels_above)

vandels_std_be = np.nanstd(d4000_vandels_below)
vandels_mean_be = np.nanmean(d4000_vandels_below)
vandels_median_be = np.nanmedian(d4000_vandels_below)

print('for 10.9 - 11.5 mass range', len(d4000_vandels_above), len(d4000_vandels_below), 'mass_above', np.nanmedian(mass_v_above), 'mass_below', np.nanmedian(mass_v_below))
print('VANDELS: \n ABOVE van der Wel z~0.75 relation: \n std = ', vandels_std_ab , '\n mean = ', vandels_mean_ab, '\n median = ', vandels_median_ab,
'\n BELOW offset van der Wel z~0.75 relation: \n std = ', vandels_std_be , '\n mean = ', vandels_mean_be, '\n median = ', vandels_median_be)


input()

print("\n Finding stats for D4000 above + below Wu z~ 0.75 relation for LEGA-C galaxies:")
mask_wu = (np.log10(Re_kpc)>wu_new_relation)#(np.log10(Re_kpc)>vdw_0_75)#
mask1_wu = (np.log10(Re_kpc)<wu_new_relation)#(np.log10(Re_kpc)<vdw_0_75)
index_masked_mass_wu = np.log10(Re_kpc)[mask_wu].index.to_list()
index_masked2_mass_wu = np.log10(Re_kpc)[mask1_wu].index.to_list()
IDs_above_wu = ID_wu[index_masked_mass_wu].values#.str.decode("utf-8").str.rstrip().values
IDs_below_wu  = ID_wu[index_masked2_mass_wu].values#.str.decode("utf-8").str.rstrip().values


#print(IDs_above_wu, IDs_below_wu)
d4000_i_above = []
mass_above = []
for id in IDs_above_wu:
    d4000_i_above.append(index_list_wu.loc[id, 'LICK_D4000_N'])
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


print('LEGA-C: \n ABOVE van Der Wel z~0.75 relation: \n std = ', legac_std_ab , '\n mean = ', legac_mean_ab, '\n median = ', legac_median_ab,
'\n BELOW van Der Wel z~0.75 relation: \n std = ', legac_std_be , '\n mean = ', legac_mean_be, '\n median = ', legac_median_be)



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
#732 objects classified as passive from the uvj_sfq =1 in table
