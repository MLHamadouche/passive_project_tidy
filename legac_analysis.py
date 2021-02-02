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


plt.rc('text', usetex=True)
lega_c = pd.read_table("/Users/massissiliahamadouche/Downloads/legac_team2018-02-16.cat", delimiter = ' ', skiprows = [i for i in range(1,129)])
#print(len(lega_c))

df = pd.DataFrame(lega_c)
#print(df.keys())
#index = df.set_index(df['id'])
#df =df.groupby((df['uvj_sfq']==1)).get_group(True)
#### reducing sample #####
df = df.groupby((df['use']==1)).get_group(True) #f_use = 1
df = df.groupby( (df['z_spec']>=0.6) &(df['z_spec']<=1.)).get_group(True) #0.6<z<1.0
df = df.groupby((df['f_sfr']==0) & (df['f_galfit']==0) & (df['f_phot']==0) & (df['f_ppxf']==0)&(df['f_spec']==0) &(df['f_z']==0)&(df['f_int']==0)).get_group(True)
#df = df.groupby((df['SN']>=20.)).get_group(True) #s/n >10
df = df.groupby((df['SN_rf_4000']>=10.)).get_group(True) #S/N SN_rf_4000 > 10
#df = df.groupby((df['eqw_obs_Hb']>=-1)).get_group(True)
df =df.groupby((df['uvj_sfq']==1)).get_group(True)
#df = df.groupby((np.log(df['lsfr_UV_IR'])<-10)).get_group(True) #S/N SN_rf_4000 > 10
#print(len(df))
df.set_index('id')
#############

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
"""
fig, ax1 = plt.subplots(figsize = [5,5])
ax1.scatter(ssfr, (np.log10(df['eqw_obs_Hb'])))
ax1.hlines(y = 0, xmin = -12, xmax = -8, ls = '-', lw = 2., color='k', alpha  = 0.5 )#1.3
ax1.vlines(x =-10,ymin = -3, ymax = 3, ls = '-', lw = 2., color='k', alpha  = 0.5 )
ax1.set_xlim(-12, -8)
ax1.set_ylim(-2,2)
plt.show()
"""
df2 = df2.rename(columns={'OBJECT':'id'})
left = df
right = df2

merged_df = left.merge(right, how = 'inner', left_on = 'id',right_on= 'id')
merged_df = merged_df.drop_duplicates('id')
merged_df = merged_df.dropna(subset=['id', 'LICK_D4000_N']) #remove rows with d4000 = NaN
merged_df= merged_df[merged_df['LICK_D4000_N'] != 0] #remove rows with d4000 = 0.
print(len(merged_df))
#print(merged_df['LICK_D4000_N'].values)

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
IDs = cat3['IDs']
df4 = pd.DataFrame(cat3)
df4 = df4.groupby((df4['log10(M*/Msun)']>10.3)&(df4['log10(M*/Msun)']<=11.5)).get_group(True)
stricter_masses = df4["log10(M*/Msun)"]
stricter_sizes= df4["Re_kpc"]
R_e_errs_4 = df4["Re_kpc_errs"]
redshifts_4 = df4['redshifts']
size_4 = np.array(np.log10(stricter_sizes))
index_4 = np.log10(stricter_sizes).index.to_list()
#x_array_4 = np.linspace(lower_lim, higher_lim, len(size_4))
#vdw_norm_model_4 = log_A + alpha*np.log10((10**x_array_4)/(5*10**10))
#mask_4 = (size_4>vdw_norm_model_4)
#mask1_4 = (size_4<vdw_norm_model_4)
#index_masked_mass_4 = np.log10(stricter_sizes)[mask_4].index.to_list()
#index_masked2_mass_4 = np.log10(stricter_sizes)[mask1_4].index.to_list()
Re_kpc = merged_df['galfit_re_kpc'].values
ssfr = merged_df['fast_lsfr'].values
mass = merged_df['fast_lmass'].values
d4000 = merged_df['LICK_D4000_N'].values


al, A = 0.71, 0.42

def vdw_relation(logA, alpha, x_values):
    logR_eff = logA + alpha*(np.log10((10**x_values)/(5*10**10)))
    return logR_eff

vdw_0_75 = vdw_relation(0.42, 0.71, x)
plt.rc('font', family='serif')
plt.rc('xtick', labelsize='small')
plt.rc('ytick', labelsize='small')

fig1, ax1 = plt.subplots(figsize=[16,10.5])
im1 = ax1.scatter(mass, np.log10(Re_kpc), s=300, c=d4000, cmap=plt.cm.magma, marker='o', linewidth=0.5, )
cbar1 = fig1.colorbar(im1, ax=ax1)
cbar1.set_label('D4000', size=18)
#ax1.plot(x, wu_relation, 'coral', lw = 1.5, label= 'Wu et al. 2018 $z \sim$ 0.7')
#ax1.plot(x, wu_relation2, 'r', lw = 2.5, label= 'Wu et al. 2018 $z \sim$ 0.7 (2)')
ax1.scatter(np.array(stricter_masses), size_4, s = 200, c = 'r', marker = '^', ec = 'k', label  = 'VANDELS PASSIVE GALAXIES')
ax1.plot(x, vdw_0_75, 'k', lw = 2., label=f'van der Wel et al. 2014 $z$ = 0.75 ETG \n relation')
#ax1.plot(x, log_Reff, 'grey' ,lw = 1.5,  label= 'van der Wel et al. 2014 ($z$ = 1.25)')

ax1.set_xlabel('$\mathrm{log_{10}{(M*/M_{\odot})}}$', size = 18)
ax1.set_ylabel('$\mathrm{log_{10}{(R_{e}/kpc)}}$', size = 18)
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.legend(prop={'size': 12}, loc='lower right')
plt.title('LEGA-C log$_{10}$(Re) v log$_{10}$(M*/M$_{\odot}$)', size = 20)
plt.xlim(9.8, 12.0)
plt.ylim(-0.8, 1.5)

plt.savefig('vdw_0_75_lega_c_wu_vandels_d4000.pdf')
plt.close()

#732 objects classified as passive from the uvj_sfq =1 in table
