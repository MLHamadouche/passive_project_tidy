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
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import photometry_stacking as ps
import legac_stacking as lg


all_passive = Table.read("/Users/Important_Tables_PhD_Project/NEW_VANDELS_passive_UVJ_inc_agn_inds.fits").to_pandas()

df = pd.DataFrame(all_passive)

df = df.groupby((df['stellar_mass_50']<=11.5)&(df['stellar_mass_50']>11.1)).get_group(True)

redshifts = df['zspec'].values
ids = [s.rstrip().decode('utf-8') for s in df['new_id']]
print(len(df))
#print(Re_kpc_vandels)
mass = df['stellar_mass_50']
d4000 = df['Dn4000']
N = len(df)
vandels_d4000_err = df['D4000_errs']
ew_hd = df['EW(Hd)']
vandels_ew_hderrs = df['EW(Hd)_errs']
age = df['exponential_age_50']

print(N, np.median(mass), np.nanmedian(d4000),np.nanstd(d4000) , np.median(age),)

import spectral_indices as si
legac = pd.read_table('merged_legac_cat.cat' ,delimiter = ',')

legac_df = pd.DataFrame(legac)
legac_df = legac_df.groupby((legac_df['fast_lmass']<=11.1)&(legac_df['fast_lmass']>10.7)).get_group(True)
#legac_df = legac_df.groupby((legac_df['z_spec']<=0.8)&(legac_df['z_spec']>0.6)).get_group(True)
#print(age)
print(legac_df)
id_legac = legac_df['id']
N_l = len(legac_df)
age_legac = (10**np.array(legac_df['fast_lage']))/10**9
legac_d4000 = legac_df['LICK_D4000_N']
legac_hd = legac_df['LICK_HD_F']
legac_mass = legac_df['fast_lmass']
legac_z = legac_df['z_spec']
legac_d4000_errs = legac_df['LICK_D4000_N_err']

print(len(id_legac), len(ids))
new_wavs_vandels = np.arange(2400, 4200, 2.5)
new_wavs_legac = np.arange(3300, 5400, 2.0)
legac_stacks, legac_stack_errs = lg.legac_stacks(id_legac)
spec_v, phot_v, med_new, med_norm  = ps.stacks_phot(ids)

vandels_stack_fluxes = spec_v[0]

D4000_index = si.Dn4000(vandels_stack_fluxes)
#Mg_UV_index.append(Mg_UV(new_spec))
H_delta_EW= si.H_delta(vandels_stack_fluxes, new_wavs_vandels)

print('d4000vandelsstacks',D4000_index)
print('hdelta vandels stacks', H_delta_EW)
input()


#print(age_legac)
print(N_l, np.median(legac_mass), np.nanmedian(legac_d4000),np.nanstd(legac_d4000), np.median(age_legac), np.median(legac_z))
input()

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('xtick', labelsize='medium')
plt.rc('ytick', labelsize='medium')

fig = plt.figure(figsize = [12,5])
gs = gridspec.GridSpec(1, 2, wspace=0.1)
cm = plt.cm.get_cmap('RdPu_r')
ax0 = plt.subplot(gs[0, 0],)
im2 = plt.scatter(d4000, ew_hd, s=60, c = age, cmap = cm, marker = 'o',edgecolors='k',  label = 'VANDELS ($z \sim $ 1.15)')#edgecolors='k',
ax0.set_xlim([1.1,2.0])

ax1 = plt.subplot(gs[0, 1])
im1 = plt.scatter(legac_d4000, legac_hd, s=60, c=age_legac,cmap = cm, edgecolors='k', marker = 'o', label = 'LEGA-C ($z \sim $ 0.7)')
ax1.set_xlim([1.2,2.25])
#fig.tight_layout()
cbaxes1 = ax1.inset_axes([0.9,0.65, 0.04, 0.3])# width="3%", height="30%", loc='upper right',

cbaxes = ax0.inset_axes([0.9, 0.65, 0.04, 0.3])#width="3%", height="30%", loc='upper right', borderpad = 0.5

cb1 = plt.colorbar(im1, cbaxes1, orientation='vertical')
cb1.set_label(r'Age (Gyr)', labelpad=-40)
cb = plt.colorbar(im2, cbaxes, orientation='vertical')
cb.set_label(r'Age (Gyr)', labelpad=-40 )
ax1.set_xticks([1.4, 1.6, 1.8,2.0,2.2])
ax0.set_xticks([1.2, 1.4,1.6,1.8])

ax1.set_xticklabels(["1.4", "1.6", "1.8", "2.0", "2.2"])
ax0.set_xticklabels(["1.2","1.4", "1.6", "1.8"])
ax0.set_ylabel('EW$_{H \delta}$ (\\AA)', labelpad=0.0, fontsize = 15)

ax0.set_xlabel('D$_n$4000', fontsize = 15)
ax0.xaxis.set_label_coords(1.05, -0.08)
leg3 = ax0.legend(loc='lower right', frameon=False, handlelength=0,handletextpad=0)
leg4 = ax1.legend(loc='lower right',frameon=False, handlelength=0,handletextpad=0)
for item in leg3.legendHandles:
    item.set_visible(False)
for item in leg4.legendHandles:
    item.set_visible(False)
plt.savefig('legac0_6_z_0_8_vandels_hd_d4000.pdf')
plt.close()

plt.rcParams['lines.dash_capstyle'] = 'butt'

fig = plt.figure(figsize = [12,5])
gs = gridspec.GridSpec(1, 2, wspace=0.1)
cm = plt.cm.get_cmap('RdPu_r')
ax0 = plt.subplot(gs[0, 0],)
error_kw = {'capsize': 5, 'capthick': 1, 'ecolor': 'black'}
ax0.errorbar(mass, d4000, yerr = vandels_d4000_err, capthick = 1, mfc = None,  marker=None, mew=0, zorder = 0, lw = 1., capsize = 5, ecolor = 'k', ls = '')
im2 = plt.scatter(mass, d4000, s=80, c = redshifts, cmap = cm, marker = 'o',edgecolors='k',label = 'VANDELS ($z \sim $ 1.15)', zorder = 100)#edgecolors='k''darkslategrey',
#ax0.set_xlim([1.1,2.0])
#ax0.axhline(y=1.35, color = 'k', ls = '--', lw = 0.5)

ax1 = plt.subplot(gs[0, 1])
ax1.errorbar(legac_mass, legac_d4000, yerr = legac_d4000_errs, capthick =1, mfc = None, ls = '', marker=None, mew=0, zorder = 0, lw =1., capsize = 5, ecolor = 'k')
im1 = plt.scatter(legac_mass, legac_d4000, s=80, c=legac_z,cmap = cm,edgecolors='k', marker = 'o',label = 'LEGA-C ($z \sim $ 0.7)', zorder = 100)#edgecolors='k',
ax0.set_ylim([1.0,2.0])
#ax1.set_ylim([1.0,2.0])
ax1.set_xlim([10.25,11.85])
ax0.set_xlim([10.25, 11.65])
#fig.tight_layout()
cbaxes1 = ax1.inset_axes([0.89,0.65, 0.04, 0.3])# width="3%", height="30%", loc='upper right',

cbaxes = ax0.inset_axes([0.89, 0.65, 0.04, 0.3])#width="3%", height="30%", loc='upper right', borderpad = 0.5

cb1 = plt.colorbar(im1, cbaxes1,  ticks=[0.61, 0.7, 0.799], orientation='vertical')

cb1.set_label(r'Redshift, $z$', labelpad= -50)
cb = plt.colorbar(im2, cbaxes,ticks = [0.97, 1.1, 1.296],  orientation='vertical')
cbaxes.tick_params(direction = 'in')
cbaxes1.tick_params(direction = 'in')
cb.set_label(r'Redshift, $z$', labelpad= -50 )
cb.set_ticklabels(['1.0', '1.1', '1.3'])
cb1.set_ticklabels(['0.6', '0.7', '0.8'])
#ax1.axhline(y=1.35, color = 'k', ls = '--', lw = 0.5)
#ax1.set_yticks([1.0,1.4, 1.6, 1.8,2.0,2.2])
#ax0.set_xticks([1.2, 1.4,1.6,1.8])
#ax1.set_xticklabels(["1.4", "1.6", "1.8", "2.0", "2.2"])
#ax0.set_xticklabels(["1.2","1.4", "1.6", "1.8"])
ax0.set_ylabel('D$_n$4000', labelpad=0.0, fontsize = 15)
ax0.set_xlabel('$\mathrm{log_{10}{(M*/M_{\odot})}}$', fontsize = 15)
ax0.xaxis.set_label_coords(1.05, -0.08)
leg1 =ax0.legend(loc='lower right', frameon=False, handlelength=0,handletextpad=0)
leg2 = ax1.legend(loc='lower right',frameon=False, handlelength=0,handletextpad=0)
for item in leg1.legendHandles:
    item.set_visible(False)
for item in leg2.legendHandles:
    item.set_visible(False)
plt.savefig('legac0_6_z_0_8_vandels_mass_d4000_z.pdf')
plt.close()
