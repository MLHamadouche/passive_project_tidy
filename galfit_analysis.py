from astropy.table import Table
from astropy.io import fits
import pandas as pd
from astropy.nddata import Cutout2D
from astropy.utils.data import get_pkg_data_filename
from astropy.io import fits
import os
import astropy.units as u
import re
from collections import OrderedDict
from glob import glob
import time
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
import numpy as np

#cdfs_hst = pd.read_table('24cdfs_hst_pixelpos.cat', delimiter = ' ')
cdfs_ground = pd.read_table('41cdfsground_pixelpos.cat', delimiter = ' ')

df = pd.DataFrame(cdfs_ground)
print(df)

ids = df['#id']
#ids = df['ID_new']


input()

def get_galfit_info(id):

    #if id == "CDFS-GROUND140672":
    #    hdulist[2].header['1_N'].split('+/-',1)[0] = np.nan()
    #    hdulist[2].header['1_N'].split('+/-',1)[1] = np.nan()

    path = '/Users/massissiliahamadouche/Downloads/galfit-example2/EXAMPLE/model_cdfs_ground_results/imgblock_'+str(id)+'.fits'
    hdulist = fits.open(path)
    arcsec_per_pix = 0.0599999
    galfit_mag = float(hdulist[2].header['1_MAG'].split('+/-',1)[0])
    galfit_mag_err = float(hdulist[2].header['1_MAG'].split('+/-',1)[1])
    galfit_axis_ratio = float(hdulist[2].header['1_AR'].split('+/-',1)[0])
    galfit_ax_ratio_err = float(hdulist[2].header['1_AR'].split('+/-',1)[1])
    galfit_sersic_ind = float(hdulist[2].header['1_N'].split('+/-',1)[0])
    galfit_sersic_ind_err = float(hdulist[2].header['1_N'].split('+/-',1)[1])
    galfit_sky = float(hdulist[2].header['2_SKY'].split('+/-',1)[0])
    sky_err = float(hdulist[2].header['2_SKY'].split('+/-',1)[1])
    pos_ang = float(hdulist[2].header['1_PA'].split('+/-',1)[0])
    pos_ang_err = float(hdulist[2].header['1_PA'].split('+/-',1)[1])
    galfit_re = hdulist[2].header['1_RE'].split('+/-',1)
    galfit_re_arc = float(galfit_re[0])*arcsec_per_pix
    galfit_re_err = float(galfit_re[1])*arcsec_per_pix

    return galfit_mag,galfit_mag_err, galfit_axis_ratio, galfit_ax_ratio_err, galfit_sersic_ind, galfit_sersic_ind_err, galfit_sky, sky_err, pos_ang, pos_ang_err, galfit_re_arc, galfit_re_err

#print(galfit_re_arc, galfit_re_err)

mag = []
mag_err = []
ax_ratio = []
ax_ratio_err = []
sersic = []
sersic_err = []
sky = []
sky_err = []
pos_ang = []
ang_err = []
re_arcsecs = []
re_err_arcsecs  =[]
new_id = []
for i in ids:
    mag_g,mag_g_err, ax_r, ax_err, n, n_err, sky_g, sky_g_err, angle, angle_err, size, size_err = get_galfit_info(i)
    new_id.append(i)
    mag.append(mag_g)
    mag_err.append(mag_g_err)
    ax_ratio.append(ax_r)
    ax_ratio_err.append(ax_err)
    sersic.append(n)
    sersic_err.append(n_err)
    sky.append(sky_g)
    sky_err.append(sky_g_err)
    pos_ang.append(angle)
    ang_err.append(angle_err)
    re_arcsecs.append(size)
    re_err_arcsecs.append(size_err)


col1 = fits.Column(name = 'ID', format='30A', array=new_id)
col2 = fits.Column(name = 'RA', format = 'E', array = df['ra'].values)
col3 = fits.Column(name = 'DEC', format = 'E', array = df['dec'].values)
col4 = fits.Column(name='mag', format='E', array=mag)
col15 = fits.Column(name='mag_err', format='E', array=mag_err)
col5 = fits.Column(name='q', format='E', array=ax_ratio)
col6 = fits.Column(name='q_err', format='E', array=ax_ratio_err)
col7 = fits.Column(name ='n', format = 'E', array = sersic)
col8 = fits.Column(name = 'n_err', format = 'E', array = sersic_err)
col9 = fits.Column(name = 'sky', format='E', array = sky)
col10 = fits.Column(name = 'sky_err', format = 'E', array = sky_err)
col11 = fits.Column(name = 'pos_ang', format = 'E', array = pos_ang)
col12 = fits.Column(name = 'pos_ang_err', format = 'E', array = ang_err)
col13 = fits.Column(name = 're_arcsecs', format = 'E', array = re_arcsecs)
col14 = fits.Column(name = 're_err_arcsecs', format = 'E', array = re_err_arcsecs)
hdu = fits.BinTableHDU.from_columns([col1, col2, col3, col4,col15,col5, col6, col7,col8,col9,col10,col11,col12,col13,col14])

file =  "galfit_model_size_info_41ground.fits"
hdu.writeto(file, overwrite = True)
