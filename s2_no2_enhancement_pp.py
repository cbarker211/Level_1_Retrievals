#!/usr/bin/python

# Import relevant packages:
import numpy as np
from netCDF4 import Dataset
from scipy import stats
import matplotlib.pyplot as plt
# from mpl_toolkits.basemap import Basemap
import argparse
from sklearn.linear_model import LinearRegression
import sys
import os
import datetime as dt
from scipy import stats

import cartopy
import cartopy.crs as ccrs
from cartopy.io.shapereader import Reader
from cartopy.feature import ShapelyFeature
import matplotlib.pyplot as plt
import matplotlib
from sklearn.metrics import mean_squared_error

from utils import plot_image

# Define data directory:
root_dir = '/home/ucfaea1/python/aria/check_coincidence-main/download/'
#root_dir = '/home/ucfaea1/Uptrop/from_folders/from_Connor/Sentinel2/update_120326/download/'

lp_name = 'PP_Riyadh08' ##'PP_Matimba'

file1 = (root_dir+'sentinel2-l1c-band-l2a-slc-'+lp_name+'-60-target.nc')

fh1 = Dataset(file1, mode='r')
#print(fh1)

time_info = fh1.variables["time"]
time_vals = fh1.variables["time"][:]
#print(time_info)

# Get vza and sza to calculate geometric AMF:
sza_info = fh1.variables["sza"]
#print(sza_info)
sza_vals = fh1.variables["sza"][:]
vza_info = fh1.variables["vza"]
#print(vza_info)
vza_vals = fh1.variables["vza"][:]
# Calculate geometric AMF:
amf_geo = ( (1.0/np.cos(np.radians(sza_vals))) +
            (1.0/np.cos(np.radians(vza_vals))) )

# TODO: need to calculate relative azimuth angle from SZA and VZA, as
# needed for lookup tables of AMF scattering weights
# RAA calculation: https://stcorp.github.io/harp/doc/html/algorithms/derivations/relative_azimuth_angle.html

# Get radiances for B01 band:
# TODO: set this to read B01 if 60m or B02 if 10m
b01_info = fh1.variables["B01"]
b01_vals = fh1.variables["B01"][:]

# Get radiances for B04 band (output with 10m and 60m S2 data):
b04_info = fh1.variables["B04"]
b04_vals = fh1.variables["B04"][:]

# Convert vals to VCDs:
#ref_10m_vals = ref_10m_vals / ref_amfg

lon_range = fh1.variables["lon_range"][:]
lat_range = fh1.variables["lat_range"][:]

#print(fh1)
fh1.close()

#print(lon_range)
#print(lat_range)

#sys.exit()

# Loop over references to find corresponding target:
for i in range(len(time_vals)-1):

    # Hard coded to read radiances for Riyadh 08 power plant
    # 4 July 2020 (same as Varon et al. 2024, Figure 1(a))
    if i!=21: continue

    #print('Target SZA, VZA: ', np.min(target_sza_vals[i,0,:,:]), np.min(target_vza_vals[i,0,:,:]))

    # Quick and dirty code to explore potential options to
    # select reference scenes. Differs from Varon approach.
    # Loops through all cloud-free scenes for 2020 and selects
    # the scenes most consistent with the target overpass.
    # Varon et al. (2024) ref scenes are for: 20220-04-10, 2020-04-25,
    #                                         2020-06-04, 2020-08-03
    # The selection approach used below identified these as ref scenes:
    #                              2020-05-30, 2020-06-14, 2020-08-08
    ind_ref = []
    for k in range(len(time_vals)):
        r = stats.pearsonr(b01_vals[i,0,:,:].flatten(), b01_vals[k,0,:,:].flatten())
        mse_b01 = mean_squared_error(b01_vals[i,0,:,:].flatten(), b01_vals[k,0,:,:].flatten())
        rmse_b01 = mse_b01**(1/2)
        mse_b04 = mean_squared_error(b04_vals[i,0,:,:].flatten(), b04_vals[k,0,:,:].flatten())
        rmse_b04 = mse_b04**(1/2)
        # VZA and SZA all the same (far as I can tell), so just selecting
        # minimum. Could instead select first value in array.
        vza_diff = np.abs(np.min(vza_vals[i,0,:,:])-np.min(vza_vals[k,0,:,:]))
        sza_diff = np.abs(np.min(sza_vals[i,0,:,:])-np.min(sza_vals[k,0,:,:]))

        # Thresholds for conditions most consistent with target scene.
        # These are selected by trial-and-error, as compromise between
        # consistency and achieving coincidence.
        #print(rvals[k],rmse_b01[k],vza_diff[k],sza_diff[k],rmse_b04[k])
        if r[0]>0.88 and rmse_b01<100 and vza_diff<0.2 and sza_diff<3 and rmse_b04<200:
            # Skip target index:
            if k==i: continue
            ind_ref.append(k)
            
    # Quick check on reference locations:
    print('Ref indices: ', ind_ref)

    # Calculate reference values simply as average of all selected.
    # Should be linear combination of these, with weights applied, as
    # in Varon et al. (2024) and Ehret et al. (2022).
    # TODO: change to linear combination.
    ref_vals = np.mean(b01_vals[ind_ref,0,:,:], axis=0)

    # Get ratio of the target scene to the reference values
    # This is Equation (1) of Varon et al. (2024) without the scaling factor
    # Or, assuming scaling factor (c) is 1.
    # TODO: Need to derive scaling factor, c.
    ratio = b01_vals[i,0,:,:] / ref_vals

    # Calculate NO2 enhancement, with some major simplifying assumptions
    # for now. Specifically, scattering weight of 1 in the AMF calculation and
    # using crude estimate of NO2 absorption cross section.
    
    # Using rough estimate of 4.0e-19 cm2/molecule for NO2
    # absorption cross-section at 294 K read from https://hitran.org/.
    # Value is ~3.5e-19 cm2/molecule at 220 K.
    # TODO: download data from HITRAN to calculate bandwidth average
    #       absorption cross section. Varon use 220 K, but 294 K
    #       far more suitable for power plant scenes, especially in
    #       Saudi Arabia.
    abs_cross_section = 24.09   # in units of m2/mol.

    # Calculate slant column density (Eq. (2) of Varon et al., 2024):
    no2_scd = (-1) * np.log(ratio) / abs_cross_section

    # Calculate VCD using only geometric AMF for now:
    no2_vcd = no2_scd / amf_geo[i,0,:,:]    

    # Test plot showing the range of the values in the plume are broadly
    # consistent with Varon et al. (2024) Fig. 1 (a):
    plot_image(no2_vcd*1e3, clip_range=(0.0,2.5))
    plt.show()
