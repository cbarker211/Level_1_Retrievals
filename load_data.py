import xarray as xr
import numpy as np
import cdsapi
import os
from haversine import haversine
import pandas as pd
import sys
from datetime import timedelta
from collections import defaultdict


def load_data(site_name, launch_ind, platform):

    data_tar, data_ref = load_satellite_data(site_name, launch_ind, platform)
    time_tar = pd.Timestamp(data_tar["time"].values).round('h')
    
    era5_dir = "/home/ucfacb0/ARIA/aria_data_analysis/data/era5"
    filename = os.path.join(era5_dir, f"era5_{site_name}_{time_tar.strftime('%Y%m%d_%H%M')}.nc")

    times_tar = [time_tar-timedelta(hours=2), time_tar-timedelta(hours=1), time_tar, time_tar+timedelta(hours=1), time_tar+timedelta(hours=2)]
    days = {(t.year, t.month, t.day) for t in times_tar}
    if len(days) > 1:
        print(f"WARNING: time window straddles multiple days: {sorted(days)}")

    era5_data = load_era5_data(filename, data_tar, times_tar)

    xsc_mean = load_xsc_data()

    return era5_data, xsc_mean, data_tar, data_ref

def load_satellite_data(site_name, launch_ind, platform):

    #####################################################################
    # First load the satellite data for the reference and target scenes.
    #####################################################################

    # Get data from target file and find launch year.
    data_dir = "/home/ucfacb0/ARIA/aria_data_download/download"
    if platform == "sn":
        platformstr = "sentinel-2-l1c-band-l2a-slc"
        res = 60
    elif platform == "ls":
        platformstr = "landsat-8-9-c2-t1-toa"
        res = 30
        
    file_tar = (f"{data_dir}/{site_name}/{platformstr}-{site_name}-{res}-tar.nc")
    data_tar = xr.open_dataset(file_tar,decode_timedelta=False)
    target_times = data_tar["time"].values + data_tar["delta_t_secs"].values.astype(int).astype('timedelta64[s]')
    target_times = target_times.astype('datetime64[h]')
    

    data_tar = data_tar.isel(time=launch_ind)
    delta_t_secs = data_tar["delta_t_secs"].values
    print(f"Overpass is {delta_t_secs/60:.0f} mins after launch.")
    launch_time = data_tar["time"].values
    overpass_time = launch_time + np.timedelta64(int(delta_t_secs), 's')
    overpass_time = overpass_time.astype('datetime64[h]')

    # Open reference file and remove launch times.
    file_ref = (f"{data_dir}/{site_name}/{platformstr}-{site_name}-{res}-ref-{str(launch_time)[:4]}.nc")
    data_ref = xr.open_dataset(file_ref,decode_timedelta=False)
    ref_times = data_ref["time"].values.astype('datetime64[h]')
    
    ref_launch_ind = np.where(np.isin(ref_times, target_times))[0]
    data_ref = data_ref.drop_isel(time=ref_launch_ind)
    print(target_times, data_ref["time"].values)

    # Calculate relative azimuth angle
    # https://stcorp.github.io/harp/doc/html/algorithms/derivations/relative_azimuth_angle.html
    delta_phi = (data_tar["saa"] - data_tar["vaa"]) % 360
    data_tar["raa"] = np.abs(delta_phi - 180)
    
    return data_tar, data_ref

def load_era5_data(filename, data_tar, times_tar):

    ##############################################
    # Load the target conditions to index AMF. 
    ##############################################

    center_lat = np.mean(data_tar["lat_range"].values)
    center_lon = np.mean(data_tar["lon_range"].values)
    delta_t_secs = data_tar["delta_t_secs"].values
    valid_time = 2
    if not os.path.exists(filename):
        extra_dict = {"variable": ["surface_pressure","total_column_ozone",
                                    "10m_u_component_of_wind","10m_v_component_of_wind",
                                    "100m_u_component_of_wind","100m_v_component_of_wind"]}
        era5_halfres = 0.125
        request = {
            "product_type": ["reanalysis"],
            "year" : [times_tar[valid_time].year],
            "month": [times_tar[valid_time].month],
            "day"  : [times_tar[valid_time].day],
            "time" : sorted(t.strftime('%H:%M') for t in times_tar),
            "data_format": "netcdf",
            "download_format": "unarchived",
            "area": [center_lat+era5_halfres,
                     center_lon-era5_halfres,
                     center_lat-era5_halfres,
                     center_lon+era5_halfres]
        }
        if extra_dict is not None:
            request.update(extra_dict)
        client = cdsapi.Client()
        result = client.retrieve("reanalysis-era5-single-levels", request)
        result.download(filename)

    ds_era5_target = xr.open_dataset(filename).isel()
    
    sp = ds_era5_target["sp"].isel(        valid_time=valid_time,latitude=0,longitude=0).values / 100 # hPa
    ozone_DU = ds_era5_target["tco3"].isel(valid_time=valid_time,latitude=0,longitude=0).values / 2.1415e-5  # kg/m² → DU
    u10 = ds_era5_target["u10"].isel(      valid_time=valid_time,latitude=0,longitude=0).values
    v10 = ds_era5_target["v10"].isel(      valid_time=valid_time,latitude=0,longitude=0).values
    u100 = ds_era5_target["u100"].isel(    valid_time=valid_time,latitude=0,longitude=0).values
    v100 = ds_era5_target["v100"].isel(    valid_time=valid_time,latitude=0,longitude=0).values
    ds_era5_target.close()

    # Then we get the plume altitude at 500m above the surface.
    h = 145366.45 * (1 - (sp / 1013.25) ** 0.190284) * 0.3048 / 1000 # km
    h_plume_km = h + 0.5

    # https://www.nasa.gov/wp-content/uploads/2021/07/falcon9_crewdragon_launch_weather_criteria_fact_sheet.pdf
    # https://www.nasa.gov/wp-content/uploads/2018/04/633165main_atlas-5-weather.pdf
    # These mention launches should only occur if the 50m wind is below 13ms-1, or launch pad wind is below 17 ms-1.
    # If an overpass is maximum 1 hour after launch with maximum wind (17 ms-1), then it could travel by up to 60 km.
    # Realistically, most launches will be at a much lower windspeed, and the overpass will be closer than 1 hour.
    wind_speed_100 = np.sqrt(u100**2 + v100**2)
    wind_direction_100 = np.degrees(np.arctan2(u100, v100)) % 360
    print(f"ERA5 100m wind speed at target site: {wind_speed_100:.1f} m/s blowing towards {wind_direction_100:.0f} degrees from north.")
    estimated_distance_100 = wind_speed_100 * delta_t_secs / 1000
    print(f"Plume may have travelled by {estimated_distance_100:.1f} km.")
    
    wind_speed_10 = np.sqrt(u10**2 + v10**2)
    wind_direction_10 = np.degrees(np.arctan2(u10, v10)) % 360
    print(f"ERA5 10m wind speed at target site: {wind_speed_10:.1f} m/s blowing towards {wind_direction_10:.0f} degrees from north.")
    estimated_distance_10 = wind_speed_10 * delta_t_secs / 1000
    print(f"Plume may have travelled by {estimated_distance_10:.1f} km.")

    # Launch site is at center, so we need the estimated distance to be smaller than half the box.
    lat_km = haversine((center_lat, center_lon), (data_tar["lat_range"].values[1], center_lon))
    lon_km = haversine((center_lat, center_lon), (center_lat, data_tar["lon_range"].values[1]))

    print(f"Grid box is {(lat_km*2):.2f} × {lon_km*2:.2f}")
    if estimated_distance_100 > min([lat_km, lon_km]):
        print(f"Warning: plume may have travelled outside of the satellite scene (estimated distance: {estimated_distance_100:.1f} km, scene size: {lat_km:.1f} km x {lon_km:.1f} km).")

    era5_data = {"h_plume_km": h_plume_km,
                 "sp":         sp,
                 "ozone_DU":   ozone_DU,
                 "u10":        u10,
                 "v10":        v10,
                 "u100":       u100,
                 "v100":       v100}
    
    return era5_data
        
def load_xsc_data():

    ##########################################
    # Determine the absorption cross section.
    ##########################################
    # TODO: Just using 294K data for now, will need to improve this.
    # bandwidths from https://sentiwiki.copernicus.eu/web/s2-mission
    # xsc data from https://hitran.org/

    xsc_file = "/home/ucfacb0/ARIA/aria_data_analysis/data/NO2_294.0_0.0_15002.0-42002.3_00.xsc.txt"
    xsc = np.loadtxt(xsc_file,max_rows=1,usecols=(1,2,3),dtype=float)
    wavenumbers = np.linspace(xsc[0],xsc[1],int(xsc[2]))

    # Using B01
    wn1 = 1e7 / 433
    wn2 = 1e7 / 453

    idx1 = np.searchsorted(wavenumbers, wn1)
    idx2 = np.searchsorted(wavenumbers, wn2)

    with open(xsc_file) as f:
        f.readline()  # skip header
        xsc_data = np.fromstring(f.read(), sep=' ')

    xsc_mean = np.mean(xsc_data[idx2:idx1]) * 1e-4 * 6.022e23 # m2/mol

    return xsc_mean