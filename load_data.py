import xarray as xr
import numpy as np
import cdsapi
import os
from haversine import haversine
import pandas as pd
import sys
from datetime import timedelta

def load_data(site_name, launch_ind, platform, center_lat, center_lon):

    data_tar, data_ref, time_tar, time_ref = load_satellite_data(site_name, launch_ind, platform)
    if data_tar is None or data_ref is None:
        raise ValueError(f"Unsupported platform: {platform}")
    
    time_era5 = time_tar.round('h')
    era5_dir = "/home/ucfacb0/ARIA/Level_1_Retrievals/data/era5"
    filename = os.path.join(era5_dir, f"era5_{site_name}_{time_era5.strftime('%Y%m%d_%H%M')}.nc")
    era5_data = load_era5_data(filename, data_tar, time_era5, center_lat, center_lon, platform)

    xsc_mean = load_xsc_data()

    return era5_data, xsc_mean, data_tar, data_ref, time_tar, time_ref

def load_satellite_data(site_name, launch_ind, platform):

    #####################################################################
    # First load the satellite data for the reference and target scenes.
    #####################################################################

    # Get data from target file and find launch year.
    
    if platform == "sn":
        data_dir = "/home/ucfacb0/ARIA/aria_data_download/download"
        filestr = f"sentinel-2-l1c-band-l2a-slc-{site_name}-60"
        file_tar = (f"{data_dir}/{site_name}/{filestr}-tar.nc")
    elif platform == "ls":
        data_dir = "/home/ucfacb0/ARIA/aria_data_download/download"
        filestr = f"landsat-8-9-c2-t1-toa-{site_name}-30"
        file_tar = (f"{data_dir}/{site_name}/{filestr}-tar.nc")
    elif platform == "sd":
        data_dir = "/home/ucfacb0/uptrop/Data/Satellites/Planet/SuperDove"
        filestr = f"superdove-8-9-c2-t1-toa-{site_name}"
        file_tar = (f"{data_dir}/{site_name}/{filestr}-tar-{launch_ind}.nc")
    else:
        raise ValueError(f"Unsupported platform: {platform}")
    
    data_tar = xr.open_dataset(file_tar,decode_timedelta=False)
    
    if platform in ["ls","sn"]:
        
        # Open target file and select launch.
        target_times = data_tar["time"].values + data_tar["delta_t_secs"].values.astype(int).astype('timedelta64[s]')
        target_times = target_times.astype('datetime64[h]')
        data_tar = data_tar.isel(time=launch_ind)
        
        # Open reference dataset. 
        launch_time = data_tar["time"].values
        file_ref = (f"{data_dir}/{site_name}/{filestr}-ref-{str(launch_time)[:4]}.nc")
        data_ref = xr.open_dataset(file_ref,decode_timedelta=False)
        ref_times = data_ref["time"].values.astype('datetime64[h]')
    
        ref_launch_ind = np.where(np.isin(ref_times, target_times))[0]
        data_ref = data_ref.drop_isel(time=ref_launch_ind)
        
    else:
        
        # Open reference dataset. 
        file_ref = (f"{data_dir}/{site_name}/{filestr}-ref-{launch_ind}.nc")
        data_ref = xr.open_dataset(file_ref,decode_timedelta=False)
        data_tar = data_tar.isel(time=0)
    
    time_tar = pd.Timestamp(data_tar["time"].values.item()) # Overpass acquisiton time
    time_ref = data_ref["time"].values
    delta_t_secs = data_tar["delta_t_secs"].item()
    print(f"Overpass is {delta_t_secs/60:.0f} mins after launch.")
    # For Landsat and Sentinel, "time" is the launch, so add delta_t_secs
    if platform != "sd":
        time_tar    += pd.Timedelta(seconds=float(delta_t_secs))

    # Calculate relative azimuth angle
    # https://stcorp.github.io/harp/doc/html/algorithms/derivations/relative_azimuth_angle.html
    delta_phi = (data_tar["saa"] - data_tar["vaa"]) % 360
    data_tar["raa"] = np.abs(delta_phi - 180)
    
    return data_tar, data_ref, time_tar, time_ref

def load_era5_data(filename, data_tar, time_era5, center_lat, center_lon, platform):

    ##############################################
    # Load the target conditions to index AMF. 
    ##############################################

    delta_t_secs = data_tar["delta_t_secs"].values
    if not os.path.exists(filename):
        extra_dict = {"variable": ["surface_pressure","total_column_ozone",
                                    "10m_u_component_of_wind","10m_v_component_of_wind",
                                    "100m_u_component_of_wind","100m_v_component_of_wind"]}
        era5_halfres = 0.125
        request = {
            "product_type": ["reanalysis"],
            "year" : [time_era5.year],
            "month": [time_era5.month],
            "day"  : [time_era5.day],
            "time" : [time_era5.strftime('%H:%M')],
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
        result.download(filename) # type: ignore

    ds_era5_target = xr.open_dataset(filename).sel(valid_time=np.datetime64(time_era5),latitude=center_lat, longitude=center_lon, method="nearest")
    sp       = ds_era5_target["sp"].values / 100 # hPa
    ozone_DU = ds_era5_target["tco3"].values / 2.1415e-5  # kg/m² → DU
    u10      = ds_era5_target["u10"].values
    v10      = ds_era5_target["v10"].values
    u100     = ds_era5_target["u100"].values
    v100     = ds_era5_target["v100"].values
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
    if platform == "sd":
        lat_km = (data_tar["y"][0].item()-data_tar["y"][-1].item())/2000
        lon_km = (data_tar["x"][0].item()-data_tar["x"][-1].item())/2000
    else:
        lat_km = haversine((center_lat, center_lon), (center_lat+0.06, center_lon))
        lon_km = haversine((center_lat, center_lon), (center_lat, center_lon+0.08))

    print(f"Grid box is {(lat_km*2):.2f} × {lon_km*2:.2f}")
    if estimated_distance_100 > min([lat_km, lon_km]):
        print(f"Warning: plume may have travelled outside of the satellite scene (estimated distance: {estimated_distance_100:.1f} km, scene size (half): {lat_km:.1f} km x {lon_km:.1f} km).")

    era5_data = {"h_plume_km": h_plume_km,
                 "sp":         sp,
                 "ozone_DU":   ozone_DU,
                 "u10":        u10,
                 "v10":        v10,
                 "u100":       u100,
                 "v100":       v100,
                 "ws_10":      wind_speed_10,
                 "wd_10":      wind_direction_10,
                 "ws_100":     wind_speed_100,
                 "wd_100":     wind_direction_100,}
    
    return era5_data
        
def load_xsc_data():

    ##########################################
    # Determine the absorption cross section.
    ##########################################
    # TODO: Just using 294K data for now, will need to improve this.
    # bandwidths from https://sentiwiki.copernicus.eu/web/s2-mission
    # xsc data from https://hitran.org/

    xsc_file = "/home/ucfacb0/ARIA/Level_1_Retrievals/data/NO2_294.0_0.0_15002.0-42002.3_00.xsc.txt"
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