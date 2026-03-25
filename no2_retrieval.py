# Import modules

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import h5py
import cdsapi
import pandas as pd
from scipy.ndimage import gaussian_filter
from scipy.optimize import nnls
import sys
import argparse
from scipy.ndimage import binary_erosion, binary_dilation
import os
from scipy import stats

class retrieve_conc:
    
    def __init__(self, res, site_name, launch_ind):
        
        self.res = res
        self.site_name = site_name
        self.launch_ind = launch_ind

        if self.res == 10:
            self.bx = 1 
        elif self.res == 60:
            self.bx = 0
        else:
            raise ValueError("Unsupported resolution.")

    def load_data(self):

        # First load the satellite data for the reference and target scenes
        data_dir = "/home/ucfacb0/ARIA/aria_data_download/download"
        file1 = (f"{data_dir}/sentinel2-l1c-band-l2a-slc-{site_name}-{self.res}-target.nc")
        data = xr.open_dataset(file1,decode_timedelta=False)
        self.sza = data["sza"].values[:,0,:,:]
        self.vza = data["vza"].values[:,0,:,:]
        self.saa = data["saa"].values[:,0,:,:]
        self.vaa = data["vaa"].values[:,0,:,:]

        self.band_data = np.zeros((*self.vaa.shape, 5))
        for count, band in enumerate(["B01","B02","B03","B04","B08"]):
            try:
                self.band_data[:,:,:,count] = data[band].values[:,0,:,:] 
            except:
                print(f"No data found for {band}.")
                continue
        
        self.lon_range = data["lon_range"].values
        self.lat_range = data["lat_range"].values
        self.times = data["time"].values
        self.target_time = pd.Timestamp(self.times[self.launch_ind]).round('h')
        data.close()

        ds_wind = xr.open_dataset("/home/ucfacb0/ARIA/aria_data_analysis/data/era5/ERA5-Riyadh08-10m-wind-fields-2020.nc")
        u10 = ds_wind["u10"]
        v10 = ds_wind["v10"]
        u10_selected = u10.sel(valid_time=self.times,
                               latitude=np.mean(self.lat_range),
                               longitude=np.mean(self.lon_range), 
                               method="nearest").values
        v10_selected = v10.sel(valid_time=self.times,
                               latitude=np.mean(self.lat_range),
                               longitude=np.mean(self.lon_range), 
                               method="nearest").values
        ds_wind.close()

        self.wind_speed = np.sqrt(u10_selected**2 + v10_selected**2)
        wind_dir = 270 - np.degrees(np.arctan2(v10_selected, u10_selected))
        angle_diff = wind_dir - wind_dir[self.launch_ind]
        self.angle_diff = np.abs((angle_diff + 180) % 360 - 180)

        # Determine the absorption cross section.
        # TODO: Just using 294K data for now, will need to improve this.
        # bandwidths from https://sentiwiki.copernicus.eu/web/s2-mission
        # xsc data from https://hitran.org/

        xsc_file = "/home/ucfacb0/ARIA/aria_data_analysis/data/NO2_294.0_0.0_15002.0-42002.3_00.xsc.txt"
        xsc = np.loadtxt(xsc_file,max_rows=1,usecols=(1,2,3),dtype=float)
        wavenumbers = np.linspace(xsc[0],xsc[1],int(xsc[2]))

        if self.res == 10: # Using B01
            wn1 = 1e7 / (442-20)
            wn2 = 1e7 / (442+20)
        elif self.res == 60: # Using B02
            wn1 = 1e7 / (492-65)
            wn2 = 1e7 / (492+65)
        else:
            raise ValueError("Resolution not recognised.")

        idx1 = np.searchsorted(wavenumbers, wn1)
        idx2 = np.searchsorted(wavenumbers, wn2)

        with open(xsc_file) as f:
            f.readline()  # skip header
            xsc_data = np.fromstring(f.read(), sep=' ')

        self.xsc_mean = np.mean(xsc_data[idx2:idx1]) * 1e-4 * 6.022e23 # m2/mol

    def compute_amf(self):

        # Load the data for the scattering weights.
        sw_path = "/home/ucfacb0/ARIA/aria_data_analysis/data/AMFs_lookup_no2_440nm.he5"
        with h5py.File(sw_path, "r") as f:
            dI0       = np.array(f["Scattering Weights/dI0"][:])  # type: ignore
            dI1       = np.array(f["Scattering Weights/dI1"][:])  # type: ignore
            dI2       = np.array(f["Scattering Weights/dI2"][:])  # type: ignore
            sza_grid  = f["Grid/SZA"][:]                              # type: ignore
            vza_grid  = f["Grid/VZA"][:]                              # type: ignore
            alb_grid  = f["Grid/Albedo"][:]                           # type: ignore
            sp_grid   = f["Grid/Surface Pressure"][:]                 # type: ignore
            alt_layer = np.array(f["Profiles/Altitude Layer"][:])     # type: ignore
            o3_labels = [s.decode() for s in f["Grid/OZO"][:]]        # type: ignore
            o3_col    = np.array(f["Profiles/Ozone Column Layer"][:]) # type: ignore

        full_totals_DU = o3_col.sum(axis=0) / 2.69e16

        target_lat = np.mean(self.lat_range)
        if abs(target_lat) < 30:
            band = "L"
        elif abs(target_lat) < 60:
            band = "M"
        else:
            band = "H"
        band_mask = np.array([label.startswith(band) for label in o3_labels])

        # Only download the target data if the file doesn't already exist.
        era5_dir = "/home/ucfacb0/ARIA/aria_data_analysis/data/era5"
        filename = os.path.join(era5_dir, f"era5_{self.site_name}_{self.target_time.strftime('%Y%m%d_%H%M')}.nc")        
        if not os.path.exists(filename):
            dataset = "reanalysis-era5-single-levels"
            request = {
                "product_type": ["reanalysis"],
                "variable": ["surface_pressure","total_column_ozone"],
                "year": [self.target_time.year],
                "month": [self.target_time.month],
                "day": [self.target_time.day],
                "time": [self.target_time.strftime('%H:%M')],
                "data_format": "netcdf",
                "download_format": "unarchived",
                "area": [np.mean(self.lat_range)+1, np.mean(self.lon_range)-1, np.mean(self.lat_range)-1, np.mean(self.lon_range)+1]
            }
            
            client = cdsapi.Client()
            result = client.retrieve(dataset, request)
            result.download(filename)
    
        ds_era5_target = xr.open_dataset(filename)
        sp = ds_era5_target["sp"].sel(latitude=np.mean(self.lat_range),
                                      longitude=np.mean(self.lon_range),
                                      method="nearest").values[0] / 100 # hPa
        
        ozone_DU = ds_era5_target["tco3"].sel(latitude=np.mean(self.lat_range),
                                              longitude=np.mean(self.lon_range),
                                              method="nearest"
                                            ).values[0] / 2.1415e-5  # kg/m² → DU
        ds_era5_target.close()

        # NOTE: Here we convert surface pressure at the target site to surface altitude.
        # Then we get the plume altitude at 500m above the surface.
        h = 145366.45 * (1 - (sp / 1013.25) ** 0.190284) * 0.3048 / 1000 # km
        h_plume_km = h + 0.5

        # Determine the albedo for the target site.
        with h5py.File("/home/ucfacb0/ARIA/aria_data_analysis/data/OMI-Aura_L3-OMLER_2005m01-2009m12_v003-2010m0503t063707.he5", "r") as f:
            base = "HDFEOS/GRIDS/EarthSurfaceReflectanceClimatology/Data Fields"
            wavelength   = np.array(f[f"{base}/Wavelength"][:]) # type: ignore
            ler          = np.array(f[f"{base}/MonthlySurfaceReflectance"][:]) # type: ignore
            lat_grid_ler = np.array(f[f"{base}/Latitude"][:]) # type: ignore
            lon_grid_ler = np.array(f[f"{base}/Longitude"][:]) # type: ignore

        i_lat = np.argmin(np.abs(lat_grid_ler - np.mean(self.lat_range)))
        i_lon = np.argmin(np.abs(lon_grid_ler - np.mean(self.lon_range)))
        i_month = self.target_time.month - 1  # 0-indexed
        mask_wav = (wavelength >= 405) & (wavelength <= 465)
        self.alb_val = np.mean(ler[i_month, mask_wav, i_lat, i_lon]) / 1000

        # Calculate relative azimuth angle
        # https://stcorp.github.io/harp/doc/html/algorithms/derivations/relative_azimuth_angle.html
        delta_phi = self.saa[self.launch_ind,:,:] - self.vaa[self.launch_ind,:,:]
        delta_phi = np.where(delta_phi >= 360, delta_phi - 360, # if above 360, then subtract 360
                    np.where(delta_phi < 0,    delta_phi + 360, # elif below 0, add 360
                                                delta_phi))     # else leave it unchanged
        raa = np.abs(delta_phi - 180) # deg

        # Calculate geometric AMF of the target
        self.amf_geo = ( (1.0/np.cos(np.radians(self.sza[self.launch_ind,:,:]))) +
                    (1.0/np.cos(np.radians(self.vza[self.launch_ind,:,:]))) )

        def nearest_idx(grid, values):

            # Create two arrays of shape (1,1,n) and (nlat,nlon,1)
            # Subtract to get the residual between grid value and values value
            diff_grid = grid[np.newaxis, np.newaxis, :] - values[..., np.newaxis]

            # Take the absolute value and find the minimum index (closest).
            min_ind = np.argmin(np.abs(diff_grid), axis=-1)
            return min_ind
    
        i_sza   = nearest_idx(sza_grid, self.sza[self.launch_ind,:,:])  # (nlat, nlon)
        i_vza   = nearest_idx(vza_grid, self.vza[self.launch_ind,:,:])  # (nlat, nlon)
        i_alb   = np.argmin(np.abs(alb_grid  - self.alb_val)) # type: ignore
        i_sp    = np.argmin(np.abs(sp_grid  - sp))
        i_z     = np.argmin(np.abs(alt_layer - h_plume_km)) # km
        i_ozone = np.argmin(np.where(band_mask, np.abs(full_totals_DU - ozone_DU), np.inf))
        print(f"Ozone profile: {i_ozone} - {o3_labels[i_ozone]} ({full_totals_DU[i_ozone]:.1f} DU, ERA5: {ozone_DU:.1f} DU)")

        # Index the dI arrays
        dI0_slice = dI0[i_sza, i_vza, i_alb, i_z, i_sp, i_ozone]
        dI1_slice = dI1[i_sza, i_vza, i_alb, i_z, i_sp, i_ozone]
        dI2_slice = dI2[i_sza, i_vza, i_alb, i_z, i_sp, i_ozone]
            
        # Calculate AMF
        # Gonzalo confirmed that SW values in the LUTs already account for the geometric AMF.
        self.amf = dI0_slice + dI1_slice * np.cos(np.radians(raa)) + dI2_slice * np.cos(2 * np.radians(raa))
        sw = self.amf / self.amf_geo
        print(f"s(w) ranges from {np.min(sw):.2f} to {np.max(sw):.2f}")

    def find_refs(self):

        print('Target SZA, VZA: ', np.mean(self.sza[self.launch_ind,:,:]), np.mean(self.vza[self.launch_ind,:,:]))

        # Loops through all cloud-free scenes for 2020 and selects the scenes most consistent with the target overpass.
        # Varon et al. (2024) ref scenes are for: 
        # - 2020-04-10
        # - 2020-04-25
        # - 2020-06-04
        # - 2020-08-03

        # But we can select references using the RMSE of the bands, SZA, and VZA.

        # For the red band, the data is different across the pixels in each individual scene.
        # So we use RMSE to work out similar scenes.
        data_len = np.shape(self.sza)[0]
        rmse_br_all = np.array([mean_squared_error(self.band_data[self.launch_ind,:,:,self.bx].flatten(), self.band_data[k,:,:,self.bx].flatten())**(1/2) for k in range(data_len)])
        rmse_bx_all = np.array([mean_squared_error(self.band_data[self.launch_ind,:,:,self.bx].flatten(), self.band_data[k,:,:,self.bx].flatten())**(1/2) for k in range(data_len)])
        pearson_all = np.array([stats.pearsonr(self.band_data[self.launch_ind,:,:,self.bx].flatten(), self.band_data[k,:,:,self.bx].flatten())[0] for k in range(data_len)])

        # SZA and VZA are essentially constant across all pixels in each individual scene.
        # So we can just look at the difference instead of using RMSE.
        sza_all = np.array([np.mean(self.sza[i,:,:]) for i in range(data_len)])
        vza_all = np.array([np.mean(self.vza[i,:,:]) for i in range(data_len)])

        # Precompute diffs
        vza_diff_all = np.abs(vza_all[self.launch_ind] - vza_all)
        sza_diff_all = np.abs(sza_all[self.launch_ind] - sza_all)

        # TODO: Daniel - RMSE threshold for selecting references?
        sza_thresh     = 3
        vza_thresh     = 0.2
        rmse_br_thresh = 200 # range_sd * np.std(rmse_br_all)
        rmse_bx_thresh = 100 # range_sd * np.std(rmse_br_all)
        pearson_thresh = 0.88

        # Loop over references to find corresponding target:
        ind_ref = []
        for k in range(data_len):

            # Skip target index:
            if k==self.launch_ind:
                continue

            if (    rmse_br_all[k] < rmse_br_thresh
                and rmse_bx_all[k] < rmse_bx_thresh
                and vza_diff_all[k] < vza_thresh
                and sza_diff_all[k] < sza_thresh
                and pearson_all[k] > pearson_thresh
                and self.wind_speed[k] > 2
                and self.angle_diff[k] > 60
                ):
                ind_ref.append(k)
                
        # Quick check on reference locations:
        print('Ref indices: ', ind_ref, self.times[ind_ref])
        print(f"rmse_br    — passing: {np.sum(rmse_br_all < rmse_br_thresh)}/{data_len}")
        print(f"rmse_bx    — passing: {np.sum(rmse_bx_all < rmse_bx_thresh)}/{data_len}")
        print(f"pearsonr   — passing: {np.sum(pearson_all < pearson_thresh)}/{data_len}")
        print(f"sza_diff   — passing: {np.sum(sza_diff_all < sza_thresh)}/{data_len}")
        print(f"vza_diff   — passing: {np.sum(vza_diff_all < vza_thresh)}/{data_len}")
        print(f"wind speed — passing: {np.sum(self.wind_speed > 2)}/{data_len}")
        print(f"wind angle — passing: {np.sum(self.angle_diff > 60)}/{data_len}")
        print(f"all        — passing: {len(ind_ref)}/{data_len}")

        return ind_ref

    def plot_image(self, no2_plume, scene_ind, site_name, target_time):

        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))

        # Set up background image
        rgb = np.stack([
            self.band_data[scene_ind, :, :, 3],  # Red
            self.band_data[scene_ind, :, :, 2],  # Green
            self.band_data[scene_ind, :, :, 1],  # Blue
        ], axis=-1)
        rgb = np.clip(rgb / 5000.0, 0, 1)

        fig.suptitle(f'{site_name} - {target_time}',fontsize=12,fontweight='normal')

        for i in range(2):
            ax[i].imshow(rgb)
            ax[i].set_xticks([])
            ax[i].set_yticks([])
            im = ax[i].imshow(no2_plume[i], cmap='magma', vmin=0, vmax=2.5)
            fig.colorbar(im, ax=ax[i], label='mmol/m2', fraction=0.039, pad=0.04)
        
        ax[0].set_title("First retrieval (no plume mask)")
        ax[1].set_title("Second retrieval (with plume mask)")
        
        fig.tight_layout()
    
        plt.savefig(f"test_{self.res}.png",dpi=300)
    
    def calc_retrieval(self, target, ref, water_mask, plume_mask):

        # Create mask
        mask = ~water_mask
        if plume_mask is not None:
            mask = mask & ~plume_mask

        # Flatten everything so its one line of pixels.
        flattened_mask = mask.flatten()
        ref_flat    = ref.flatten()[flattened_mask]
        target_flat = target.flatten()[flattened_mask]
        c = (ref_flat @ target_flat) / (ref_flat @ ref_flat)

        # Equation 1 - Normalized Reflectance Ratio
        ratio   = (c * target) / ref

        # Equation 2 - NO2 slant column enhancements  [mol m-2]
        no2_scd = (-1) * np.log(ratio) / self.xsc_mean

        # Equation 3 - Vertical columns  [mmol m-2]
        no2_vcd = (no2_scd / self.amf_geo) * 1e3

        # Derive plume mask from preliminary retrieval
        # TODO: Daniel - What did you use for the parameters here? Is there a way to automate this?
        sigma      = 2
        percentile = 92
        morph_struct_size = 5
        morph_iterations = 2
        if plume_mask is None:
            no2_smooth = gaussian_filter(no2_vcd, sigma=sigma)
            threshold  = np.nanpercentile(no2_smooth, percentile)
            plume_mask = (no2_vcd > threshold)

            # Alternating erosion/dilation of the plume mask to remove artifacts.
            struct = np.ones((morph_struct_size, morph_struct_size))
            for _ in range(morph_iterations):
                # Opening: removes small isolated artifact pixels
                plume_mask = binary_dilation(binary_erosion(plume_mask, struct), struct)
            # Closing: fills small holes inside the plume body
            plume_mask = binary_erosion(binary_dilation(plume_mask, struct), struct)

        display_mask = plume_mask & ~water_mask
        no2_plume = np.ma.masked_where(~display_mask, no2_vcd)

        return no2_vcd, no2_plume, plume_mask

    def compute_water_mask(self,scene_ind):
        B03 = self.band_data[scene_ind, :, :, 2].astype(float)
        B08 = self.band_data[scene_ind, :, :, 4].astype(float)
        ndwi = (B03 - B08) / (B03 + B08 + 1e-10) # small value avoids division by zero
        water_mask = ndwi > 0   # True where water
    
        # Expand the water mask outward to exclude edge-contaminated pixels
        dilation_radius = 3
        struct = np.ones((2 * dilation_radius + 1, 2 * dilation_radius + 1))
        water_mask_dilated = binary_dilation(water_mask, structure=struct)

        return water_mask_dilated
    
    def get_no2_conc(self):

        self.load_data()
        self.compute_amf()
        ind_ref = self.find_refs()
        if len(ind_ref) == 0:
            raise ValueError("No references found.")
        water_mask = self.compute_water_mask(self.launch_ind)

        # Define the scenes.
        target = self.band_data[self.launch_ind, :, :, self.bx]  # (H, W) - target scene
        refs   = self.band_data[ind_ref, :, :, self.bx]          # (N_refs, H, W) - reference scenes
        
        # Flatten the arrays.
        X = refs.reshape(refs.shape[0], -1).T
        Y = target.flatten()

        # TODO: Daniel - Is this the correct approach? 
        # Do we remove the water pixels from the references before calculating the linear weighted combination?
        # And then remove the plume pixels the next time?
        background_mask = ~water_mask.flatten()
        w, _ = nnls(X[background_mask], Y[background_mask])
        ref = sum(w[i] * refs[i] for i in range(refs.shape[0]))
        print(w)

        no2_plume_list = []
        # First retrieval to get the plume mask.
        no2_vcd, no2_plume, plume_mask = self.calc_retrieval(target, ref, water_mask, None)
        no2_plume_list.append(no2_plume)

        background_mask = ~water_mask.flatten() & ~plume_mask.flatten()
        w, _ = nnls(X[background_mask], Y[background_mask])
        ref = sum(w[i] * refs[i] for i in range(refs.shape[0]))
        print(w)

        # Second retrieval to get the final plume.
        no2_vcd, no2_plume, plume_mask = self.calc_retrieval(target, ref, water_mask, plume_mask)
        no2_plume_list.append(no2_plume)
        
        self.plot_image(no2_plume_list, self.launch_ind, site_name, self.target_time)

if __name__ == "__main__":  

    # Load data
    site_name = 'PP_Riyadh08'
    launch_ind = 21

    # Set up the arguments for each function.
    parser = argparse.ArgumentParser()
    parser.add_argument('-res', "--resolution", default = "10", choices=(["10","60"]), help="Resolution (10 or 60).")
    args = parser.parse_args() 

    #Loop over all years and run functions depending on input arguments.
    Retrieval = retrieve_conc(int(args.resolution), site_name, launch_ind)  
    Retrieval.get_no2_conc()