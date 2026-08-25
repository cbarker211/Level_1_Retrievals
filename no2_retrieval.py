# Import modules

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.metrics import mean_squared_error
import h5py
import pandas as pd
from scipy.ndimage import gaussian_filter
from scipy.optimize import nnls
import sys
import argparse
from scipy.ndimage import binary_erosion, binary_dilation
import os
from scipy import stats
import json
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from haversine import haversine
from matplotlib.patches import Rectangle
from load_data import load_data

class NO2Retrieval:
    
    def __init__(self, res, site_name, launch_ind):
        
        self.res = res
        self.site_name = site_name
        self.launch_ind = launch_ind

    def compute_amf(self, sat_data, era5_data):

        # Load the data for the scattering weights.
        sw_path = "/home/ucfacb0/ARIA/aria_data_analysis/data/AMFs_lookup_no2_440nm.he5"
        with h5py.File(sw_path, "r") as f:
            dI0       = np.array(f["Scattering Weights/dI0"][:])  # type: ignore
            dI1       = np.array(f["Scattering Weights/dI1"][:])  # type: ignore
            dI2       = np.array(f["Scattering Weights/dI2"][:])  # type: ignore
            sza_grid  = np.array(f["Grid/SZA"][:])                # type: ignore
            vza_grid  = np.array(f["Grid/VZA"][:])                # type: ignore
            alb_grid  = np.array(f["Grid/Albedo"][:])             # type: ignore   
            sp_grid   = np.array(f["Grid/Surface Pressure"][:])   # type: ignore
            alt_layer = np.array(f["Profiles/Altitude Layer"][:])     # type: ignore
            o3_labels = [s.decode() for s in f["Grid/OZO"][:]]        # type: ignore
            o3_col    = np.array(f["Profiles/Ozone Column Layer"][:]) # type: ignore

        full_totals_DU = o3_col.sum(axis=0) / 2.69e16

        target_lat = sat_data["center_lat"]
        if abs(target_lat) < 30:
            band = "L"
        elif abs(target_lat) < 60:
            band = "M"
        else:
            band = "H"
        band_mask = np.array([label.startswith(band) for label in o3_labels])

        # Determine the albedo for the target site.
        with h5py.File("/home/ucfacb0/ARIA/aria_data_analysis/data/OMI-Aura_L3-OMLER_2005m01-2009m12_v003-2010m0503t063707.he5", "r") as f:
            base = "HDFEOS/GRIDS/EarthSurfaceReflectanceClimatology/Data Fields"
            wavelength   = np.array(f[f"{base}/Wavelength"][:]) # type: ignore
            ler          = np.array(f[f"{base}/MonthlySurfaceReflectance"][:]) # type: ignore
            lat_grid_ler = np.array(f[f"{base}/Latitude"][:]) # type: ignore
            lon_grid_ler = np.array(f[f"{base}/Longitude"][:]) # type: ignore

        i_lat = np.argmin(np.abs(lat_grid_ler - sat_data["center_lat"]))
        i_lon = np.argmin(np.abs(lon_grid_ler - sat_data["center_lon"]))
        i_month = sat_data["time_tar"].month - 1  # 0-indexed
        mask_wav = (wavelength >= 405) & (wavelength <= 465)
        alb_val = np.mean(ler[i_month, mask_wav, i_lat, i_lon]) / 1000

        # Calculate geometric AMF of the target
        self.amf_geo = ( (1.0/np.cos(np.radians(sat_data["sza_tar"][self.launch_ind,:,:]))) +
                    (1.0/np.cos(np.radians(sat_data["sza_tar"][self.launch_ind,:,:]))) )

        def nearest_idx(grid, values):

            # Create two arrays of shape (1,1,n) and (nlat,nlon,1)
            # Subtract to get the residual between grid value and values value
            diff_grid = grid[np.newaxis, np.newaxis, :] - values[..., np.newaxis]

            # Take the absolute value and find the minimum index (closest).
            min_ind = np.argmin(np.abs(diff_grid), axis=-1)
            return min_ind
    
        i_sza   = nearest_idx(sza_grid, sat_data["sza_tar"][self.launch_ind,:,:])  # (nlat, nlon)
        i_vza   = nearest_idx(vza_grid, sat_data["vza_tar"][self.launch_ind,:,:])  # (nlat, nlon)
        i_alb   = np.argmin(np.abs(alb_grid  - alb_val))
        i_sp    = np.argmin(np.abs(sp_grid   - era5_data["sp"]))
        i_z     = np.argmin(np.abs(alt_layer - era5_data["h_plume_km"])) # km
        i_ozone = np.argmin(np.where(band_mask, np.abs(full_totals_DU - era5_data["ozone_DU"]), np.inf))
        print(f"Ozone profile: {i_ozone} - {o3_labels[i_ozone]} ({full_totals_DU[i_ozone]:.1f} DU, ERA5: {era5_data['ozone_DU']:.1f} DU)")

        # Index the dI arrays
        dI0_slice = dI0[i_sza, i_vza, i_alb, i_z, i_sp, i_ozone]
        dI1_slice = dI1[i_sza, i_vza, i_alb, i_z, i_sp, i_ozone]
        dI2_slice = dI2[i_sza, i_vza, i_alb, i_z, i_sp, i_ozone]
            
        # Calculate AMF
        # Gonzalo confirmed that SW values in the LUTs already account for the geometric AMF.
        self.amf = dI0_slice + dI1_slice * np.cos(np.radians(sat_data["raa_tar"])) + dI2_slice * np.cos(2 * np.radians(sat_data["raa_tar"]))

    def find_refs(self, mask, number_refs):
        ''' Loops through all cloud-free scenes for the same year and selects the scenes most consistent with the target overpass.

        Only look over land pixels, and find references with a similar B04 band using RMSE.
        For sza and vza, use a absolute degree difference.'''       

        # Define the scenes. The launch scenes are already removed from the references when the data is loaded. 

        b04_tar  = self.band_data_tar[self.launch_ind,:,:,3][mask] # Land pixels only.
        b04_refs = self.band_data_ref[:,:,:,3][:,mask] # Land pixels only.
        data_len = np.shape(self.sza_ref)[0]
        rmse_br_all = np.array([mean_squared_error(b04_tar, b04_refs[k,...])**(1/2) for k in range(data_len)])

        # Find the absolute difference in mean sza between target and reference scenes.
        mean_sza_tar = np.nanmean(self.sza_tar[self.launch_ind, :, :])
        sza_diff_all = np.array([
            abs(mean_sza_tar - np.nanmean(self.sza_ref[k, :, :]))
            for k in range(data_len)
        ])

        # For vza, sometimes there are NaNs on the edge of the data, so we have to mask these.
        vza_diff_all = []
        for k in range(data_len):
            tar = self.vza_tar[self.launch_ind, :, :]
            ref = self.vza_ref[k, :, :]
            mask = np.isfinite(tar) & np.isfinite(ref)
            vza_diff_all.append(abs(np.nanmean(tar[mask]) - np.nanmean(ref[mask])))
        vza_diff_all = np.array(vza_diff_all)

        # Hard threshold for vza. 
        # The viewing angle difference is typically close to zero or around 3 degrees.
        vza_thresh = 1.0
        valid_idx = np.where(vza_diff_all < vza_thresh)[0]
        if len(valid_idx) == 0:
            ind_ref = []
        else:

            # Now find the best four references of whats left.
            sza_valid = sza_diff_all[valid_idx]
            rmse_valid = rmse_br_all[valid_idx]

            sza_rank = np.argsort(np.argsort(sza_valid))
            rmse_rank = np.argsort(np.argsort(rmse_valid))

            score = sza_rank + rmse_rank

            best_local = np.argsort(score)[:number_refs]
            ind_ref = valid_idx[best_local]
                
        # Quick check on reference locations:
        print('Ref indices: ', ind_ref, self.time_ref[ind_ref])
        print(f"vza_diff   — passing: {np.sum(vza_diff_all < vza_thresh)}/{data_len}")

        return ind_ref

    def plot_diag_image(self, ratio, validland_mask):

        fig, ax = plt.subplots(1, 4, figsize = (24, 8))

        # Plot the RGB image of the target scene.      
        rgb = np.stack([
            self.band_data_tar[self.launch_ind, :, :, 3],  # Red
            self.band_data_tar[self.launch_ind, :, :, 2],  # Green
            self.band_data_tar[self.launch_ind, :, :, 1],  # Blue
        ], axis=-1)
        rgb = np.clip(rgb / 5000.0, 0, 1)
        im = ax[0].imshow(rgb, extent=self.extent)
        ax[0].set_title("RGB")

        # Plot the ratio of the target to reference scene.
        im = ax[1].imshow(ratio, cmap='RdBu_r', vmin=0.9, vmax=1.1, extent=self.extent)
        ax[1].set_title("Ratio = c * target / reference")
        plt.colorbar(im, ax=ax[1], fraction=0.039, pad=0.04)

        cluster_plot = np.ma.masked_where(self.mf_clusters == -1, self.mf_clusters)
        im = ax[2].imshow(cluster_plot, cmap=plt.colormaps['gist_rainbow'].resampled(self.n_clusters), interpolation='nearest', extent=self.extent)
        ax[2].set_title(f'K-means ({self.mf_clusters.max()+1} clusters)')
        plt.colorbar(im, ax=ax[2], fraction=0.039, pad=0.04)

        im = ax[3].imshow(np.where(validland_mask, 1, 0), cmap='gray', vmin=0, vmax=1.1, extent=self.extent)
        ax[3].set_title("Water mask")
        plt.colorbar(im, ax=ax[3], fraction=0.039, pad=0.04)

        # Save fig.
        fig.tight_layout()
        if not os.path.exists(f"out/{self.site_name.replace('/', '')}"):
            os.makedirs(f"out/{self.site_name.replace('/', '')}")
        plt.savefig(f"./out/{self.site_name.replace('/', '')}/{self.site_name}_{self.time_tar.strftime('%Y%m%d_%H%M')}_{self.res}_diag.png")
        
    def plot_final_image(self, no2_vcd, scene_ind, source_rate):

        fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(16, 16))

        # Set up background image
        rgb = np.stack([
            self.band_data_tar[scene_ind, :, :, 3],  # Red
            self.band_data_tar[scene_ind, :, :, 2],  # Green
            self.band_data_tar[scene_ind, :, :, 1],  # Blue
        ], axis=-1)
        rgb = np.clip(rgb / 5000.0, 0, 1)

        # Convert fractional bounds to pixel indices
        h_min_frac, h_max_frac, w_min_frac, w_max_frac = self.crop
        h, w = rgb.shape[:2]
        y_start = int(h_min_frac * h)
        y_end   = int(h_max_frac * h)
        x_start = int(w_min_frac * w)
        x_end   = int(w_max_frac * w)

        # Update geographic extent
        xmin, xmax, ymin, ymax = self.extent

        crop_extent = [
            xmin + w_min_frac * (xmax - xmin),  # x min
            xmin + w_max_frac * (xmax - xmin),  # x max
            ymin + h_min_frac * (ymax - ymin),  # y min
            ymin + h_max_frac * (ymax - ymin),  # y max
        ]

        fig.suptitle(f'{self.site_name} - {self.time_tar} - {source_rate:.0f} kg',fontsize=12,fontweight='normal')
        for i in range(2):
            for j in range(2):
                
                ax[i,j].imshow(rgb[y_start:y_end, x_start:x_end], extent=crop_extent)
                ax[i,j].set_xticks([])
                ax[i,j].set_yticks([])
                
                # Print over land only, and set vmin to 0.
                no2_crop = no2_vcd[j][i][y_start:y_end, x_start:x_end]
                im = ax[i,j].imshow(no2_crop, cmap='magma', extent=crop_extent, vmin=0)
                cbar = fig.colorbar(im, ax=ax[i,j], fraction=0.039, pad=0.04)
                cbar.set_label('NO$_2$ column enhancement (mol m$^{-2}$)',fontsize=18)
                cbar.ax.tick_params(labelsize=18)

        # Plot scalebar
        length_m = 750
        scale_pixels = length_m / 60
        crop_width_pixels = x_end - x_start
        print(crop_width_pixels*60/1000)
        crop_width_degrees = crop_extent[1] - crop_extent[0]
        bar_width = scale_pixels * crop_width_degrees / crop_width_pixels
        xmin_c, xmax_c, ymin_c, ymax_c = crop_extent
        bar_height = (ymax_c - ymin_c) * 0.02
        x0 = xmin_c + 0.85 * (xmax_c - xmin_c)
        y0 = ymin_c + 0.06 * (ymax_c - ymin_c)

        # Background box dimensions
        padding_x = bar_width * 0.25
        bg_x = x0 - padding_x
        bg_y = y0 - bar_height * 1.5
        bg_width = bar_width + 2 * padding_x

        # Semi-transparent background box
        bg_rect = Rectangle((bg_x, bg_y), bg_width, bar_height * 5.5,
                            facecolor="black", edgecolor="none",
                            alpha=0.35, zorder=18
        )
        ax[1,1].add_patch(bg_rect)

        # Draw rectangle
        rect = Rectangle((x0, y0), bar_width, bar_height,
                        facecolor="white", edgecolor='black', lw=1.5, zorder=20)
        ax[1,1].add_patch(rect)

        # Label
        ax[1,1].text(x0 + bar_width  / 2, y0 + bar_height * 1.5,
                f"{int(length_m)} m", color='white',
                ha='center', va='bottom',
                fontsize=14, fontweight='bold', zorder=20) 

        ax[0,0].set_title("First retrieval (no plume mask)")
        ax[0,1].set_title("Second retrieval (with plume mask)")
        
        fig.tight_layout()
        if not os.path.exists(f"out/{self.site_name.replace('/', '')}"):
            os.makedirs(f"out/{self.site_name.replace('/', '')}")
        plt.savefig(f"./out/{self.site_name.replace('/', '')}/{self.site_name}_{self.time_tar.strftime('%Y%m%d_%H%M')}_{self.res}.png")
    
    def calc_retrieval(self, target, ref, validland_mask, plume_mask, apply_morphology=False):

        # If theres a plume mask, create a non-plume, valid land pixel mask.
        mask = validland_mask.copy()
        if plume_mask is not None:
            mask = mask & ~plume_mask

        # Flatten everything so its one line of pixels and calculate the scale factor (2D array).
        flattened_mask = mask.flatten()
        ref_flat    = ref.flatten()[flattened_mask]
        target_flat = target.flatten()[flattened_mask]
        c = (ref_flat @ target_flat) / (ref_flat @ ref_flat)

        plt.figure(figsize=(6,6))
        plt.scatter(ref_flat, target_flat, s=5, alpha=0.5)
        x = np.linspace(ref_flat.min(), ref_flat.max(), 100)
        plt.plot(x, c * x, label=f'Best fit: y={c:.6f}x')
        plt.plot(x, x, linestyle='--', label='1:1 line')
        plt.xlabel("Reflectance (Target-weighted Reference)")
        plt.ylabel("Reflectance (Target)")
        plt.legend()
        plt.grid(True)
        if not apply_morphology:
            plt.title("Reflectance Comparison over Valid Land Pixels")
            plt.savefig("test1.png")
        else:
            plt.title("Reflectance Comparison over Valid Land Non-Plume Pixels")
            plt.savefig("test2.png")

        # Equation 1 - Normalized Reflectance Ratio
        ratio   = (c * target) / ref
        ratio[~validland_mask] = np.nan

        # Equation 2 - NO2 slant column enhancements  [mol m-2]
        no2_scd = (-1) * np.log(ratio) / self.xsc_mean

        # Equation 3 - Vertical columns  [mmol m-2]
        no2_vcd = (no2_scd / self.amf)
        # Check how similar the target and reference are in the background (non-plume, non-water) pixels.
        valid = np.isfinite(ratio)
        q = np.nanpercentile(ratio[valid], [5, 95])
        print("5-95 ratio percentiles:", q)

        # Save an image showing some diagnostics.
        if not apply_morphology:
            self.plot_diag_image(ratio, validland_mask)

        # Derive plume mask from preliminary retrieval
        sigma      = 1.5
        percentile = 90
        morph_struct_size = 5
        morph_iterations = 3
        no2_smooth = gaussian_filter(no2_vcd, sigma=sigma)
        threshold  = np.nanpercentile(no2_smooth, percentile)
        plume_mask = (no2_smooth > threshold)

        # Alternating erosion/dilation of the plume mask to remove artifacts (2nd retrieval only)
        if apply_morphology:
            struct = np.ones((morph_struct_size, morph_struct_size))
            for _ in range(morph_iterations):
                # Opening: removes small isolated artifact pixels
                plume_mask = binary_dilation(binary_erosion(plume_mask, struct), struct)
            # Closing: fills small holes inside the plume body
            plume_mask = binary_erosion(binary_dilation(plume_mask, struct), struct)

        display_mask = plume_mask & validland_mask
        no2_plume = np.ma.masked_where(~display_mask, no2_vcd)

        return no2_vcd, no2_plume, plume_mask.astype(bool)
    
    def _compute_water_mask(self, data, scene_ind):

        # If working at 60m resolution, coarsen from 10m to 60m.
        B03 = data[scene_ind, :, :, 2].astype(float)
        B08 = data[scene_ind, :, :, 7].astype(float)
        ndwi = (B03 - B08) / (B03 + B08 + 1e-10)
        water_mask = ndwi > 0
            
        # Expand the water mask outward to exclude edge-contaminated pixels
        dilation_radius = 0 # We need this to be smaller when we have a lot of inland water features.
        # This is because it can end up removing a lot of land.
        struct = np.ones((2 * dilation_radius + 1, 2 * dilation_radius + 1))
        water_mask_dilated = binary_dilation(water_mask, structure=struct)

        return water_mask_dilated.astype(bool)
    
    def _compute_validland_mask(self, crop, ind_ref=None):
        if ind_ref is None:
            ref_idx = np.arange(self.datamask_ref.shape[0])
        else:
            ref_idx = ind_ref
        data_mask_union = self.datamask_tar & np.all(self.datamask_ref[ref_idx, ...], axis=0)
        water_mask_tar = self._compute_water_mask(self.band_data_tar, self.launch_ind)
        water_mask_refs = np.array([self._compute_water_mask(self.band_data_ref, k) for k in ref_idx])
        water_mask_union_10 = water_mask_tar | np.any(water_mask_refs, axis=0)
        validland_mask = data_mask_union & (~water_mask_union_10)

        h_min_frac, h_max_frac, w_min_frac, w_max_frac = crop

        height, width = validland_mask.shape
        h_start = int(h_min_frac * height)
        h_end   = int(h_max_frac * height)

        w_start = int(w_min_frac * width)
        w_end   = int(w_max_frac * width)

        center_mask = np.zeros_like(validland_mask, dtype=bool)
        center_mask[h_start:h_end, w_start:w_end] = True

        # Combine with valid land mask
        validland_mask = validland_mask & center_mask

        return validland_mask

    def compute_clusters(self, band_data, validland_mask):

        # Add some extra features to help the clustering - NDVI, NDWI, NDSI.
        NDVI = (band_data[..., 7] - band_data[..., 3]) / (band_data[..., 7] + band_data[..., 3] + 1e-6)
        NDWI = (band_data[..., 2] - band_data[..., 7]) / (band_data[..., 2] + band_data[..., 7] + 1e-6)
        NDSI = (band_data[..., 2] - band_data[..., 9]) / (band_data[..., 2] + band_data[..., 9] + 1e-6)
        features = np.concatenate([band_data, NDVI[..., None], NDWI[..., None], NDSI[..., None]],axis=-1)

        # Apply a gaussian smoothing kernel to help with the clustering, only smoothing over pixels not bands.
        features = gaussian_filter(features, sigma=(1,1,0))
        
        """K-means clustering on RxCxB hyperspectral image. Returns RxC class map."""
        R, C, B = features.shape
        pixels = features.reshape(-1, B)
        land_mask = validland_mask.reshape(-1)
        mask = np.all(np.isfinite(pixels), axis=1) & land_mask
        X = pixels[mask]
        labels = np.full(R * C, -1, dtype=np.int32)

        # Keep the PCA components that explain 99% of the variance.
        pca = PCA(n_components=0.99)
        pix_reduced = pca.fit_transform(X)
        print(f"PCs for 99% variance: {pca.n_components_}.")

        self.n_clusters = 5
        labels[mask] = MiniBatchKMeans(n_clusters=self.n_clusters, batch_size=1024, n_init=10, random_state=0).fit_predict(pix_reduced)
        clusters = labels.reshape(R, C)

        unique, counts = np.unique(labels[mask], return_counts=True)
        pixel_counts = dict(zip(unique, counts))

        max_count = max(pixel_counts.values())
        bar_width = 75  # Max width of bars in characters
        print("Pixels per cluster:")
        # Print a bar to the terminal for each cluster for a visual check.
        for cluster, count in pixel_counts.items():
            bar_len = int((count / max_count) * bar_width)
            bar = "█" * bar_len
            print(f"Cluster {cluster}: {count:>8} | {bar}")
        
        return clusters
        
    def get_no2_conc(self):

        sat_data, era5_data, self.xsc_mean = load_data(self.site_name, self.launch_ind)
        self.band_data_tar = sat_data["band_data_tar"]
        self.band_data_ref = sat_data["band_data_ref"]
        self.datamask_tar = sat_data["datamask_tar"]
        self.datamask_ref = sat_data["datamask_ref"]
        self.sza_tar = sat_data["sza_tar"]
        self.vza_tar = sat_data["vza_tar"]
        self.sza_ref = sat_data["sza_ref"]
        self.vza_ref = sat_data["vza_ref"]
        self.time_ref = sat_data["time_ref"]
        self.time_tar = sat_data["time_tar"]
        self.extent = (sat_data["lon"][0], sat_data["lon"][1], sat_data["lat"][0], sat_data["lat"][1]) 
        self.u10 = era5_data["u10"]
        self.v10 = era5_data["v10"]

        # Launch site is at center, so we need the estimated distance to be smaller than half the box.
        lat_km = haversine((sat_data["center_lat"], sat_data["center_lon"]), (sat_data["lat"][1], sat_data["center_lon"]))
        lon_km = haversine((sat_data["center_lat"], sat_data["center_lon"]), (sat_data["center_lat"], sat_data["lon"][1]))

        self.compute_amf(sat_data, era5_data)

        # We need to compare to the refs over valid land pixels only, so create a union mask. 
        self.crop = (0.3,0.55,0.35,0.65)  
        validland_mask = self._compute_validland_mask(self.crop)

        #################################
        # Split the target into clusters.
        #################################

        band_data        = self.band_data_tar[self.launch_ind, :,:,:]
        self.mf_clusters = self.compute_clusters(band_data, validland_mask)

        '''So the workflow is:

        build clusters on the target
        
        for each cluster k in the target:
            extract the target pixels belonging to cluster k
            look at the same pixels in each candidate reference scene
            compute a similarity score
            pick the best reference date for that cluster

        build a composite reference scene where each target cluster uses its own best reference date'''

        ############
        # Find refs
        ############

        # TODO: Normalize first, because otherwise we find refs before calculating c.
        # But whole scene might be a good match but just 20% darker.

        ind_ref = self.find_refs(validland_mask, number_refs=2)
        if len(ind_ref) == 0:
            raise ValueError("No references found.")

        ###############
        # Retrieval 1
        ###############

        # TODO: Cluster analysis - do linear weighted approach but scene dependent.
        # Use PCA 12D - top 5 PCAs  delete band 10
        # Which pixels have the most similar pca coefficients. 
        # clustered matched filter for hyperspectral, but not for broadband.

        # Calculate the linear weighted references with over valid land pixels.
        target = self.band_data_tar[self.launch_ind, :, :, 0]  # (H, W) - target scene
        refs   = self.band_data_ref[ind_ref, :, :, 0]          # (N_refs, H, W) - reference scenes

        # Now we know what our target and refs are, we can create a new union mask.
        # This stops us unecessarily losing pixels.

        validland_mask = self._compute_validland_mask(self.crop, ind_ref)

        # Calculate the linear weighted references with over non-plume valid land pixels.
        background_mask = validland_mask
        
        w, _ = nnls(refs[:, background_mask].T, target[background_mask])
        ref = sum(w[i] * refs[i] for i in range(refs.shape[0]))
        print(f"Reference weightings are {w}")

        no2_plume_list = []
        # First retrieval to get the plume mask.
        no2_vcd, no2_plume, plume_mask = self.calc_retrieval(target, ref, validland_mask, None, apply_morphology=False)
        no2_plume_list.append((no2_vcd,no2_plume))

        ###############
        # Retrieval 2
        ###############

        # Calculate the linear weighted references with over non-plume valid land pixels.
        background_mask = validland_mask & ~plume_mask
        w, _ = nnls(refs[:, background_mask].T, target[background_mask])
        ref = sum(w[i] * refs[i] for i in range(refs.shape[0]))
        print(w)

        # Second retrieval to get the final plume.
        no2_vcd, no2_plume, plume_mask = self.calc_retrieval(target, ref, validland_mask, plume_mask, apply_morphology=True)
        no2_plume_list.append((no2_vcd,no2_plume))
        
        #mw_no2 = 0.046  # kg/mol
        #pixel_area = self.res ** 2 # m2
        #ime_no2 = np.ma.sum((no2_plume / 1000) * pixel_area)  # mol
        #factor = np.exp(delta_t / tau)
        #ime_nox = factor * mw_no2 * ime_no2  # kg

        self.plot_final_image(no2_plume_list, self.launch_ind, 0)

if __name__ == "__main__":  

    # Load data
    platform = "sn"
    site_data = json.load(open(f"/home/ucfacb0/ARIA/aria_data_download/matches/matches_{platform}.json"))

    # Set up the arguments for each function.
    parser = argparse.ArgumentParser()
    parser.add_argument('-res', "--resolution", default = "60", choices=(["10","60"]), help="Resolution (10 or 60).")
    parser.add_argument('-site',"--site",       default = "all",help="Launch site.")
    args = parser.parse_args()

    site_name = args.site
    if site_name == "all":
        site_names = list(site_data.keys())
    else:
        site_names = [site_name]

    for site_name in site_names:

        launches = len(site_data[site_name]["launch_times"])
        #Loop over all years and run functions depending on input arguments.
        for launch in range(launches):

            launch_time = pd.to_datetime(site_data[site_name]["launch_times"][launch])

            print(f"Processing launch {launch+1}/{launches} for site {site_name} at {args.resolution}m resolution.")
            Retrieval = NO2Retrieval(int(args.resolution), site_name, launch)  
            Retrieval.get_no2_conc()
            print("\n")
            print("#"*40)
            print("\n")
            sys.exit()