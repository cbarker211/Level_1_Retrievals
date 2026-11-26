import numpy as np
import pandas as pd
from datetime import timedelta
from pathlib import Path
import xarray as xr
import os

from scipy.ndimage import binary_dilation
from omnicloudmask import predict_from_array
from scipy.ndimage import gaussian_filter
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA

from sentinelhub.data_collections import DataCollection
from sentinelhub.config import SHConfig
from sentinelhub.constants import MimeType
from sentinelhub.geo_utils import bbox_to_dimensions
from sentinelhub.api.process import SentinelHubRequest
from sentinelhub.constants import CRS
from sentinelhub.geometry import BBox

###################
# Helper functions.
###################

def create_rgb(data,clip):
    
    rgb = np.stack([
        data["red"],
        data["green"],
        data["blue"],
    ], axis=-1)
          
    return np.clip(rgb / clip, 0, 1)

def compute_water_mask(data, filters, platform, scene_ind=None):

    '''This uses the band data to detect water and snow/salinity, 
    and remove the brightest pixels.'''
        
    band1 = data["green"].values.astype(np.float32)
    band2 = data["nir"].values.astype(np.float32)
    if platform != "sd":
        band3 = data["swir1"].values.astype(np.float32)
    else:
        band3 = data["red"] # Just a placeholder, we can't do NDSI for Planet
    b02   = data["blue"].values.astype(np.float32)
    b03   = data["green"].values.astype(np.float32)
    b04   = data["red"].values.astype(np.float32)

    # --- handle optional time dimension ---
    if scene_ind is not None:
        band1 = band1[scene_ind]
        band2 = band2[scene_ind]
        if platform != "sd":
            band3 = band3[scene_ind]
        b02 = b02[scene_ind]
        b03 = b03[scene_ind]
        b04 = b04[scene_ind]
            
    # Calculate the water and snow/salinity mask
    ndwi = (band1 - band2) / (band1 + band2 + 1e-10)
    if platform != "sd":
        ndsi = (band1 - band3) / (band1 + band3 + 1e-10)
        water_mask = (ndwi > filters[1]) | (ndsi > filters[2])
    else:
        water_mask = ndwi > filters[1]

    # Calculate the brightness mask
    brightness = np.mean(np.stack([b02,b03,b04]),axis=0)
    struct = np.ones((2 * filters[3] + 1, 2 * filters[3] + 1))
    if brightness.ndim == 2:
        threshold = np.nanpercentile(brightness, filters[0])
        bright_mask = brightness > threshold
        mask = water_mask | bright_mask
        masks = binary_dilation(mask, structure=struct)
    else:
        thresholds = np.nanpercentile(brightness,filters[0],axis=(1, 2))
        bright_mask = brightness > thresholds[:, None, None]
        masks = water_mask | bright_mask
        
        masks = np.array([binary_dilation(m, structure=struct) for m in masks])
        
    return masks.astype(bool)

def compute_validland_mask(data_tar, data_ref, target_band, crop, filters, platform, ind_ref=None):

    if ind_ref is None:
        ref_idx = np.arange(data_ref[target_band].shape[0])
    else:
        ref_idx = ind_ref
    
    # Water mask
    water_mask_tar  = compute_water_mask(data_tar, filters, platform)
    water_mask_refs = compute_water_mask(data_ref, filters, platform, ref_idx)
    water_mask = water_mask_tar | np.any(water_mask_refs, axis=0)
    
    # Valid mask
    valid_mask = compute_center_mask(crop, data_tar[target_band])
    if platform == "sn":
        valid_mask = valid_mask & data_tar["dataMask"].values.astype(bool) & np.all(data_ref["dataMask"].values[ref_idx, ...].astype(bool), axis=0)
    elif platform == "ls":
        valid_mask = valid_mask
    elif platform == "sd":
        valid_mask = valid_mask & ~data_tar["udm2"].sel(type='udm1').values.astype(bool) & np.all(~data_ref["udm2"].sel(type='udm1').values[ref_idx,...].astype(bool), axis=0)
    else:
        raise ValueError(f"Unknown platform {platform}.")
    
    total_px = data_tar[target_band].shape[0]*data_tar[target_band].shape[1]
    valid_frac = np.sum(valid_mask) / total_px * 100
    print(f"{valid_frac:.2f}% Valid Pixels")

    valid_mask = (valid_mask &
                  np.isfinite(data_tar[target_band].values) & 
                  np.all(np.isfinite(data_ref[target_band].isel(time=ref_idx).values), axis=0))
    
    return valid_mask, water_mask

def compute_center_mask(crop,data):

    # Center mask
    h_min_frac, h_max_frac, w_min_frac, w_max_frac = crop
    height, width = data.shape
    h_start = int(h_min_frac * height)
    h_end   = int(h_max_frac * height)
    w_start = int(w_min_frac * width)
    w_end   = int(w_max_frac * width)
    center_mask = np.zeros_like(data, dtype=bool)
    center_mask[h_start:h_end, w_start:w_end] = True

    return center_mask

def compute_clusters(data_tar, validland_mask, n_clusters, platform, target_band):

    # Add some extra features to help the clustering - NDVI, NDWI, NDSI.
    if platform != "sd":
        NDSI = ((data_tar["green"] - data_tar["swir1"]) / (data_tar["green"] + data_tar["swir1"] + 1e-6)).values
    else:
        NDSI = None

    NDVI = ((data_tar["nir"] - data_tar["red"]) / (data_tar["nir"] + data_tar["red"] + 1e-6)).values
    NDWI = ((data_tar["green"] - data_tar["nir"]) / (data_tar["green"] + data_tar["nir"] + 1e-6)).values
    if platform != "sd":
        band_vars = [v for v in data_tar.data_vars if v.startswith("B") and v != target_band]
    else:
        band_vars = [v for v in data_tar.data_vars if data_tar[v].ndim == 2 and v != target_band]
    bands = np.stack([data_tar[v].values for v in band_vars], axis=-1)  # (H, W, nB)
    
    feature_list = [bands, NDVI[..., None], NDWI[..., None]]
    if platform != "sd" and NDSI is not None:
        feature_list.append(NDSI[..., None])
    features = np.concatenate(feature_list, axis=-1)
    
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

    labels[mask] = MiniBatchKMeans(n_clusters=n_clusters, batch_size=1024, n_init=3, random_state=0).fit_predict(pix_reduced)
    clusters = labels.reshape(R, C)

    unique, counts = np.unique(labels[mask], return_counts=True)
    pixel_counts = dict(zip(unique, counts))

    max_count = max(pixel_counts.values())
    bar_width = 75  # Max width of bars in characters
    #print("Pixels per cluster:")
    ## Print a bar to the terminal for each cluster for a visual check.
    #for cluster, count in pixel_counts.items():
    #    bar_len = int((count / max_count) * bar_width)
    #    bar = "█" * bar_len
    #    print(f"Cluster {cluster}: {count:>8} | {bar}")
    
    return clusters

def compute_cloudmask(ind_ref, platform, data_tar, time_tar, time_ref, site_name, config, center_lat, center_lon):

    cld_dir = "/home/ucfacb0/ARIA/Level_1_Retrievals/data/cld"
    times = [time_tar]
    for ind in ind_ref:
        times.append(pd.Timestamp(time_ref[ind]))
    union, input_array = None, None
    for time in times:
        filename = os.path.join(cld_dir, f"cld_{site_name}_{time.strftime('%Y%m%d_%H%M')}.nc")
        if platform == "sn":
            if not os.path.exists(filename):

                with xr.open_dataset(f"/home/ucfacb0/ARIA/aria_data_download/download/{site_name}/sentinel-2-l1c-band-l2a-slc-{site_name}-60-tar.nc") as ds:
                    crs = ds.attrs["crs"]

                time_start = (time-timedelta(hours=0.5)).strftime("%Y-%m-%dT%H:%M:%SZ")
                time_end   = (time+timedelta(hours=0.5)).strftime("%Y-%m-%dT%H:%M:%SZ")
                bbox = BBox((center_lon - 0.08, center_lat - 0.06,
                             center_lon + 0.08, center_lat + 0.06), crs=CRS.WGS84)
                bbox_utm = bbox.transform(CRS(crs))

                request_res_dep_data = SentinelHubRequest(
                    data_folder='download', evalscript=Path(f"/home/ucfacb0/ARIA/aria_data_download/evalscripts/evalscript_10m.js").read_text(),
                    input_data=[SentinelHubRequest.input_data(
                                data_collection=DataCollection.SENTINEL2_L1C.define_from("s2l1c", service_url=config.sh_base_url),
                                time_interval=(time_start, time_end),
                                other_args={"id":"S2L1C"})],
                    responses=[SentinelHubRequest.output_response("default", MimeType.TIFF)],
                    bbox=bbox_utm, size=bbox_to_dimensions(bbox_utm, resolution=10), config=config,
                )   

                requested_data_10 = request_res_dep_data.get_data()[0]
                shape = np.shape(requested_data_10)

                # Convert the data_out dict into xarray Variables with appropriate dimensions and attributes.
                data_vars = {}
                for ind, var in enumerate(["red", "green", "red", "nir"]):
                    data_vars[var] = xr.Variable(
                        dims=("y", "x"),
                        data=requested_data_10[..., ind],
                        attrs={"long_name": 'Sentinel-2-l1c TOA reflectance', "units": "DN"}
                    )

                # --- Construct Dataset ---
                ds_10m = xr.Dataset(
                    data_vars=data_vars,
                    coords={
                        "x":         ("x",         np.arange(shape[1], dtype=np.int32), {"long_name": "pixel col index", "units": "1"}),
                        "y":         ("y",         np.arange(shape[0], dtype=np.int32), {"long_name": "pixel row index", "units": "1"}),
                    },
                    attrs={"Data_Attribution": f"Sentinel-2 data for Site: {site_name}",
                            "crs": crs})
                
                ds_10m.to_netcdf(filename)

            ds_10m = xr.open_dataset(filename)

            input_array = np.stack([
                ds_10m["red"],   # Red
                ds_10m["green"],   # Green
                ds_10m["nir"],   # NIR
            ]).astype(np.float32)
        
        elif platform == "ls":
            
            input_array = np.stack([
                data_tar["red"],   # Red
                data_tar["green"],   # Green
                data_tar["nir"],   # NIR
            ]).astype(np.float32)
            nodata_px = ~np.isfinite(input_array).all(axis=0)  # catches both inf and nan
            input_array = np.where(np.isfinite(input_array), input_array, 0.0).astype(np.float32)
        if input_array:
            cloud_mask = predict_from_array(input_array)[0] > 0
        else:
            return None
        union = cloud_mask if union is None else (union | cloud_mask)
    
    return union

def find_refs(mask, data_tar, data_ref, time_ref, number_refs):
    ''' Loops through all cloud-free scenes for the same year and selects the scenes most consistent with the target overpass.

    Only look over land pixels, and find references with a similar B04 band using RMSE.
    For sza and vza, use a absolute degree difference.'''       

    # Define the scenes. The launch scenes are already removed from the references when the data is loaded. 

    b04_tar  = data_tar["red"].values[mask]   # Valid land pixels only.
    b04_refs = data_ref["red"].values[:,mask] # Valid land pixels only.

    # Normalize so that a scene that's spectrally identical but globally darker now scores ~0.
    tar_norm  = b04_tar / b04_tar.mean()
    refs_norm = b04_refs / b04_refs.mean(axis=1, keepdims=True)
    rmse_br_all = np.sqrt(np.mean((tar_norm - refs_norm) ** 2, axis=1))
    data_len = data_ref["sza"].shape[0]

    tar_mean = data_tar["vza"].mean(skipna=True).item()
    ref_means = data_ref["vza"].mean(dim=["y","x"], skipna=True).values
    vza_diff_all = np.abs(tar_mean - ref_means)
    vza_diff_all = np.array(vza_diff_all)

    # Hard threshold for vza. 
    # The viewing angle difference is typically close to zero or around 3 degrees.
    vza_thresh = 2.0
    valid_idx = np.where(vza_diff_all < vza_thresh)[0]
    if len(valid_idx) == 0:
        ind_ref = []
    else:

        # Now find the best references of whats left.
        rmse_valid = rmse_br_all[valid_idx]
        best_local = np.argsort(rmse_valid)[:number_refs]
        ind_ref = valid_idx[best_local]
            
    # Quick check on reference locations:
    print('Ref indices: ', ind_ref, time_ref[ind_ref])
    print(f"VZA Filter: {np.sum(vza_diff_all < vza_thresh)}/{data_len}")

    return ind_ref