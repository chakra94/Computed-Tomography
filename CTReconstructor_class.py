#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 12 10:46:03 2025

@author: ankitc
"""

import os
import time
import numpy as np
import logging
from joblib import Parallel, delayed
from scipy.interpolate import interp1d
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)

class CTReconstructor:
    def __init__(self, geom, n_jobs=-1, memmap_path='/tmp/recon_vol.dat'):
        self.geom = geom
        self.n_jobs = n_jobs
        self.memmap_path = memmap_path
        
        
        
    def filtering_3D_cone(self, data, gamma, alpha, L, sample_spacing=1, smoothing=True):   # cone beam projection
        """ 3D cone beam data filtering 
        
        Parameters:
            - data (3D array): [num_frames, num_slices, num_detector]
            - gamma (2D array): [num_slices, num_detector] Denotes the fan angles
            - alpha (2D array): [num_slices, num_detector] Denotes the cone angles
            - L (2D array): [num_slices, num_detector] Distance of pixel center to source, denotes the scaling factor for each slice-detector pair
            - sample_spacing (float): Optional. Sampling distance in the detector direction (e.g., 2.5 mm), default is 1.
            - smoothing (bool): Optional. A flag, denotes whether to apply smoothing filter for superior reconstruction or not.
        
        Returns:
            - Filtered projections (3D array): [num_frames, num_slices, num_detector]
        """
        num_views, num_slices, num_detector = data.shape
        
        # Compute the scaling factor for each pixel as :   (L^2) * cos(gamma) * cos(alpha)
        scaling_factor = (L**2) * (np.cos(gamma) / np.cos(alpha))  # Shape: [num_slices, num_detector]
        
        # Apply scaling to the projection data
        scaled_data = data * scaling_factor[None,:,:]
        # scaled_data = np.zeros_like(data)  # [num_frames, num_slices, num_detector]
        # for frame in range(num_views):  # Loop over views
        #     scaled_data[frame, :, :] = data[frame, :, :] * scaling_factor  # Element-wise scaling
        
        # Frequency array for filtering
        fft_size = 2 ** int(np.ceil(np.log2(2 * num_detector - 1)))  # Zero-padding for FFT to avoid aliasing
        freq = np.fft.fftfreq(fft_size, d=sample_spacing)
        
        # Perform FFT along the detector direction (gamma axis)
        projections_fft = np.fft.fft(scaled_data, n=fft_size, axis=2, norm=None)
        
        # Apply the ramp filter (|freq| in frequency domain)
        filtered_fft = projections_fft * np.abs(freq)
        
        # Optional Hamming window for smoother reconstruction
        if smoothing:
            hamming_window = np.fft.fftshift(np.hamming(fft_size))
            filtered_fft = filtered_fft * hamming_window
        
        # Perform IFFT along the detector direction and consider the real part as filtered projection
        filtered_projections = np.fft.ifft(filtered_fft, n=fft_size, axis=2, norm=None).real
        
        # Return the filtered projection cropped to original detector size
        return filtered_projections[:, :, 0:num_detector]




    def pixel_driven_backprojection_3d(self, projections, theta, source_coord, detector_pixel_coords):
        """
        Reconstructs a 3D volume from cone-beam CT projections using a pixel-driven backprojection algorithm.
        
        Args:
            projections (numpy.ndarray): Filtered projection data (num_frames, num_slices, num_detectors).
            theta (numpy.ndarray): Projection angles in radians (num_frames,).
            source_coord (numpy.ndarray): 3D coordinate of the X-ray source (3,).
            detector_pixel_coords (numpy.ndarray): Detector pixel coordinates (num_slices, num_detectors, 3).

        Returns:
            numpy.ndarray or np.memmap: Reconstructed 3D volume.
        """
        geom = self.geom
        img_size = geom.vol_size_spatial
        voxel_resolution = geom.spatial_resolution
        z_resolution = geom.axial_resolution
        image_center = geom.vol_cent
        img_px = int(np.round(img_size / voxel_resolution))

        num_frames, num_input_slices, num_detectors = projections.shape
        num_output_slices = int(np.round((num_input_slices * geom.detector_pixel_size[1]) / z_resolution))

        # Generate voxel grid
        center = img_px / 2
        x_c_px = image_center[0] / voxel_resolution
        y_c_px = image_center[1] / voxel_resolution
        x = np.linspace(-center + x_c_px + voxel_resolution / 2, center + x_c_px - voxel_resolution / 2, img_px)
        y = np.linspace(-center + y_c_px + voxel_resolution / 2, center + y_c_px - voxel_resolution / 2, img_px)
        X, Y = np.meshgrid(x, y)

        logging.info("Interpolating projections and detector geometry...")
        detector_z_extent = num_input_slices * geom.detector_pixel_size[1]
        z_input = np.linspace(0, detector_z_extent, num_input_slices)
        z_output = np.linspace(0, detector_z_extent, num_output_slices)

        proj_interp_func = interp1d(z_input, projections, axis=1, kind='linear', bounds_error=False, fill_value='extrapolate')
        projections = proj_interp_func(z_output)

        det_interp_func = interp1d(z_input, detector_pixel_coords, axis=0, kind='linear', bounds_error=False, fill_value='extrapolate')
        detector_pixel_coords = det_interp_func(z_output)
        logging.info("Interpolation complete.")

        if os.path.exists(self.memmap_path):
            os.remove(self.memmap_path)
        reconstructed_volume = np.memmap(self.memmap_path, dtype='float32', mode='w+', shape=(num_output_slices, img_px, img_px))

        def process_slice_batch(batch_indices):
            results = []
            for slice_idx in batch_indices:
                start = time.time()
                det_coords = detector_pixel_coords[slice_idx]
                det_x = det_coords[:, 0]

                if np.any(np.isnan(det_x)) or np.allclose(det_x, det_x[0]) or np.any(np.diff(np.sort(det_x)) == 0):
                    results.append((slice_idx, np.zeros((img_px, img_px), dtype='float32')))
                    continue

                sorted_idx = np.argsort(det_x)
                det_x_sorted = det_x[sorted_idx]
                idx_sorted = np.arange(len(det_x))[sorted_idx]
                interp_func = interp1d(det_x_sorted, idx_sorted, bounds_error=False, fill_value=(0, len(det_x) - 1))

                source_to_det_dist = np.mean(np.linalg.norm(det_coords - source_coord, axis=1))

                slice_volume = np.zeros((img_px, img_px), dtype='float32')

                for i, angle in enumerate(theta):
                    x_rot = X * np.cos(angle) + Y * np.sin(angle)
                    y_rot = -X * np.sin(angle) + Y * np.cos(angle)
                    z_val = z_output[slice_idx] + geom.vol_cent[2] - (detector_z_extent / 2)
                    voxel_coords = np.stack([x_rot * voxel_resolution, y_rot * voxel_resolution, np.full_like(X, z_val) * z_resolution], axis=-1)

                    source_to_voxel_dist = np.linalg.norm(voxel_coords - source_coord, axis=-1)
                    source_to_voxel_dist = np.where(source_to_voxel_dist == 0, 1e-6, source_to_voxel_dist)

                    magnification = source_to_det_dist / source_to_voxel_dist
                    magnification = np.clip(magnification, 0, np.inf)

                    x_proj = x_rot * magnification
                    x_proj = np.clip(x_proj, det_x_sorted.min(), det_x_sorted.max())
                    detector_idx = interp_func(x_proj * geom.spatial_resolution * magnification).astype(int)
                    detector_idx = np.clip(detector_idx, 0, len(det_x) - 1)

                    proj_values = projections[i, slice_idx, detector_idx] * (magnification ** 2)
                    slice_volume += proj_values.astype('float32')

                slice_volume /= len(theta)
                logging.info(f"Slice {slice_idx} processed in {time.time() - start:.2f}s")
                results.append((slice_idx, slice_volume))
            return results

        batch_size = 8
        all_batches = [list(range(i, min(i + batch_size, num_output_slices))) for i in range(0, num_output_slices, batch_size)]

        logging.info("Starting parallel reconstruction...")
        results = Parallel(n_jobs=self.n_jobs, backend='loky')(
            delayed(process_slice_batch)(batch) for batch in tqdm(all_batches)
        )

        for batch in results:
            for slice_idx, slice_volume in batch:
                reconstructed_volume[slice_idx] = slice_volume

        logging.info("Reconstruction complete.")
        return np.array(reconstructed_volume)




# Example usage:
# geom = YourGeometryClass(...)  # Define this class with attributes like vol_size_spatial, spatial_resolution, etc.
# recon = CTReconstructor(geom)
# filtered_projection = recon.filtering_3D_cone(projections, gamma_angles, alpha_angles, source_to_detector_distances, sample_spacing=inPlane_detector_pixel_size+dead_cell_region, smoothing=True)
# volume = recon.pixel_driven_backprojection_3d(projections, theta, source_coord, detector_pixel_coords)
