# Define the volume geometry for a single sample taken from a cetrain angle at a cetrain time. 
# Volumes depth along the temporal direction (along the belt) is constrained by the width of the detector along z-axis (i.e. depends on pixel sizes and number of detector slices)

class vol_geom():
    vol_length = 256   # [mm]  along x-axis (tunnel_width)
    vol_width = 256   # [mm]  along y-axis  (tunnel_height)
    vol_height = proj_geom.detector_width   # [mm]  along z-axis  (width of the detector-slice)
    
    spatial_resolution = 1   # [mm]  side length of each voxel in the x-y plane
    temporal_resolution = 1   # [mm]  side length of each voxel along z-axis  (reconstruction quality constrained by detector slices)
    
    # Number of voxels in the reconstructed volume along x-,y- and z-axes respectively
    Nx = int(vol_length/spatial_resolution)
    Ny = int(vol_width/spatial_resolution)
    Nz = int(vol_height/axial_resolution)
