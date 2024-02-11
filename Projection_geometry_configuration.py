import numpy as np

# Define the projection Geometry
class proj_geom():
    distance_source_origin = 650  # [mm] Source to isocenter distance
    distance_detector_origin = 400   # [mm] Center of detector box to isocenter 
    detector_pixel_size = 2  # [mm]
    detector_height = 64 # [mm]
    detector_width = detector_pixel_size # [mm]
    proj_per_helix = 9
    detector_cards = 19   # number of detector card
    det_cent_x = [425.27, 406.73, 374.77, 333.81, 286.35, 234.10, 178.32, 120.08, 60.31, 0]   # x-coordinates of detector card centers (starting from left, considering the symmetricity)
    det_cent_y = [25.57, 115.77, 189.19, 249.57, 298.69, 337.56, 366.73, 386.55, 397.19, 398.69]   # y-coordinates of detector card centers (starting from left, considering the symmetricity)    
    
    det_cent_x = np.array(det_cent_x)
    det_cent_y = np.array(det_cent_y)
    detector_rows = int(detector_width/detector_pixel_size)  # Vertical size of detector [pixels], i.e. number of detector slices
    detector_cols = int(detector_height/detector_pixel_size)  # Horizontal size of detector [pixels]. i.e. number of detector pixels in one slice
    angles = np.linspace(0, 2*np.pi, num=proj_per_helix, endpoint=False)
