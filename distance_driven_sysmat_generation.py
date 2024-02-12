### This function takes the projection geometry and the volume geometry as inputs, along with the angle (in radians) and returns the System Matrix in sparse format
#
# ***** Inputs *****
# proj_geom: A class contaning the projection geometry
# vol_geom : A class containing the volume geometry
# theta: A scalar denoting the angle in radians
#
# ***** Output *****
# W: System matrix of shape (#det_pix_num * #voxel_num) in the scipy.sparse format

def sysMat_pipeline(proj_geom, vol_geom, theta):
    total_det_pix_number = proj_geom.detector_rows * proj_geom.detector_cols * proj_geom.detector_cards  # Total number of pixels in the detector array
    total_vol_vox_number = vol_geom.Nx*vol_geom.Ny*vol_geom.Nz
    threshold = thresh_calc(proj_geom, vol_geom)
    voxel_centers = voxel_centers_calculation(vol_geom)
    source, det_cent, det_xy_unit, det_z_unit = source_detector_location(proj_geom, theta)
    index = int(det_cent.shape[0]/2)
    common_plane_center = det_cent[index,:]
    voxel_proj = lines_plane_intersection(source, common_plane_center, voxel_centers)
    pixel_centers = pixel_centers_calculation(proj_geom, det_cent, det_xy_unit, det_z_unit)
    pixel_proj = lines_plane_intersection(source, common_plane_center, pixel_centers)
    roi_voxel_proj, idx = ROI_volume_calculation(pixel_proj, voxel_proj)
    with parallel_backend('multiprocessing'):
        sysMat_view = Parallel(n_jobs=16)(delayed(create_sysMat)(pixel_proj, roi_voxel_proj, threshold, p, theta, idx) for p in range(total_det_pix_number))
    sysMat_info = [i for i in sysMat_view if i is not None]
    W = sparse_mat_generation(sysMat_info, total_det_pix_number, total_vol_vox_number)
    return W





###   Find the intersection points of a plane passing through detector_center with multiple 
###   lines connecting xray_source and voxel_points  
#
# Source:  https://stackoverflow.com/questions/5666222/3d-line-plane-intersection
#
# ***** Inputs *****
# xray_source :  A point on the line in 3-D coordinate, and also the normal to the plane from detector_center
# detector_center :  A point on the plane in 3-D coordinate
# point :  Another point on the line (same dimension as xray_source)
#
# ***** Output *****
# Psi :  Point of intersection on the plane in 3-D coordinate

def lines_plane_intersection(xray_source, detector_center, voxel_points):
    #Define plane
    planeNormal = detector_center   # xray_source - detector_center
    planePoint = detector_center   

    #Define ray
    rayDirection = xray_source - voxel_points
    rayPoint = voxel_points   # Any point along the ray

    ndotu = planeNormal.dot(rayDirection.transpose()) 

    w = rayPoint - planePoint
    si = (-planeNormal.dot(w.transpose()) / ndotu).reshape([-1,1])
    Psi = w + np.multiply(np.repeat(si,3,axis=1) , rayDirection) + planePoint
    return Psi





###   Find all the data_points which are within the 'threshold' Eucledian distance from the center_point   ###
#
# ***** Inputs *****
# center_point :  A vector of shape [1,n], where n is the dimension
# data_points :  An array of shape [p,n], where p is the number of data points
# threshold :  A scalar value
# idx: Actual voxel indecies of the data_points
#
# ***** outputs *****
# All the points in the data_points which are within the 'threshold' Eucledian distance {shape-> [m,n], where m<=p}
# neighbour_index :  Index of all those points in the array 'data_points' which are its neighbours

def dist_constraint_neighbourhood(center_points, data_points, threshold, idx):
    out = np.linalg.norm(data_points - center_points, ord=2, axis=1) 
    neighbour = np.where(out<=threshold)[0]
    if len(neighbour)>0:
        neighbour_index = idx[neighbour]
    else:
        neighbour_index = []
    return data_points[neighbour,:], neighbour_index





###   Calculate the threshold used in fixed_radius_neighbour algorithm
#
# ***** Inputs *****
# proj_geom : A class containing the projection geometry
# vol_geom : A class containing the volume geometry
#
# ***** Outputs *****
# threshold : A scalar value calculated based on the defined geometry

def thresh_calc(proj_geom, vol_geom):
    magnification = (proj_geom.distance_source_origin + proj_geom.distance_detector_origin)/proj_geom.distance_source_origin
    det_th = proj_geom.detector_pixel_size/ 2
    vox_th = max(vol_geom.axial_resolution, vol_geom.spatial_resolution) / 2
    thresh = np.sqrt((det_th**2) + (vox_th**2))
    threshold = magnification * thresh * 2
    return threshold





###   Generate the system matrix from the projections of voxels and detector pixels on a common plane
#
# ***** Inputs *****
# pixel_centers : A [m*3] vector containing all the 3D points of the detector pixel centers' projection on a common plane, where m is the total number of detector pixels
# voxel_proj : A [n*3] vector containing all the 3D points of the voxel centers' projection on that common plane, where n is the total number of voxels
# threshold : A positive scalar value defining the fixed-radius of neighbourhood selection
# p : An integer denoting the index of detector pixel
#
# ***** Outputs *****
# Returns the system matrix information in the most compact manner

def create_sysMat(pixel_centers, voxel_proj, threshold, p, view, idx):
    nearest_voxels, voxel_indices = dist_constraint_neighbourhood(pixel_centers[p,:], voxel_proj, threshold, idx)
    if len(voxel_indices)==0:
        return None
    else:
        omega = (threshold - np.sqrt(np.sum(np.square(nearest_voxels - pixel_centers[p,:]),1)))/threshold
        # omega = 1/(np.sqrt(np.sum(np.square(nearest_voxels - pixel_centers[p,:]),1)))
        return [view, p, voxel_indices, omega]
    
    
    
    

###   Voxel Centers Calculation by assuming the volume center as the origin of the coordinate system
#
# ***** Inputs *****
# vol_geom : A class containing the volume geometry
#
# ***** Outputs *****
# voxel_centers : A [n*3] vector containing the 3D location of the voxel centers

def voxel_centers_calculation(vol_geom):
    vol_xunit = np.array([vol_geom.spatial_resolution, 0, 0]).reshape([1,3])
    vol_yunit = np.array([0, vol_geom.spatial_resolution, 0]).reshape([1,3])
    vol_zunit = np.array([0, 0, vol_geom.axial_resolution]).reshape([1,3])
    vol_unit = np.concatenate((vol_xunit, vol_yunit, vol_zunit), axis=0)   # 1st_row: x,  2nd_row: y,  3rd_row: z
    
    total_voxel_number = vol_geom.Nx * vol_geom.Ny * vol_geom.Nz
    
    vrow = np.arange(0,vol_geom.Nx, 1)
    vcol = np.arange(0,vol_geom.Ny, 1)
    vdep = np.arange(0,vol_geom.Nz, 1)
    xx,yy,zz = np.array(np.meshgrid(vrow,vcol,vdep))
    xx = np.ravel(xx).reshape([-1,1])
    yy = np.ravel(yy).reshape([-1,1])
    zz = np.ravel(zz).reshape([-1,1])
    vcoord = np.concatenate((xx,yy,zz),axis=1)
    vcenter_shift = np.matmul(vcoord, vol_unit)
    corner_voxel_cent = np.array([-(vol_geom.vol_length/2)+(vol_geom.spatial_resolution/2), -(vol_geom.vol_width/2)+(vol_geom.spatial_resolution/2), -(vol_geom.vol_height/2)+(vol_geom.axial_resolution/2)]).reshape([1,3])   # left-bottom corner
    voxel_centers = corner_voxel_cent + vcenter_shift
    return voxel_centers





###   Detector Pixel Centers Calculation for each pixel by assuming the volume center as the origin of the coordinate system
#
# ***** Inputs *****
# proj_geom : A class containing the projection geometry
# det_xy_unit : A [3*1] vector defining the unit-vector of the detector pixel along the direction in x-y plane
# det_z_unit : A [3*1] vector defining the unit-vector of the detector pixel along the z-axis
#
# ***** Outputs *****
# pixel_centers : A [m*3] vector containing the 3D location of every detector pixel centers

def pixel_centers_calculation(proj_geom, det_cent, det_xy_unit, det_z_unit):
    pixel_corner = det_cent - (det_xy_unit * (proj_geom.detector_height/(2*proj_geom.detector_pixel_size))) - (det_z_unit * (proj_geom.detector_width/(2*proj_geom.detector_pixel_size)))   # size=[19*3]
    corner_pixel_center = pixel_corner + (det_xy_unit/proj_geom.detector_pixel_size) + (det_z_unit/proj_geom.detector_pixel_size)
    pcol = np.arange(0, proj_geom.detector_cols, 1)
    prow = np.arange(0, proj_geom.detector_rows, 1)
    dx,dy = np.array(np.meshgrid(pcol,prow))
    dx = np.ravel(dx).reshape([-1,1])
    dy = np.ravel(dy).reshape([-1,1])
    dcoord = np.concatenate((dx,dy),axis=1)   # size=[768*2]
    pixel_centers = np.zeros([proj_geom.detector_cards*proj_geom.detector_cols*proj_geom.detector_rows, 3])
    for card in range(proj_geom.detector_cards):
        xyUnit = det_xy_unit[card,:]
        zUnit = det_z_unit[card,:]
        det_unit = np.vstack((xyUnit,zUnit))
        dcenter_shift = np.matmul(dcoord, det_unit)
        pix_cent = corner_pixel_center[card,:] + dcenter_shift
        pixel_centers[card*proj_geom.detector_rows*proj_geom.detector_cols:(card+1)*proj_geom.detector_rows*proj_geom.detector_cols] = pix_cent

    return pixel_centers  





### Rotation Matrix Creation (Counter-clockwise)
#
# ***** Input *****
# angle: in radians
#
# ***** Output *****
# A 2*2 rotation matrix

def get_rot_matrix(angle):
#     angle = angle*np.pi/180
    # counter-clockwise
    return np.array([[ np.cos(angle), -np.sin(angle)],
                     [ np.sin(angle),  np.cos(angle)]
                    ])






###   Define the location of the source and detector center. 
###   Also define the unit vectors of the detector pixels along the xy-plane and z-axis.
#
# ***** Inputs *****
# proj_geom : A class containing the projection geometry
# theta : A scalar value defining the view angle of the source from origin in radians
#
# ***** Outputs *****
# source : A list of size (3,) defining the 3D location of the point-source
# det_cent : A list of size (3,) defining the 3D location of the detector center
# det_xy_unit : A vector of size [3,1] defining the unit vector along the xy-plane
# det_z_unit : A vector of size [3,1] defining the unit vector along the z-axis

def source_detector_location(proj_geom, theta):
    # Source location
    source = np.zeros([3,])
    source[0] = proj_geom.distance_source_origin * np.sin(theta)   # Source_x
    source[1] = -proj_geom.distance_source_origin * np.cos(theta)   # Source_y
    source[2] = 0   # Source_z
    # print("Source location for view ", view, " (angle ",theta*180/np.pi,") is ", source)

    # Detector centers' locations
    xxx = proj_geom.det_cent_x
    yyy = proj_geom.det_cent_y
    det_cent = np.zeros([proj_geom.detector_cards, 3])
    det_cent[:,2] = source[2]
    rot_matrix = get_rot_matrix(theta)
    det_pos = np.zeros([proj_geom.detector_cards,2])
    if proj_geom.detector_cards%2==0:
        det_pos[:,0] = np.concatenate([-xxx, np.flip(xxx)], axis=0)
        det_pos[:,1] = np.concatenate([yyy, np.flip(yyy)], axis=0)
    else:
        xx = xxx[:-1]
        yy = yyy[:-1]
        det_pos[:,0] = np.concatenate([-xx, np.array([xxx[-1]]), np.flip(xx)], axis=0)
        det_pos[:,1] = np.concatenate([yy, np.array([yyy[-1]]), np.flip(yy)], axis=0)
    det_cent[:,0:2] = np.transpose(rot_matrix @ det_pos.transpose())

    # Unit vectors along XY-plane
    det_xy_unit = np.zeros([proj_geom.detector_cards, 3])
    dx = source[0] - det_cent[:,0]   # source[x] - det_cent[x]
    dy = source[1] - det_cent[:,1]   # source[y] - det_cent[y]
    norm = np.sqrt(dx**2 + dy**2)
    dx = np.divide(dx,norm) * proj_geom.detector_pixel_size
    dy = np.divide(dy,norm) * proj_geom.detector_pixel_size
    dx = np.expand_dims(dx, axis=1)
    dy = np.expand_dims(dy, axis=1)
    det_xy_unit[:,0:2] = np.concatenate((-dy, dx), axis=1)


    # Unit vector along Z-axis
    det_z_unit = np.zeros([proj_geom.detector_cards, 3])
    det_z_unit[:,2] = proj_geom.detector_pixel_size   # det_rows_z

    return source, det_cent, det_xy_unit, det_z_unit





### This function converts the system matrix from struct format to sparse format
#
# ***** Inputs *****
# system_matrix : System matrix for a particular view in struct format   [theta, p, {v_i}, {w_i}]   (optimal format to store data)
# det_pix_num : Number of total detector pixels  (an integer value)
# helix_voxel_num : Total number of voxels in one helix (for reconstruction) or one view (for projection)   [vol_geom.Nx * vol_geom.Ny * vol_geom.Nz]
#
# ***** Outputs *****
# W : Sparse system matrix for a particular view of size [det_pix_num * helix_voxel_num]

def sparse_mat_generation(system_matrix, det_pix_num, helix_voxel_num):
    row_count = 0
    for i in range(len(system_matrix)):
        sysMat = system_matrix[i]
        angle = sysMat[0]   # in radians
        pixel_index = sysMat[1]   # index of vector format prjector
        voxel_index = sysMat[2]   # index of vector format volume
        weights = sysMat[3]   # weights corresponding to voxel index
        
        pixel_index_array = np.asarray(np.reshape(np.ones(weights.shape)*pixel_index,[-1,1]), dtype=int)
        voxel_index_array = np.asarray(voxel_index.reshape([-1,1]), dtype=int)
        weights_array = weights.reshape([-1,1])
        
        if row_count==0:
            triplet = np.concatenate((pixel_index_array, voxel_index_array, weights_array), axis=1)
        else:
            triplet_ = np.concatenate((pixel_index_array, voxel_index_array, weights_array), axis=1)
            triplet = np.concatenate((triplet, triplet_), axis=0)
        row_count += 1
        
    W = coo_matrix((triplet[:,2],( np.asarray(triplet[:,0],dtype=int), np.asarray(triplet[:,1],dtype=int) )),shape=(det_pix_num, helix_voxel_num)).tocsr()
    return W





# This function removes all those voxel projections which are not falling on the detector pixel at that certain projection
#
# ***** Inputs *****
#
# pixel_proj: A [m0*3] array defining the detector pixel's projection on the common plane
# voxel_proj: A [n*3] array depicting the voxel's projection on the the same common plane
#
# ***** Output *****
#
# roi_voxel_proj: A [n0*4] array depicting the voxel's projection on the the same common plane which are wihtin the z-range of pixel_proj, and the last column shows its index value
# idx: An array containing the index values of the voxels' falling within the ROI area

def ROI_volume_calculation(pixel_proj, voxel_proj, eps=proj_geom.detector_pixel_size):
    z_min = np.min(pixel_proj[:,2])
    z_max = np.max(pixel_proj[:,2])

    a = np.argwhere(voxel_proj[:,2]>=z_min-eps)
    b = np.argwhere(voxel_proj[:,2]<=z_max+eps)
    idx = np.intersect1d(a, b)
    roi_voxel_proj = voxel_proj[idx,:]
    return roi_voxel_proj, idx
