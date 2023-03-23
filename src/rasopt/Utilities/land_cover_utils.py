

# Library imports.
import numpy as np
import h5py
import geojson
import os
import osgeo
from osgeo import gdal
import rasterio
from rasterio.enums import Resampling
import subprocess

# Local imports.
from rasopt.Utilities import utils

# Functions

def percent_points_on_land_cover(land_cover_raster, land_cover_dict, cell_ids, cell_coords):
    """
    Returns the percent of the points that fall on each land cover type.
    E.g., 30% of the points are on LC 1, 70% of the points on LC 2. The percentages sum to 100%
    :param land_cover_raster:
    :param land_cover_dict: 
    :param cell_ids:
    :param cell_coords:
    :return:
    """


def percent_land_covers(land_cover_raster, land_cover_dict, cell_ids, cell_coords):
    """
    Returns the percent of each land cover type that the points cover.
    E.g., LC 1 is 10% covered by the points, LC 2 is 15 % covered. The percentages do not necessarily sum to 100%.
    # TODO: Figure out how to calculate coverage by a group of points. Convex hull?
    :param land_cover_raster:
    :param land_cover_dict:
    :param cell_ids:
    :param cell_coords:
    :return:
    """

def terrain_water_depth(dem_fp, plan_hdf_fp, cell_fp_idx_path, facepoint_coord_path, depth_path, cell_coord_path,
                                      timestep, geo_dir, out_fname_prefix='', nodata=-999.0, cleanup_rasters=False,
                        resample=1.0):
    """
    Computes the water depth at each cell in the terrain file.
    :param dem_fp: Path to the DEM file.
    :param plan_hdf_fp: Path to the plan hdf file.
    :return: Array with the same shape as the DEM containing the depth at each cell.
    """
    # Create the depth geojson.
    depth_geojson = utils.depth_geojson(plan_hdf_fp, cell_fp_idx_path, facepoint_coord_path, depth_path, cell_coord_path,
                                      timestep, elevation=True)

    # Save depth geojson.
    gjson_fp = os.path.join(geo_dir, 'mesh_gjson.geojson')
    with open(gjson_fp, 'w') as gf:
        geojson.dump(depth_geojson, gf)

    # Open the dem raster and get the extent and resolution.
    with rasterio.open(dem_fp, 'r') as ds:
        bounds = ds.bounds
        transform = ds.transform
        cell_width_X = transform[0]
        cell_width_Y = -transform[4]
        x_min = bounds.left
        x_max = bounds.right
        y_min = bounds.bottom
        y_max = bounds.top
        dem_array = ds.read(1)
        dem_crs = ds.crs

        # Resampling the DEM raster.
        if resample != 1:
            # resample data to target shape
            dem_array = ds.read(
                out_shape=(
                    ds.count,
                    int(ds.height * resample),
                    int(ds.width * resample)
                ),
                resampling=Resampling.bilinear
            )

            # scale image transform
            transform = ds.transform * ds.transform.scale(
                (ds.width / dem_array.shape[-1]),
                (ds.height / dem_array.shape[-2])
            )

            # Reduce array to 2D.
            dem_array = np.squeeze(dem_array)

            cell_width_X = transform[0]
            cell_width_Y = -transform[4]

    # Raster path.
    print('DEM SHAPE', dem_array.shape)
    raster_fp = os.path.join(geo_dir, f'mesh_raster_{timestep}.tif')

    # Rasterize the geojson feature collection.
    layer_name = 'mesh_gjson'
    property_name = 'Depth'

    gdal_command = ("gdal_rasterize -at -l {} -a {} -tr {} {} -a_nodata {} -te {} {} {} {} -ot Float32 "
                    "-of GTiff {} {}".format(layer_name, property_name, cell_width_X,
                                             cell_width_Y, nodata, x_min, y_min, x_max,
                                             y_max, gjson_fp, raster_fp))
    print(gdal_command)
    subprocess.call(gdal_command, shell=True)

    # Open back up the raster and read out the array.
    with rasterio.open(raster_fp) as dataset:
        raster_array = dataset.read(1)

    # Open the DEM raster and replace the raster with the depth array.
    depth_array = raster_array - dem_array
    depth_array[depth_array < 0] = 0
    depth_array[depth_array > 100] = 0 # Handle no data value from dem_array causing extremely large values.

    # Raster path.
    depth_raster_fp = os.path.join(geo_dir, out_fname_prefix + f'depth_raster_{timestep}.tif')

    with rasterio.open(
        depth_raster_fp,
        'w',
        driver='GTiff',
        height=depth_array.shape[0],
        width=depth_array.shape[1],
        count=1,
        dtype=depth_array.dtype,
        crs=dem_crs,
        transform=transform
    ) as dst:
        dst.write(depth_array, indexes=1)

    print(raster_array.shape)

    # Delete rasters to save space.
    if cleanup_rasters is True:
        os.remove(depth_raster_fp)
        os.remove(raster_fp)

    return depth_array, depth_raster_fp



if __name__ == '__main__':
    dem_fp = r"C:\Users\ay434\Documents\10_min_Runs\Secchia_flood_2014_10min_Orig_GT\DTM_1m.tif"
    plan_hdf_fp = r"C:\Users\ay434\Documents\10_min_Runs\Secchia_flood_2014_10min_Orig_GT\Secchia_Panaro.p23.hdf"
    # Cell coordinate path.
    cell_coord_path = 'Geometry/2D Flow Areas/Secchia_Panaro/Cells Center Coordinate'

    # Water depths path.
    depth_path = ('Results/Unsteady/Output/Output Blocks/Base Output/Unsteady Time Series/'
                  '2D Flow Areas/Secchia_Panaro/Depth')

    # Manning's n calibration table path.
    cal_table_path = "Geometry/Land Cover (Manning's n)/Calibration Table"

    # Cell facepoint index path.
    cell_facepoint_idx_path = 'Geometry/2D Flow Areas/Secchia_Panaro/Cells FacePoint Indexes'

    # Facepoint coordinate path.
    facepoint_coord_path = 'Geometry/2D Flow Areas/Secchia_Panaro/FacePoints Coordinate'

    timestep = 120

    raster, raster_fp = terrain_water_depth_memory(dem_fp, plan_hdf_fp, cell_facepoint_idx_path, facepoint_coord_path, depth_path, cell_coord_path,
                                      timestep, out_fname_prefix='HEC_GT')

    print(raster_fp)