############################################################################
# Flooding hotspot algorithm.
############################################################################

# Library imports.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from scipy.optimize import curve_fit
from pathlib import Path
from datetime import datetime, timedelta
import geopandas as gpd
from shapely import Point
import rasterio

# Package imports.
from rasopt.Utilities import utils, alter_files

def find_hotspots(breach_names, breach_dfs, start_dts, timesteps, levee_fp, threshold_depth):
    """
    Find hotspots in floodplain that share many overlapping breach scenarios.
    :param breach_names: Names of breach scenarios.
    :param breach_dfs: Data frames of breach scenario depths.
    :param start_dts: Breach start datetimes.
    :param timesteps: Timesteps at which to extract hotspots.
    :param levee_fp: File path to levee geometry.
    :param threshold_depth: Inundation threshold depth.
    :return: Data frame of hotspots for each time step.
    """

    # Get latitude and longitude from the breach data frames.
    lats = breach_dfs[0].lat.to_numpy()
    lons = breach_dfs[0].lon.to_numpy()

    # Align timesteps.
    aligned_depth_arrays, start_datetime = align_timesteps(breach_names, breach_dfs, start_dts)

    # Create binary inundation arrays with 1s where depth is above threshold value.
    inun_arrays = []
    for ar in aligned_depth_arrays:
        inun_ar = ar > threshold_depth
        inun_arrays.append(inun_ar)

    # Compute location breach sets.
    hotspots = []
    for end_idx in timesteps:
        # Breach scenarios that have not yet been accounted for by a sensor location.
        unaccounted_breaches = breach_names.copy()
        while len(unaccounted_breaches) > 0:
            S = np.zeros((inun_arrays[0].shape[0], len(inun_arrays)))
            for i, ar in enumerate(inun_arrays):
                # Only process breaches that are yet unaccounted for.
                if breach_names[i] in unaccounted_breaches:
                    # Find the rows that are inundated within the period.
                    any_inun = np.any(ar[:, :end_idx], axis=1).astype(int)
                    S[:, i] = any_inun

            # If no inundated points are found, skip the timestep.
            if np.all(S == 0) and len(unaccounted_breaches) == len(breach_names):
                print(f'No inundated locations for timestep {end_idx}')
                break
            # If some breaches have been accounted for, but no more inundated locations are found, end loop.
            elif np.all(S == 0) and len(unaccounted_breaches) < len(breach_names):
                print(f'Breaches {unaccounted_breaches} have not produced inundated locations for timestep {end_idx}')
                break

            # Maximum cardinality of elements of S.
            S_sums = np.sum(S, axis=1)
            M = np.max(S_sums)

            # Row indices of elements of S that share maximum cardinality.
            S_max_card_idx = np.where(S_sums == M)

            # Locations with maximum cardinality.
            S_max_card = S[S_max_card_idx]

            # Latitude and longitude of points with maximum cardinality.
            lat_max_card = lats[S_max_card_idx]
            lon_max_card = lons[S_max_card_idx]

            # Furthest point from the levee.
            furthest_coords, furthest_index = furthest_point_from_levee(levee_fp, lat_max_card, lon_max_card)

            # Breach scenarios accounted for by the furthest point.
            furthest_point_breaches = S_max_card[furthest_index,:]
            selected_breach_names = np.array(breach_names)[furthest_point_breaches == 1]

            # Remove breach names that have already been accounted for.
            [unaccounted_breaches.remove(i) for i in selected_breach_names]

            # Actual datetime for time step.
            dt = start_datetime + timedelta(minutes=10 * end_idx)

            # Add hotspot.
            hotspot = [end_idx, dt, furthest_coords[0], furthest_coords[1], *list(furthest_point_breaches)]
            hotspots.append(hotspot)

    # Create hotspot data frame.
    hotspot_columns = ['timestep', 'timestamp', 'longitude', 'latitude', *breach_names]
    hotspot_df = pd.DataFrame(hotspots, columns=hotspot_columns)

    return hotspot_df


def align_timesteps(breach_names, breach_dfs, start_dts):
    # Get timesteps for each dataframe.
    timesteps = {}
    for i, breach_name in enumerate(breach_names):
        ts = [start_dts[i] + timedelta(minutes=j * 10) for j in range(breach_dfs[i].shape[1])]
        timesteps[breach_name] = ts

    # Make arrays of depth dfs and align datetimes.
    start_datetime = np.max(start_dts) # Latest datetime.
    aligned_depth_arrays = []
    for i, breach_name in enumerate(breach_names):
        # Get the start index for each dataframe.
        ts = timesteps[breach_name]
        start_index = ts.index(start_datetime)

        # Create aligned depth array.
        unaligned_array = breach_dfs[i].to_numpy()
        aligned_array = unaligned_array[:, start_index + 2:]
        aligned_depth_arrays.append(aligned_array)

    return aligned_depth_arrays, start_datetime


def furthest_point_from_levee(levee_fp, lat_max_card, lon_max_card):
    # Load in the levee shapefile.
    levee_df = gpd.read_file(levee_fp)

    # Convert them to a list of Point objects
    points = [Point(lon, lat) for lat, lon in zip(lat_max_card, lon_max_card)]

    # Assuming linestring_geometry is your linestring geometry in GeoPandas
    # Convert it to a Shapely LineString object
    line = levee_df.at[0, 'geometry']

    # Calculate the distance from each point to the linestring
    distances = [point.distance(line) for point in points]

    # Find the index of the point with the maximum distance
    furthest_point_index = distances.index(max(distances))

    # Get the point with the maximum distance
    furthest_point = points[furthest_point_index]
    furthest_point_coords = (furthest_point.x, furthest_point.y)

    return furthest_point_coords, furthest_point_index