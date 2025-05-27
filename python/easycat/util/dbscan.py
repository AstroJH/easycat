from sklearn.cluster import DBSCAN
from astropy.coordinates import SkyCoord
from astropy.units import Quantity
import numpy as np
import pandas as pd
import numba


def filter_dbscan(lcurve:pd.DataFrame, pos_ref:SkyCoord, radius:Quantity, min_neighbors:int=4, min_cluster_size=15):
    clusters, _ = dbscan(lcurve, pos_ref, radius, min_neighbors)

    for c in clusters:
        if len(c) >= min_cluster_size:
            c.reset_index(drop=True, inplace=True)
            return c
    
    return lcurve.iloc[0:0]


def calc_distance(positions:SkyCoord):
    # N = len(positions)
    # distance = np.empty(shape=(N, N), dtype=np.float64)

    # for i in range(0, N):
    #     for j in range(i, N):
    #         pos_i:SkyCoord = positions[i]
    #         pos_j:SkyCoord = positions[j]

    #         if i != j:
    #             sep = pos_i.separation(pos_j).to_value("arcsec")
    #         else:
    #             sep = 0

    #         distance[i, j] = sep
    #         distance[j, i] = sep

    pos_ra_deg  = positions.ra.to("deg").value
    pos_dec_deg = positions.dec.to("deg").value
    distance = calc_distance_njit(pos_ra_deg, pos_dec_deg)
    
    return distance


@numba.njit
def calc_separation_njit(ra1, dec1, ra2, dec2):
    rad_ra1  = np.deg2rad(ra1)
    rad_ra2  = np.deg2rad(ra2)
    rad_dec1 = np.deg2rad(dec1)
    rad_dec2 = np.deg2rad(dec2)

    rad_theta = np.sqrt(
        (rad_ra1-rad_ra2)**2 * np.cos(rad_dec1)**2
            +
        (rad_dec1-rad_dec2)**2
    )

    return np.rad2deg(rad_theta)*3600


@numba.njit
def calc_distance_njit(pos_ra_deg, pos_dec_deg):
    N = len(pos_ra_deg)
    distance = np.empty(shape=(N, N), dtype=np.float64)

    for i in range(0, N):
        for j in range(i, N):
            if i != j:
                sep = calc_separation_njit(
                    pos_ra_deg[i], pos_dec_deg[i],
                    pos_ra_deg[j], pos_dec_deg[j]
                )
            else:
                sep = 0

            distance[i, j] = sep
            distance[j, i] = sep
    
    return distance


def dbscan(lcurve:pd.DataFrame, pos_ref:SkyCoord, radius:Quantity, min_neighbors:int=4):
    raj2000 = lcurve["raj2000"]
    dej2000 = lcurve["dej2000"]

    positions = SkyCoord(raj2000, dej2000, unit="deg", frame="fk5")
    distance = calc_distance(positions)
    model = DBSCAN(eps=radius.to_value("arcsec"), min_samples=min_neighbors, metric="precomputed")

    result = model.fit(distance)

    return handle_dbscan_result(lcurve, pos_ref, result)



def handle_dbscan_result(lcurve:pd.DataFrame, pos_ref:SkyCoord, result:DBSCAN):
    clusters:list[pd.DataFrame] = []
    noise = None
    
    labels_ = result.labels_
    labels = np.unique(labels_)

    for label in labels:
        data = lcurve[labels_==label]
        
        if label == -1:
            noise = data
        else:
            clusters.append(data)
    
    def sep(cluster:pd.DataFrame):
        raj2000 = cluster["raj2000"]
        dej2000 = cluster["dej2000"]
        ra_center = np.median(raj2000)
        dec_center = np.median(dej2000)

        center = SkyCoord(ra_center, dec_center, unit="deg", frame="fk5")

        return center.separation(pos_ref)
    
    clusters.sort(key=sep)

    return clusters, noise
    