"""
Distance / travel-time matrices.

Primary: OSRM road network (real driving distance + duration), via the public
demo server or a self-hosted OSRM_URL. Result is cached to disk so it's fetched
once. Fallback: Haversine x detour factor at a fixed urban speed -- always works
offline (used automatically if `requests`/OSRM are unavailable).

get_matrices(df, inst) -> (dist_km[N,N], time_min[N,N], source_str)
"""
import os
import numpy as np
from config import GEN_DIR, OSRM_URL

try:
    import requests
    _HAS_REQ = True
except Exception:
    _HAS_REQ = False


def haversine_matrix(lats, lons):
    R = 6371.0
    la = np.radians(lats)[:, None]; lo = np.radians(lons)[:, None]
    dla = la - la.T; dlo = lo - lo.T
    a = np.sin(dla/2)**2 + np.cos(la)*np.cos(la.T)*np.sin(dlo/2)**2
    return 2 * R * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def _osrm_matrix(lats, lons):
    """Query OSRM /table for durations (s) and distances (m). Raises on failure."""
    coords = ";".join(f"{lo},{la}" for la, lo in zip(lats, lons))
    url = (f"{OSRM_URL}/table/v1/driving/{coords}"
           f"?annotations=duration,distance")
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    j = r.json()
    if j.get("code") != "Ok":
        raise RuntimeError(f"OSRM: {j.get('code')}")
    dur = np.array(j["durations"], float) / 60.0        # minutes
    dist = np.array(j["distances"], float) / 1000.0     # km
    return dist, dur


def get_matrices(df, inst, use_osrm=True, cache=True):
    lats = df["lat"].to_numpy(); lons = df["lon"].to_numpy()
    cache_f = os.path.join(GEN_DIR, f"matrix_{inst.name}.npz")
    if cache and os.path.exists(cache_f):
        z = np.load(cache_f, allow_pickle=True)
        return z["dist"], z["time"], str(z["source"])

    source = None
    if use_osrm and _HAS_REQ:
        try:
            dist, time = _osrm_matrix(lats, lons)
            source = "OSRM road network"
        except Exception as e:
            reason = type(e).__name__
            print(f"[distances] OSRM unavailable ({reason}); using Haversine fallback.")
    if source is None:
        dist = haversine_matrix(lats, lons) * inst.detour_factor
        time = dist / inst.avg_speed_kmph * 60.0        # minutes
        source = f"Haversine x{inst.detour_factor} @ {inst.avg_speed_kmph}km/h"

    np.fill_diagonal(dist, 0.0); np.fill_diagonal(time, 0.0)
    if cache:
        os.makedirs(GEN_DIR, exist_ok=True)
        np.savez(cache_f, dist=dist, time=time, source=source)
    return dist, time, source


if __name__ == "__main__":
    from config import get_instance
    import data_gen
    inst = get_instance()
    df = data_gen.generate(inst)
    dist, time, src = get_matrices(df, inst)
    print(f"Matrix source: {src}  shape={dist.shape}")
    print(f"Depot->stop1: {dist[0,1]:.2f} km, {time[0,1]:.1f} min")
    print(f"Mean pairwise dist: {dist[dist>0].mean():.2f} km; max: {dist.max():.2f} km")
