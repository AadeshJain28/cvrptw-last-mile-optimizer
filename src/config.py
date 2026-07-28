"""
Configuration for the last-mile CVRPTW project.

DEMO   : ~40 stops   -- fast; exact Gurobi check runs on a sub-instance.
FULL   : ~150 stops  -- the resume-grade instance.
Switch:  python main.py full   (also honours env var INSTANCE=full)
"""
import os, sys
from dataclasses import dataclass


@dataclass
class Instance:
    name: str
    n_stops: int            # customers (excl. depot)
    vehicle_capacity: int   # parcels per vehicle
    n_vehicles: int         # fleet size available
    horizon_min: int = 480          # 8-hour shift
    service_min: int = 8            # per-stop service time
    avg_speed_kmph: float = 24.0    # urban driving speed (for Haversine fallback time)
    detour_factor: float = 1.30     # road/straight-line ratio for Haversine fallback
    seed: int = 7


DEMO = Instance(name="demo", n_stops=40, vehicle_capacity=120, n_vehicles=6)
MID = Instance(name="mid", n_stops=95, vehicle_capacity=140, n_vehicles=12)   # <=100 -> fits public OSRM
FULL = Instance(name="full", n_stops=150, vehicle_capacity=140, n_vehicles=16)
_BY_NAME = {"demo": DEMO, "mid": MID, "full": FULL}


def get_instance() -> Instance:
    val = os.environ.get("INSTANCE", "").lower()
    if val not in _BY_NAME:
        val = "demo"
        for a in sys.argv[1:]:
            if a.lower() in _BY_NAME:
                val = a.lower(); break
    return _BY_NAME[val]


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_NODES = os.path.join(ROOT, "data", "raw", "bhiwandi_tsp_nodes.csv")
GEN_DIR = os.path.join(ROOT, "data", "generated")
RESULTS_DIR = os.path.join(ROOT, "results")
FIG_DIR = os.path.join(RESULTS_DIR, "figures")
OSRM_URL = os.environ.get("OSRM_URL", "https://router.project-osrm.org")
