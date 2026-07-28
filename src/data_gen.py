"""
Generate a realistic ~150-stop Mumbai last-mile instance.

Stops are clustered around the 20 real towns in the original TSP dataset (used as
locality centres), each given a demand (parcels), a delivery time window, and a
service time. Node 0 is the Bhiwandi depot. Everything is documented and seeded.
"""
import os
import numpy as np
import pandas as pd
from config import get_instance, RAW_NODES, GEN_DIR


def _load_centres():
    if os.path.exists(RAW_NODES):
        df = pd.read_csv(RAW_NODES)
        return df[["Location", "Latitude", "Longitude"]].values.tolist()
    # fallback centres (Bhiwandi depot first) if raw file missing
    return [["Bhiwandi (Depot)", 19.2813, 73.0483], ["Thane", 19.2183, 72.9781],
            ["Kalyan", 19.2403, 73.1305], ["Vashi", 19.0771, 72.9986],
            ["Kurla", 19.0726, 72.8796], ["Panvel", 18.9894, 73.1175]]


def generate(inst=None, out_dir=GEN_DIR):
    inst = inst or get_instance()
    rng = np.random.default_rng(inst.seed)
    centres = _load_centres()
    depot = centres[0]
    town_centres = centres[1:]                      # exclude depot as a cluster centre

    def depot_minutes(lat, lon):
        """Approx earliest arrival from depot (Haversine x detour / speed)."""
        R = 6371.0
        p1, p2 = np.radians(depot[1]), np.radians(lat)
        dphi = np.radians(lat - depot[1]); dl = np.radians(lon - depot[2])
        a = np.sin(dphi/2)**2 + np.cos(p1)*np.cos(p2)*np.sin(dl/2)**2
        km = 2*R*np.arcsin(np.sqrt(a)) * inst.detour_factor
        return km / inst.avg_speed_kmph * 60.0

    rows = [dict(node=0, name=depot[0], lat=depot[1], lon=depot[2],
                 demand=0, tw_start=0, tw_end=inst.horizon_min, service=0)]
    for k in range(1, inst.n_stops + 1):
        c = town_centres[rng.integers(len(town_centres))]
        lat = c[1] + rng.normal(0, 0.012)           # ~1.3 km sd around the town
        lon = c[2] + rng.normal(0, 0.012)
        demand = int(np.clip(rng.gamma(3.0, 4.0), 1, inst.vehicle_capacity // 3))
        ea = depot_minutes(lat, lon)                # min feasible arrival from depot
        # ~40% unrestricted; rest a 90-210 min window that is REACHABLE from the depot
        if rng.random() < 0.40:
            e, l = 0, inst.horizon_min
        else:
            width = int(rng.integers(90, 210))
            lo = int(min(ea + 45, inst.horizon_min - 45))     # latest-time floor keeps it reachable
            l = int(rng.integers(lo, inst.horizon_min))
            e = max(0, l - width)
        rows.append(dict(node=k, name=f"{c[0]}_{k}", lat=round(lat, 5), lon=round(lon, 5),
                         demand=demand, tw_start=e, tw_end=l,
                         service=int(rng.integers(5, inst.service_min + 4))))
    df = pd.DataFrame(rows)

    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(os.path.join(out_dir, f"stops_{inst.name}.csv"), index=False)
    _write_assumptions(inst, df, out_dir)
    return df


def _write_assumptions(inst, df, out_dir):
    txt = f"""# Instance assumptions ({inst.name})

Depot: {df.iloc[0]['name']} (node 0). Customers: {len(df)-1}.
Fleet: {inst.n_vehicles} vehicles x capacity {inst.vehicle_capacity} parcels; shift {inst.horizon_min} min.

- Stops are SYNTHETIC but geographically realistic: sampled around the 20 real towns
  in the original dataset (used as locality centres) with ~1.3 km spread.
- Demand ~ Gamma(3, 4) parcels, clipped. Total demand = {int(df.demand.sum())} parcels
  (>= {int(np.ceil(df.demand.sum()/inst.vehicle_capacity))} vehicles needed by capacity alone).
- ~40% of stops accept any time; the rest have a 90-210 min delivery window inside the shift.
- Service time 5-{inst.service_min+3} min per stop.
- Travel times come from OSRM road data when available, else Haversine x {inst.detour_factor}
  detour at {inst.avg_speed_kmph} km/h (see distances.py).
"""
    with open(os.path.join(out_dir, f"assumptions_{inst.name}.md"), "w") as f:
        f.write(txt)


if __name__ == "__main__":
    df = generate()
    print(f"Generated instance with {len(df)-1} stops + depot.")
    print(df.head(8).to_string(index=False))
    print(f"\nTotal demand: {int(df.demand.sum())} parcels; "
          f"windowed stops: {int((df.tw_end < df.tw_end.max()).sum())}")
