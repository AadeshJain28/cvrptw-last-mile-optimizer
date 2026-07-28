"""Per-stop ETA table for a solution (vehicle, sequence, arrival, window, on-time)."""
import pandas as pd
from cvrptw import route_eta_table


def eta_table(prob, routes):
    rows = []
    for v, route in enumerate(routes):
        for seq, rec in enumerate(route_eta_table(prob, route)):
            rec.update(vehicle=v, seq=seq)
            rows.append(rec)
    cols = ["vehicle", "seq", "node", "name", "eta_min", "tw_start", "tw_end", "on_time"]
    return pd.DataFrame(rows)[cols]
