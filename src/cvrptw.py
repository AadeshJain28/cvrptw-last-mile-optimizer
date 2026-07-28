"""
CVRPTW core: instance holder, route evaluation (distance + time-window ETAs +
capacity), feasibility, and the penalized cost used by every metaheuristic.

A Solution is a list of routes; each route is a list of customer indices (1..n),
implicitly starting and ending at the depot (node 0). Empty routes = unused vehicles.
"""
from dataclasses import dataclass
import numpy as np

VEHICLE_COST = 2000.0     # soft cost per vehicle used (minimize fleet first)
PEN_CAP = 500.0           # penalty per unit of capacity overflow
PEN_TW = 200.0            # penalty per minute of customer-window lateness (HARD constraint)
PEN_DUR = 30.0            # penalty per minute a route exceeds the shift (SOFT target)


@dataclass
class CVRPTW:
    dist: np.ndarray       # [N,N] km  (N = n_customers+1, node 0 = depot)
    time: np.ndarray       # [N,N] min
    demand: np.ndarray     # [N]
    e: np.ndarray          # [N] earliest service start
    l: np.ndarray          # [N] latest service start
    service: np.ndarray    # [N] service duration
    Q: int                 # vehicle capacity
    K: int                 # fleet size
    horizon: int           # shift length (min)
    names: list = None

    @property
    def n(self):
        return len(self.demand) - 1     # number of customers

    @classmethod
    def from_frames(cls, df, dist, time, inst):
        return cls(dist=dist, time=time,
                   demand=df["demand"].to_numpy(),
                   e=df["tw_start"].to_numpy(), l=df["tw_end"].to_numpy(),
                   service=df["service"].to_numpy(),
                   Q=inst.vehicle_capacity, K=inst.n_vehicles,
                   horizon=inst.horizon_min, names=df["name"].tolist())


def route_metrics(prob: CVRPTW, route):
    """Return (distance, cust_lateness, arrival_times, load, duration) for one route.
    cust_lateness = sum of customer time-window violations (the HARD constraint);
    duration = clock at depot return (compared to the shift as a SOFT target)."""
    if not route:
        return 0.0, 0.0, [], 0, 0.0
    d = prob.dist; t = prob.time
    dist = d[0, route[0]]
    clock = t[0, route[0]]                       # arrival at first stop
    lateness = 0.0
    arrivals = []
    prev = route[0]
    for idx, node in enumerate(route):
        if idx > 0:
            dist += d[prev, node]
            clock += t[prev, node]
        start = max(clock, prob.e[node])         # wait if early
        lateness += max(0.0, clock - prob.l[node])
        arrivals.append(clock)
        clock = start + prob.service[node]       # departure
        prev = node
    dist += d[prev, 0]
    clock += t[prev, 0]                          # back to depot
    load = int(prob.demand[route].sum())
    return dist, lateness, arrivals, load, clock


def evaluate(prob: CVRPTW, routes):
    """Aggregate a full solution -> dict of metrics + penalized cost.
    feasible = customer windows met + capacity ok + fleet not exceeded.
    Route-duration overrun of the shift is a soft, penalized target (reported)."""
    total_dist = total_late = cap_over = over_dur = 0.0
    max_dur = 0.0; used = 0
    for r in routes:
        if not r:
            continue
        used += 1
        dist, late, _, load, dur = route_metrics(prob, r)
        total_dist += dist; total_late += late
        cap_over += max(0, load - prob.Q)
        over_dur += max(0.0, dur - prob.horizon)
        max_dur = max(max_dur, dur)
    veh_over = max(0, used - prob.K)
    cost = (total_dist + VEHICLE_COST * used
            + PEN_CAP * (cap_over + veh_over * prob.Q)
            + PEN_TW * total_late + PEN_DUR * over_dur)
    feasible = (cap_over == 0 and total_late < 1e-6 and veh_over == 0)
    return dict(cost=cost, distance=total_dist, vehicles=used,
                lateness=total_late, cap_over=cap_over, feasible=feasible,
                over_dur=over_dur, max_dur=max_dur)


def all_customers(prob):
    return list(range(1, prob.n + 1))


def route_eta_table(prob, route):
    """Per-stop ETA (clock minutes) + window + on-time flag, for reporting."""
    _, _, arrivals, _, _ = route_metrics(prob, route)
    out = []
    for node, arr in zip(route, arrivals):
        out.append(dict(node=node, name=prob.names[node] if prob.names else node,
                        eta_min=round(arr, 1), tw_start=int(prob.e[node]),
                        tw_end=int(prob.l[node]),
                        on_time=bool(arr <= prob.l[node] + 1e-6)))
    return out
