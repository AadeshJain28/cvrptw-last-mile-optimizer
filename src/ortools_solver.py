"""
OR-Tools CVRPTW reference solver (the strong benchmark to beat/match).

Builds a routing model with a Distance objective, a Time dimension carrying the
hard time windows, and a Capacity dimension, then runs guided local search.
Returns routes in OUR format (list of customer lists) so the benchmark scores it
with the same cvrptw.evaluate() as the metaheuristics -- a fair comparison.

Gracefully returns None if ortools is not installed.
"""
import numpy as np

try:
    from ortools.constraint_solver import routing_enums_pb2, pywrapcp
    _HAS_ORT = True
except Exception:
    _HAS_ORT = False

SCALE = 100   # OR-Tools needs integer arcs; scale km/min by this


def solve_ortools(prob, time_limit_s=15):
    if not _HAS_ORT:
        return None
    n = prob.n + 1
    mgr = pywrapcp.RoutingIndexManager(n, prob.K, 0)
    routing = pywrapcp.RoutingModel(mgr)

    dist = (prob.dist * SCALE).astype(int)
    tmat = (prob.time * SCALE).astype(int)
    serv = (prob.service * SCALE).astype(int)

    def dist_cb(a, b):
        return int(dist[mgr.IndexToNode(a)][mgr.IndexToNode(b)])
    dist_idx = routing.RegisterTransitCallback(dist_cb)
    routing.SetArcCostEvaluatorOfAllVehicles(dist_idx)
    routing.SetFixedCostOfAllVehicles(int(2000 * SCALE))     # prefer fewer vehicles

    # Time dimension (travel + service) carries the time windows
    def time_cb(a, b):
        i, j = mgr.IndexToNode(a), mgr.IndexToNode(b)
        return int(tmat[i][j] + serv[i])
    time_idx = routing.RegisterTransitCallback(time_cb)
    routing.AddDimension(time_idx, int(prob.horizon * SCALE),
                         int(prob.horizon * SCALE), False, "Time")
    time_dim = routing.GetDimensionOrDie("Time")
    for node in range(n):
        idx = mgr.NodeToIndex(node)
        time_dim.CumulVar(idx).SetRange(int(prob.e[node] * SCALE),
                                        int(prob.l[node] * SCALE))

    # Capacity dimension
    def dem_cb(a):
        return int(prob.demand[mgr.IndexToNode(a)])
    dem_idx = routing.RegisterUnaryTransitCallback(dem_cb)
    routing.AddDimensionWithVehicleCapacity(dem_idx, 0, [prob.Q] * prob.K, True, "Cap")

    params = pywrapcp.DefaultRoutingSearchParameters()
    params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    params.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    params.time_limit.FromSeconds(time_limit_s)

    sol = routing.SolveWithParameters(params)
    if sol is None:
        return None
    routes = []
    for v in range(prob.K):
        idx = routing.Start(v); route = []
        while not routing.IsEnd(idx):
            node = mgr.IndexToNode(idx)
            if node != 0:
                route.append(node)
            idx = sol.Value(routing.NextVar(idx))
        if route:
            routes.append(route)
    return routes


def available():
    return _HAS_ORT
