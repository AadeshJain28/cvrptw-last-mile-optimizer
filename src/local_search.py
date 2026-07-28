"""
Local-search operators for CVRPTW: intra-route 2-opt and Or-opt, inter-route
relocate and exchange. Moves use ROUTE-LEVEL cost deltas (only the routes that
change are re-evaluated), so a full pass is fast even at 150 stops. A route's cost
includes a fixed vehicle charge, so relocate/exchange also drive down fleet size.
"""
import numpy as np
from cvrptw import route_metrics, evaluate, PEN_TW, PEN_CAP, PEN_DUR, VEHICLE_COST


def route_cost(prob, route):
    if not route:
        return 0.0
    dist, late, _, load, dur = route_metrics(prob, route)
    return (dist + PEN_TW * late + PEN_CAP * max(0, load - prob.Q)
            + PEN_DUR * max(0.0, dur - prob.horizon) + VEHICLE_COST)


def two_opt(prob, route):
    if len(route) < 3:
        return route
    best = route[:]; best_c = route_cost(prob, best); improved = True
    while improved:
        improved = False
        for i in range(len(best) - 1):
            for j in range(i + 1, len(best)):
                cand = best[:i] + best[i:j + 1][::-1] + best[j + 1:]
                c = route_cost(prob, cand)
                if c < best_c - 1e-9:
                    best, best_c = cand, c; improved = True
    return best


def or_opt(prob, route):
    best = route[:]; best_c = route_cost(prob, best)
    for seg in (1, 2, 3):
        for i in range(len(best) - seg + 1):
            block = best[i:i + seg]; rest = best[:i] + best[i + seg:]
            for k in range(len(rest) + 1):
                cand = rest[:k] + block + rest[k:]
                c = route_cost(prob, cand)
                if c < best_c - 1e-9:
                    best, best_c = cand, c
    return best


def _intra(prob, routes):
    return [or_opt(prob, two_opt(prob, r)) if r else r for r in routes]


def relocate(prob, routes, max_sweeps=40):
    """Best-improvement per sweep: find the single best customer move, apply, repeat."""
    routes = [r[:] for r in routes if r]
    for _ in range(max_sweeps):
        best_delta, best = -1e-6, None
        for ri in range(len(routes)):
            for pos in range(len(routes[ri])):
                cust = routes[ri][pos]
                r2 = routes[ri][:pos] + routes[ri][pos + 1:]
                base, new_ri = route_cost(prob, routes[ri]), route_cost(prob, r2)
                for rj in range(len(routes) + 1):        # +1 allows opening a new route
                    if rj < len(routes) and rj == ri:
                        continue
                    target = routes[rj] if rj < len(routes) else []
                    old_rj = route_cost(prob, target)
                    for k in range(len(target) + 1):
                        cand = target[:k] + [cust] + target[k:]
                        delta = (new_ri + route_cost(prob, cand)) - (base + old_rj)
                        if delta < best_delta:
                            best_delta, best = delta, (ri, r2, rj, cand)
        if best is None:
            break
        ri, r2, rj, cand = best
        routes[ri] = r2
        if rj < len(routes):
            routes[rj] = cand
        else:
            routes.append(cand)
        routes = [r for r in routes if r]
    return [r for r in routes if r]


def exchange(prob, routes, max_sweeps=20):
    routes = [r[:] for r in routes if r]
    for _ in range(max_sweeps):
        best_delta, best = -1e-6, None
        for ri in range(len(routes)):
            for rj in range(ri + 1, len(routes)):
                base = route_cost(prob, routes[ri]) + route_cost(prob, routes[rj])
                for a in range(len(routes[ri])):
                    for b in range(len(routes[rj])):
                        R1 = routes[ri][:]; R2 = routes[rj][:]
                        R1[a], R2[b] = R2[b], R1[a]
                        delta = route_cost(prob, R1) + route_cost(prob, R2) - base
                        if delta < best_delta:
                            best_delta, best = delta, (ri, rj, R1, R2)
        if best is None:
            break
        ri, rj, R1, R2 = best
        routes[ri], routes[rj] = R1, R2
    return [r for r in routes if r]


def local_search(prob, routes, max_passes=4):
    routes = [r[:] for r in routes if r]
    prev = np.inf
    for _ in range(max_passes):
        routes = _intra(prob, routes)
        routes = relocate(prob, routes)
        routes = _intra(prob, routes)
        routes = exchange(prob, routes)
        c = evaluate(prob, routes)["cost"]
        if c >= prev - 1e-6:
            break
        prev = c
    return [r for r in routes if r]
