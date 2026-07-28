"""
Constructive heuristics: nearest-neighbour and Clarke-Wright savings.
Both respect capacity and time windows and return a list of feasible routes.
"""
import numpy as np
from cvrptw import CVRPTW, route_metrics


def _feasible(prob, route):
    if not route:
        return True
    _, late, _, load, _ = route_metrics(prob, route)
    return late < 1e-6 and load <= prob.Q


def nearest_neighbour(prob: CVRPTW):
    unvisited = set(range(1, prob.n + 1))
    routes = []
    while unvisited:
        route = []
        while True:
            last = route[-1] if route else 0
            best, bestc = None, np.inf
            for j in unvisited:
                cand = route + [j]
                _, late, _, load, _ = route_metrics(prob, cand)
                if load <= prob.Q and late < 1e-6 and prob.time[last, j] < bestc:
                    best, bestc = j, prob.time[last, j]
            if best is None:
                break
            route.append(best); unvisited.discard(best)
        if not route:                       # safety: force-place one customer
            j = unvisited.pop(); route = [j]
        routes.append(route)
    return routes


def clarke_wright(prob: CVRPTW):
    """Parallel savings; merges route endpoints while feasible."""
    routes = [[i] for i in range(1, prob.n + 1)]
    route_of = {i: idx for idx, i in enumerate(range(1, prob.n + 1))}
    d = prob.dist
    savings = []
    for i in range(1, prob.n + 1):
        for j in range(i + 1, prob.n + 1):
            savings.append((d[0, i] + d[0, j] - d[i, j], i, j))
    savings.sort(reverse=True)

    for s, i, j in savings:
        ri, rj = route_of.get(i), route_of.get(j)
        if ri is None or rj is None or ri == rj:
            continue
        Ri, Rj = routes[ri], routes[rj]
        if not Ri or not Rj:
            continue
        # merge only if i,j are route endpoints (keep orientation options)
        merged = None
        if Ri[-1] == i and Rj[0] == j:
            merged = Ri + Rj
        elif Ri[0] == i and Rj[-1] == j:
            merged = Rj + Ri
        elif Ri[-1] == i and Rj[-1] == j:
            merged = Ri + Rj[::-1]
        elif Ri[0] == i and Rj[0] == j:
            merged = Ri[::-1] + Rj
        if merged is None or not _feasible(prob, merged):
            continue
        routes[ri] = merged; routes[rj] = []
        for node in merged:
            route_of[node] = ri
    return [r for r in routes if r]


if __name__ == "__main__":
    from config import get_instance
    import data_gen, distances
    inst = get_instance(); df = data_gen.generate(inst)
    dist, time, src = distances.get_matrices(df, inst)
    prob = CVRPTW.from_frames(df, dist, time, inst)
    from cvrptw import evaluate
    for name, fn in [("nearest_neighbour", nearest_neighbour), ("clarke_wright", clarke_wright)]:
        routes = fn(prob)
        m = evaluate(prob, routes)
        print(f"{name:18s} vehicles={m['vehicles']:2d} dist={m['distance']:8.1f} "
              f"feasible={m['feasible']} lateness={m['lateness']:.1f}")
