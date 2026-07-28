"""
Metaheuristics for CVRPTW: Simulated Annealing, Tabu Search, Genetic Algorithm.
All share the penalized cost in cvrptw.evaluate and the moves below; each returns
a Result with best routes, metrics, a convergence curve, and wall-clock time.
"""
import time
import numpy as np
from cvrptw import evaluate, route_metrics
from construct import clarke_wright, nearest_neighbour, _feasible
from local_search import local_search, two_opt, or_opt, _intra


class Result(dict):
    __getattr__ = dict.get


def _pack(prob, routes, curve, t0, name):
    m = evaluate(prob, routes)
    return Result(name=name, routes=[r for r in routes if r], cost=m["cost"],
                  distance=m["distance"], vehicles=m["vehicles"],
                  lateness=m["lateness"], feasible=m["feasible"],
                  seconds=round(time.perf_counter() - t0, 2), curve=curve)


# ---------- shared moves ----------
def _best_insertion(prob, routes, cust):
    best, bc = None, np.inf
    for ri in range(len(routes) + 1):
        target = routes[ri] if ri < len(routes) else []
        for k in range(len(target) + 1):
            cand = target[:k] + [cust] + target[k:]
            _, late, _, load, _ = route_metrics(prob, cand)
            c = late * 200 + max(0, load - prob.Q) * 500 + (0 if ri < len(routes) else 2000)
            if c < bc:
                best, bc = (ri, k), c
    ri, k = best
    routes = [r[:] for r in routes]
    if ri == len(routes):
        routes.append([cust])
    else:
        routes[ri] = routes[ri][:k] + [cust] + routes[ri][k:]
    return [r for r in routes if r]


def _random_move(prob, routes, rng):
    routes = [r[:] for r in routes if r]
    kind = rng.integers(3)
    if kind == 0 and routes:                       # relocate
        ri = rng.integers(len(routes))
        if routes[ri]:
            pos = rng.integers(len(routes[ri]))
            cust = routes[ri].pop(pos)
            routes = [r for r in routes if r]
            routes = _best_insertion(prob, routes, cust)
    elif kind == 1 and routes:                     # 2-opt a random route
        ri = rng.integers(len(routes))
        if len(routes[ri]) > 2:
            routes[ri] = two_opt(prob, routes[ri])
    else:                                          # exchange two customers
        if len(routes) >= 2:
            ri, rj = rng.choice(len(routes), 2, replace=False)
            if routes[ri] and routes[rj]:
                a, b = rng.integers(len(routes[ri])), rng.integers(len(routes[rj]))
                routes[ri][a], routes[rj][b] = routes[rj][b], routes[ri][a]
    return [r for r in routes if r]


# ---------- Simulated Annealing ----------
def simulated_annealing(prob, init=None, iters=6000, T0=250.0, alpha=0.9975, seed=0):
    rng = np.random.default_rng(seed); t0 = time.perf_counter()
    cur = local_search(prob, init or clarke_wright(prob), max_passes=2)
    cur_c = evaluate(prob, cur)["cost"]
    best, best_c = cur, cur_c; T = T0; curve = []
    for it in range(iters):
        cand = _random_move(prob, cur, rng)
        c = evaluate(prob, cand)["cost"]
        if c < cur_c or rng.random() < np.exp(-(c - cur_c) / max(T, 1e-6)):
            cur, cur_c = cand, c
            if c < best_c:
                best, best_c = cand, c
        T *= alpha
        if it % 50 == 0:
            curve.append(best_c)
    best = local_search(prob, best, max_passes=3)
    return _pack(prob, best, curve, t0, "SA")


# ---------- Tabu Search ----------
def tabu_search(prob, init=None, iters=400, tenure=15, sample=40, seed=0):
    rng = np.random.default_rng(seed); t0 = time.perf_counter()
    cur = local_search(prob, init or clarke_wright(prob), max_passes=2)
    best = cur; best_c = evaluate(prob, cur)["cost"]; cur_c = best_c
    tabu = {}; curve = []
    for it in range(iters):
        best_move, best_move_c, best_key = None, np.inf, None
        for _ in range(sample):
            cand = _random_move(prob, cur, rng)
            key = (len(cand), round(evaluate(prob, cand)["distance"], 1))  # coarse move signature
            c = evaluate(prob, cand)["cost"]
            tabued = tabu.get(key, 0) > it
            if (not tabued or c < best_c) and c < best_move_c:
                best_move, best_move_c, best_key = cand, c, key
        if best_move is None:
            break
        cur, cur_c = best_move, best_move_c
        tabu[best_key] = it + tenure
        if cur_c < best_c:
            best, best_c = cur, cur_c
        curve.append(best_c)
    best = local_search(prob, best, max_passes=3)
    return _pack(prob, best, curve, t0, "Tabu")


# ---------- Genetic Algorithm ----------
def _route_crossover(prob, p1, p2, rng):
    take = [r for r in p1 if rng.random() < 0.5]         # inherit some routes from p1
    taken = {c for r in take for c in r}
    child = [r[:] for r in take]
    leftover = [c for r in p2 for c in r if c not in taken]
    for c in leftover:
        child = _best_insertion(prob, child, c)
    return child


def _tournament(pop, costs, rng):
    i, j = rng.integers(0, len(pop), 2)
    return pop[i] if costs[i] <= costs[j] else pop[j]


def genetic_algorithm(prob, pop_size=16, gens=25, seed=0):
    """Children get only cheap intra-route optimisation; full inter-route local
    search is applied periodically to the incumbent (keeps GA fast at scale)."""
    rng = np.random.default_rng(seed); t0 = time.perf_counter()
    pop = [_intra(prob, clarke_wright(prob))]
    while len(pop) < pop_size:
        r = nearest_neighbour(prob); rng.shuffle(r)
        pop.append(_intra(prob, r))
    costs = [evaluate(prob, s)["cost"] for s in pop]
    best_i = int(np.argmin(costs)); best, best_c = pop[best_i], costs[best_i]
    curve = [best_c]
    for g in range(gens):
        new = [best]                                     # elitism
        while len(new) < pop_size:
            a, b = _tournament(pop, costs, rng), _tournament(pop, costs, rng)
            child = _route_crossover(prob, a, b, rng)
            if rng.random() < 0.6:
                child = _random_move(prob, child, rng)
            new.append(_intra(prob, child))              # cheap improvement only
        pop = new
        costs = [evaluate(prob, s)["cost"] for s in pop]
        i = int(np.argmin(costs))
        if costs[i] < best_c:
            best, best_c = pop[i], costs[i]
        if g % 6 == 5:                                    # periodic intensification
            best = local_search(prob, best, max_passes=2)
            best_c = evaluate(prob, best)["cost"]
        curve.append(best_c)
    best = local_search(prob, best, max_passes=3)
    return _pack(prob, best, curve, t0, "GA")


ALGORITHMS = {"SA": simulated_annealing, "Tabu": tabu_search, "GA": genetic_algorithm}


if __name__ == "__main__":
    from config import get_instance
    import data_gen, distances
    from cvrptw import CVRPTW
    inst = get_instance(); df = data_gen.generate(inst)
    dist, time_m, src = distances.get_matrices(df, inst)
    prob = CVRPTW.from_frames(df, dist, time_m, inst)
    init = clarke_wright(prob)
    print("init:", evaluate(prob, init))
    for name, fn in ALGORITHMS.items():
        res = fn(prob)
        print(f"{name:5s} dist={res.distance:8.1f} veh={res.vehicles:2d} "
              f"feasible={res.feasible} lateness={res.lateness:.2f} time={res.seconds}s")
