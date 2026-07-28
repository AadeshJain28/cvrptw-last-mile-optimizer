"""
Verification tests (run: python tests/test_cvrptw.py).
Guards the constraints that make routing results credible: every customer served
once, capacity respected, time windows met, and metaheuristics beat construction.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np
from config import Instance
import data_gen, distances
from cvrptw import CVRPTW, evaluate, route_metrics
from construct import nearest_neighbour, clarke_wright
from local_search import local_search
import metaheuristics as MH

TINY = Instance(name="tiny", n_stops=18, vehicle_capacity=110, n_vehicles=6)


def _prob():
    df = data_gen.generate(TINY)
    dist, tmat, _ = distances.get_matrices(df, TINY, cache=False)
    return CVRPTW.from_frames(df, dist, tmat, TINY), df


def test_all_customers_served_once():
    prob, _ = _prob()
    routes = local_search(prob, clarke_wright(prob))
    served = [c for r in routes for c in r]
    assert sorted(served) == list(range(1, prob.n + 1)), "customers missing/duplicated"
    print("PASS  every customer served exactly once")


def test_capacity_respected():
    prob, _ = _prob()
    routes = local_search(prob, clarke_wright(prob))
    for r in routes:
        assert prob.demand[r].sum() <= prob.Q, "capacity exceeded"
    print("PASS  capacity respected on every route")


def test_time_windows_met():
    prob, _ = _prob()
    routes = local_search(prob, clarke_wright(prob))
    total_late = sum(route_metrics(prob, r)[1] for r in routes)
    assert total_late < 1e-6, f"time-window lateness = {total_late}"
    assert evaluate(prob, routes)["vehicles"] <= prob.K, "fleet exceeded"
    print("PASS  all customer time windows met; fleet not exceeded")


def test_metaheuristics_beat_construction():
    prob, _ = _prob()
    nn = evaluate(prob, nearest_neighbour(prob))["cost"]
    sa = MH.simulated_annealing(prob, iters=1500)
    ga = MH.genetic_algorithm(prob, pop_size=10, gens=8)
    assert sa.cost <= nn + 1e-6, "SA did not improve on nearest-neighbour"
    assert ga.cost <= nn + 1e-6, "GA did not improve on nearest-neighbour"
    assert sa.feasible and ga.feasible, "metaheuristic returned infeasible solution"
    print(f"PASS  SA/GA beat NN construction (NN cost {nn:.0f} -> "
          f"SA {sa.cost:.0f}, GA {ga.cost:.0f})")


if __name__ == "__main__":
    print(f"OR-Tools:{__import__('ortools_solver').available()} "
          f"Gurobi:{__import__('gurobi_exact').available()}")
    test_all_customers_served_once()
    test_capacity_respected()
    test_time_windows_met()
    test_metaheuristics_beat_construction()
    print("\nAll tests passed.")
