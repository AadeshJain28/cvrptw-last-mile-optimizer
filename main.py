"""
End-to-end CVRPTW benchmark: generate a Mumbai instance -> build road/Haversine
matrices -> solve with constructions, GA/SA/Tabu, OR-Tools, and an exact Gurobi
check on a sub-instance -> tabulate, plot convergence + route map, export ETAs.

Runs anywhere: OR-Tools / Gurobi / folium / OSRM are optional (graceful skips).
Switch size:  python main.py full
"""
import os, sys, json, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
import numpy as np, pandas as pd

from config import get_instance, RESULTS_DIR, FIG_DIR
import data_gen, distances
from cvrptw import CVRPTW, evaluate
import construct, metaheuristics as MH, ortools_solver as ORT, gurobi_exact as GX
import eta as ETA, viz

os.makedirs(FIG_DIR, exist_ok=True)
inst = get_instance()
PARAMS = dict(demo=dict(sa=3500, tabu=200, ga=(12, 12)),
              mid=dict(sa=8000, tabu=350, ga=(16, 18)),
              full=dict(sa=12000, tabu=500, ga=(16, 20)))[inst.name]

print(f"=== CVRPTW | instance={inst.name} ({inst.n_stops} stops, {inst.n_vehicles} vehicles) ===")
df = data_gen.generate(inst)
dist, tmat, src = distances.get_matrices(df, inst)
prob = CVRPTW.from_frames(df, dist, tmat, inst)
print(f"distance source: {src}")
print(f"OR-Tools: {'yes' if ORT.available() else 'no'} | Gurobi: {'yes' if GX.available() else 'no'} "
      f"| folium: {'yes' if viz.has_folium() else 'no'}\n")

rows, curves, sols = [], [], {}

def record(name, routes, seconds, curve=None):
    m = evaluate(prob, routes)
    sols[name] = routes
    rows.append(dict(method=name, cost=round(m["cost"], 0), distance_km=round(m["distance"], 1),
                     vehicles=m["vehicles"], feasible=bool(m["feasible"]),
                     lateness_min=round(m["lateness"], 2), max_route_min=round(m["max_dur"], 0),
                     seconds=round(seconds, 1)))
    if curve:
        curves.append(dict(name=name, curve=curve))

# ---- constructions ----
t = time.perf_counter(); record("NearestNeighbour", construct.nearest_neighbour(prob), time.perf_counter()-t)
t = time.perf_counter(); record("ClarkeWright", construct.clarke_wright(prob), time.perf_counter()-t)

# ---- metaheuristics ----
r = MH.simulated_annealing(prob, iters=PARAMS["sa"]);              record("SA", r.routes, r.seconds, r.curve)
r = MH.tabu_search(prob, iters=PARAMS["tabu"]);                    record("Tabu", r.routes, r.seconds, r.curve)
r = MH.genetic_algorithm(prob, pop_size=PARAMS["ga"][0], gens=PARAMS["ga"][1]); record("GA", r.routes, r.seconds, r.curve)

# ---- OR-Tools reference ----
if ORT.available():
    t = time.perf_counter(); ort = ORT.solve_ortools(prob, time_limit_s=15)
    if ort:
        record("OR-Tools", ort, time.perf_counter()-t)

bench = pd.DataFrame(rows)
feas = bench[bench.feasible]
best_cost = feas.cost.min() if len(feas) else bench.cost.min()
# objective = fleet-then-distance: cost = distance + fixed vehicle cost per truck
bench["gap_%"] = ((bench.cost - best_cost) / best_cost * 100).round(1)
bench = bench.sort_values(["feasible", "cost"], ascending=[False, True])
print("BENCHMARK (ranked by the optimized objective: distance + per-vehicle cost; "
      "all use the same evaluate()):")
print(bench[["method", "cost", "distance_km", "vehicles", "feasible", "gap_%",
             "seconds"]].to_string(index=False))

# ---- exact Gurobi validation on a sub-instance ----
exact_note = "Gurobi not available (skipped)"
if GX.available():
    res = GX.solve_exact(prob, max_nodes=12)
    if res:
        ex_routes, ex_dist, m = res
        # solve the SAME sub-instance with GA for an apples-to-apples optimality gap
        sub_df = df.iloc[:m+1].reset_index(drop=True)
        sd, stime, _ = dist[:m+1, :m+1], tmat[:m+1, :m+1], None
        subprob = CVRPTW(sd, stime, prob.demand[:m+1], prob.e[:m+1], prob.l[:m+1],
                         prob.service[:m+1], prob.Q, prob.K, prob.horizon, prob.names[:m+1])
        gres = MH.genetic_algorithm(subprob, pop_size=14, gens=15)
        gap = (gres.distance - ex_dist) / ex_dist * 100
        exact_note = (f"Sub-instance ({m} stops): Gurobi optimum={ex_dist:.1f} km, "
                      f"GA={gres.distance:.1f} km -> gap {gap:+.1f}%")
print("\nEXACT CHECK:", exact_note)

# ---- pick best feasible solution, export artifacts ----
best_name = bench[bench.feasible].iloc[0]["method"] if len(feas) else bench.iloc[0]["method"]
best_routes = sols[best_name]
os.makedirs(RESULTS_DIR, exist_ok=True)
bench.to_csv(os.path.join(RESULTS_DIR, "benchmark.csv"), index=False)
eta_df = ETA.eta_table(prob, best_routes)
eta_df.to_csv(os.path.join(RESULTS_DIR, "best_eta.csv"), index=False)
viz.plot_convergence(curves, os.path.join(FIG_DIR, "01_convergence.png"))
viz.plot_routes(df, best_routes, os.path.join(FIG_DIR, "02_best_routes.png"),
                title=f"Best CVRPTW solution ({best_name}, {len(best_routes)} vehicles)")
wrote_map = viz.folium_map(df, best_routes, os.path.join(RESULTS_DIR, "route_map.html"))

json.dump(dict(instance=inst.name, n_stops=inst.n_stops, distance_source=src,
               best_method=best_name, best_distance_km=round(evaluate(prob, best_routes)["distance"], 1),
               best_vehicles=evaluate(prob, best_routes)["vehicles"],
               all_time_windows_met=bool(evaluate(prob, best_routes)["feasible"]),
               exact_check=exact_note, ortools=ORT.available(), gurobi=GX.available()),
          open(os.path.join(RESULTS_DIR, "summary.json"), "w"), indent=2)

print(f"\nBest: {best_name} | {evaluate(prob, best_routes)['vehicles']} vehicles, "
      f"{evaluate(prob, best_routes)['distance']:.1f} km, all TW met="
      f"{evaluate(prob, best_routes)['feasible']}")
print(f"On-time stops in best ETA: {int(eta_df.on_time.sum())}/{len(eta_df)}")
print(f"Artifacts -> results/ (benchmark.csv, best_eta.csv, summary.json, "
      f"{'route_map.html, ' if wrote_map else ''}figures/). Done.")
