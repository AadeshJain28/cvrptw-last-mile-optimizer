# Last-Mile Delivery Optimization — CVRPTW with Metaheuristics & Road Travel-Times

A capacitated, time-windowed vehicle-routing solver for a **150-stop Mumbai** last-mile
network, using **real OSRM road travel-times**, with custom **GA / SA / Tabu**
metaheuristics benchmarked against **OR-Tools** and an exact **Gurobi** model — and
per-stop **ETAs** on an interactive map.


## What it does

- **CVRPTW** — a fleet of capacitated vehicles serves ~150 stops, each with a demand,
  a hard delivery **time window**, and a service time; objective: fewest vehicles, then
  least travel-time.
- **Real road times** — an asymmetric distance/duration matrix from **OSRM**
  (OpenStreetMap); automatic **Haversine × detour** fallback offline.
- **Five solvers, one scorer** — nearest-neighbour, Clarke-Wright savings, **GA**, **SA**,
  **Tabu**, and **OR-Tools**; every solution is scored by the *same* `evaluate()` for a
  fair comparison, plus an exact **Gurobi** optimum on a sub-instance to measure the gap.
- **ETAs + map** — per-stop arrival times with on-time flags, and an interactive
  **folium** route map.

---

## CVRPTW formulation

Sets: depot `0`, customers `1..n`. Vehicle capacity `Q`, fleet `K`, shift `H`.
Vars: `x[i,j] ∈ {0,1}` (arc used), `t[i]` (arrival time), `u[i]` (cumulative load).

```
min   Σ_ij dist[i,j]·x[i,j]   + f · Σ_j x[0,j]          (distance + per-vehicle fixed cost)
s.t.  Σ_j x[i,j] = 1,  Σ_j x[j,i] = 1          ∀ customer i          (visit once)
      Σ_j x[0,j] = Σ_j x[j,0] ≤ K                                    (fleet size)
      t[j] ≥ t[i] + s[i] + τ[i,j] − M(1−x[i,j])   ∀ i, j≠0           (time propagation)
      e[i] ≤ t[i] ≤ l[i]                          ∀ i                (hard time windows)
      u[j] ≥ u[i] + d[j] − Q(1−x[i,j]);  d[i] ≤ u[i] ≤ Q             (capacity)
```

The exact model above is solved by Gurobi on a small sub-instance; OR-Tools and the
metaheuristics optimize the same objective on the full instance. Customer time windows
are **hard**; exceeding the shift `H` on the return leg is a **soft**, penalized target.

---

## Results (real runs — full ML/OR stack: OR-Tools + Gurobi + OSRM + folium)

**Objective = fleet-then-distance** (`cost = distance + fixed cost per vehicle`), which is
realistic for last-mile (a truck's fixed cost dwarfs its per-km cost). Every method is
scored by the same `evaluate()`; all meet **100% of delivery time windows**.

**`mid` — 95 stops, real OSRM road travel-times** (ranked by objective cost):

| Method | Cost | Distance (km) | Vehicles | TW met | Gap vs best |
|---|---|---|---|---|---|
| **OR-Tools** | 14,784 | 783.8 | **7** | ✓ | 0.0% |
| **GA** | 15,160 | 1160.4 | **7** | ✓ | +2.5% |
| SA | 16,739 | **738.6** | 8 | ✓ | +13.2% |
| Tabu | 16,739 | **738.6** | 8 | ✓ | +13.2% |
| Nearest-Neighbour | 17,127 | 1126.5 | 8 | ✓ | +15.8% |
| Clarke-Wright | 18,867 | 866.9 | 9 | ✓ | +27.6% |

> **Exact check (Gurobi):** on a 12-stop sub-instance the metaheuristic is within **+0.5%**
> of the proven optimum (266.6 km vs 268.0 km) — near-optimal, validated.

**`full` — 150 stops** (Haversine; public OSRM caps at ~100 pts) — ranked by objective cost:

| Method | Cost | Distance (km) | Vehicles | TW met | Gap vs best |
|---|---|---|---|---|---|
| **SA** | 24,974 | 973.9 | **12** | ✓ | 0.0% |
| **GA** | 24,974 | 973.9 | **12** | ✓ | 0.0% |
| Tabu | 26,969 | **968.8** | 13 | ✓ | +8.0% |
| Clarke-Wright | 28,993 | 992.5 | 14 | ✓ | +16.1% |
| Nearest-Neighbour | 31,473 | 1473.3 | 15 | ✓ | +26.0% |

**What the tables show honestly:** OR-Tools and GA find the fewest-vehicle (lowest-cost)
plans; SA/Tabu find the **shortest routes** (738 km on `mid`, beating OR-Tools' 784 km) at
one extra vehicle — a genuine fleet-vs-distance trade-off. Best plans cut distance **~34%**
vs nearest-neighbour, with **100% of time windows met** (95/95 and 150/150 on-time).

---

## Run it

```bash
pip install -r requirements.txt      # numpy/pandas/matplotlib are enough for the fallback path
python main.py                       # demo (40 stops), end-to-end
python main.py full                  # 150 stops + OR-Tools + Gurobi exact check + folium map
python tests/test_cvrptw.py          # feasibility + solver-quality tests
```

To use **real road times**, just have internet + `requests` (public OSRM) or set
`OSRM_URL` to a self-hosted server. **Graceful degradation:** OR-Tools, Gurobi, folium,
and OSRM are all optional — the console header prints what's active, and GA/SA/Tabu +
Haversine always run.

## Repo structure

```
cvrptw-last-mile-opt/
├── src/  config · data_gen · distances(OSRM+Haversine) · cvrptw(core+ETA)
│         construct(NN, savings) · local_search(2opt/Or-opt/relocate/exchange)
│         metaheuristics(GA/SA/Tabu) · ortools_solver · gurobi_exact · eta · viz
├── main.py            end-to-end benchmark + artifacts
├── tests/             feasibility + quality tests
├── data/  raw/ (original 20 nodes) · generated/ (instance + matrix cache + assumptions)
├── results/  benchmark.csv · best_eta.csv · route_map.html · summary.json · figures/
└── requirements.txt
```
