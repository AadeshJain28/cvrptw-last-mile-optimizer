"""
Exact CVRPTW (compact 2-index MILP) for a SMALL sub-instance -- used only to
validate that the metaheuristics reach / stay near the true optimum.

Formulation: arc vars x[i,j]; MTZ-style time propagation carries the windows,
a load variable carries capacity. Minimise distance + a per-vehicle fixed cost.
For tractability we restrict to the first `max_nodes` customers (default 12).

Returns (routes, objective_distance) or None if Gurobi is unavailable.
"""
import numpy as np

try:
    import gurobipy as gp
    from gurobipy import GRB
    _HAS_GRB = True
except Exception:
    _HAS_GRB = False


def solve_exact(prob, max_nodes=12, time_limit_s=60):
    if not _HAS_GRB:
        return None
    m = min(max_nodes, prob.n)
    V = list(range(m + 1))                      # 0 = depot, 1..m customers
    d = prob.dist; t = prob.time
    e, l, s, dem = prob.e, prob.l, prob.service, prob.demand
    Q, K, H = prob.Q, prob.K, prob.horizon
    bigM_t = H + t.max() + s.max()

    mdl = gp.Model("cvrptw_exact"); mdl.Params.OutputFlag = 0
    mdl.Params.TimeLimit = time_limit_s
    x = mdl.addVars(V, V, vtype=GRB.BINARY, name="x")
    tv = mdl.addVars(V, lb=0, name="t")         # arrival time
    u = mdl.addVars(V, lb=0, ub=Q, name="u")    # cumulative load
    for i in V:
        x[i, i].UB = 0

    mdl.setObjective(gp.quicksum(d[i, j] * x[i, j] for i in V for j in V if i != j)
                     + 2000 * gp.quicksum(x[0, j] for j in V if j != 0), GRB.MINIMIZE)

    for i in V:
        if i == 0:
            continue
        mdl.addConstr(gp.quicksum(x[i, j] for j in V if j != i) == 1)
        mdl.addConstr(gp.quicksum(x[j, i] for j in V if j != i) == 1)
    mdl.addConstr(gp.quicksum(x[0, j] for j in V if j != 0) <= K)
    mdl.addConstr(gp.quicksum(x[0, j] for j in V if j != 0)
                  == gp.quicksum(x[j, 0] for j in V if j != 0))

    for i in V:
        for j in V:
            if i != j and j != 0:
                mdl.addConstr(tv[j] >= tv[i] + s[i] + t[i, j] - bigM_t * (1 - x[i, j]))
                mdl.addConstr(u[j] >= u[i] + dem[j] - Q * (1 - x[i, j]))
    for i in V:
        mdl.addConstr(tv[i] >= e[i]); mdl.addConstr(tv[i] <= l[i])
        mdl.addConstr(u[i] >= dem[i])

    mdl.optimize()
    if mdl.SolCount == 0:
        return None
    arcs = {(i, j) for i in V for j in V if i != j and x[i, j].X > 0.5}
    routes = []
    for j in V:
        if (0, j) in arcs:
            route = [j]; cur = j
            while True:
                nxt = [b for (a, b) in arcs if a == cur and b != 0]
                if not nxt:
                    break
                cur = nxt[0]; route.append(cur)
            routes.append(route)
    dist_obj = sum(d[i, j] for (i, j) in arcs)
    return routes, dist_obj, m


def available():
    return _HAS_GRB
