"""Visualization: convergence curves, a static route map (matplotlib), and an
interactive folium map (when folium is installed)."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import folium
    _HAS_FOLIUM = True
except Exception:
    _HAS_FOLIUM = False


def plot_convergence(results, path):
    plt.figure(figsize=(6, 4))
    for r in results:
        if r.get("curve"):
            plt.plot(np.linspace(0, 1, len(r["curve"])), r["curve"], label=r["name"])
    plt.xlabel("search progress"); plt.ylabel("best penalized cost")
    plt.title("Metaheuristic convergence"); plt.legend(); plt.tight_layout()
    plt.savefig(path, dpi=130); plt.close()


def plot_routes(df, routes, path, title="Best CVRPTW solution"):
    lat = df["lat"].to_numpy(); lon = df["lon"].to_numpy()
    plt.figure(figsize=(7, 7))
    plt.scatter(lon[1:], lat[1:], s=14, c="#555", zorder=3)
    plt.scatter([lon[0]], [lat[0]], s=140, marker="*", c="red", zorder=5, label="depot")
    cmap = plt.cm.tab20(np.linspace(0, 1, max(len(routes), 1)))
    for v, route in enumerate(routes):
        seq = [0] + route + [0]
        plt.plot(lon[seq], lat[seq], "-", color=cmap[v % len(cmap)], lw=1.2, alpha=0.8)
    plt.xlabel("Longitude"); plt.ylabel("Latitude"); plt.title(title)
    plt.legend(); plt.tight_layout(); plt.savefig(path, dpi=130); plt.close()


def folium_map(df, routes, path):
    """Interactive route map; returns True if written (folium available)."""
    if not _HAS_FOLIUM:
        return False
    lat = df["lat"].to_numpy(); lon = df["lon"].to_numpy()
    m = folium.Map(location=[lat.mean(), lon.mean()], zoom_start=11, tiles="cartodbpositron")
    folium.Marker([lat[0], lon[0]], tooltip="Depot",
                  icon=folium.Icon(color="red", icon="home")).add_to(m)
    colors = ["blue", "green", "purple", "orange", "darkred", "cadetblue",
              "darkgreen", "darkblue", "black", "pink", "gray", "beige"]
    for v, route in enumerate(routes):
        col = colors[v % len(colors)]
        pts = [(lat[0], lon[0])] + [(lat[i], lon[i]) for i in route] + [(lat[0], lon[0])]
        folium.PolyLine(pts, color=col, weight=2.5, opacity=0.8,
                        tooltip=f"Vehicle {v+1} ({len(route)} stops)").add_to(m)
        for i in route:
            folium.CircleMarker([lat[i], lon[i]], radius=3, color=col, fill=True,
                                tooltip=df.iloc[i]["name"]).add_to(m)
    m.save(path)
    return True


def has_folium():
    return _HAS_FOLIUM
