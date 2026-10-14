#!/usr/bin/env python3
"""
Disegna, su slice 2D, i contorni di:
  - foglie accettate che intersecano la slice;
  - rami e tronchi appartenenti agli alberi delle foglie accettate,
    se il nodo stesso interseca la slice.
Richiede nella directory CATALOG_DIR:
  metadata.json
  clumps_leaves.csv
  tree_hierarchy.json
  membership_leaves.npz
  membership_nodes.npz
membership_leaves.npz contiene:
  xyz_kpc, widths_kpc, clump_id
membership_nodes.npz contiene un array node_<node_id> per ogni nodo,
con gli indici delle celle appartenenti a quel nodo.
"""
import csv
import gc
import json
import os
import imageio
import matplotlib as mpl
import numpy as np
import yt
from scipy.ndimage import gaussian_filter
# ============================================================
# CONFIGURAZIONE
# ============================================================
FILENAME = "/mnt/O1/otello/parthenon.restart.00180.rhdf"
CATALOG_DIR = "/home/core/otello/prime_sim/cold_clumps_box"
OUT_DIR = "/mnt/O1/otello/slices_cold_box"
HALF_WIDTH_KPC = 10.0
N_GRID = 1200
# Esempio: una sola slice a offset 0.
# Per una tomografia, ad esempio:
# OFFSETS = np.arange(-0.020, 0.0201, 0.001)
OFFSETS = np.array([0.000])
os.makedirs(OUT_DIR, exist_ok=True)
# ============================================================
# 1. CARICAMENTO CATALOGO, ALBERO E MEMBERSHIP
# ============================================================
print("--- 1. Caricamento dati dell'albero ---", flush=True)
with open(os.path.join(CATALOG_DIR, "metadata.json")) as f:
    meta = json.load(f)
center_kpc = np.asarray(meta["center_kpc"], dtype=float)
leaf_mem_path = os.path.join(CATALOG_DIR, "membership_leaves.npz")
node_mem_path = os.path.join(CATALOG_DIR, "membership_nodes.npz")
if not os.path.exists(node_mem_path):
    raise FileNotFoundError(
        "Manca membership_nodes.npz. Il file deve essere creato da "
        "cf4_tree.py salvando, per ogni nodo, gli indici delle sue celle."
    )
leaf_mem = np.load(leaf_mem_path)
xyz_kpc = np.asarray(leaf_mem["xyz_kpc"], dtype=float)
widths_kpc = np.asarray(leaf_mem["widths_kpc"], dtype=float)
leaf_membership = np.asarray(leaf_mem["clump_id"], dtype=np.int64)
node_membership = np.load(node_mem_path)
with open(os.path.join(CATALOG_DIR, "tree_hierarchy.json")) as f:
    raw_tree = json.load(f)
tree = {int(k): v for k, v in raw_tree.items()}
if not tree:
    raise RuntimeError("tree_hierarchy.json non contiene nodi.")
# Legge il CSV delle foglie e mantiene la corrispondenza:
# leaf_id = ID presente in membership_leaves.npz
# node_id = ID presente in tree_hierarchy.json e membership_nodes.npz
leaf_to_node = {}
accepted_leaf_ids = set()
with open(os.path.join(CATALOG_DIR, "clumps_leaves.csv"), newline="") as f:
    reader = csv.DictReader(f)
    required = {"clump_id", "node_id", "accepted"}
    missing = required - set(reader.fieldnames or [])
    if missing:
        raise ValueError(
            f"Colonne mancanti in clumps_leaves.csv: {sorted(missing)}"
        )
    for row in reader:
        leaf_id = int(row["clump_id"])
        node_id = int(row["node_id"])
        leaf_to_node[leaf_id] = node_id
        accepted = row["accepted"].strip().lower() in {
            "true", "1", "yes", "y"
        }
        if accepted:
            accepted_leaf_ids.add(leaf_id)
accepted_leaf_node_ids = {
    leaf_to_node[leaf_id]
    for leaf_id in accepted_leaf_ids
    if leaf_id in leaf_to_node
}
# Verifica che gli ID usati dal catalogo siano coerenti con l'albero.
for node_id in accepted_leaf_node_ids:
    if node_id not in tree:
        raise ValueError(
            f"Il node_id {node_id} del catalogo non è presente nell'albero."
        )
# Cache dei discendenti di ciascun nodo.
def descendants(node_id):
    result = [node_id]
    for child in tree[node_id].get("children", []):
        child = int(child)
        result.extend(descendants(child))
    return result
_descendants_cache = {
    node_id: descendants(node_id)
    for node_id in tree
}
# Nodi da considerare: foglie accettate e tutti i loro antenati.
"""valid_nodes = set()
for leaf_node_id in accepted_leaf_node_ids:
    current = leaf_node_id
    while current is not None:
        current = int(current)
        valid_nodes.add(current)
        parent = tree[current].get("parent")
        current = None if parent is None else int(parent)"""
valid_nodes = set(tree.keys())
levels_total = max(
    int(node.get("level", 0)) for node in tree.values()
) + 1
def get_style_level(level, total_levels):
    """Stile: tronco grigio tratteggiato; rami colorati per livello."""
    level = int(level)
    if level == 0:
        return "dimgrey", 0.8, "--"
    norm = (level - 1) / max(1, total_levels - 2)
    cmap = mpl.colormaps["coolwarm"]
    return cmap(norm), 0.9, "-"
# ============================================================
# 2. SFONDO 3D, CARICATO UNA SOLA VOLTA
# ============================================================
print("--- 2. Estrazione dello sfondo AMR 3D ---", flush=True)
ds_raw = yt.load(FILENAME)
left_edge_kpc = center_kpc - HALF_WIDTH_KPC
right_edge_kpc = center_kpc + HALF_WIDTH_KPC
left_edge = ds_raw.arr(left_edge_kpc, "kpc")
right_edge = ds_raw.arr(right_edge_kpc, "kpc")
dims = np.array([N_GRID, N_GRID, N_GRID], dtype=int)
bbox = np.column_stack((left_edge_kpc, right_edge_kpc))
cg = ds_raw.arbitrary_grid(
    left_edge=left_edge,
    right_edge=right_edge,
    dims=dims
)
ds_uni = yt.load_uniform_grid(
    data={
        ("gas", "density"): (
            cg[("parthenon", "cons_density")]
            .in_units("g/cm**3")
            .d,
            "g/cm**3"
        )
    },
    domain_dimensions=dims,
    length_unit="kpc",
    bbox=bbox,
    geometry="cartesian"
)
del cg, ds_raw
gc.collect()
# ============================================================
# 3. FUNZIONI PER LA PROIEZIONE DELLE CELLE
# ============================================================
def pixel_bounds(c0, c1, minimum, width, n):
    """Converte i bordi di una cella in indici pixel validi."""
    start = int(np.clip(
        np.floor((c0 - minimum) / width * n),
        0,
        n - 1
    ))
    end = int(np.clip(
        np.ceil((c1 - minimum) / width * n),
        0,
        n
    ))
    if end <= start:
        end = min(n, start + 1)
    return start, end
def put_cells_on_grid(ids, h_idx, v_idx, h_min, v_min, width, n):
    """Crea una maschera 2D usando le celle indicate da ids."""
    mask = np.zeros((n, n), dtype=float)
    if len(ids) == 0:
        return mask
    h = xyz_kpc[ids, h_idx]
    v = xyz_kpc[ids, v_idx]
    dh = widths_kpc[ids, h_idx]
    dv = widths_kpc[ids, v_idx]
    for hc, vc, dhc, dvc in zip(h, v, dh, dv):
        hs, he = hc - dhc / 2.0, hc + dhc / 2.0
        vs, ve = vc - dvc / 2.0, vc + dvc / 2.0
        ih0, ih1 = pixel_bounds(hs, he, h_min, width, n)
        iv0, iv1 = pixel_bounds(vs, ve, v_min, width, n)
        mask[iv0:iv1, ih0:ih1] = 1.0
    return mask
def draw_contour(ax, H_grid, V_grid, mask, color, linewidth,
                 linestyle="-", zorder=5, alpha=0.9):
    """Disegna un contorno se la maschera contiene celle."""
    if not np.any(mask):
        return
    smooth = gaussian_filter(mask, sigma=0.5)
    ax.contour(
        H_grid,
        V_grid,
        smooth,
        levels=[0.5],
        colors=[color],
        linewidths=linewidth,
        linestyles=linestyle,
        alpha=alpha,
        zorder=zorder
    )
# ============================================================
# 4. GENERAZIONE DI UNA SLICE
# ============================================================
def create_slice(axis, offset_kpc):
    axes = {
        "x": (0, 1, 2),
        "y": (1, 2, 0),
        "z": (2, 0, 1)
    }
    ax_idx, h_idx, v_idx = axes[axis]
    slice_center = center_kpc.copy()
    slice_center[ax_idx] += float(offset_kpc)
    h_min = center_kpc[h_idx] - HALF_WIDTH_KPC
    v_min = center_kpc[v_idx] - HALF_WIDTH_KPC
    width = 2.0 * HALF_WIDTH_KPC
    edges_h = np.linspace(h_min, h_min + width, N_GRID + 1)
    edges_v = np.linspace(v_min, v_min + width, N_GRID + 1)
    H_grid, V_grid = np.meshgrid(
        (edges_h[:-1] + edges_h[1:]) / 2.0,
        (edges_v[:-1] + edges_v[1:]) / 2.0
    )
    # Una cella interseca il piano se il piano passa dentro il suo spessore.
    in_slice = np.abs(
        xyz_kpc[:, ax_idx] - slice_center[ax_idx]
    ) <= 0.5 * widths_kpc[:, ax_idx] * 1.01
    plot = yt.SlicePlot(
        ds_uni,
        axis,
        ("gas", "density"),
        center=slice_center,
        width=(width, "kpc"),
        origin="native"
    )
    plot.set_unit(("gas", "density"), "g/cm**3")
    plot.set_axes_unit("kpc")
    plot.annotate_scale(size_bar_args={"color": "white"})
    plot.annotate_title(
        f"Hierarchical tree - axis {axis.upper()} "
        f"(offset {offset_kpc:+.3f} kpc)"
    )
    plot.render()
    ax = plot.plots[("gas", "density")].axes
    if np.any(in_slice):
        slice_indices = np.flatnonzero(in_slice)
        # --------------------------------------------------------
        # Tronchi e rami: usano le celle proprie del nodo.
        # Non dipendono dalle foglie che attraversano la slice.
        # --------------------------------------------------------
        for node_id in sorted(valid_nodes):
            node = tree[node_id]
            if bool(node.get("is_leaf", False)):
                continue
            key = f"node_{node_id}"
            if key not in node_membership.files:
                raise KeyError(
                    f"Manca {key} in membership_nodes.npz"
                )
            node_ids = np.asarray(node_membership[key], dtype=np.int64)
            node_ids = node_ids[
                in_slice[node_ids]
            ]
            if len(node_ids) == 0:
                continue
            mask = put_cells_on_grid(
                node_ids, h_idx, v_idx, h_min, v_min, width, N_GRID
            )
            color, linewidth, linestyle = get_style_level(
                node.get("level", 0), levels_total
            )
            draw_contour(
                ax, H_grid, V_grid, mask,
                color, linewidth, linestyle,
                zorder=3 + int(node.get("level", 0)),
                alpha=0.9
            )
        # --------------------------------------------------------
        # Foglie: usano la membership delle foglie e sono rosse.
        # Solo foglie accettate e che attraversano la slice.
        # --------------------------------------------------------
        for leaf_id in sorted(accepted_leaf_ids):
            leaf_ids = np.flatnonzero(
                (leaf_membership == leaf_id) & in_slice
            )
            if len(leaf_ids) == 0:
                continue
            mask = put_cells_on_grid(
                leaf_ids, h_idx, v_idx, h_min, v_min, width, N_GRID
            )
            draw_contour(
                ax, H_grid, V_grid, mask,
                "crimson", 1.2, "-", zorder=20, alpha=1.0
            )
    filename = (
        f"Slice_Tree_{axis.upper()}_off{offset_kpc:+.3f}.png"
    )
    filepath = os.path.join(OUT_DIR, filename)
    plot.save(filepath)
    print(f"Salvata: {filepath}", flush=True)
    return filepath
# ============================================================
# 5. VIDEO
# ============================================================
def make_video(paths, filename):
    if len(paths) < 2:
        return
    output = os.path.join(OUT_DIR, filename)
    try:
        with imageio.get_writer(output, fps=10) as writer:
            for path in paths:
                writer.append_data(imageio.v2.imread(path))
        print(f"Video salvato: {output}", flush=True)
    except Exception as exc:
        print(f"Errore durante il video {filename}: {exc}", flush=True)
# ============================================================
# 6. MAIN
# ============================================================
print("--- Generazione slices ---", flush=True)
frames = {"x": [], "y": [], "z": []}
for axis in ("x", "y", "z"):
    for offset in OFFSETS:
        frames[axis].append(create_slice(axis, offset))
# Decommenta se OFFSETS contiene almeno due valori.
# make_video(frames["x"], "Tomografia_Gerarchica_X.mp4")
# make_video(frames["y"], "Tomografia_Gerarchica_Y.mp4")
# make_video(frames["z"], "Tomografia_Gerarchica_Z.mp4")
print("Pipeline delle slices completata.", flush=True)
