"""Native Cartesian factor-two SMR/AMR face graph; bypasses yt contour linking.
Requires temperature_clump_utils.py beside this file, plus scipy. No 3-D regridding.
Outputs:
  clumps_leaves.csv    final clumps only (leaves)
  tree_nodes.csv       complete catalogue: trunks, branches and leaves
  tree_hierarchy.json  parent/children relationships
  membership_leaves.npz cell membership of final leaves
"""
import csv
import json
import math
from pathlib import Path
import resource
import sys
import time
import numpy as np
import scipy
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
import yt
import temperature_clump_utils as common
tree_hierarchy = {}
node_counter = 0
final_membership = None
def read_cells(region, a):
    pieces = []
    total = 0
    tf = tuple(a.temperature_field)
    fields = [('gas', 'density'), tf, ('index', 'cell_volume')]
    fields += [('gas', f'velocity_{axis}') for axis in 'xyz']
    fields += [('index', x) for x in ['x', 'y', 'z', 'dx', 'dy', 'dz']]

    for chunk in region.chunks(fields, 'io'):
        rho = chunk['gas', 'density'].to_value('g/cm**3')
        temp = chunk[tf].to_value('K')
        total += len(temp)
        if not np.all(np.isfinite(rho) & (rho > 0) &
                      np.isfinite(temp) & (temp > 0)):
            raise ValueError('Invalid density/temperature in ROI')
        mask = (temp >= a.tmin) & (temp < a.tmax)
        if np.any(mask):
            volume = chunk['index', 'cell_volume'][mask].to_value('pc**3')
            mass = (
                chunk['gas', 'density'][mask] *
                chunk['index', 'cell_volume'][mask]
            ).to_value('Msun')
            xyz = np.column_stack([
                chunk['index', x][mask].to_value('kpc')
                for x in 'xyz'
            ])
            widths = np.column_stack([
                chunk['index', 'd' + x][mask].to_value('kpc')
                for x in 'xyz'
            ])
            velocity = np.column_stack([
                chunk['gas', f'velocity_{axis}'][mask].to_value('km/s')
                for axis in 'xyz'
            ])

            pieces.append({
                'rho': rho[mask],
                'temp': temp[mask],
                'volume': volume,
                'mass': mass,
                'xyz': xyz,
                'widths': widths,
                'velocity': velocity

            })
        chunk.clear_data()
    if not pieces:
        return {
            k: np.empty((0, 3) if k in ['xyz', 'widths', 'velocity'] else 0)
            for k in ['rho', 'temp', 'volume', 'mass', 'xyz', 'widths', 'velocity']
        }, total

    cells = {k: np.concatenate([p[k] for p in pieces]) for k in pieces[0]}
    for name in ['volume', 'mass']:
        if not np.all(np.isfinite(cells[name]) & (cells[name] > 0)):
            raise ValueError(f'Invalid cell {name}')
        if not np.all(np.isfinite(cells['velocity'])):
            raise ValueError('Invalid cell velocities')
    return cells, total
def build_graph(xyz, widths, left, root_widths, domain_dimensions,
                batch=250000):
    n = len(xyz)
    if not np.all(np.isfinite(xyz)) or not np.all(
            np.isfinite(widths) & (widths > 0)):
        raise ValueError('Invalid cell positions or widths')
    if not n:
        return coo_matrix((0, 0), dtype=np.uint8).tocsr()
    ratio = root_widths / widths
    lev3 = np.rint(np.log2(ratio)).astype(np.int64)
    if (np.any(lev3 < 0) or
        np.any(lev3 != lev3[:, 0, None]) or
        not np.allclose(ratio, np.exp2(lev3), rtol=1e-8, atol=1e-8)):
        raise ValueError(
            'Requires isotropic factor-two refinement of Cartesian root cells'
        )
    level = lev3[:, 0]
    if level.max() > 30:
        raise ValueError('Unexpected refinement level >30')
    logical = (xyz - left) / widths - 0.5
    ijk = np.rint(logical).astype(np.int64)
    if np.max(np.abs(logical - ijk)) > 1e-5:
        raise ValueError('Cell centers are not aligned with the dyadic lattice')
    groups = {}
    def key(q, dims):
        return q[:, 0] + dims[0] * (q[:, 1] + dims[1] * q[:, 2])
    for lev in np.unique(level):
        ids = np.flatnonzero(level == lev)
        dims = np.asarray(domain_dimensions, dtype=np.int64) * (2 ** int(lev))
        if math.prod(map(int, dims)) >= np.iinfo(np.int64).max:
            raise ValueError('Logical cell keys exceed int64')
        q = ijk[ids]
        if np.any(q < 0) or np.any(q >= dims):
            raise ValueError('Cell outside domain')
        keys = key(q, dims)
        order = np.argsort(keys)
        keys = keys[order]
        ids = ids[order]
        if np.any(np.diff(keys) == 0):
            raise ValueError('Duplicate active cells')
        groups[int(lev)] = (ids, keys, dims)
    edge_i = []
    edge_j = []
    def lookup(q, source, target_group, check_overlap=False):
        ids, keys, dims = target_group
        valid = np.all((q >= 0) & (q < dims), axis=1)
        q = q[valid]
        source = source[valid]
        k = key(q, dims)
        pos = np.searchsorted(keys, k)
        match = pos < len(keys)
        match[match] &= keys[pos[match]] == k[match]
        if check_overlap:
            if np.any(match):
                raise ValueError(
                    'Overlapping active coarse/fine cells; '
                    'cannot build graph safely'
                )
        else:
            edge_i.append(source[match])
            edge_j.append(ids[pos[match]])
    for lev, (ids, keys, dims) in groups.items():
        for start in range(0, len(ids), batch):
            source = ids[start:start + batch]
            q = ijk[source]
            for axis in range(3):
                neighbor = q.copy()
                neighbor[:, axis] += 1
                lookup(neighbor, source, groups[lev])
            for coarse in groups:
                if coarse >= lev:
                    continue
                factor = 2 ** (lev - coarse)
                parent = q // factor
                lookup(parent, source, groups[coarse], check_overlap=True)
                for axis in range(3):
                    for side in [-1, 1]:
                        face = q[:, axis] + (side == 1)
                        aligned = (face % factor) == 0
                        neighbor = parent[aligned].copy()
                        neighbor[:, axis] += side
                        lookup(neighbor, source[aligned], groups[coarse])
    i = np.concatenate(edge_i)
    j = np.concatenate(edge_j)
    graph = coo_matrix(
        (np.ones(len(i), dtype=np.uint8), (i, j)),
        shape=(n, n)
    ).tocsr()
    return graph.maximum(graph.T)
def components(graph, ids, min_cells):
    if not len(ids):
        return []
    sub = graph[ids][:, ids]
    count, labels = connected_components(sub, directed=False)
    sizes = np.bincount(labels, minlength=count)
    order = np.argsort(labels, kind='stable')
    ends = np.cumsum(sizes)
    starts = ends - sizes
    return [
        ids[order[s:e]]
        for s, e, size in zip(starts, ends, sizes)
        if size >= min_cells
    ]
def segment(graph, rho, a, lower):
    """Build the tree with exactly the cf4_clumps leaf algorithm."""
    global tree_hierarchy, node_counter, final_membership
    tree_hierarchy = {}
    node_counter = 0
    final_membership = np.full(len(rho), -1, dtype=np.int64)
    all_nodes, found_leaves = [], []
    calls = 1
    min_cells = a.min_cells
    def add_node(ids, birth, parent, level, leaf):
        global node_counter
        nid = node_counter
        node_counter += 1
        node = {'node_id': int(nid),
                'parent': None if parent is None else int(parent),
                'children': [], 'level': int(level),
                'birth_density': float(birth),
                'is_leaf': bool(leaf),
                'cell_ids': np.asarray(ids, dtype=np.int64)}
        all_nodes.append(node)
        tree_hierarchy[nid] = node
        if parent is not None:
            tree_hierarchy[parent]['children'].append(nid)
        return node
    roots = components(graph, np.arange(len(rho)), min_cells)
    if a.mode == 'connected':
        for root in roots:
            node = add_node(root, lower, None, 0, True)
            node['leaf_id'] = len(found_leaves)
            found_leaves.append(node)
            final_membership[root] = node['leaf_id']
        return all_nodes, found_leaves, calls
    # Identical stack and threshold logic to cf4_clumps.py.
    stack = [(root, lower, lower * a.density_step, None, 0)
             for root in roots]
    while stack:
        ids, birth, threshold, parent, level = stack.pop()
        split = False
        upper = rho[ids].max()
        while threshold <= upper:
            eligible = ids[rho[ids] >= threshold]
            if len(eligible) < 2 * min_cells:
                break
            calls += 1
            if calls > a.max_contour_calls:
                raise RuntimeError('Component-call limit exceeded; no completed catalog')
            kids = components(graph, eligible, min_cells)
            if len(kids) >= 2:
                branch = add_node(ids, birth, parent, level, False)
                stack.extend((kid, threshold, threshold * a.density_step,
                              branch['node_id'], level + 1) for kid in kids)
                split = True
                break
            threshold *= a.density_step
        if not split:
            leaf = add_node(ids, birth, parent, level, True)
            leaf['leaf_id'] = len(found_leaves)
            found_leaves.append(leaf)
            final_membership[ids] = leaf['leaf_id']
    return all_nodes, found_leaves, calls
def measure(cells, ids, birth, clump_id, center, a):
    v = cells['volume'][ids]
    m = cells['mass'][ids]
    w = cells['widths'][ids] * 1000
    xyz = cells['xyz'][ids]
    t = cells['temp'][ids]
    pos = np.average(xyz, axis=0, weights=m)
    r = float(np.cbrt(3 * v.sum() / (4 * np.pi)))
    max_edge = float(w.max())
    boundary = bool(np.any(
        np.linalg.norm(np.abs(xyz - center) + w / 2000, axis=1)
        >= a.radius_kpc
    ))
    resolved = bool(
        r >= a.min_radius_cells * max_edge and
        r >= a.min_radius_pc
    )
    values = [
        clump_id, float(birth), len(ids), float(m.sum()), float(v.sum()),
        r, float(w.min()), max_edge, r / max_edge,
        float(np.average(t, weights=m)), *map(float, pos),
        float(np.linalg.norm(pos - center)), boundary, resolved,
        bool(resolved and (a.include_boundary or not boundary))
    ]
    return dict(zip(common.COLUMNS, values))
def node_type(node):
    if node['is_leaf']:
        return 'leaf'
    if node['parent'] is None:
        return 'trunk'
    return 'branch'
def measure_node(cells, node, center, a):
    row = measure(
        cells,
        node['cell_ids'],
        node['birth_density'],
        node.get('leaf_id', node['node_id']),
        center,
        a
    )
    row['node_id'] = int(node['node_id'])
    row['node_type'] = node_type(node)
    row['parent_id'] = -1 if node['parent'] is None else node['parent']
    row['level'] = node['level']
    return row
TREE_COLUMNS = list(common.COLUMNS) + [
    'node_id', 'node_type', 'parent_id', 'level'
]
def serializable_hierarchy():
    return {
        str(node_id): {
            'parent': node['parent'],
            'children': node['children'],
            'level': node['level'],
            'birth_density': node['birth_density'],
            'is_leaf': node['is_leaf']
        }
        for node_id, node in tree_hierarchy.items()
    }
def plot_true_tree_slice(ds, a, c, cells, rows, out):
    import matplotlib as mpl
    from scipy.ndimage import gaussian_filter
    import gc
    print('--- Generazione Tomografia Esatta (Asse Y, offset 0) ---',
          flush=True)
    N_GRID = 1000
    HALF_WIDTH_KPC = a.radius_kpc
    left_edge_kpc = c - HALF_WIDTH_KPC
    right_edge_kpc = c + HALF_WIDTH_KPC
    left_edge = ds.arr(left_edge_kpc, 'kpc')
    right_edge = ds.arr(right_edge_kpc, 'kpc')
    dims = np.array([N_GRID, N_GRID, N_GRID])
    bbox = np.array([
        [left_edge_kpc[i], right_edge_kpc[i]] for i in range(3)
    ])
    cg = ds.arbitrary_grid(
        left_edge=left_edge,
        right_edge=right_edge,
        dims=dims
    )
    ds_uni = yt.load_uniform_grid(
        data={
            ('gas', 'density'): (
                cg[('parthenon', 'cons_density')].in_units(
                    'g/cm**3'
                ).d,
                'g/cm**3'
            )
        },
        domain_dimensions=dims,
        length_unit='kpc',
        bbox=bbox,
        geometry='cartesian'
    )
    del cg
    gc.collect()
    asse = 'y'
    ax_idx, h_idx, v_idx = 1, 2, 0
    slice_center = c.copy()
    h_min, h_max = c[h_idx] - HALF_WIDTH_KPC, c[h_idx] + HALF_WIDTH_KPC
    v_min, v_max = c[v_idx] - HALF_WIDTH_KPC, c[v_idx] + HALF_WIDTH_KPC
    h_edges = np.linspace(h_min, h_max, N_GRID + 1)
    v_edges = np.linspace(v_min, v_max, N_GRID + 1)
    H_grid, V_grid = np.meshgrid(
        (h_edges[:-1] + h_edges[1:]) / 2,
        (v_edges[:-1] + v_edges[1:]) / 2
    )
    p = yt.SlicePlot(
        ds_uni,
        asse,
        ('gas', 'density'),
        center=slice_center,
        width=(2.0 * HALF_WIDTH_KPC, 'kpc'),
        origin='native'
    )
    p.set_unit(('gas', 'density'), 'g/cm**3')
    p.set_axes_unit('kpc')
    p.annotate_scale(size_bar_args={'color': 'white'})
    p.annotate_title('True Hierarchical Tree - Axis Y (Offset: 0.000 kpc)')
    p.render()
    ax = p.plots[('gas', 'density')].axes
    accepted_leaf_ids = {
        int(r['clump_id']) for r in rows if r['accepted']
    }
    leaf_to_node = {
        int(r['clump_id']): int(r['node_id'])
        for r in rows
    }
    accepted_ids = {
        leaf_to_node[leaf_id]
        for leaf_id in accepted_leaf_ids
    }
    xyz_kpc = cells['xyz']
    widths_kpc = cells['widths']
    in_slice = np.abs(
        xyz_kpc[:, ax_idx] - slice_center[ax_idx]
    ) <= (widths_kpc[:, ax_idx] / 2.0 * 1.01)
    if np.any(in_slice):
        id_2d = np.full((N_GRID, N_GRID), -1, dtype=int)
        h_c = xyz_kpc[in_slice, h_idx]
        v_c = xyz_kpc[in_slice, v_idx]
        dh = widths_kpc[in_slice, h_idx]
        dv = widths_kpc[in_slice, v_idx]
        id_slice = final_membership[in_slice]
        for i in range(len(h_c)):
            if id_slice[i] < 0:
                continue
            h_s, h_e = h_c[i] - dh[i] / 2, h_c[i] + dh[i] / 2
            v_s, v_e = v_c[i] - dv[i] / 2, v_c[i] + dv[i] / 2
            ih_s = int(np.clip(
                np.floor((h_s - h_min) / (2 * HALF_WIDTH_KPC) * N_GRID),
                0, N_GRID - 1
            ))
            ih_e = int(np.clip(
                np.ceil((h_e - h_min) / (2 * HALF_WIDTH_KPC) * N_GRID),
                0, N_GRID
            ))
            iv_s = int(np.clip(
                np.floor((v_s - v_min) / (2 * HALF_WIDTH_KPC) * N_GRID),
                0, N_GRID - 1
            ))
            iv_e = int(np.clip(
                np.ceil((v_e - v_min) / (2 * HALF_WIDTH_KPC) * N_GRID),
                0, N_GRID
            ))
            if ih_e == ih_s:
                ih_e = min(N_GRID, ih_e + 1)
            if iv_e == iv_s:
                iv_e = min(N_GRID, iv_e + 1)
            id_2d[iv_s:iv_e, ih_s:ih_e] = id_slice[i]
        def get_descendants(node_id):
            descendants = [node_id]
            for child in tree_hierarchy[node_id]['children']:
                descendants.extend(get_descendants(child))
            return descendants
        valid_nodes_global = set()
        for leaf_node_id in accepted_ids:
            current = leaf_node_id
            while current is not None:
                valid_nodes_global.add(current)
                current = tree_hierarchy[current]['parent']
        levels_tot = (
            max(node['level'] for node in tree_hierarchy.values()) + 1
            if tree_hierarchy else 1
        )
        def get_style_livello(level, total_levels):
            if level == 0:
                return 'dimgrey', 0.5, '--'
            norm_val = (level - 1) / max(1, total_levels - 2)
            cmap = mpl.colormaps['coolwarm']
            return cmap(norm_val), 0.7, 'solid'
        leaf_node_to_leaf_id = {
            int(node['node_id']): int(node['leaf_id'])
            for node in tree_hierarchy.values()
            if node['is_leaf'] and 'leaf_id' in node
        }
        for ancestor_id in valid_nodes_global:
            if ancestor_id in accepted_ids:
                continue
            family_node_ids = get_descendants(ancestor_id)
            family_leaf_ids = [
                leaf_node_to_leaf_id[node_id]
                for node_id in family_node_ids
                if node_id in leaf_node_to_leaf_id
            ]
            mask_branch = np.isin(id_2d, family_leaf_ids).astype(float)
            if mask_branch.sum() > 0:
                level = tree_hierarchy[ancestor_id]['level']
                branch_smooth = gaussian_filter(mask_branch, sigma=0.5)
                color, lw, ls = get_style_livello(level, levels_tot)
                ax.contour(
                    H_grid, V_grid, branch_smooth, levels=[0.5],
                    colors=[color], linewidths=lw, linestyles=ls,
                    alpha=0.85, zorder=3 + level
                )
        for leaf_id in accepted_leaf_ids:
            mask_leaf = (id_2d == leaf_id).astype(float)
            if mask_leaf.sum() > 0:
                leaf_smooth = gaussian_filter(mask_leaf, sigma=0.5)
                ax.contour(
                    H_grid, V_grid, leaf_smooth, levels=[0.5],
                    colors=['crimson'], linewidths=0.9, zorder=15
                )
    filepath = out / 'Slice_TrueTree_Y_offset0.png'
    p.save(filepath)
    print(f'Slice salvata con successo in: {filepath}', flush=True)
def main():
    a = common.arguments()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    ds = None
    meta = dict(
        arguments=vars(a),
        backend='native_face_graph_v1',
        status='started',
        yt_version=yt.__version__,
        scipy_version=scipy.__version__
    )
    def checkpoint(stage):
        meta['stage'] = stage
        meta['elapsed_seconds'] = time.perf_counter() - start
        common.write_json(out / 'metadata.json', meta)
        print(f"{stage} ({meta['elapsed_seconds']:.1f} s)", flush=True)
    try:
        options = (
            json.loads(Path(a.load_options).read_text())
            if a.load_options else {}
        )
        meta['load_options'] = options
        meta['temperature_selection'] = {
            'Tmin_K': a.tmin,
            'Tmax_K': a.tmax,
            'interval': 'Tmin <= T < Tmax'
        }
        filename = common.choose_snapshot(a, options)
        ds = yt.load(filename, **options)
        if ds.dimensionality != 3 or \
                str(ds.geometry).lower().split('.')[-1] != 'cartesian':
            raise ValueError('3D Cartesian mesh required')
        center = (
            ds.domain_center
            if a.center_kpc is None
            else ds.arr(a.center_kpc, 'kpc')
        )
        c = center.to_value('kpc')
        left = ds.domain_left_edge.to_value('kpc')
        right = ds.domain_right_edge.to_value('kpc')
        if np.any(c - a.radius_kpc <= left) or \
                np.any(c + a.radius_kpc >= right):
            raise ValueError(
                'ROI must lie strictly inside domain; no periodic wrapping'
            )
        region = ds.sphere(center, (a.radius_kpc, 'kpc'))
        checkpoint('reading temperature-selected cells')
        cells, total = read_cells(region, a)
        n = len(cells['rho'])
        lower = (
            a.rho_min
            if a.rho_min is not None
            else (float(cells['rho'].min()) if n else 1.)
        )
        upper = float(cells['rho'].max()) if n else lower
        levels = (
            max(1, int(math.ceil(
                (math.log(upper) - math.log(lower)) /
                math.log(a.density_step)
            )))
            if upper >= lower else 0
        )
        meta.update(
            snapshot=filename,
            time_Myr=float(ds.current_time.to_value('Myr')),
            center_kpc=c.tolist(),
            periodicity=list(ds.periodicity),
            length_unit=str(ds.length_unit),
            mass_unit=str(ds.mass_unit),
            time_unit=str(ds.time_unit),
            mu=str(getattr(ds, 'mu', None)),
            region_cells=total,
            selected_cells=n,
            invalid_cells=0,
            selected_mass_Msun=float(cells['mass'].sum()),
            rho_start_g_cm3=lower,
            rho_upper_g_cm3=upper,
            threshold_levels=levels
        )
        eligible = cells['rho'] >= lower
        cells = {k: v[eligible] for k, v in cells.items()}
        meta['eligible_cells'] = len(cells['rho'])
        meta['preflight_seconds'] = time.perf_counter() - start
        region.clear_data()
        if a.mode == 'hierarchical' and levels > a.max_levels:
            raise ValueError('Density-level limit exceeded')
        if a.preflight_only:
            meta['status'] = 'preflight_complete'
            return
        checkpoint('building native face graph')
        graph = build_graph(
            cells['xyz'],
            cells['widths'],
            left,
            (right - left) / ds.domain_dimensions,
            ds.domain_dimensions
        )
        meta['graph_edges'] = int(graph.nnz // 2)
        meta['graph_seconds'] = (
            time.perf_counter() - start - meta['preflight_seconds']
        )
        checkpoint('finding components')
        all_nodes, found_leaves, calls = segment(
            graph, cells['rho'], a, lower
        )
        del graph
        leaf_rows = [
            measure_node(cells, node, c, a)
            for node in found_leaves
        ]
        tree_rows = [
            measure_node(cells, node, c, a)
            for node in all_nodes
        ]
        with (out / 'clumps_leaves.csv').open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=TREE_COLUMNS)
            writer.writeheader()
            writer.writerows(leaf_rows)
        with (out / 'tree_nodes.csv').open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=TREE_COLUMNS)
            writer.writeheader()
            writer.writerows(tree_rows)
        leaf_membership = np.full(
            len(cells['rho']), -1, dtype=np.int64
        )
        for node in found_leaves:
            leaf_membership[node['cell_ids']] = node['leaf_id']
        np.savez_compressed(
            out / 'membership_leaves.npz',
            xyz_kpc=cells['xyz'],
            widths_kpc=cells['widths'],
            clump_id=leaf_membership,
            rho_g_cm3=cells['rho'],
            temp_K=cells['temp'],
            cell_mass_Msun=cells['mass'],
            cell_volume_pc3=cells['volume'],
            velocity_km_s=cells['velocity'],
        )
        node_cell_ids = {
            f"node_{int(node['node_id'])}":
                np.asarray(node["cell_ids"], dtype=np.int64)
            for node in all_nodes
        }

        # Membership con gli indici delle celle appartenenti a ogni nodo
        node_membership = {
            f"node_{int(node['node_id'])}":
                np.asarray(node["cell_ids"], dtype=np.int64)
            for node in all_nodes
        }

        np.savez_compressed(
            out / "membership_nodes.npz",
            **node_membership
        )

        with (out / 'tree_hierarchy.json').open('w') as f:
            json.dump(serializable_hierarchy(), f, indent=4)
        meta.update(
            component_calls=calls,
            total_tree_nodes=len(tree_rows),
            tree_trunks=sum(r['node_type'] == 'trunk' for r in tree_rows),
            tree_branches=sum(r['node_type'] == 'branch' for r in tree_rows),
            tree_leaves=sum(r['node_type'] == 'leaf' for r in tree_rows),
            candidate_clumps=len(leaf_rows),
            accepted_clumps=sum(r['accepted'] for r in leaf_rows),
            accepted_mass_Msun=sum(
                r['mass_Msun'] for r in leaf_rows if r['accepted']
            ),
            domain_left_kpc=left.tolist(),
            domain_right_kpc=right.tolist(),

        )
        meta.update(common.histogram(leaf_rows, a, out))
        meta['overlay_errors'] = []
        checkpoint('catalog saved')
        
        if a.overlay_axes != 'none':
            try:
                plot_true_tree_slice(ds, a, c, cells, leaf_rows, out)
            except Exception as exc:
                meta['overlay_errors'].append(f'Plotting error: {exc}')
                print(f'Errore durante il plot della slice: {exc}')
                
        meta['status'] = (
            'complete'
            if not meta['overlay_errors']
            else 'catalog_complete_overlay_failed'
        )
    except Exception as exc:
        meta.update(status='failed', error=str(exc))
        raise
    finally:
        meta['elapsed_seconds'] = time.perf_counter() - start
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        divisor = 1024 ** 3 if sys.platform == 'darwin' else 1024 ** 2
        meta['peak_RSS_GiB'] = rss / divisor
        common.write_json(out / 'metadata.json', meta)
        print(
            f"Status: {meta['status']}; "
            f"elapsed {meta['elapsed_seconds']:.1f} s",
            flush=True
        )
        if ds is not None:
            common.close_dataset(ds)
if __name__ == '__main__':
    main()
