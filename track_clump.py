#!/usr/bin/env python3
"""Track one leaf clump across consecutive cf4_tree.py output directories."""
import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

KMS_MYR_TO_KPC = 1.022712165e-3


def read_snapshot(folder):
    folder = Path(folder)
    with (folder / 'metadata.json').open() as f:
        meta = json.load(f)
    with (folder / 'tree_nodes.csv').open(newline='') as f:
        rows = list(csv.DictReader(f))

    leaves = {}
    for row in rows:
        if row.get('node_type') != 'leaf':
            continue
        # In cf4_tree.py clump_id in a leaf is the leaf_id.
        cid = int(row['clump_id'])
        if row.get('accepted', '').lower() not in ('true', '1'):
            continue
        leaves[cid] = row

    with np.load(folder / 'membership_leaves.npz') as z:
        required = {
            'xyz_kpc', 'widths_kpc', 'clump_id', 'rho_g_cm3',
            'temp_K', 'cell_mass_Msun', 'cell_volume_pc3', 'velocity_km_s'
        }
        missing = required.difference(z.files)
        if missing:
            raise ValueError(
                f'{folder}: membership_leaves.npz does not contain {sorted(missing)}. '
                'Regenerate the results with the modified version of the finder.'
            )
        arrays = {name: z[name].copy() for name in required}

    cell_ids = arrays['clump_id']
    for cid in list(leaves):
        ix = np.flatnonzero(cell_ids == cid)
        if not len(ix):
            del leaves[cid]
            continue
        leaves[cid]['_ix'] = ix

    return {'folder': folder, 'meta': meta, 'arrays': arrays, 'leaves': leaves}


def number(row, key, default=np.nan):
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return default


def leaf_properties(snapshot, cid):
    row = snapshot['leaves'][cid]
    a = snapshot['arrays']
    ix = row['_ix']
    xyz = a['xyz_kpc'][ix]
    widths = a['widths_kpc'][ix]
    mass = a['cell_mass_Msun'][ix]
    volume = a['cell_volume_pc3'][ix]
    rho = a['rho_g_cm3'][ix]
    temp = a['temp_K'][ix]
    vel = a['velocity_km_s'][ix]
    mtot = float(mass.sum())
    vtot = float(volume.sum())
    if mtot > 0:
        center = np.average(xyz, axis=0, weights=mass)
        vmean = np.average(vel, axis=0, weights=mass)
        tmean = float(np.average(temp, weights=mass))
        rhobar = float(np.average(rho, weights=mass))
    else:
        center = np.mean(xyz, axis=0)
        vmean = np.mean(vel, axis=0)
        tmean = float(np.mean(temp))
        rhobar = float(np.mean(rho))
    req_kpc = float(np.cbrt(3 * vtot / (4 * math.pi)) / 1000.0)
    return {
        'cid': cid, 'row': row, 'ix': ix, 'xyz': xyz, 'widths': widths,
        'mass_cells': mass, 'mass': mtot, 'volume_pc3': vtot,
        'center': center, 'velocity': vmean, 'radius_kpc': req_kpc,
        'rho_mean_g_cm3': rhobar, 'temp_mean_K': tmean,
        'n_cells': int(len(ix)),
    }


def directed_mass_coverage(source, target, predicted_source_xyz):
    """Mass fraction of source cells spatially represented in the target."""
    if not len(source['xyz']) or not len(target['xyz']):
        return 0.0
    tree = cKDTree(target['xyz'])
    dist, j = tree.query(predicted_source_xyz, k=1)
    # Due allowance for the half-diagonal of each AMR cell.
    ds = 0.5 * np.linalg.norm(source['widths'], axis=1)
    dt = 0.5 * np.linalg.norm(target['widths'][j], axis=1)
    tol = ds + dt + 1.0e-8
    hit = dist <= tol
    denom = float(source['mass_cells'].sum())
    return float(source['mass_cells'][hit].sum() / denom) if denom > 0 else 0.0


def pair_metrics(source, target, dt_myr):
    shift = source['velocity'] * (dt_myr * KMS_MYR_TO_KPC)
    predicted_center = source['center'] + shift
    predicted_cells = source['xyz'] + source['velocity'][None, :] * (
        dt_myr * KMS_MYR_TO_KPC
    )
    distance = float(np.linalg.norm(predicted_center - target['center']))
    src_cover = directed_mass_coverage(source, target, predicted_cells)
    # Reverse coverage asks whether target material is near the advected source.
    # Use a temporary source geometry at its velocity-shifted position.
    advected = dict(source)
    advected['xyz'] = predicted_cells
    tgt_cover = directed_mass_coverage(target, advected, target['xyz'])
    overlap_f1 = (
        2 * src_cover * tgt_cover / (src_cover + tgt_cover)
        if src_cover + tgt_cover > 0 else 0.0
    )
    mass_ratio = target['mass'] / source['mass'] if source['mass'] > 0 else np.inf
    log_mass_penalty = (
        abs(math.log(mass_ratio)) if mass_ratio > 0 and np.isfinite(mass_ratio)
        else 99.0
    )
    scale = max(source['radius_kpc'] + target['radius_kpc'], 1.0e-3)
    distance_score = math.exp(-distance / scale)
    mass_score = math.exp(-log_mass_penalty)
    # Spatial membership is the strongest signal; advection and mass are
    # useful when AMR regridding makes exact cell overlap unavailable.
    score = 0.55 * overlap_f1 + 0.25 * distance_score + 0.20 * mass_score
    return {
        'score': score, 'distance_kpc': distance,
        'overlap_source_fraction': src_cover,
        'overlap_target_fraction': tgt_cover,
        'overlap_f1': overlap_f1, 'mass_ratio': mass_ratio,
        'predicted_center': predicted_center,
    }


def outside_box(point, meta):
    left = meta.get('domain_left_kpc')
    right = meta.get('domain_right_kpc')
    if left is None or right is None:
        return False, 'box_limits_missing'
    p = np.asarray(point)
    return bool(np.any(p < np.asarray(left)) or np.any(p > np.asarray(right))), 'domain'


def outside_roi(point, meta):
    args = meta.get('arguments', {})
    center = meta.get('center_kpc')
    radius = args.get('radius_kpc')
    if center is None or radius is None:
        return False
    return bool(np.linalg.norm(np.asarray(point) - np.asarray(center)) > float(radius))


def output_row(snapshot, cid, props, status, event, match=None):
    meta = snapshot['meta']
    r = props['row'] if props else {}
    result = {
        'snapshot': str(snapshot['meta'].get('snapshot', snapshot['folder'])),
        'time_Myr': meta.get('time_Myr', np.nan),
        'status': status, 'event': event,
        'clump_id': cid if cid is not None else '',
        'mass_Msun': props['mass'] if props else np.nan,
        'volume_pc3': props['volume_pc3'] if props else np.nan,
        'radius_kpc': props['radius_kpc'] if props else np.nan,
        'x_kpc': props['center'][0] if props else np.nan,
        'y_kpc': props['center'][1] if props else np.nan,
        'z_kpc': props['center'][2] if props else np.nan,
        'vx_km_s': props['velocity'][0] if props else np.nan,
        'vy_km_s': props['velocity'][1] if props else np.nan,
        'vz_km_s': props['velocity'][2] if props else np.nan,
        'n_cells': props['n_cells'] if props else np.nan,
        'mean_density_g_cm3': props['rho_mean_g_cm3'] if props else np.nan,
        'mean_temperature_K': props['temp_mean_K'] if props else np.nan,
        'finder_accepted': r.get('accepted', ''),
        'finder_birth_density': r.get('birth_density_g_cm3', ''),
        'finder_level': r.get('level', ''),
        'finder_score': match['score'] if match else np.nan,
        'match_distance_kpc': match['distance_kpc'] if match else np.nan,
        'overlap_source_fraction': match['overlap_source_fraction'] if match else np.nan,
        'overlap_target_fraction': match['overlap_target_fraction'] if match else np.nan,
        'mass_ratio_next_over_previous': match['mass_ratio'] if match else np.nan,
    }
    # Include every additional aggregate property from tree_nodes.csv, without
    # overwriting the consistently recomputed quantities above.
    for key, value in r.items():
        if key not in result and not key.startswith('_'):
            result['finder_' + key] = value
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('snapshot_dirs', nargs='+', help='Output folders, in temporal order')
    parser.add_argument('--start-id', type=int, required=True,
                        help='clump_id (leaf_id) in the first snapshot')
    parser.add_argument('--output', default='clump_history.csv')
    parser.add_argument('--events', default='clump_events.json')
    parser.add_argument('--min-score', type=float, default=0.20)
    parser.add_argument('--min-overlap', type=float, default=0.03)
    parser.add_argument('--max-distance-kpc', type=float, default=10.0)
    parser.add_argument('--max-mass-ratio', type=float, default=20.0)
    parser.add_argument('--event-overlap', type=float, default=0.10,
                        help='minimum fraction to report split/merge')
    args = parser.parse_args()

    snapshots = [read_snapshot(p) for p in args.snapshot_dirs]
    if len(snapshots) < 1:
        raise SystemExit('At least one snapshot is required')
    first = snapshots[0]
    if args.start_id not in first['leaves']:
        available = sorted(first['leaves'])
        raise SystemExit(
            f'clump_id {args.start_id} is not an accepted leaf in the first snapshot. '
            f'Available IDs: {available}'
        )

    current_id = args.start_id
    current_props = leaf_properties(first, current_id)
    history = [output_row(first, current_id, current_props, 'TRACK_START', 'start')]
    events = []
    active = True

    for index in range(1, len(snapshots)):
        prev, nxt = snapshots[index - 1], snapshots[index]
        prev_time = float(prev['meta']['time_Myr'])
        next_time = float(nxt['meta']['time_Myr'])
        dt = next_time - prev_time
        if dt <= 0:
            raise ValueError(f'Non-increasing times: {prev_time} -> {next_time} Myr')

        if not active:
            history.append(output_row(
                nxt, None, None, 'NOT_TRACKED_AFTER_TERMINATION', 'not_followed'
            ))
            continue

        source = current_props
        targets = {
            cid: leaf_properties(nxt, cid) for cid in nxt['leaves']
        }
        # Calculate scores between the tracked leaf and all subsequent leaves.
        pairs = {
            cid: pair_metrics(source, target, dt)
            for cid, target in targets.items()
        }
        ranked = sorted(pairs, key=lambda cid: pairs[cid]['score'], reverse=True)
        viable = []
        for cid in ranked:
            m = pairs[cid]
            ratio = m['mass_ratio']
            passes = (
                m['score'] >= args.min_score and
                (m['overlap_f1'] >= args.min_overlap or
                 m['distance_kpc'] <= args.max_distance_kpc) and
                ratio <= args.max_mass_ratio and
                ratio >= 1.0 / args.max_mass_ratio
            )
            if passes:
                viable.append(cid)

        if not viable:
            predicted = source['center'] + source['velocity'] * (dt * KMS_MYR_TO_KPC)
            out_box, _ = outside_box(predicted, nxt['meta'])
            if out_box:
                status, event = 'EXITED_BOX', 'predicted_center_outside_domain'
            elif outside_roi(predicted, nxt['meta']):
                status, event = 'OUTSIDE_ANALYSIS_REGION', 'predicted_center_outside_finder_roi'
            else:
                status, event = 'NOT_FOUND', 'no_plausible_leaf_match'
            history.append(output_row(nxt, None, None, status, event))
            events.append({'step': index, 'time_Myr': next_time,
                           'status': status, 'predicted_center_kpc': predicted.tolist()})
            active = False
            continue

        chosen = viable[0]
        chosen_metric = pairs[chosen]
        event_parts = []

        # Split: multiple leaves of the new snapshot receive a significant share
        # of the spatially advected mass of the previous leaf.
        split_ids = [
            cid for cid in viable
            if pairs[cid]['overlap_source_fraction'] >= args.event_overlap
        ]
        if len(split_ids) >= 2:
            event_parts.append('SPLIT')
            events.append({
                'step': index, 'time_Myr': next_time, 'event': 'SPLIT',
                'parent_clump_id': current_id,
                'child_clump_ids': split_ids,
                'continued_as_clump_id': chosen,
            })

        # Merge: checks if another leaf from the previous catalog contributes
        # spatially in a significant way to the chosen leaf in the new snapshot.
        merge_sources = []
        for old_id in prev['leaves']:
            if old_id == current_id:
                continue
            old = leaf_properties(prev, old_id)
            other_dt = dt
            other = pair_metrics(old, targets[chosen], other_dt)
            if other['overlap_target_fraction'] >= args.event_overlap:
                merge_sources.append(old_id)
        if merge_sources:
            event_parts.append('MERGER')
            events.append({
                'step': index, 'time_Myr': next_time, 'event': 'MERGER',
                'merged_into_clump_id': chosen,
                'other_previous_clump_ids': merge_sources,
            })

        status = 'TRACKED'
        event = '+'.join(event_parts) if event_parts else 'CONTINUATION'
        next_props = targets[chosen]
        history.append(output_row(nxt, chosen, next_props, status, event, chosen_metric))
        current_id, current_props = chosen, next_props

    fieldnames = list(dict.fromkeys(k for row in history for k in row))
    with Path(args.output).open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)
    Path(args.events).write_text(json.dumps(events, indent=2))
    print(f'History written to {args.output}')
    print(f'Events written to {args.events}')
    for row in history:
        print(f"t={row['time_Myr']} Myr | {row['status']} | {row['event']} | id={row['clump_id']}")


if __name__ == '__main__':
    main()