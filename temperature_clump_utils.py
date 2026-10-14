#!/usr/bin/env python3
"""Cold AMR clouds: yt connectivity, explicit density hierarchy, CSV and overlays.
See README_clumps.md. Run one process per snapshot, not mpirun.
"""
import argparse
import csv
import glob
import json
import math
from pathlib import Path
import resource
import sys
import time

import numpy as np
import yt

COLD = ('gas', 'clump_cold_density')


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument('--snapshot', help='Explicit .phdf or .rhdf; fastest selection')
    src.add_argument('--pattern', help='Quoted glob; read metadata to select nearest time')
    p.add_argument('--time-myr', type=float, default=45.)
    p.add_argument('--time-tolerance-myr', type=float, default=0.5)
    p.add_argument('--out', required=True)
    p.add_argument('--label', default='dust')
    p.add_argument('--radius-kpc', type=float, default=2.)
    p.add_argument('--center-kpc', nargs=3, type=float, help='SMBH position; default domain center')
    p.add_argument('--logt-center', type=float, required=True,
                   help='Center of temperature band: log10(T/K), e.g. 4 or 5.5')
    p.add_argument('--logt-halfwidth', type=float, default=0.25,
                   help='Half-width in dex; same value for both phases')
    p.add_argument('--temperature-field', nargs=2, default=['gas', 'temperature'])
    p.add_argument('--load-options', help='JSON with yt.load kwargs, e.g. units_override, parameters')
    p.add_argument('--mode', choices=['hierarchical', 'connected'], default='hierarchical')
    p.add_argument('--rho-min', type=float, help='Fixed starting density in g/cm^3; default cold minimum')
    p.add_argument('--density-step', type=float, default=4.)
    p.add_argument('--min-cells', type=int, default=32)
    p.add_argument('--min-radius-cells', type=float, default=2., help='r_eff / max cell edge')
    p.add_argument('--min-radius-pc', type=float, default=0., help='Common physical cut for run comparisons')
    p.add_argument('--include-boundary', action='store_true')
    p.add_argument('--max-levels', type=int, default=64)
    p.add_argument('--max-contour-calls', type=int, default=10000)
    p.add_argument('--preflight-only', action='store_true')
    p.add_argument('--overlay-axes', default='z', help='z, xyz, or none')
    p.add_argument('--overlay-limit', type=int, default=100, help='Largest accepted clumps to outline')
    p.add_argument('--pixels', type=int, default=800)
    p.add_argument('--log-bins', nargs=3, type=float, default=[-2., 5., .2], metavar=('MIN','MAX','STEP'))
    a = p.parse_args()
    for name in ['radius_kpc','density_step','min_radius_cells','time_tolerance_myr']:
        if not np.isfinite(getattr(a, name)) or getattr(a, name) <= 0:
            p.error(f'{name} must be finite and positive')
    if a.density_step <= 1 or a.min_cells < 1 or a.max_levels < 1 or a.max_contour_calls < 1:
        p.error('step must exceed 1; cell and safety limits must be positive')
    if a.rho_min is not None and (not np.isfinite(a.rho_min) or a.rho_min <= 0):
        p.error('--rho-min must be positive')
    if not np.isfinite(a.min_radius_pc) or a.min_radius_pc < 0:
        p.error('--min-radius-pc must be finite and nonnegative')
    lo, hi, step = a.log_bins
    if not all(np.isfinite(a.log_bins)) or hi <= lo or step <= 0 or (hi-lo)/step > 10000:
        p.error('Invalid log bins')
    if a.overlay_axes != 'none' and (not a.overlay_axes or set(a.overlay_axes) - set('xyz')):
        p.error('Invalid overlay axes')
    if a.pixels < 32 or a.overlay_limit < 1:
        p.error('Invalid overlay size/limit')
    if not np.isfinite(a.logt_center) or not np.isfinite(a.logt_halfwidth) or a.logt_halfwidth <= 0:
        p.error('Temperature center must be finite and half-width positive')
    if not -2 < a.logt_center-a.logt_halfwidth < a.logt_center+a.logt_halfwidth < 12:
        p.error('Temperature interval is outside supported numerical range')
    a.tmin = 10.**(a.logt_center-a.logt_halfwidth)
    a.tmax = 10.**(a.logt_center+a.logt_halfwidth)
    return a


def write_json(path, obj):
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')


def close_dataset(ds):
    # Metadata-only selection must not trigger construction of the AMR index.
    handle = getattr(ds, '_handle', None)
    if handle is not None and hasattr(handle, 'close'):
        handle.close()


def choose_snapshot(a, options):
    if a.snapshot:
        return str(Path(a.snapshot).resolve())
    files = sorted(glob.glob(a.pattern))
    if not files:
        raise FileNotFoundError(a.pattern)
    times = []
    for i, f in enumerate(files):
        ds = yt.load(f, **options)
        try:
            t = float(ds.current_time.to_value('Myr'))
            if not np.isfinite(t):
                raise ValueError(f'Invalid time: {f}')
            times.append(t)
        finally:
            close_dataset(ds)
        if i % 50 == 0:
            print(f'Metadata: {i+1}/{len(files)}', flush=True)
    i = int(np.argmin(np.abs(np.array(times)-a.time_myr)))
    if abs(times[i]-a.time_myr) > a.time_tolerance_myr:
        raise ValueError(f'Nearest snapshot is at {times[i]} Myr, outside tolerance')
    return str(Path(files[i]).resolve())


COLUMNS = ['clump_id','threshold_g_cm3','n_cells','mass_Msun','volume_pc3','r_eff_pc',
           'cell_edge_min_pc','cell_edge_max_pc','radius_over_max_edge','mean_temperature_K',
           'x_kpc','y_kpc','z_kpc','r_center_kpc','touches_roi','resolved','accepted']


def histogram(rows, a, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    lo, hi, step = a.log_bins
    edges = np.arange(lo, hi+step*.5, step)
    if edges[-1] < hi:
        edges = np.append(edges, hi)
    values = np.log10([r['r_eff_pc'] for r in rows if r['accepted']])
    fig, ax = plt.subplots(figsize=(7,5))
    ax.hist(values, bins=edges, histtype='step', linewidth=2)
    ax.set(xlabel=r'$\log_{10}(r_{\rm eff}/{\rm pc})$', ylabel='N clumps', title=a.label)
    ax.text(.03,.97,f'N = {len(values)}',transform=ax.transAxes,va='top')
    fig.tight_layout()
    fig.savefig(out/'radius_distribution.png',dpi=200)
    plt.close(fig)
    counts, _ = np.histogram(values, edges)
    np.savetxt(out/'histogram.csv',np.column_stack([edges[:-1],edges[1:],counts]),delimiter=',',
               header='log10_radius_left,log10_radius_right,N',comments='')
    return {'below_bins':int(np.sum(values<edges[0])), 'above_bins':int(np.sum(values>edges[-1]))}

