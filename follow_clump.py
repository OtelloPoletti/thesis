#!/usr/bin/env python3
"""
Fixed-Cam: Fissa l'inquadratura al centro del dominio spaziale (es. il buco nero)
mentre la profondità della slice si adatta per intersecare sempre il clump.
"""
import os
import re
import yt
import pandas as pd
import numpy as np
import imageio
from scipy.ndimage import gaussian_filter
from PIL import Image

# ================= CONFIGURAZIONE =================
HISTORY_FILE = "/mnt/O1/otello/risultati/cold_clumps_1050_1080/clump_history.csv"
CATALOG_ROOT = "/mnt/O1/otello/risultati/cold_clumps_1050_1080"
OUT_DIR = "./fixed_cam_frames"
VIDEO_NAME = "fixed_clump_Z.mp4"
ZOOM_WIDTH_KPC = 1.0  # Campo visivo largo 1 kpc
AXIS = "z"
FPS = 10
FIXED_ORIGIN = [0.0, 0.0, 0.0]  # Il punto attorno a cui fissare l'inquadratura
# ==================================================

AXES_MAP = {"x": (0, 1, 2), "y": (1, 2, 0), "z": (2, 0, 1)}

def pixel_bounds(c0, c1, minimum, width, n):
    start = int(np.clip(np.floor((c0 - minimum) / width * n), 0, n - 1))
    end = int(np.clip(np.ceil((c1 - minimum) / width * n), 0, n))
    if end <= start: end = min(n, start + 1)
    return start, end

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    
    df = pd.read_csv(HISTORY_FILE)
    df = df[df['status'].isin(['TRACK_START', 'TRACKED', 'CONTINUATION'])]
    
    if df.empty:
        print("Nessun dato valido nella history.")
        return

    ax_idx, h_idx, v_idx = AXES_MAP[AXIS]
    N_GRID = 800
    frame_paths = []

    print(f"Generazione di {len(df)} frame Fixed-Cam in corso...")

    for i, row in df.iterrows():
        rhdf_path = str(row['snapshot'])
        clump_id = int(row['clump_id'])
        t_myr = row['time_Myr']
        
        # Centro reale del clump
        clump_center = np.array([row['x_kpc'], row['y_kpc'], row['z_kpc']])
        
        # CENTRO TELECAMERA: X e Y fissi a 0, Z segue il clump
        cam_center = np.array(FIXED_ORIGIN)
        cam_center[ax_idx] = clump_center[ax_idx]
        
        match = re.search(r'(\d+)\.rhdf', rhdf_path)
        if not match:
            continue
        snap_num = match.group(1)
        catalog_dir = os.path.join(CATALOG_ROOT, f"cold_clumps_box_{snap_num}")
        mem_file = os.path.join(catalog_dir, "membership_leaves.npz")
        
        if not os.path.exists(mem_file):
            continue

        print(f"[{i:03d}] Snap {snap_num} | Inquadratura={cam_center.round(3)} | Clump={clump_center.round(3)}")
        
        mem = np.load(mem_file)
        xyz_kpc = np.asarray(mem["xyz_kpc"], dtype=float)
        widths_kpc = np.asarray(mem["widths_kpc"], dtype=float)
        membership_ids = np.asarray(mem["clump_id"], dtype=np.int64)
        
        ds = yt.load(rhdf_path)
        slc = yt.SlicePlot(
            ds, AXIS, ("gas", "density"),
            center=ds.arr(cam_center, "kpc"),
            width=(ZOOM_WIDTH_KPC, "kpc"),
            origin="native"
        )
        slc.set_unit(("gas", "density"), "g/cm**3")
        slc.set_axes_unit("kpc")
        slc.set_cmap(("gas", "density"), "viridis")
        slc.annotate_scale(size_bar_args={"color": "white"})
        slc.annotate_title(f"Fixed-Cam: t = {t_myr:.2f} Myr")
        
        slc.render()
        ax_plot = slc.plots[("gas", "density")].axes
        
        # Calcolo dei limiti basati sulla telecamera fissa
        h_min = cam_center[h_idx] - ZOOM_WIDTH_KPC / 2.0
        v_min = cam_center[v_idx] - ZOOM_WIDTH_KPC / 2.0
        
        in_slice = np.abs(xyz_kpc[:, ax_idx] - cam_center[ax_idx]) <= 0.5 * widths_kpc[:, ax_idx] * 1.01
        leaf_mask = (membership_ids == clump_id) & in_slice
        cell_indices = np.flatnonzero(leaf_mask)
        
        if len(cell_indices) > 0:
            mask_2d = np.zeros((N_GRID, N_GRID), dtype=float)
            h_c = xyz_kpc[cell_indices, h_idx]
            v_c = xyz_kpc[cell_indices, v_idx]
            dh = widths_kpc[cell_indices, h_idx]
            dv = widths_kpc[cell_indices, v_idx]
            
            for hc, vc, dhc, dvc in zip(h_c, v_c, dh, dv):
                hs, he = hc - dhc / 2.0, hc + dhc / 2.0
                vs, ve = vc - dvc / 2.0, vc + dvc / 2.0
                ih0, ih1 = pixel_bounds(hs, he, h_min, ZOOM_WIDTH_KPC, N_GRID)
                iv0, iv1 = pixel_bounds(vs, ve, v_min, ZOOM_WIDTH_KPC, N_GRID)
                mask_2d[iv0:iv1, ih0:ih1] = 1.0
                
            smooth = gaussian_filter(mask_2d, sigma=0.5)
            edges_h = np.linspace(h_min, h_min + ZOOM_WIDTH_KPC, N_GRID)
            edges_v = np.linspace(v_min, v_min + ZOOM_WIDTH_KPC, N_GRID)
            H_grid, V_grid = np.meshgrid(edges_h, edges_v)
            
            ax_plot.contour(
                H_grid, V_grid, smooth,
                levels=[0.5], colors=['crimson'], linewidths=1.5, zorder=10
            )

        filename = os.path.join(OUT_DIR, f"frame_{i:04d}.png")
        slc.save(filename)
        frame_paths.append(filename)
        
        del ds, mem
        import gc
        gc.collect()

    print("Generazione frame completata. Creazione del video in corso...")

    # 6. Generazione del Video con imageio e PIL
    if len(frame_paths) >= 2:
        output_video = os.path.join(OUT_DIR, VIDEO_NAME)
        try:
            # Prendi le dimensioni esatte del primo frame
            first_frame = imageio.v2.imread(frame_paths[0])
            w, h = first_frame.shape[1], first_frame.shape[0]
            
            # Forza larghezza e altezza a essere numeri pari (divisibili per 2) per libx264
            w = w if w % 2 == 0 else w - 1
            h = h if h % 2 == 0 else h - 1
            target_size = (w, h)

            with imageio.get_writer(output_video, fps=FPS, macro_block_size=None) as writer:
                for path in frame_paths:
                    img = imageio.v2.imread(path)
                    
                    # Ridimensiona il frame solo se non coincide con il target_size pari
                    if (img.shape[1], img.shape[0]) != target_size:
                        pil_img = Image.fromarray(img)
                        img = np.array(pil_img.resize(target_size, Image.Resampling.LANCZOS))
                        
                    writer.append_data(img)
            print(f"\n✅ Video completato con successo: {output_video}")
        except Exception as exc:
            print(f"\n❌ Errore durante la creazione del video: {exc}")
    else:
        print("\n⚠️ Non ci sono abbastanza frame per generare un video.")

if __name__ == "__main__":
    main()