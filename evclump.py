import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
""""properties of the single clump in time"""
HISTORY_FILE = "/mnt/O1/otello/risultati/cold_clumps_1050_1080/clump_history.csv"
OUT_PLOT = "./clump_evolution_mergers.png"

def main():
    if not os.path.exists(HISTORY_FILE):
        print(f"Error: The file {HISTORY_FILE} does not exist.")
        return

    df = pd.read_csv(HISTORY_FILE)
    df = df[df['status'].isin(['TRACK_START', 'TRACKED', 'CONTINUATION'])]
    
    if df.empty:
        print("No valid data to plot.")
        return

    t = df['time_Myr']
    dist = np.sqrt(df['x_kpc']**2 + df['y_kpc']**2 + df['z_kpc']**2)
    vel = np.sqrt(df['vx_km_s']**2 + df['vy_km_s']**2 + df['vz_km_s']**2)

    # Identify the times when a merger occurred
    merger_times = df[df['event'].str.contains('MERGER', na=False)]['time_Myr']

    fig, axs = plt.subplots(3, 2, figsize=(12, 12), sharex=True)
    fig.suptitle("Clump time evolution", fontsize=16)

    # Plot the physical quantities
    axs[0, 0].plot(t, df['mass_Msun'], marker='o', color='tab:blue')
    axs[0, 0].set_ylabel(r'Mass [$M_\odot$]')
    axs[0, 0].set_yscale('log')

    axs[0, 1].plot(t, dist, marker='o', color='tab:red')
    axs[0, 1].set_ylabel('Distance from center [kpc]')

    axs[1, 0].plot(t, df['mean_density_g_cm3'], marker='o', color='tab:green')
    axs[1, 0].set_ylabel('Mean density [g/cm$^3$]')
    axs[1, 0].set_yscale('log')

    axs[1, 1].plot(t, df['mean_temperature_K'], marker='o', color='tab:orange')
    axs[1, 1].set_ylabel('Mean temperature [K]')

    axs[2, 0].plot(t, df['radius_kpc'], marker='o', color='tab:purple')
    axs[2, 0].set_ylabel('Effective radius [kpc]')
    axs[2, 0].set_xlabel('Time [Myr]')

    axs[2, 1].plot(t, vel, marker='o', color='tab:brown')
    axs[2, 1].set_ylabel('Velocity [km/s]')
    axs[2, 1].set_xlabel('Time [Myr]')

    # Add merger lines to all subplots
    for ax in axs.flat:
        ax.grid(True, linestyle='--', alpha=0.6)
        for mt in merger_times:
            ax.axvline(mt, color='red', linestyle=':', linewidth=2, alpha=0.7)
            
    # Add a dummy legend for the merger line on the first plot
    axs[0, 0].plot([], [], color='red', linestyle=':', linewidth=2, label='Merger Event')
    axs[0, 0].legend(loc='upper left')

    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    plt.savefig(OUT_PLOT, dpi=300)
    print(f"Plot saved in: {OUT_PLOT}")

if __name__ == "__main__":
    main()