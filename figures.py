"""The figures of the paper (Figs. 2-5), from the figure data written by analysis.py.

    import figures
    figures.fig2("results/figures/data")                      # show it
    figures.fig4("results/figures/data", out_dir="results/figures", IMAGE_LIMIT=20)

Each function reads the figure data in `data_dir`, draws the figure, saves <name>.png and .pdf in
`out_dir` when it is given, shows it when `show` is true, and returns the matplotlib figure. The
keyword arguments in capitals are the plot settings (sizes, colours, fonts, limits); their
defaults reproduce the figures of the paper. The plotting code is the co-author's, as used for
the paper. Fig. 1 (the pipeline schematic) is an image, assets/fig1-pipeline.png.
"""
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


def fig2(
        data_dir,
        out_dir=None,
        show=True,
        FIGSIZE=(8, 8),
        DPI=300,
        OUTPUT_BASENAME='fig2-noise-and-example',
        LEFT=0.08,
        RIGHT=0.965,
        TOP=0.965,
        BOTTOM=0.055,
        GROUP_HEIGHT_RATIOS=[0.93, 1.72],
        GROUP_VERTICAL_SPACE=0.32,
        GROUP_A_WSPACE=0.34,
        SQUARE_AB=True,
        GROUP_A_LABEL='(a)',
        GROUP_A_LABEL_X=0.005,
        GROUP_A_LABEL_Y=0.973,
        CURVE_COLOR='#0000cd',
        CURVE_LINEWIDTH=1.35,
        MARKER_SIZE=3.0,
        WHITE_BASELINE_COLOR='#008000',
        WHITE_BASELINE_STYLE='--',
        WHITE_BASELINE_WIDTH=1.8,
        FILTERGRAM_REF_MIN=4.5,
        FILTERGRAM_REF_LABEL='270 s\n(no shared\nfiltergrams)',
        FILTERGRAM_REF_COLOR='#ff3030',
        FILTERGRAM_REF_STYLE=':',
        FILTERGRAM_REF_WIDTH=1.8,
        TEMPORAL_XLIM=(0, 15),
        TEMPORAL_XTICKS=[0, 5, 10, 15],
        ONE_OVER_E_COLOR='#8a2be2',
        ONE_OVER_E_STYLE='-.',
        ONE_OVER_E_WIDTH=1.8,
        GROUP_B_WSPACE=0.07,
        GROUP_B_HSPACE=0.20,
        COLORBAR_WIDTH_RATIO=0.055,
        GROUP_B_LABEL='(b)',
        GROUP_B_LABEL_X=0.005,
        GROUP_B_LABEL_Y=0.600,
        TIME_TITLE_X=0.515,
        TIME_TITLE_Y=0.600,
        SHOW_TIME_TITLE=True,
        TOP_IMAGE_LIMIT=30,
        BOTTOM_IMAGE_LIMIT=15,
        SHOW_REFERENCE_NOTE=True,
        SHOW_PSNR_NOTES=True,
        SHOW_SIGMA_NOTE=True,
        BASE_FONT=9.5,
        TITLE_FONT=10.0,
        AXIS_LABEL_FONT=9.0,
        TICK_FONT=9.0,
        LEGEND_FONT=8.5,
        GROUP_LABEL_FONT=10.5,
        IMAGE_TITLE_FONT=9.5,
        IMAGE_NOTE_FONT=10.0,
        TIME_TITLE_FONT=10.5,
        COLORBAR_TICK_FONT=8.5,
        COLORBAR_LABEL_FONT=9.5,
        GROUP_LABEL_BOLD=False,
        COLORBAR_SHOW_ZERO=True,
        COLORBAR_UNIT='Gauss',
        COLORBAR_UNIT_ROTATION=90,
        COLORBAR_UNIT_LABELPAD=5):
    """Fig. 2: noise structure (a) and the denoising example (b)."""
    data_dir = Path(data_dir)
    def psnr(a, b, data_range):
        mse = float(np.mean((a - b) ** 2))
        return 10 * np.log10(data_range ** 2 / mse)


    # Load saved figure data
    ns = np.load(data_dir / 'fig-noise-structure_data.npz')
    ex = np.load(data_dir / 'fig-baseline-example_data.npz', allow_pickle=True)

    stacked = ex['stacked']
    single = ex['single']
    den = ex['n2n_mixsigma']
    tv = ex['adaptive_tv']
    dr = float(ex['data_range'])
    hour = str(ex['hour'])
    time_label = hour[:16] + " UT"

    print('Loaded example hour:', hour)
    print('Top-row PSNR values:')
    print('  Single frame   :', f'{psnr(single, stacked, dr):.2f} dB')
    print('  N2N mixed-sigma:', f'{psnr(den, stacked, dr):.2f} dB')
    print('  Adaptive TV    :', f'{psnr(tv, stacked, dr):.2f} dB')


    plt.rcParams.update({
        'font.size': BASE_FONT,
        'axes.titlesize': TITLE_FONT,
        'axes.labelsize': AXIS_LABEL_FONT,
        'xtick.labelsize': TICK_FONT,
        'ytick.labelsize': TICK_FONT,
        'legend.fontsize': LEGEND_FONT,
        'lines.linewidth': CURVE_LINEWIDTH,
    })

    fig = plt.figure(figsize=FIGSIZE)

    outer = fig.add_gridspec(
        2, 1,
        height_ratios=GROUP_HEIGHT_RATIOS,
        hspace=GROUP_VERTICAL_SPACE,
    )

    # ============================================================
    # Group (a): noise structure
    # ============================================================
    gs_a = outer[0].subgridspec(1, 3, wspace=GROUP_A_WSPACE)
    ax0, ax1, ax2 = [fig.add_subplot(gs_a[0, i]) for i in range(3)]

    # 1) Temporal autocorrelation
    ax0.plot(
        ns['lag_minutes'], ns['temporal_corr'],
        'o-', ms=MARKER_SIZE, color=CURVE_COLOR,
    )
    ax0.axhline(
        float(ns['white_baseline']),
        ls=WHITE_BASELINE_STYLE,
        lw=WHITE_BASELINE_WIDTH,
        color=WHITE_BASELINE_COLOR,
        label=f"White-noise \nbaseline\n({float(ns['white_baseline']):+.3f})",
    )
    ax0.axvline(
        FILTERGRAM_REF_MIN,
        ls=FILTERGRAM_REF_STYLE,
        lw=FILTERGRAM_REF_WIDTH,
        color=FILTERGRAM_REF_COLOR,
        label=FILTERGRAM_REF_LABEL,
    )
    ax0.set_xlim(*TEMPORAL_XLIM)
    ax0.set_xticks(TEMPORAL_XTICKS)
    ax0.set_xlabel('Time lag (min)')
    ax0.set_ylabel(r'Residual correlation $C(\Delta t)$')
    ax0.set_title('Temporal autocorrelation', pad=6)
    ax0.legend(
        loc='upper right',
        frameon=False,
        handlelength=2.2,
        labelspacing=1.0,
    )
    if SQUARE_AB:
        ax0.set_box_aspect(1)

    # 2) Spatial autocorrelation
    r = ns['radial_autocorr']
    ax1.plot(np.arange(len(r)), r, color=CURVE_COLOR)
    ax1.axhline(
        1 / np.e,
        ls=ONE_OVER_E_STYLE,
        lw=ONE_OVER_E_WIDTH,
        color=ONE_OVER_E_COLOR,
        label=r'$1/e$',
    )
    ax1.set_xlim(0, 30)
    ax1.set_xlabel('Spatial lag (px)')
    ax1.set_ylabel('Radial autocorrelation')
    ax1.set_title('Spatial autocorrelation', pad=6)
    ax1.legend(
        loc='upper right',
        frameon=False,
    )
    if SQUARE_AB:
        ax1.set_box_aspect(1)

    # 3) Example residual
    ax2.imshow(
        ns['example_residual'],
        cmap='gray',
        vmin=-BOTTOM_IMAGE_LIMIT,
        vmax=BOTTOM_IMAGE_LIMIT,
        interpolation='none',
    )
    ax2.set_title('Example residual', pad=6)
    ax2.axis('off')

    if SHOW_SIGMA_NOTE:
        ax2.text(
            0.5, -0.06,
            rf"$\sigma \approx {float(ns['sigma_G']):.1f}$ G",
            ha='center', va='top',
            transform=ax2.transAxes,
            fontsize=IMAGE_NOTE_FONT,
        )

    # Group (a) label
    fig.text(
        GROUP_A_LABEL_X, GROUP_A_LABEL_Y,
        GROUP_A_LABEL,
        ha='left', va='top',
        fontsize=GROUP_LABEL_FONT,
        fontweight='bold' if GROUP_LABEL_BOLD else 'normal',
    )

    # ============================================================
    # Group (b): one 2 x 4 image matrix + row-aligned colorbars
    # ============================================================
    gs_b = outer[1].subgridspec(
        2, 5,
        width_ratios=[1, 1, 1, 1, COLORBAR_WIDTH_RATIO],
        wspace=GROUP_B_WSPACE,
        hspace=GROUP_B_HSPACE,
    )

    axs = [[fig.add_subplot(gs_b[r, c]) for c in range(4)] for r in range(2)]

    # Top row
    top_panels = [
        ('21-frame stack', 'Reference', stacked),
        ('Single frame', f'PSNR = {psnr(single, stacked, dr):.1f} dB', single),
        (r'N2N mixed-$\sigma$', f'PSNR = {psnr(den, stacked, dr):.1f} dB', den),
        ('Adaptive TV', f'PSNR = {psnr(tv, stacked, dr):.1f} dB', tv),
    ]

    top_im = None
    for c, (title, note, img) in enumerate(top_panels):
        ax = axs[0][c]
        top_im = ax.imshow(
            img,
            cmap='gray',
            vmin=-TOP_IMAGE_LIMIT,
            vmax=TOP_IMAGE_LIMIT,
            interpolation='none',
        )
        ax.set_title(title, pad=4, fontsize=IMAGE_TITLE_FONT)
        ax.axis('off')

        show_note = ((c == 0 and SHOW_REFERENCE_NOTE) or
                     (c > 0 and SHOW_PSNR_NOTES))
        if show_note:
            ax.text(
                0.5, -0.035, note,
                ha='center', va='top',
                transform=ax.transAxes,
                fontsize=IMAGE_NOTE_FONT,
            )

    # Bottom row
    # Column 2 is intentionally blank.
    # bottom_panels = {
    #     0: (r'Single $-$ stack', rf"$\sigma = {float(ex['sigma_G']):.1f}$ G", single - stacked),
    #     2: (r'Removed by N2N mixed-$\sigma$', '', single - den),
    #     3: ('Removed by TV', '', single - tv),
    # }
    bottom_panels = {
        0: (r'Single $-$ Stack', rf"$\sigma = {float(ex['sigma_G']):.1f}$ G", single - stacked),
        2: ('Removed by\n' + r'N2N mixed-$\sigma$', '', single - den),
        3: ('Removed by\nAdaptive TV', '', single - tv),
    }

    bottom_im = None
    for c in range(4):
        ax = axs[1][c]
        ax.axis('off')

        if c not in bottom_panels:
            continue

        title, note, img = bottom_panels[c]
        bottom_im = ax.imshow(
            img,
            cmap='gray',
            vmin=-BOTTOM_IMAGE_LIMIT,
            vmax=BOTTOM_IMAGE_LIMIT,
            interpolation='none',
        )
        ax.set_title(title, pad=4, fontsize=IMAGE_TITLE_FONT)

        if note and SHOW_SIGMA_NOTE:
            ax.text(
                0.5, -0.035, note,
                ha='center', va='top',
                transform=ax.transAxes,
                fontsize=IMAGE_NOTE_FONT,
            )

    # Top-row colorbar: ±30 G by default
    cax_top = fig.add_subplot(gs_b[0, 4])
    cb_top = fig.colorbar(top_im, cax=cax_top)
    top_ticks = [-TOP_IMAGE_LIMIT, TOP_IMAGE_LIMIT]
    if COLORBAR_SHOW_ZERO:
        top_ticks = [-TOP_IMAGE_LIMIT, 0, TOP_IMAGE_LIMIT]
    cb_top.set_ticks(top_ticks)
    cb_top.ax.tick_params(labelsize=COLORBAR_TICK_FONT, length=2.5)
    cb_top.set_label(
        COLORBAR_UNIT,
        rotation=COLORBAR_UNIT_ROTATION,
        labelpad=COLORBAR_UNIT_LABELPAD,
        fontsize=COLORBAR_LABEL_FONT,
    )

    # Bottom-row colorbar: ±15 G by default
    cax_bottom = fig.add_subplot(gs_b[1, 4])
    cb_bottom = fig.colorbar(bottom_im, cax=cax_bottom)
    bottom_ticks = [-BOTTOM_IMAGE_LIMIT, BOTTOM_IMAGE_LIMIT]
    if COLORBAR_SHOW_ZERO:
        bottom_ticks = [-BOTTOM_IMAGE_LIMIT, 0, BOTTOM_IMAGE_LIMIT]
    cb_bottom.set_ticks(bottom_ticks)
    cb_bottom.ax.tick_params(labelsize=COLORBAR_TICK_FONT, length=2.5)
    cb_bottom.set_label(
        COLORBAR_UNIT,
        rotation=COLORBAR_UNIT_ROTATION,
        labelpad=COLORBAR_UNIT_LABELPAD,
        fontsize=COLORBAR_LABEL_FONT,
    )

    # Group (b) label
    fig.text(
        GROUP_B_LABEL_X, GROUP_B_LABEL_Y,
        GROUP_B_LABEL,
        ha='left', va='center',
        fontsize=GROUP_LABEL_FONT,
        fontweight='bold' if GROUP_LABEL_BOLD else 'normal',
    )

    # Time-only title
    if SHOW_TIME_TITLE:
        fig.text(
            TIME_TITLE_X, TIME_TITLE_Y,
            time_label,
            ha='center', va='center',
            fontsize=TIME_TITLE_FONT,
        )

    fig.subplots_adjust(
        left=LEFT,
        right=RIGHT,
        top=TOP,
        bottom=BOTTOM,
    )

    # Resize top colorbar
    pos = cax_top.get_position()
    new_h = pos.height * 0.85
    new_y = pos.y0 + (pos.height - new_h) / 2
    cax_top.set_position([pos.x0, new_y, pos.width* 1.4, new_h])

    # Resize bottom colorbar
    pos = cax_bottom.get_position()
    new_h = pos.height * 0.85
    new_y = pos.y0 + (pos.height - new_h) / 2
    cax_bottom.set_position([pos.x0, new_y, pos.width* 1.4, new_h])

    GROUP_A_X_SHIFT = 0.065
    GROUP_B_X_SHIFT = -0.02   # (-) 왼쪽, (+) 오른쪽

    # Move all three panels in group (a) horizontally
    for ax in [ax0, ax1, ax2]:
        pos = ax.get_position()
        ax.set_position([
            pos.x0 + GROUP_A_X_SHIFT,
            pos.y0,
            pos.width,
            pos.height
        ])

    # Move group (b) horizontally
    for row in axs:
        for ax in row:
            pos = ax.get_position()
            ax.set_position([
                pos.x0 + GROUP_B_X_SHIFT,
                pos.y0,
                pos.width,
                pos.height
            ])

    for cax in [cax_top, cax_bottom]:
        pos = cax.get_position()
        cax.set_position([
            pos.x0 + GROUP_B_X_SHIFT,
            pos.y0,
            pos.width,
            pos.height
        ])


    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        # Save after you are satisfied with the preview above.
        fig.savefig(out_dir / f'{OUTPUT_BASENAME}.png', dpi=DPI, bbox_inches='tight')
        fig.savefig(out_dir / f'{OUTPUT_BASENAME}.pdf', bbox_inches='tight')

        print('Saved:')
        print(out_dir / f'{OUTPUT_BASENAME}.png')
        print(out_dir / f'{OUTPUT_BASENAME}.pdf')

    if show:
        plt.show()
    return fig


def fig3(
        data_dir,
        out_dir=None,
        show=True,
        FIGSIZE=(16.0, 4.8),
        DPI=300,
        OUTPUT_BASENAME="fig3-noise-level",
        BASE_FONT=14,
        AXIS_LABEL_FONT=14,
        TICK_FONT=13,
        SUBPLOT_TITLE_FONT=16,
        LEGEND_FONT=15,
        LEFT=0.065,
        RIGHT=0.985,
        TOP=0.90,
        BOTTOM=0.27,
        WSPACE=0.28,
        TITLES=[
    "PSNR vs. Stack (dB)",
    "SSIM",
    r"Residual $\sigma$ (Gauss)",
],
        XLABEL=r"Effective input $\sigma$ (Gauss)",
        XLIM=(6.5, 23.5),
        XTICKS=[7.5, 10, 12.5, 15, 17.5, 20, 22.5],
        COLOR_NOISY="#ffa500",
        COLOR_MIXED="#8a2be2",
        COLOR_NOAUG="#cd3700",
        COLOR_TV="#009E73",
        STYLE_NOISY="--",
        STYLE_MIXED="-",
        STYLE_NOAUG=":",
        STYLE_TV="-.",
        MARKER_NOISY="o",
        MARKER_MIXED="o",
        MARKER_NOAUG="s",
        MARKER_TV="^",
        LINEWIDTH=2.5,
        MARKERSIZE=5.5,
        TRAINING_SIGMA=7.9,
        TRAINING_LINE_COLOR="#777777",
        TRAINING_LINE_STYLE=":",
        TRAINING_LINE_WIDTH=1.4,
        TRAINING_LABEL=r"Training $\sigma \approx 7.9$ Gauss",
        LEGEND_ANCHOR=(0.50, 0.075),
        LEGEND_NCOL=5,
        LEGEND_HANDLELENGTH=2.8,
        LEGEND_COLUMNSPACING=1.8,
        SHOW_GRID=False,
        GRID_ALPHA=0.20):
    """Fig. 3: performance against the input noise level."""
    data_dir = Path(data_dir)
    # Load the unchanged noise-level data
    nl = np.genfromtxt(
        data_dir / "fig-noise-level_data.csv",
        delimiter=",",
        names=True,
    )

    x = nl["effective_sigma_G"]

    print("Noise-level rows:", len(nl))
    print("Effective input sigma (G):", x)
    print("First data point:", f"{x[0]:.2f} G")


    # ============================================================
    # PLOT
    # ============================================================

    plt.rcParams.update({
        "font.size": BASE_FONT,
        "axes.titlesize": SUBPLOT_TITLE_FONT,
        "axes.labelsize": AXIS_LABEL_FONT,
        "xtick.labelsize": TICK_FONT,
        "ytick.labelsize": TICK_FONT,
    })

    fig, ax = plt.subplots(1, 3, figsize=FIGSIZE)

    series = [
        ("Noisy input", "in", COLOR_NOISY, STYLE_NOISY, MARKER_NOISY),
        (r"N2N mixed-$\sigma$ (this work)", "n2n", COLOR_MIXED, STYLE_MIXED, MARKER_MIXED),
        ("N2N (no augmentation)", "fix", COLOR_NOAUG, STYLE_NOAUG, MARKER_NOAUG),
        ("Adaptive TV", "tv", COLOR_TV, STYLE_TV, MARKER_TV),
    ]

    metrics = ["p", "s", "r"]

    for j, metric in enumerate(metrics):

        for label, key, color, ls, marker in series:
            col = f"{key}_{metric}"

            if col in nl.dtype.names:
                ax[j].plot(
                    x,
                    nl[col],
                    label=label,
                    color=color,
                    linestyle=ls,
                    marker=marker,
                    linewidth=LINEWIDTH,
                    markersize=MARKERSIZE,
                )

        # Noisy-input residual sigma equals the effective input-noise sigma.
        if metric == "r":
            ax[j].plot(
                x,
                x,
                color=COLOR_NOISY,
                linestyle=STYLE_NOISY,
                marker=MARKER_NOISY,
                linewidth=LINEWIDTH,
                markersize=MARKERSIZE,
            )

        # Training-noise reference at 7.9 G.
        ax[j].axvline(
            TRAINING_SIGMA,
            linestyle=TRAINING_LINE_STYLE,
            linewidth=TRAINING_LINE_WIDTH,
            color=TRAINING_LINE_COLOR,
            label=TRAINING_LABEL if j == 0 else None,
            zorder=0,
        )

        ax[j].set_xlabel(XLABEL)
        ax[j].set_title(TITLES[j], pad=8)
        ax[j].set_xlim(*XLIM)
        ax[j].set_xticks(XTICKS)

        if SHOW_GRID:
            ax[j].grid(alpha=GRID_ALPHA)

    # Shared legend below the single row.
    handles, labels = ax[0].get_legend_handles_labels()

    fig.legend(
        handles,
        labels,
        loc="center",
        bbox_to_anchor=LEGEND_ANCHOR,
        frameon=False,
        fontsize=LEGEND_FONT,
        ncol=LEGEND_NCOL,
        handlelength=LEGEND_HANDLELENGTH,
        columnspacing=LEGEND_COLUMNSPACING,
    )

    fig.subplots_adjust(
        left=LEFT,
        right=RIGHT,
        top=TOP,
        bottom=BOTTOM,
        wspace=WSPACE,
    )


    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        # Save the final files.
        fig.savefig(
            out_dir / f"{OUTPUT_BASENAME}.png",
            dpi=DPI,
            bbox_inches="tight",
        )

        fig.savefig(
            out_dir / f"{OUTPUT_BASENAME}.pdf",
            bbox_inches="tight",
        )

        print("Saved:")
        print(out_dir / f"{OUTPUT_BASENAME}.png")
        print(out_dir / f"{OUTPUT_BASENAME}.pdf")

    if show:
        plt.show()
    return fig


def fig4(
        data_dir,
        out_dir=None,
        show=True,
        FIGSIZE=(16.5, 8.5),
        DPI=300,
        OUTPUT_BASENAME="fig4-cross-instrument",
        SHOW_OVERALL_TITLE=False,
        OVERALL_TITLE="Cross-instrument transfer",
        OVERALL_TITLE_FONT=15,
        OVERALL_TITLE_Y=0.985,
        BASE_FONT=14,
        SCATTER_TITLE_FONT=16,
        AXIS_LABEL_FONT=16,
        TICK_FONT=15,
        IMAGE_TITLE_FONT=17,
        GROUP_LABEL_FONT=17,
        DATE_FONT=14,
        COLORBAR_TICK_FONT=15,
        COLORBAR_LABEL_FONT=16,
        LEFT=0.075,
        RIGHT=0.955,
        TOP=0.90,
        BOTTOM=0.08,
        WSPACE=0.15,
        HSPACE=0.35,
        PAIRED_WIDTH_RATIO=1.00,
        IMAGE_WIDTH_RATIO=1.00,
        COLORBAR_WIDTH_RATIO=0.055,
        GROUP_A_LABEL="(a)",
        GROUP_B_LABEL="(b)",
        GROUP_LABEL_BOLD=False,
        GROUP_A_LABEL_X=0.020,
        GROUP_A_LABEL_Y=0.945,
        GROUP_B_LABEL_X=0.020,
        GROUP_B_LABEL_Y=0.465,
        PAIRED_MEAN_COLOR_MDI="#0072B2",
        PAIRED_MEAN_COLOR_PHI="#D55E00",
        PAIR_LINE_COLOR="#9A9A9A",
        PAIR_LINE_WIDTH=0.9,
        PAIR_LINE_ALPHA=0.35,
        PAIR_POINT_COLOR="#808080",
        PAIR_POINT_SIZE=18,
        PAIR_POINT_ALPHA=0.55,
        MEAN_LINE_WIDTH=2.6,
        MEAN_MARKER="o",
        MEAN_MARKER_SIZE=8.5,
        PAIRED_YLABEL="Disagreement (Gauss)",
        PAIRED_XLABELS=["Before", "After"],
        SQUARE_PAIRED=True,
        HIDE_TOP_RIGHT_SPINES=True,
        SHOW_PAIR_TITLE=True,
        SHOW_SUMMARY_LINE=True,
        PAIR_TITLE_MDI="HMI – MDI",
        PAIR_TITLE_PHI="HMI – PHI",
        IMAGE_LIMIT=30,
        IMAGE_CMAP="gray",
        IMAGE_TITLES_MDI=[
    "HMI raw",
    "MDI raw",
    "HMI denoised",
    "MDI denoised",
],
        IMAGE_TITLES_PHI=[
    "HMI raw",
    "PHI raw",
    "HMI denoised",
    "PHI denoised",
],
        SHOW_DATE=False,
        DATE_X=0.03,
        DATE_Y=0.04,
        DATE_COLOR="white",
        COLORBAR_UNIT="Gauss",
        COLORBAR_UNIT_ROTATION=90,
        COLORBAR_LABELPAD=5,
        COLORBAR_SHOW_ZERO=True,
        TOP_COLORBAR_HEIGHT_SCALE=0.80,
        BOTTOM_COLORBAR_HEIGHT_SCALE=0.80,
        COLORBAR_WIDTH_SCALE=1.20,
        SHOW_GRID=False,
        GRID_ALPHA=0.20):
    """Fig. 4: cross-instrument transfer (HMI-MDI, HMI-PHI)."""
    data_dir = Path(data_dir)
    # Load data
    mdi = np.load(
        data_dir / "fig-cross-instrument-mdi_data.npz",
        allow_pickle=True,
    )
    phi = np.load(
        data_dir / "fig-cross-instrument-phi_data.npz",
        allow_pickle=True,
    )

    rows = [
        ("MDI", mdi),
        ("PHI", phi),
    ]

    for name, z in rows:
        before = z["before_G"]
        after = z["after_G"]
        print(
            f"{name}: {len(before)} pairs, "
            f"{before.mean():.2f} -> {after.mean():.2f} G, "
            f"{100*(after.mean()-before.mean())/before.mean():+.1f}%"
        )
        print("  example:", str(z["example"]))


    # ============================================================
    # PLOT
    # ============================================================

    plt.rcParams.update({
        "font.size": BASE_FONT,
        "axes.labelsize": AXIS_LABEL_FONT,
        "xtick.labelsize": TICK_FONT,
        "ytick.labelsize": TICK_FONT,
    })

    fig = plt.figure(figsize=FIGSIZE)

    # Outer grid: two rows
    outer = fig.add_gridspec(
        2, 1,
        hspace=HSPACE,
    )

    axes = []
    colorbar_axes = []

    for i in range(2):

        # One row = paired before–after plot + image group
        row_grid = outer[i].subgridspec(
            1, 2,
            width_ratios=[PAIRED_WIDTH_RATIO, 4.25],
            wspace=0.15,   # paired plot ↔ first image gap
        )

        # Paired before–after plot
        ax_paired = fig.add_subplot(row_grid[0, 0])

        # Image block: 4 images + colorbar
        image_grid = row_grid[0, 1].subgridspec(
            1, 5,
            width_ratios=[1, 1, 1, 1, COLORBAR_WIDTH_RATIO],
            wspace=0.05,   # image ↔ image gap
        )

        row_axes = [ax_paired]

        for j in range(4):
            row_axes.append(fig.add_subplot(image_grid[0, j]))

        axes.append(row_axes)
        colorbar_axes.append(fig.add_subplot(image_grid[0, 4]))

    cax_top = colorbar_axes[0]
    cax_bottom = colorbar_axes[1]

    row_specs = [
        {
            "name": "MDI",
            "data": mdi,
            "mean_color": PAIRED_MEAN_COLOR_MDI,
            "pair_title": PAIR_TITLE_MDI,
            "image_titles": IMAGE_TITLES_MDI,
        },
        {
            "name": "PHI",
            "data": phi,
            "mean_color": PAIRED_MEAN_COLOR_PHI,
            "pair_title": PAIR_TITLE_PHI,
            "image_titles": IMAGE_TITLES_PHI,
        },
    ]

    row_mappables = []

    for i, spec in enumerate(row_specs):
        z = spec["data"]
        before = np.asarray(z["before_G"], dtype=float)
        after = np.asarray(z["after_G"], dtype=float)

        before_mean = float(before.mean())
        after_mean = float(after.mean())
        reduction_pct = 100 * (before_mean - after_mean) / before_mean

        # --------------------------------------------------------
        # Paired before–after disagreement plot
        # --------------------------------------------------------
        ax_pair = axes[i][0]

        # Individual observation pairs:
        # each line connects the same pair before and after denoising.
        for bval, aval in zip(before, after):
            ax_pair.plot(
                [0, 1],
                [bval, aval],
                color=PAIR_LINE_COLOR,
                linewidth=PAIR_LINE_WIDTH,
                alpha=PAIR_LINE_ALPHA,
                zorder=1,
            )

            ax_pair.scatter(
                [0, 1],
                [bval, aval],
                s=PAIR_POINT_SIZE,
                color=PAIR_POINT_COLOR,
                alpha=PAIR_POINT_ALPHA,
                edgecolors="none",
                zorder=2,
            )

        # Mean before → after, highlighted with the row color
        ax_pair.plot(
            [0, 1],
            [before_mean, after_mean],
            color=spec["mean_color"],
            linewidth=MEAN_LINE_WIDTH,
            marker=MEAN_MARKER,
            markersize=MEAN_MARKER_SIZE,
            zorder=4,
        )

        ax_pair.set_xlim(-0.25, 1.25)
        ax_pair.set_xticks([0, 1])
        ax_pair.set_xticklabels(PAIRED_XLABELS)
        ax_pair.set_ylabel(PAIRED_YLABEL)

        ymax = max(float(before.max()), float(after.max())) * 1.10
        ax_pair.set_ylim(0, ymax)

        if SQUARE_PAIRED:
            ax_pair.set_box_aspect(1)

        if HIDE_TOP_RIGHT_SPINES:
            ax_pair.spines["top"].set_visible(False)
            ax_pair.spines["right"].set_visible(False)

        if SHOW_GRID:
            ax_pair.grid(axis="y", alpha=GRID_ALPHA)

        # Row title and summary
        title_lines = []

        if SHOW_PAIR_TITLE:
            title_lines.append(
                f'{spec["pair_title"]}, {len(before)} pairs'
            )

        if SHOW_SUMMARY_LINE:
            title_lines.append(
                f"{before_mean:.1f} → {after_mean:.1f} G "
                f"(-{reduction_pct:.0f}%)"
            )

        if title_lines:
            ax_pair.set_title(
                "\n".join(title_lines),
                fontsize=SCATTER_TITLE_FONT,
                pad=7,
            )

            # ax_pair.text(
            #     0.5, 0.97,
            #     f"{len(before)} pairs, "
            #     f"{before_mean:.1f} → {after_mean:.1f} G "
            #     f"({reduction_pct:.0f}% reduction)",
            #     transform=ax_pair.transAxes,
            #     ha="center",
            #     va="top",
            #     fontsize=10,
            # )

        # --------------------------------------------------------
        # Four magnetogram panels
        # --------------------------------------------------------
        ex = z["extent_arcsec"] if "extent_arcsec" in z.files else None

        imgs = [
            z["hmi_raw"],
            z["other_raw"],
            z["hmi_denoised"],
            z["other_denoised"],
        ]

        mappable = None

        for j, (title, img) in enumerate(zip(spec["image_titles"], imgs)):
            ax_img = axes[i][j + 1]

            mappable = ax_img.imshow(
                img,
                cmap=IMAGE_CMAP,
                vmin=-IMAGE_LIMIT,
                vmax=IMAGE_LIMIT,
                interpolation="none",
                origin="lower",
                extent=ex,
            )

            ax_img.set_title(
                title,
                fontsize=IMAGE_TITLE_FONT,
                pad=5,
            )

            ax_img.tick_params(labelsize=TICK_FONT)
            ax_img.locator_params(nbins=3)

            if j == 0:
                ax_img.set_xlabel("Solar X (arcsec)", fontsize=AXIS_LABEL_FONT)
                ax_img.set_ylabel("Solar Y (arcsec)", fontsize=AXIS_LABEL_FONT)
            else:
                ax_img.set_xticklabels([])
                ax_img.set_yticklabels([])

        row_mappables.append(mappable)

        if SHOW_DATE:
            stem = str(z["example"])
            date_string = f"{stem[:4]}-{stem[4:6]}-{stem[6:8]}"

            axes[i][1].text(
                DATE_X,
                DATE_Y,
                date_string,
                transform=axes[i][1].transAxes,
                fontsize=DATE_FONT,
                color=DATE_COLOR,
                ha="left",
                va="bottom",
            )

    # ------------------------------------------------------------
    # Separate colorbars
    # ------------------------------------------------------------
    ticks = [-IMAGE_LIMIT, IMAGE_LIMIT]

    if COLORBAR_SHOW_ZERO:
        ticks = [-IMAGE_LIMIT, 0, IMAGE_LIMIT]

    cb_top = fig.colorbar(row_mappables[0], cax=cax_top)
    cb_top.set_ticks(ticks)
    cb_top.ax.tick_params(labelsize=COLORBAR_TICK_FONT, length=2.5)
    cb_top.set_label(
        COLORBAR_UNIT,
        rotation=COLORBAR_UNIT_ROTATION,
        labelpad=COLORBAR_LABELPAD,
        fontsize=COLORBAR_LABEL_FONT,
    )

    cb_bottom = fig.colorbar(row_mappables[1], cax=cax_bottom)
    cb_bottom.set_ticks(ticks)
    cb_bottom.ax.tick_params(labelsize=COLORBAR_TICK_FONT, length=2.5)
    cb_bottom.set_label(
        COLORBAR_UNIT,
        rotation=COLORBAR_UNIT_ROTATION,
        labelpad=COLORBAR_LABELPAD,
        fontsize=COLORBAR_LABEL_FONT,
    )

    # ------------------------------------------------------------
    # Group labels
    # ------------------------------------------------------------
    fig.text(
        GROUP_A_LABEL_X,
        GROUP_A_LABEL_Y,
        GROUP_A_LABEL,
        ha="left",
        va="center",
        fontsize=GROUP_LABEL_FONT,
        fontweight="bold" if GROUP_LABEL_BOLD else "normal",
    )

    fig.text(
        GROUP_B_LABEL_X,
        GROUP_B_LABEL_Y,
        GROUP_B_LABEL,
        ha="left",
        va="center",
        fontsize=GROUP_LABEL_FONT,
        fontweight="bold" if GROUP_LABEL_BOLD else "normal",
    )

    # ------------------------------------------------------------
    # Group date/time titles
    # ------------------------------------------------------------
    mdi_stem = str(mdi["example"])
    mdi_date = (
        f"{mdi_stem[:4]}-{mdi_stem[4:6]}-{mdi_stem[6:8]} "
        f"{mdi_stem[9:11]}:{mdi_stem[11:13]} UT"
    )

    phi_stem = str(phi["example"])
    phi_date = (
        f"{phi_stem[:4]}-{phi_stem[4:6]}-{phi_stem[6:8]} "
        f"{phi_stem[9:11]}:{phi_stem[11:13]} UT"
    )

    fig.text(
        0.52,
        0.945,
        mdi_date,
        ha="center",
        va="center",
        fontsize=18,
    )

    fig.text(
        0.52,
        0.465,
        phi_date,
        ha="center",
        va="center",
        fontsize=18,
    )

    if SHOW_OVERALL_TITLE:
        fig.suptitle(
            OVERALL_TITLE,
            fontsize=OVERALL_TITLE_FONT,
            y=OVERALL_TITLE_Y,
        )

    fig.subplots_adjust(
        left=LEFT,
        right=RIGHT,
        top=TOP,
        bottom=BOTTOM,
    )

    # ------------------------------------------------------------
    # Optional manual resize of each colorbar AFTER layout
    # ------------------------------------------------------------
    for cax, scale in [
        (cax_top, TOP_COLORBAR_HEIGHT_SCALE),
        (cax_bottom, BOTTOM_COLORBAR_HEIGHT_SCALE),
    ]:
        pos = cax.get_position()
        new_h = pos.height * scale
        new_w = pos.width * COLORBAR_WIDTH_SCALE
        new_y = pos.y0 + (pos.height - new_h) / 2

        cax.set_position([
            pos.x0,
            new_y,
            new_w,
            new_h,
        ])


    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        # Save after you are satisfied with the preview.
        fig.savefig(
            out_dir / f"{OUTPUT_BASENAME}.png",
            dpi=DPI,
            bbox_inches="tight",
        )
        fig.savefig(
            out_dir / f"{OUTPUT_BASENAME}.pdf",
            bbox_inches="tight",
        )

        print("Saved:")
        print(out_dir / f"{OUTPUT_BASENAME}.png")
        print(out_dir / f"{OUTPUT_BASENAME}.pdf")

    if show:
        plt.show()
    return fig


def fig5(
        data_dir,
        out_dir=None,
        show=True,
        FIGSIZE=(15, 10),
        DPI=300,
        OUTPUT_BASENAME="fig5-signal-preservation",
        BASE_FONT=15,
        AXIS_LABEL_FONT=15,
        TICK_FONT=15,
        TITLE_FONT=16,
        LEGEND_FONT=17,
        PANEL_LABEL_FONT=18,
        LEFT=0.075,
        RIGHT=0.985,
        TOP=0.955,
        BOTTOM=0.155,
        WSPACE_TOP=0.23,
        WSPACE_BOTTOM=0.25,
        HSPACE=0.40,
        PANEL_LABEL_X=-0.2,
        PANEL_LABEL_Y=1.02,
        LEGEND_ANCHOR=(0.50, 0.015),
        LEGEND_NCOL=5,
        LEGEND_HANDLELENGTH=2.6,
        LEGEND_COLUMNSPACING=1.8,
        LEGEND_LABELSPACING=0.8,
        LINEWIDTH=2.2,
        MARKERSIZE=7.0,
        REFERENCE_LINE_WIDTH=1.0,
        COLOR_NOISY="darkorange",
        COLOR_N2N="#8A2BE2",
        COLOR_TV="#009E73",
        COLOR_BM3D="dodgerblue",
        COLOR_BLUR="deeppink",
        STYLE_NOISY="--",
        STYLE_N2N="-",
        STYLE_TV="--",
        STYLE_BM3D="-.",
        STYLE_BLUR=":",
        MARKER_NOISY="s",
        MARKER_N2N="o",
        MARKER_TV="s",
        MARKER_BM3D="^",
        MARKER_BLUR="D",
        REFERENCE_COLOR="#9A9A9A",
        SHOW_GRID=False,
        GRID_ALPHA=0.18,
        PSD_YLIM=(0.0, 2.0),
        INJECTION_YLIM=(0.6, 1.15)):
    """Fig. 5: signal preservation (flux by field, power spectrum, injected elements)."""
    data_dir = Path(data_dir)
    # Load data
    with open(data_dir / "fig-signal-preservation_data.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    def series(panel, method):
        r = [q for q in rows if q["panel"] == panel and q["method"] == method]
        return [q["x"] for q in r], np.array([float(q["y"]) for q in r], dtype=float)

    print("Loaded rows:", len(rows))
    print("Panels:", sorted(set(r["panel"] for r in rows)))


    # ============================================================
    # STYLE DEFINITIONS
    # ============================================================

    STYLE = {
        "n2n_mixsigma": {
            "label": r"N2N mixed-$\sigma$ (this work)",
            "color": COLOR_N2N,
            "marker": MARKER_N2N,
            "linestyle": STYLE_N2N,
        },
        "tv": {
            "label": "Adaptive TV",
            "color": COLOR_TV,
            "marker": MARKER_TV,
            "linestyle": STYLE_TV,
        },
        "bm3d": {
            "label": "BM3D",
            "color": COLOR_BM3D,
            "marker": MARKER_BM3D,
            "linestyle": STYLE_BM3D,
        },
        "blur2": {
            "label": "Gaussian blur, 2 px",
            "color": COLOR_BLUR,
            "marker": MARKER_BLUR,
            "linestyle": STYLE_BLUR,
        },
        "input": {
            "label": "Noisy input",
            "color": COLOR_NOISY,
            "marker": MARKER_NOISY,
            "linestyle": STYLE_NOISY,
        },
    }


    # ============================================================
    # PLOT
    # ============================================================

    plt.rcParams.update({
        "font.size": BASE_FONT,
        "axes.titlesize": TITLE_FONT,
        "axes.labelsize": AXIS_LABEL_FONT,
        "xtick.labelsize": TICK_FONT,
        "ytick.labelsize": TICK_FONT,
    })

    fig = plt.figure(figsize=FIGSIZE)

    # Outer grid: top row + bottom row
    outer = fig.add_gridspec(
        2, 1,
        height_ratios=[1.0, 1.0],
        hspace=HSPACE,
    )

    top = outer[0].subgridspec(1, 2, wspace=WSPACE_TOP)
    bottom = outer[1].subgridspec(1, 3, wspace=WSPACE_BOTTOM)

    ax_f = fig.add_subplot(top[0, 0])
    ax_p = fig.add_subplot(top[0, 1])
    ax_i = [fig.add_subplot(bottom[0, k]) for k in range(3)]

    # ------------------------------------------------------------
    # Helper: panel label at upper-left, aligned to y-axis top
    # ------------------------------------------------------------
    def add_panel_label(ax, letter):
        ax.text(
            PANEL_LABEL_X,
            PANEL_LABEL_Y,
            f"({letter})",
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=PANEL_LABEL_FONT,
            clip_on=False,
        )

    # ------------------------------------------------------------
    # (a) Flux preservation by field-strength bin
    # ------------------------------------------------------------
    for m in ("n2n_mixsigma", "tv", "bm3d"):
        st = STYLE[m]
        x, y = series("flux", m)

        ax_f.plot(
            range(len(x)),
            y,
            color=st["color"],
            marker=st["marker"],
            linestyle=st["linestyle"],
            linewidth=LINEWIDTH,
            markersize=MARKERSIZE,
            label=st["label"],
        )

    ax_f.axhline(
        1.0,
        color=REFERENCE_COLOR,
        linewidth=REFERENCE_LINE_WIDTH,
    )

    ax_f.set_xticks(
        range(7),
        ["0–5", "5–10", "10–20", "20–30", "30–50", "50–100", ">100"],
    )

    ax_f.set_xlabel(r"$|B|$ of the stacked reference (Gauss)", labelpad=12)
    ax_f.set_ylabel("Unsigned flux,\ndenoised / reference", labelpad=12)
    add_panel_label(ax_f, "a")

    if SHOW_GRID:
        ax_f.grid(alpha=GRID_ALPHA)

    # ------------------------------------------------------------
    # (b) Spatial-power preservation
    # ------------------------------------------------------------
    for m in ("input", "n2n_mixsigma", "tv", "bm3d"):
        st = STYLE[m]
        x, y = series("psd", m)

        ax_p.plot(
            np.array(x, dtype=float),
            y,
            color=st["color"],
            linestyle=st["linestyle"],
            linewidth=LINEWIDTH,
            label=st["label"],
        )

    ax_p.axhline(
        1.0,
        color=REFERENCE_COLOR,
        linewidth=REFERENCE_LINE_WIDTH,
    )

    ax_p.set_xscale("log")
    ax_p.set_ylim(*PSD_YLIM)
    ax_p.invert_xaxis()
    ax_p.set_xticks(
        [256, 64, 16, 8, 4, 2],
        ["256", "64", "16", "8", "4", "2"],
    )
    ax_p.minorticks_off()

    ax_p.set_xlabel("Spatial wavelength (pixels)", labelpad=12)
    ax_p.set_ylabel("Power,\nimage / reference", labelpad=12)
    add_panel_label(ax_p, "b")

    if SHOW_GRID:
        ax_p.grid(alpha=GRID_ALPHA)

    # ------------------------------------------------------------
    # (c)–(e) Injected-element recovery
    # ------------------------------------------------------------
    sigmas = ("1", "2", "4")
    panel_letters = ("c", "d", "e")

    for ax, sig, letter in zip(ax_i, sigmas, panel_letters):

        for m in ("n2n_mixsigma", "tv", "bm3d", "blur2"):
            st = STYLE[m]
            x, y = series(f"injection_sigma{sig}", m)

            ax.plot(
                np.array(x, dtype=float),
                y,
                color=st["color"],
                marker=st["marker"],
                linestyle=st["linestyle"],
                linewidth=LINEWIDTH,
                markersize=MARKERSIZE,
                label=st["label"],
            )

        ax.axhline(
            1.0,
            color=REFERENCE_COLOR,
            linewidth=REFERENCE_LINE_WIDTH,
        )

        ax.set_ylim(*INJECTION_YLIM)
        ax.set_xticks([10, 20, 30, 50])
        ax.set_xlabel("Injected peak field (Gauss)", labelpad=12)

        # Centered title only; panel label is placed independently.
        ax.set_title(
            rf"Element width $\sigma$ = {sig} px",
            loc="center",
            pad=8,
        )

        # add_panel_label(ax, letter)

        if SHOW_GRID:
            ax.grid(alpha=GRID_ALPHA)

    ax_i[0].set_ylabel("Aperture flux,\nrecovered / injected", labelpad=12)

    # (c): (a)와 같은 왼쪽 위치
    ax_i[0].text(
        -0.3, 1.02, "(c)",
        transform=ax_i[0].transAxes,
        ha="left", va="bottom",
        fontsize=PANEL_LABEL_FONT,
        clip_on=False,
    )

    # (d), (e): y-axis 왼쪽 spine에 맞춤
    ax_i[1].text(
        -0.115, 1.02, "(d)",
        transform=ax_i[1].transAxes,
        ha="left", va="bottom",
        fontsize=PANEL_LABEL_FONT,
        clip_on=False,
    )

    ax_i[2].text(
        -0.12, 1.02, "(e)",
        transform=ax_i[2].transAxes,
        ha="left", va="bottom",
        fontsize=PANEL_LABEL_FONT,
        clip_on=False,
    )

    # ------------------------------------------------------------
    # One common legend below the whole figure
    # ------------------------------------------------------------
    legend_handles = [
        Line2D(
            [0], [0],
            color=STYLE["input"]["color"],
            linestyle=STYLE["input"]["linestyle"],
            linewidth=LINEWIDTH,
            label=STYLE["input"]["label"],
        ),
        Line2D(
            [0], [0],
            color=STYLE["n2n_mixsigma"]["color"],
            marker=STYLE["n2n_mixsigma"]["marker"],
            linestyle=STYLE["n2n_mixsigma"]["linestyle"],
            linewidth=LINEWIDTH,
            markersize=MARKERSIZE,
            label=STYLE["n2n_mixsigma"]["label"],
        ),
        Line2D(
            [0], [0],
            color=STYLE["tv"]["color"],
            marker=STYLE["tv"]["marker"],
            linestyle=STYLE["tv"]["linestyle"],
            linewidth=LINEWIDTH,
            markersize=MARKERSIZE,
            label=STYLE["tv"]["label"],
        ),
        Line2D(
            [0], [0],
            color=STYLE["bm3d"]["color"],
            marker=STYLE["bm3d"]["marker"],
            linestyle=STYLE["bm3d"]["linestyle"],
            linewidth=LINEWIDTH,
            markersize=MARKERSIZE,
            label=STYLE["bm3d"]["label"],
        ),
        Line2D(
            [0], [0],
            color=STYLE["blur2"]["color"],
            marker=STYLE["blur2"]["marker"],
            linestyle=STYLE["blur2"]["linestyle"],
            linewidth=LINEWIDTH,
            markersize=MARKERSIZE,
            label=STYLE["blur2"]["label"],
        ),
    ]

    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=LEGEND_ANCHOR,
        frameon=False,
        fontsize=LEGEND_FONT,
        ncol=LEGEND_NCOL,
        handlelength=LEGEND_HANDLELENGTH,
        columnspacing=LEGEND_COLUMNSPACING,
        labelspacing=LEGEND_LABELSPACING,
    )

    # ------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------
    fig.subplots_adjust(
        left=LEFT,
        right=RIGHT,
        top=TOP,
        bottom=BOTTOM,
    )


    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        # Save after checking the preview.
        fig.savefig(
            out_dir / f"{OUTPUT_BASENAME}.png",
            dpi=DPI,
            bbox_inches="tight",
        )

        fig.savefig(
            out_dir / f"{OUTPUT_BASENAME}.pdf",
            bbox_inches="tight",
        )

        print("Saved:")
        print(out_dir / f"{OUTPUT_BASENAME}.png")
        print(out_dir / f"{OUTPUT_BASENAME}.pdf")

    if show:
        plt.show()
    return fig
