"""
Publication-quality figures for SecHarness paper.
Generates Figure 2 (Tool Usage) and Figure 3 (Chain Length Distribution).
Target: Computers & Security (Elsevier), elsarticle preprint single-column.
"""

import matplotlib.pyplot as plt
import numpy as np
import os

# ============================================================
# Style Configuration (house plotting standard)
# ============================================================
STYLE = {
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'font.size': 9,
    'axes.labelsize': 10,
    'axes.titlesize': 10,
    'axes.linewidth': 0.6,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'xtick.major.width': 0.5,
    'ytick.major.width': 0.5,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'legend.fontsize': 8,
    'legend.framealpha': 0.8,
    'legend.edgecolor': '0.8',
    'lines.linewidth': 1.0,
    'lines.markersize': 4,
    'grid.alpha': 0.3,
    'grid.linewidth': 0.4,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.02,
    'figure.constrained_layout.use': True,
    'pdf.fonttype': 42,       # embed fonts (Elsevier requirement)
    'ps.fonttype': 42,
}
plt.rcParams.update(STYLE)

# Accessible color palette
COLORS = {
    '3B': '#0077BB',   # blue
    '7B': '#EE7733',   # orange
    'E4': '#009988',   # teal
}

# Elsevier single-column preprint: ~6.5in text width
FIG_SINGLE = (5.5, 3.4)
FIG_WIDE = (5.5, 2.8)

# Output directories
PAPER_FIG_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'paper', 'figures')
RESULT_FIG_DIR = os.path.join(os.path.dirname(__file__), '..', 'results', 'figures')
os.makedirs(PAPER_FIG_DIR, exist_ok=True)
os.makedirs(RESULT_FIG_DIR, exist_ok=True)


def verify_figure(fig, target='single'):
    """Verify figure meets publication standards."""
    w, h = fig.get_size_inches()
    expected_w = 5.5 if target == 'single' else 6.5
    if abs(w - expected_w) > 1.5:
        print(f"[WARN] Width {w:.2f} differs from expected {expected_w} for {target}")
    # Font size check
    for ax in fig.get_axes():
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            fs = label.get_fontsize()
            if fs < 6:
                print(f"[WARN] Font too small: {fs}pt")
    # Font type check
    ft = plt.rcParams.get('pdf.fonttype', None)
    if ft != 42:
        print(f"[WARN] pdf.fonttype={ft}, should be 42 for embedded fonts")
    print(f"[OK] Figure verified: {w:.2f}x{h:.2f} inches, {target}-column")


# ============================================================
# Figure 2: Tool Usage Comparison (3B vs 7B vs E4)
# ============================================================
def plot_tool_usage():
    """Grouped bar chart: tool invocation rates by model size."""

    tools = [
        'check_anomaly',
        'lookup_signature',
        'query_history',
        'load_knowledge',
    ]
    tool_labels = [
        'check\nanomaly',
        'lookup\nsignature',
        'query\nhistory',
        'load\nknowledge',
    ]

    # Data from v2_8b_scaling_comparison.csv and E4_unsw_results.csv
    rates_3b = [1.0, 0.0, 0.0, 0.0]        # E3: 3B — single-tool strategy
    rates_7b = [1.0, 0.155, 0.32, 0.125]    # E3-8B: 7B — multi-tool strategy
    # E4 (fine-tuned 3B) also single-tool
    rates_e4 = [1.0, 0.0, 0.0, 0.0]

    x = np.arange(len(tools))
    width = 0.25

    fig, ax = plt.subplots(figsize=FIG_SINGLE)

    bars1 = ax.bar(x - width, rates_3b, width, label='E3 (3B, zero-shot)',
                   color=COLORS['3B'], edgecolor='white', linewidth=0.5,
                   hatch='')
    bars2 = ax.bar(x, rates_7b, width, label='E3-7B (7B, zero-shot)',
                   color=COLORS['7B'], edgecolor='white', linewidth=0.5,
                   hatch='///')
    bars3 = ax.bar(x + width, rates_e4, width, label='E4 (3B, fine-tuned)',
                   color=COLORS['E4'], edgecolor='white', linewidth=0.5,
                   hatch='...')

    ax.set_ylabel('Invocation Rate')
    ax.set_xlabel('Tool')
    ax.set_xticks(x)
    ax.set_xticklabels(tool_labels)
    ax.set_ylim(0, 1.15)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.legend(loc='upper right', frameon=True)
    ax.grid(axis='y', alpha=0.3)

    # Add value labels on bars > 0
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            if height > 0.01:
                ax.annotate(f'{height:.0%}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3), textcoords="offset points",
                           ha='center', va='bottom', fontsize=7)

    verify_figure(fig)

    for ext in ['pdf', 'png']:
        fig.savefig(os.path.join(PAPER_FIG_DIR, f'fig2_tool_usage.{ext}'),
                    bbox_inches='tight', facecolor='white')
        fig.savefig(os.path.join(RESULT_FIG_DIR, f'fig2_tool_usage.{ext}'),
                    bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print("[DONE] Figure 2: Tool Usage Comparison saved.")


# ============================================================
# Figure 3: Chain Length Distribution (3B vs 7B)
# ============================================================
def plot_chain_distribution():
    """Bar chart: chain length distribution showing 3B uniform vs 7B bimodal."""

    # Chain length counts from E3-8B audit JSONL analysis (n=200)
    # 3B (E3): all 200 samples have chain_length = 2
    counts_3b = {2: 200, 3: 0, 4: 0, 5: 0}

    # 7B (E3-8B): bimodal distribution from actual audit data
    # 52.5% chain=2 (105), 4.0% chain=3 (8), 2.0% chain=4 (4), 41.5% chain=5 (83)
    # Verify: 105*2 + 8*3 + 4*4 + 83*5 = 210+24+16+415 = 665 / 200 = 3.325
    counts_7b = {2: 105, 3: 8, 4: 4, 5: 83}

    chain_lengths = [2, 3, 4, 5]
    width = 0.35

    fig, ax = plt.subplots(figsize=FIG_SINGLE)

    x = np.arange(len(chain_lengths))
    bars_3b = ax.bar(x - width/2, [counts_3b[c] for c in chain_lengths], width,
                     alpha=0.7, label='E3 (3B)',
                     color=COLORS['3B'], edgecolor='black', linewidth=0.5)
    bars_7b = ax.bar(x + width/2, [counts_7b[c] for c in chain_lengths], width,
                     alpha=0.7, label='E3-7B (7B)',
                     color=COLORS['7B'], edgecolor='black', linewidth=0.5,
                     hatch='///')

    ax.set_xlabel('Chain Length (tool calls + classify)')
    ax.set_ylabel('Number of Samples')
    ax.set_xticks(x)
    ax.set_xticklabels([str(c) for c in chain_lengths])
    ax.legend(loc='upper right', frameon=True)
    ax.grid(axis='y', alpha=0.3)

    # Annotate key insight
    ax.annotate('3B: single-tool\nstrategy (100%)',
               xy=(0 - width/2, 200), xytext=(1.5, 170),
               arrowprops=dict(arrowstyle='->', color='#0077BB', lw=1.2),
               fontsize=7, color='#0077BB', ha='center')

    ax.annotate('7B: bimodal\n(multi-tool exploration)',
               xy=(3 + width/2, 83), xytext=(2.8, 130),
               arrowprops=dict(arrowstyle='->', color='#EE7733', lw=1.2),
               fontsize=7, color='#EE7733', ha='center')

    verify_figure(fig)

    for ext in ['pdf', 'png']:
        fig.savefig(os.path.join(PAPER_FIG_DIR, f'fig3_chain_distribution.{ext}'),
                    bbox_inches='tight', facecolor='white')
        fig.savefig(os.path.join(RESULT_FIG_DIR, f'fig3_chain_distribution.{ext}'),
                    bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print("[DONE] Figure 3: Chain Length Distribution saved.")


# ============================================================
# Figure 1: SecHarness Architecture (simplified block diagram)
# ============================================================
def plot_architecture():
    """SecHarness architecture diagram using matplotlib patches."""

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 9)
    ax.axis('off')

    from matplotlib.patches import FancyBboxPatch

    # Color scheme
    C_AGENT = '#E8F4FD'    # light blue - agent
    C_TOOLS = '#FFF3E0'    # light orange - tools
    C_KNOW = '#E8F5E9'     # light green - knowledge
    C_OBS = '#F3E5F5'      # light purple - observation
    C_ACT = '#FBE9E7'      # light red - action
    C_PERM = '#FFF9C4'     # light yellow - permissions
    C_BORDER = '#333333'
    C_INPUT = '#B0BEC5'
    C_OUTPUT = '#B0BEC5'

    def draw_box(x, y, w, h, color, label, sublabels=None, fontsize=9):
        box = FancyBboxPatch((x, y), w, h,
                            boxstyle="round,pad=0.1",
                            facecolor=color, edgecolor=C_BORDER,
                            linewidth=1.0)
        ax.add_patch(box)
        if sublabels:
            ax.text(x + w/2, y + h - 0.25, label,
                   ha='center', va='top', fontsize=fontsize,
                   fontweight='bold', color='#333333')
            for i, sub in enumerate(sublabels):
                ax.text(x + w/2, y + h - 0.55 - i*0.28, sub,
                       ha='center', va='top', fontsize=7,
                       fontfamily='monospace', color='#555555')
        else:
            ax.text(x + w/2, y + h/2, label,
                   ha='center', va='center', fontsize=fontsize,
                   fontweight='bold', color='#333333')

    def draw_arrow(x1, y1, x2, y2, color='#666666', style='->', lw=1.2):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                   arrowprops=dict(arrowstyle=style, color=color,
                                  lw=lw, connectionstyle='arc3,rad=0'))

    # === Central Agent Box ===
    draw_box(3.8, 3.8, 4.4, 1.6, C_AGENT, 'LLM Agent',
            sublabels=['Observe → Reason → Act',
                      'Iterative loop (max 5 steps)'])

    # === Input (left) ===
    draw_box(0.2, 4.1, 2.6, 1.0, C_INPUT, 'Traffic Sample',
            sublabels=['(kv-format)'], fontsize=8)
    draw_arrow(2.8, 4.6, 3.8, 4.6)

    # === Output (right) ===
    draw_box(9.2, 4.1, 2.6, 1.0, C_OUTPUT, 'Audit Record',
            sublabels=['verdict + chain'], fontsize=8)
    draw_arrow(8.2, 4.6, 9.2, 4.6)

    # === Tools (top-left) ===
    draw_box(0.3, 6.5, 5.0, 1.8, C_TOOLS, 'Tools (T)',
            sublabels=['check_anomaly (ML classifier)',
                      'lookup_signature (signature DB)',
                      'query_history (detection history)',
                      'load_knowledge (domain KB)'])
    draw_arrow(3.5, 6.5, 4.8, 5.4, color='#EE7733')

    # === Knowledge (top-right) ===
    draw_box(6.7, 6.5, 5.0, 1.8, C_KNOW, 'Knowledge (K)',
            sublabels=['Attack pattern descriptions',
                      'Protocol baselines',
                      'Alert interpretation guides',
                      '8 curated documents'])
    draw_arrow(8.5, 6.5, 7.2, 5.4, color='#009988')

    # === Observation (bottom-left) ===
    draw_box(0.1, 0.5, 3.5, 1.8, C_OBS, 'Observation (O)',
            sublabels=['Traffic pre-analysis features',
                      'Sliding-window history',
                      'Per-step tool outputs',
                      'Intermediate decisions'])
    draw_arrow(2.5, 2.3, 4.5, 3.8, color='#9C27B0')

    # === Action (bottom-center) — now a proper box ===
    draw_box(4.2, 0.5, 3.6, 1.8, C_ACT, 'Action (A)',
            sublabels=['classify (final verdict)',
                      'escalate (human referral)',
                      'log_decision (reasoning)'])
    draw_arrow(6.0, 2.3, 6.0, 3.8, color='#B71C1C')

    # === Permissions (bottom-right) ===
    draw_box(8.4, 0.5, 3.5, 1.8, C_PERM, 'Permissions (P)',
            sublabels=['FormatValidator (JSON schema)',
                      'ConfidenceGate (threshold)',
                      'ConsistencyChecker (conflicts)',
                      'OutputLimiter (max steps)'])
    draw_arrow(9.5, 2.3, 7.5, 3.8, color='#F44336')

    # === Title ===
    ax.text(6.0, 8.75, 'SecHarness: $H = (T, K, O, A, P)$',
           ha='center', va='top', fontsize=11, fontweight='bold')

    verify_figure(fig)

    for ext in ['pdf', 'png']:
        fig.savefig(os.path.join(PAPER_FIG_DIR, f'fig1_architecture.{ext}'),
                    bbox_inches='tight', facecolor='white')
        fig.savefig(os.path.join(RESULT_FIG_DIR, f'fig1_architecture.{ext}'),
                    bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print("[DONE] Figure 1: SecHarness Architecture saved.")


# ============================================================
# Figure 4: 2×2 Factorial Decomposition (Grouped Bar)
# Template: Multi-Metric Grouped Bar
# ============================================================
def plot_factorial():
    """2×2 factorial bar chart showing ceiling effect."""
    import seaborn as sns

    # Paul Tol bright palette
    TOL_BRIGHT = ['#4477AA', '#EE6677', '#228833', '#CCBB44', '#66CCEE', '#AA3377', '#BBBBBB']

    conditions = ['Zero-shot', 'Fine-tuned']
    no_harness = [27.9, 87.7]
    with_harness = [92.4, 92.4]  # E3, E4: identical RF-relay predictions

    fig, ax = plt.subplots(figsize=FIG_SINGLE)
    x = np.arange(len(conditions))
    width = 0.3

    # Hatching for B&W distinguishability (house standard)
    bars1 = ax.bar(x - width/2, no_harness, width, label='No Harness',
                   color=TOL_BRIGHT[6], edgecolor='black', linewidth=0.5,
                   hatch='///')
    bars2 = ax.bar(x + width/2, with_harness, width, label='SecHarness',
                   color=TOL_BRIGHT[0], edgecolor='black', linewidth=0.5,
                   hatch='')

    # Value labels on top
    for bar in list(bars1) + list(bars2):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.0,
               f'{bar.get_height():.1f}', ha='center', va='bottom', fontsize=6)

    # Harness-effect: diagonal arrow between the two bars of each group,
    # with the pp label placed between them (white bbox avoids overlap)
    for i in range(len(conditions)):
        diff = with_harness[i] - no_harness[i]
        ax.annotate('', xy=(x[i] + width/2, with_harness[i]),
                    xytext=(x[i] - width/2, no_harness[i]),
                    arrowprops=dict(arrowstyle='->', color=TOL_BRIGHT[1], lw=1.0))
        ax.text(x[i], (no_harness[i] + with_harness[i]) / 2,
                f'+{diff:.1f}pp', fontsize=6, fontweight='bold', color=TOL_BRIGHT[1],
                ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='none', alpha=0.85))

    # Ceiling annotation: E3 and E4 are identical, centered near top, clear of bars/edges
    ax.text(0.5, 103.5, 'E3 $=$ E4: identical (92.4)', fontsize=6, color='dimgray',
            ha='center', va='center')

    ax.set_ylabel('Accuracy (%)')
    ax.set_xticks(x)
    ax.set_xticklabels(conditions)
    ax.set_ylim(0, 112)
    # legend outside the plot (top), so it never overlaps the bars/annotations
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.14), ncol=2,
              fontsize=6, frameon=False)

    verify_figure(fig)
    for ext in ['pdf', 'png']:
        fig.savefig(os.path.join(PAPER_FIG_DIR, f'fig4_factorial.{ext}'),
                    bbox_inches='tight', facecolor='white')
        fig.savefig(os.path.join(RESULT_FIG_DIR, f'fig4_factorial.{ext}'),
                    bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print("[DONE] Figure 4: Factorial Decomposition saved.")


# ============================================================
# Figure 5: Confusion Matrices (E1, E3, E4)
# Template: Confusion Matrix (Heatmap) using seaborn
# ============================================================
def plot_confusion_matrices():
    """Side-by-side confusion matrices using sns.heatmap."""
    import seaborn as sns

    # Data: rows=true [Normal, Attack], cols=pred [Normal, Attack]
    cm_e1 = np.array([[257, 63],   # Normal: TN, FP
                      [658, 22]])  # Attack: FN, TP
    cm_e3 = np.array([[282, 38],
                      [45, 635]])
    cm_e4 = np.array([[282, 38],
                      [38, 642]])

    labels = ['Normal', 'Attack']
    cms = [cm_e1, cm_e3, cm_e4]
    titles = ['(a) E1: Zero-shot', '(b) E3: ZS + SecHarness', '(c) E4: FT + SecHarness']

    fig, axes = plt.subplots(1, 3, figsize=(5.5, 2.2))

    for ax, cm, title in zip(axes, cms, titles):
        # Normalize by row (per-class recall) — house template pattern
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                    xticklabels=labels, yticklabels=labels,
                    linewidths=0.5, linecolor='white',
                    cbar=False, vmin=0, vmax=1.0,
                    annot_kws={'size': 8}, ax=ax)
        ax.set_xlabel('Predicted', fontsize=8)
        if ax == axes[0]:
            ax.set_ylabel('Actual', fontsize=8)
        else:
            ax.set_ylabel('')
        ax.set_title(title, fontsize=8, pad=4)
        ax.tick_params(labelsize=7)

    verify_figure(fig)
    for ext in ['pdf', 'png']:
        fig.savefig(os.path.join(PAPER_FIG_DIR, f'fig5_confusion_matrix.{ext}'),
                    bbox_inches='tight', facecolor='white')
        fig.savefig(os.path.join(RESULT_FIG_DIR, f'fig5_confusion_matrix.{ext}'),
                    bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print("[DONE] Figure 5: Confusion Matrices saved.")


# ============================================================
# Figure 6: Latency Comparison
# Template: Method Comparison (Bar Chart)
# ============================================================
def plot_latency():
    """Bar chart comparing inference latency with hatching for B&W."""
    # Paul Tol bright palette
    TOL_BRIGHT = ['#4477AA', '#EE6677', '#228833', '#CCBB44', '#66CCEE', '#AA3377', '#BBBBBB']

    conditions = ['E1\nZero-shot', 'E2\nFine-tuned', 'E3\nZS+Harness', 'E4\nFT+Harness',
                  'E3-7B\n7B+Harness']
    latencies = [3.733, 4.558, 11.098, 4.350, 39.300]  # seconds
    colors_lat = [TOL_BRIGHT[6], TOL_BRIGHT[6], TOL_BRIGHT[0], TOL_BRIGHT[2], TOL_BRIGHT[1]]
    hatches = ['///', '\\\\\\', '', 'xxx', '...']  # B&W distinguishability

    fig, ax = plt.subplots(figsize=FIG_SINGLE)
    bars = ax.bar(range(len(conditions)), latencies, color=colors_lat,
                  edgecolor='black', linewidth=0.5, width=0.6)

    # Apply hatching
    for bar, hatch in zip(bars, hatches):
        bar.set_hatch(hatch)

    # Value labels on top (house template pattern)
    for bar, val in zip(bars, latencies):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
               f'{val:.1f}s', ha='center', va='bottom', fontsize=6)

    # Speedup annotation E3→E4
    ax.annotate('', xy=(3, 4.350), xytext=(2, 11.098),
               arrowprops=dict(arrowstyle='->', color=TOL_BRIGHT[1], lw=1.5,
                              connectionstyle='arc3,rad=-0.3'))
    ax.text(2.8, 8.5, '2.6\u00d7 faster', fontsize=7, color=TOL_BRIGHT[1],
           fontweight='bold', ha='center')

    ax.set_ylabel('Latency per sample (s)')
    ax.set_xticks(range(len(conditions)))
    ax.set_xticklabels(conditions, fontsize=7)
    ax.set_ylim(0, 45)

    verify_figure(fig)
    for ext in ['pdf', 'png']:
        fig.savefig(os.path.join(PAPER_FIG_DIR, f'fig6_latency.{ext}'),
                    bbox_inches='tight', facecolor='white')
        fig.savefig(os.path.join(RESULT_FIG_DIR, f'fig6_latency.{ext}'),
                    bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print("[DONE] Figure 6: Latency Comparison saved.")


# ============================================================
# Figure 7: Cross-Dataset Comparison (UNSW vs CIC)
# Template: Ablation Results (Subplot Grid) + Grouped Bar
# ============================================================
def plot_cic_comparison():
    """Grouped bar chart with hatching, sharey, Paul Tol palette."""
    # PAIR_COLORS: blue + orange, deuteranopia-safe
    PAIR_COLORS = ('#0077BB', '#EE7733')

    metrics = ['Accuracy', 'F1', 'ZD Rate']
    e3_unsw = [92.4, 94.4, 97.0]  # E3 := RF relay (identical to E4 on UNSW)
    e3_cic = [89.6, 71.4, 0.0]
    e4_unsw = [92.4, 94.4, 97.0]
    e4_cic = [89.2, 70.0, 0.0]

    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.8), sharey=True)
    x = np.arange(len(metrics))
    width = 0.3

    for ax, unsw, cic, title in [
        (axes[0], e3_unsw, e3_cic, '(a) E3: Zero-shot + SecHarness'),
        (axes[1], e4_unsw, e4_cic, '(b) E4: Fine-tuned + SecHarness'),
    ]:
        b1 = ax.bar(x - width/2, unsw, width, label='UNSW-NB15',
                    color=PAIR_COLORS[0], edgecolor='black', linewidth=0.4)
        b2 = ax.bar(x + width/2, cic, width, label='CIC-IDS2017',
                    color=PAIR_COLORS[1], edgecolor='black', linewidth=0.4,
                    hatch='///')  # hatching for B&W

        # Value labels
        for bar, val in zip(list(b1) + list(b2), list(unsw) + list(cic)):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                       f'{val:.1f}', ha='center', va='bottom', fontsize=7)

        # Highlight ZD=0%
        ax.annotate('0%', xy=(2 + width/2, 1), fontsize=7, ha='center',
                   va='bottom', color='#EE6677', fontweight='bold')

        ax.set_title(title, fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels(metrics, fontsize=7)
        ax.set_ylim(0, 128)  # headroom keeps the legend clear of the 97.0 ZD bar
        ax.legend(fontsize=7, loc='upper right')

    axes[0].set_ylabel('Score (%)')

    verify_figure(fig)
    for ext in ['pdf', 'png']:
        fig.savefig(os.path.join(PAPER_FIG_DIR, f'fig7_cic_comparison.{ext}'),
                    bbox_inches='tight', facecolor='white')
        fig.savefig(os.path.join(RESULT_FIG_DIR, f'fig7_cic_comparison.{ext}'),
                    bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print("[DONE] Figure 7: Cross-Dataset Comparison saved.")


# ============================================================
# Main
# ============================================================
if __name__ == '__main__':
    print("Generating paper figures...")
    plot_architecture()
    plot_tool_usage()
    plot_chain_distribution()
    plot_factorial()
    plot_confusion_matrices()
    plot_latency()
    plot_cic_comparison()
    print("\nAll figures saved to:")
    print(f"  Paper: {os.path.abspath(PAPER_FIG_DIR)}")
    print(f"  Results: {os.path.abspath(RESULT_FIG_DIR)}")
