# -*- coding: utf-8 -*-
"""
Generate a self-contained data summary (xlsx) + figures (png) + a full report
(docx & md) for the NO-validation rolling backtest, into a dedicated folder
meant to be sent to the supervisor.

Input : results_wf_noval/  (7 windows x 3 seeds, no validation)
Output: ../report_no_validation_rolling/
          - no_validation_rolling_data_summary.xlsx   (multi-sheet)
          - figures/*.png
          - no_validation_rolling_results_report.docx
          - no_validation_rolling_results_report.md
          - source_csv/        (copies of the source csvs for transparency)

Run: python -X utf8 report_rollback_noval.py
"""

import os
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import docx
from docx import Document
from docx.shared import Pt, Inches, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, 'results_wf_noval')
OUT = os.path.join(HERE, '..', 'report_no_validation_rolling')
FIG = os.path.join(OUT, 'figures')

WINDOWS = ['rolling1', 'rolling2', 'rolling3', 'main', 'rolling5', 'rolling6', 'second']
SEEDS = [42, 123, 456]
MODELS = ['a2c', 'ddpg', 'ppo', 'td3', 'sac', 'equal_weight', 'close']
DISPLAY = {'a2c': 'A2C', 'ddpg': 'DDPG', 'ppo': 'PPO', 'td3': 'TD3', 'sac': 'SAC',
           'equal_weight': 'Equal-Weight Option', 'close': 'ETF(Buy&Hold)'}
DRL = ['a2c', 'ddpg', 'ppo', 'td3', 'sac']

# Original single-window run: recorded out-of-sample (test-period) values, verbatim
# from the initial notebook (2026-05-09, single window, NO validation, NO seed).
RECORDED = {
    'tvm':    {'a2c': 1.270552, 'ddpg': 4.278962, 'ppo': 2.947627,
               'td3': 0.266330, 'sac': 0.927517, 'close': 0.931863},
    'mdd':    {'a2c': -0.203995, 'ddpg': -0.279185, 'ppo': -0.177951,
               'td3': -0.795856, 'sac': -0.419028, 'close': -0.084297},
    'sharpe': {'a2c': 1.826308, 'ddpg': 4.989122, 'ppo': 4.705803,
               'td3': -4.004336, 'sac': 0.231964, 'close': -3.659669},
}

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


# ---------------- data loading / recomputation ----------------
def load_test_returns(win, seed):
    p = os.path.join(SRC, f'{win}_seed{seed}', 'test_returns.csv')
    df = pd.read_csv(p, index_col=0, parse_dates=True)
    return df.sort_index()


def pooled_by_seed():
    """{seed: DataFrame(index=date, columns=model)} pooled over windows in time order."""
    out = {}
    for seed in SEEDS:
        cols = {}
        for m in MODELS:
            segs = [load_test_returns(w, seed)[f'{m}_r'].dropna() for w in WINDOWS]
            s = pd.concat(segs).sort_index()
            s = s[~s.index.duplicated(keep='last')]
            cols[m] = s
        out[seed] = pd.DataFrame(cols)
    return out


def max_drawdown(x):
    nav = (1 + x).cumprod()
    return float((nav / nav.cummax() - 1).min())


def compound(x):
    return float((1 + x).prod())


# ---------------- figures ----------------
def fig_nav(pooled, windows_metrics):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    # (a) cross-seed MEAN nav, all models
    ax = axes[0]
    mean_nav = None
    navs = {}
    for m in MODELS:
        n = pd.concat([(1 + pooled[s][m]).cumprod() for s in SEEDS], axis=1).mean(axis=1)
        navs[m] = n
        ax.plot(n.index, n.values, label=DISPLAY[m], lw=1.8,
                ls='--' if m == 'close' else ('-.' if m == 'equal_weight' else '-'))
    ax.axhline(1.0, color='gray', lw=0.8, alpha=0.6)
    ax.set_title('(a) Pooled full-period NAV (mean across 3 seeds)')
    ax.set_xlabel('Date')
    ax.set_ylabel('NAV (initial=1)')
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.3)
    # (b) per-seed nav for DRL algorithms
    ax = axes[1]
    for m in DRL:
        for s in SEEDS:
            n = (1 + pooled[s][m]).cumprod()
            ax.plot(n.index, n.values, lw=0.8, alpha=0.5)
        ax.plot(navs[m].index, navs[m].values, lw=2.0, label=DISPLAY[m])
    ax.axhline(1.0, color='gray', lw=0.8, alpha=0.6)
    ax.set_title('(b) NAV of the 5 DRL algorithms per seed (thin=single seed, thick=mean)')
    ax.set_xlabel('Date')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    p = os.path.join(FIG, 'fig1_pooled_nav.png')
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def fig_tvm_by_window(windows_metrics):
    piv = windows_metrics.pivot_table(index='window', columns='model', values='tvm',
                                      aggfunc='mean').reindex(WINDOWS)[MODELS]
    x = np.arange(len(WINDOWS))
    w = 0.11
    fig, ax = plt.subplots(figsize=(12, 5))
    for i, m in enumerate(MODELS):
        ax.bar(x + (i - 3) * w, piv[m].values, w, label=DISPLAY[m])
    ax.axhline(1.0, color='k', lw=0.8, ls='--')
    ax.set_xticks(x)
    ax.set_xticklabels(WINDOWS)
    ax.set_ylabel('TVM (final / initial NAV)')
    ax.set_title('TVM by window (mean across 3 seeds; dashed=1, below means loss)')
    ax.legend(fontsize=8, ncol=4)
    ax.grid(alpha=0.3, axis='y')
    fig.tight_layout()
    p = os.path.join(FIG, 'fig2_tvm_by_window.png')
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def fig_main_by_seed(windows_metrics):
    main = windows_metrics[windows_metrics['window'] == 'main']
    piv = main.pivot_table(index='model', columns='seed', values='tvm').reindex(MODELS)
    x = np.arange(len(MODELS))
    w = 0.25
    fig, ax = plt.subplots(figsize=(9, 5))
    for i, s in enumerate(SEEDS):
        ax.bar(x + (i - 1) * w, piv[s].values, w, label=f'seed={s}')
    ax.axhline(1.0, color='k', lw=0.8, ls='--')
    ax.set_xticks(x)
    ax.set_xticklabels([DISPLAY[m] for m in MODELS], rotation=20)
    ax.set_ylabel('TVM')
    ax.set_title('main (= main experiment): TVM by algorithm x 3 seeds')
    ax.legend()
    ax.grid(alpha=0.3, axis='y')
    fig.tight_layout()
    p = os.path.join(FIG, 'fig3_main_by_seed.png')
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def _heatmap(ax, data, title, fmt='{:.2f}', cmap='RdYlGn', center=1.0):
    im = ax.imshow(data.values, cmap=cmap, aspect='auto',
                   vmin=np.nanmin(data.values), vmax=np.nanmax(data.values))
    ax.set_xticks(range(data.shape[1]))
    ax.set_xticklabels([DISPLAY[c] for c in data.columns], rotation=20)
    ax.set_yticks(range(data.shape[0]))
    ax.set_yticklabels(data.index)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            v = data.values[i, j]
            ax.text(j, i, fmt.format(v), ha='center', va='center', fontsize=8,
                    color='black')
    ax.set_title(title)
    return im


def fig_heatmaps(windows_metrics):
    tvm = windows_metrics.pivot_table(index='window', columns='model', values='tvm',
                                      aggfunc='mean').reindex(WINDOWS)[MODELS]
    shp = windows_metrics.pivot_table(index='window', columns='model', values='sharpe',
                                      aggfunc='mean').reindex(WINDOWS)[MODELS]
    fig, axes = plt.subplots(2, 1, figsize=(11, 9))
    _heatmap(axes[0], tvm, 'TVM by window (mean across 3 seeds)')
    _heatmap(axes[1], shp, 'Sharpe by window (mean across 3 seeds)')
    fig.tight_layout()
    p = os.path.join(FIG, 'fig4_window_heatmaps.png')
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def fig_pooled_bars(pooled_summary):
    ps = pooled_summary.set_index('model').reindex(MODELS)
    x = np.arange(len(MODELS))
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, col, ttl in [
            (axes[0], 'tvm', 'Pooled full-period TVM (3-seed mean±std)'),
            (axes[1], 'sharpe', 'Pooled full-period Sharpe (3-seed mean±std)')]:
        ax.bar(x, ps[f'{col}_mean'].values, yerr=ps[f'{col}_std'].values,
               capsize=4, color='#4C72B0')
        ax.axhline(1.0 if col == 'tvm' else 0.0, color='k', lw=0.8, ls='--')
        ax.set_xticks(x)
        ax.set_xticklabels([DISPLAY[m] for m in MODELS], rotation=20)
        ax.set_title(ttl)
        ax.grid(alpha=0.3, axis='y')
    fig.tight_layout()
    p = os.path.join(FIG, 'fig5_pooled_metrics.png')
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


# ---------------- excel ----------------
def build_config_sheet():
    rows = [
        ['Item', 'Original single-window run', 'This rolling experiment'],
        ['Data range', '2022-07-01 ~ 2023-05-31 (223 trading days)',
         '2022-01-01 ~ 2023-12-31 (sliced per window)'],
        ['Window split', 'Single window: 9 months train + 2 months test',
         'Rolling: 9 months train + 2 months test, step 2 months, 7 windows'],
        ['Train/test boundary', 'train <2023-04-01 (184 days), test >=2023-04-01 (39 days)',
         'per window train=[start, +9m), test=[start+9m, +11m)'],
        ['Validation set', 'None', 'None'],
        ['Validation selection', 'None (final model used directly)', 'None (final model used directly)'],
        ['State-space dimension', '37 (4 indicators)', '67 (9 indicators) * see "Limitations"'],
        ['PPO training steps', '200,000', '200,000'],
        ['Steps of other algorithms', 'a2c/ddpg/td3 50,000; sac 70,000',
         'a2c/ddpg/td3 50,000; sac 70,000'],
        ['Random seeds', 'None (single training run)', '42 / 123 / 456 (3 seeds per window)'],
        ['Number of rolling windows', '1', '7'],
        ['Initial capital / other env params', '60,000; others as left', 'Same as left'],
        ['Reproduction anchor', '-', 'main train/test boundaries identical to the original single-window'],
    ]
    return pd.DataFrame(rows[1:], columns=rows[0])


def build_window_pivot(windows_metrics, value, agg='mean'):
    return windows_metrics.pivot_table(index='window', columns='model',
                                       values=value, aggfunc=agg).reindex(WINDOWS)[MODELS]


def build_main_repro(windows_metrics):
    main = windows_metrics[windows_metrics['window'] == 'main']
    rows = []
    for m in MODELS:
        sub = main[main['model'] == m]
        r = {'Model': DISPLAY[m]}
        r['seed42 TVM (this run)'] = sub[sub['seed'] == 42]['tvm'].iloc[0]
        r['seed123 TVM (this run)'] = sub[sub['seed'] == 123]['tvm'].iloc[0]
        r['seed456 TVM (this run)'] = sub[sub['seed'] == 456]['tvm'].iloc[0]
        r['Mean TVM (this run)'] = sub['tvm'].mean()
        r['Recorded TVM'] = RECORDED['tvm'].get(m, np.nan)
        r['TVM diff (mean - record)'] = r['Mean TVM (this run)'] - r['Recorded TVM']
        r['Mean Sharpe (this run)'] = sub['sharpe'].mean()
        r['Recorded Sharpe'] = RECORDED['sharpe'].get(m, np.nan)
        r['Mean MDD (this run)'] = sub['mdd'].mean()
        r['Recorded MDD'] = RECORDED['mdd'].get(m, np.nan)
        rows.append(r)
    return pd.DataFrame(rows)


def write_excel(cfg, wm, pooled_summary, seed_pooled, ttest, main_repro, path):
    with pd.ExcelWriter(path, engine='openpyxl') as xw:
        cfg.to_excel(xw, sheet_name='1_experiment_config', index=False)
        main_repro.to_excel(xw, sheet_name='2_main_repro_vs_5-9', index=False)
        # per-window cross-seed summary
        for value, label in [('tvm', 'TVM'), ('sharpe', 'Sharpe'), ('mdd', 'MDD')]:
            mean = build_window_pivot(wm, value, 'mean')
            std = build_window_pivot(wm, value, 'std')
            comb = mean.rename(columns=lambda c: f'{c}_mean')
            for c in std.columns:
                comb[f'{c}_std'] = std[c]
            comb.index.name = f'{label}_window_x_model(cross-seed)'
            comb.to_excel(xw, sheet_name=f'3_per_window_{label}_cross_seed')
        wm.to_excel(xw, sheet_name='4_per_window_metrics_all_seeds', index=False)
        seed_pooled.to_excel(xw, sheet_name='5_pooled_by_seed', index=False)
        pooled_summary.to_excel(xw, sheet_name='6_pooled_cross_seed', index=False)
        ttest.to_excel(xw, sheet_name='7_paired_t_test_by_seed', index=False)


# ---------------- report text ----------------
def md_tables():
    return None


def build_report(figs, tables, path_docx, path_md):
    wm, pooled_summary, seed_pooled, ttest, main_repro, pooled = tables

    def mstat(m, col='tvm'):
        return pooled_summary.set_index('model').loc[m, f'{col}_mean'], \
               pooled_summary.set_index('model').loc[m, f'{col}_std']

    # ---- markdown ----
    L = []
    A = L.append
    A('# No-validation rolling backtest: data summary and results report\n')
    A('**Objective**: test whether the experiment in the initial submission (2026-05-09, single window, '
      'no validation set) is reproducible, and whether its "9-month train + 2-month test" split is stable '
      'over an extended time series.\n')
    A('**Subject**: deep reinforcement learning (A2C / DDPG / PPO / TD3 / SAC) asset-allocation backtest '
      'on an HS300 ETF option portfolio.\n')
    A('\n## 1. Experiment design\n')
    A('- Uses **exactly the same window size** as the original experiment (9 months train + 2 months test, '
      '**no validation set**), slid over the extended data (2022-01-01 ~ 2023-12-31) with a **2-month step**, '
      'for **7 windows** in total.\n')
    A('- Each window is trained independently with **3 random seeds (42 / 123 / 456)** to avoid results that '
      'are merely an artefact of a single seed.\n')
    A('- **main** has train/test boundaries **identical** to the main experiment (train 2022-07~2023-03, '
      'test 2023-04~2023-05) and serves as the reproducibility anchor.\n')
    A('- Benchmarks: equal-weight option portfolio (Equal-Weight) and ETF buy-and-hold (Buy & Hold).\n')
    A('- Metrics: terminal value multiple TVM (final/initial), maximum drawdown MDD (based on daily returns), '
      'annualised Sharpe ratio (based on daily returns, 252 trading days, risk-free rate 0).\n')
    A('\n### Table 1  Experiment configuration comparison (original single-window vs this rolling run)\n')
    A(_md_table(build_config_sheet()))
    A('> Note: this run keeps the current **67-dimensional state space** (9 indicators), which differs from '
      'the **37 dimensions** (4 indicators) of the main experiment; the PPO training steps have been '
      'restored to the original **200,000**.\n')

    A('\n## 2. Per-window results\n')
    A('### Table 2  TVM by window (mean across 3 seeds)\n')
    A(_md_table(build_window_pivot(wm, 'tvm').round(3).reset_index(), index=False))
    A('\n![TVM by window](figures/%s)\n' % os.path.basename(figs['tvm_window']))
    A('### Table 3  Sharpe by window (mean across 3 seeds)\n')
    A(_md_table(build_window_pivot(wm, 'sharpe').round(2).reset_index(), index=False))
    A('\n![Window heatmaps](figures/%s)\n' % os.path.basename(figs['heat']))

    A('\n## 3. Reproducibility check: main (= main experiment)\n')
    A('### Table 4  main TVM per algorithm across 3 seeds vs the original record\n')
    A(_md_table(main_repro.round(4)))
    A('\n![main three seeds](figures/%s)\n' % os.path.basename(figs['main']))
    A('- **ETF benchmark**: all three seeds give **0.93186**, digit-by-digit identical to the original record '
      '**0.931863** -> the data pipeline, trading calendar and metric definitions are fully aligned.\n')
    A('- **PPO**: TVM across three seeds = 1.803 / 2.502 / 1.130 (Sharpe 5.62 / 5.53 / 3.80), '
      'the **only reinforcement-learning algorithm with positive returns on all 3 seeds**; consistent with the '
      'initial memory that "PPO performed well on main".\n')
    A('- **Other algorithms**: among the three seeds of a2c/ddpg/td3/sac, **at least one seed blows up** '
      '(e.g. ddpg seed42 TVM ~ 0, td3 seed456 TVM = 0.242).\n')
    A('- However, compared with the original record the values of the 5 DRL algorithms still differ greatly '
      '(e.g. DDPG recorded 4.279 while seed42 here is ~0) -> '
      '**the original single-run, seedless experimental record is not reproducible**.\n')

    A('\n## 4. Pooled full-period out-of-sample results (274 trading days)\n')
    A('### Table 5  Pooled full-period metrics (3-seed mean ± std)\n')
    ps = pooled_summary.copy()
    ps['Model'] = ps['model'].map(DISPLAY)
    show = ps[['Model', 'tvm_mean', 'tvm_std', 'mdd_mean', 'sharpe_mean', 'sharpe_std']]
    A(_md_table(show.round(4), index=False))
    A('\n![Pooled metrics](figures/%s)\n' % os.path.basename(figs['pooled']))
    A('\n![Pooled NAV](figures/%s)\n' % os.path.basename(figs['nav']))
    A('- All 5 reinforcement-learning algorithms **lose heavily** on all seeds, with TVM <= 0.10, far below the '
      'ETF benchmark (0.920); PPO averages **0.0001** (close to zero).\n')
    A('- Benchmarks: equal-weight option portfolio 0.0205, ETF buy-and-hold 0.9203 (identical across the three seeds).\n')

    A('\n## 5. Statistical tests\n')
    A('### Table 6  Paired t-tests of PPO against each comparator (per seed, daily returns)\n')
    A(_md_table(ttest.round(5), index=False))
    A('- Across the 3 seeds, the paired t-tests of PPO against each comparator all have **p > 0.10**, '
      'i.e. **no statistical significance**.\n')

    A('\n## 6. Conclusions\n')
    A('1. **Reproducibility**: checked against the main anchor, the ETF benchmark is fully reproduced; but the results '
      'of the 5 DRL algorithms are not reproducible - the original code used **no random seed and trained only once**, '
      'so its recorded values are highly accidental.\n')
    A('2. **Stability**: after rolling the same window size 7 times x 3 seeds over the extended period, '
      'algorithm performance fluctuates violently across windows, most windows blow up out of sample, no algorithm '
      'beats the ETF over the pooled period, and the results are not statistically significant -> '
      '**the "9-month train + 2-month test, no validation set" split is not robust**.\n')
    A('3. **On main**: PPO is profitable on all three main seeds, showing that the algorithm works in **this particular '
      'window**, but this is a **window-specific favourable case** and cannot be extrapolated into a general advantage.\n')

    A('\n## 7. Limitations and notes\n')
    A('- This rolling experiment keeps the current **67-dimensional state space** (including 5 new option features), '
      'which differs from the 37 dimensions of the main experiment; the "reproduction comparison" is therefore '
      'strictly a test under "**same data, same window, same training steps, but a different observation space and no '
      'seeds**".\n')
    A('- No validation set means the final model at the end of training is used directly; evidence shows that this '
      'setting easily degenerates into trend-following / gambling-style strategies, which is one of the main reasons '
      'for the widespread out-of-sample blow-ups in this round.\n')
    A('- This report is a diagnostic analysis, intended to document the reproducibility and split-robustness issues '
      'of the original experiment.\n')

    A('\n## Appendix: list of deliverables\n')
    A('- `no_validation_rolling_data_summary.xlsx`: 7 worksheets (configuration comparison, main reproduction comparison, '
      'per-window TVM/Sharpe/MDD cross-seed, per-window all-seed details, pooled by seed, pooled cross-seed, paired t-tests).\n')
    A('- `figures/`: fig1 pooled NAV, fig2 TVM by window, fig3 main three seeds, '
      'fig4 window heatmaps, fig5 pooled metrics.\n')
    A('- `source_csv/`: copies of the original summary files from the results directory.\n')

    with open(path_md, 'w', encoding='utf-8') as f:
        f.write('\n'.join(L))

    # ---- docx ----
    doc = Document()
    doc.styles['Normal'].font.name = 'Microsoft YaHei'
    doc.styles['Normal'].font.size = Pt(10.5)

    def H(text, level=1):
        doc.add_heading(text, level=level)

    def P(text, bold=False):
        p = doc.add_paragraph()
        r = p.add_run(text)
        r.bold = bold
        return p

    def PIC(path, width=6.3):
        doc.add_picture(path, width=Inches(width))
        doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER

    def TBL(df, index=False):
        cols = ([''] if index else []) + list(df.columns)
        t = doc.add_table(rows=1, cols=len(cols))
        t.style = 'Light Grid Accent 1'
        for j, c in enumerate(cols):
            t.rows[0].cells[j].text = str(c)
        for _, row in df.iterrows():
            cells = t.add_row().cells
            vals = ([str(row.name)] if index else []) + list(row.values)
            for j, v in enumerate(vals):
                cells[j].text = ('' if pd.isna(v) else
                                 (f'{v:.4f}' if isinstance(v, (float, np.floating)) else str(v)))

    doc.add_heading('No-validation rolling backtest: data summary and results report', 0)
    P('Objective: test whether the initial submission (2026-05-09, single window, no validation set) is reproducible, '
      'and whether its "9-month train + 2-month test" split is stable over an extended time series.')
    P('Subject: deep reinforcement learning (A2C / DDPG / PPO / TD3 / SAC) asset-allocation backtest on an HS300 ETF '
      'option portfolio.')

    H('1. Experiment design', 1)
    for s in ['Uses exactly the same window size as the original experiment (9 months train + 2 months test, no '
              'validation set), slid over the extended data (2022-01-01 ~ 2023-12-31) with a 2-month step, for 7 '
              'windows in total.',
              'Each window is trained independently with 3 random seeds (42 / 123 / 456) to avoid results that are '
              'merely an artefact of a single seed.',
              'main has train/test boundaries identical to the main experiment (train 2022-07~2023-03, '
              'test 2023-04~2023-05) and serves as the reproducibility anchor.',
              'Benchmarks: equal-weight option portfolio and ETF buy-and-hold.',
              'Metrics: TVM (final/initial NAV), maximum drawdown MDD, annualised Sharpe ratio (252 days, risk-free '
              'rate 0).']:
        doc.add_paragraph(s, style='List Bullet')
    P('Table 1  Experiment configuration comparison (original single-window vs this rolling run)', bold=True)
    TBL(build_config_sheet())
    P('Note: this run keeps the current 67-dimensional state space (9 indicators), which differs from the 37 '
      'dimensions (4 indicators) of the main experiment; the PPO training steps have been restored to the '
      'original 200,000.')

    H('2. Per-window results', 1)
    P('Table 2  TVM by window (mean across 3 seeds)', bold=True)
    TBL(build_window_pivot(wm, 'tvm').round(3))
    PIC(figs['tvm_window'])
    P('Table 3  Sharpe by window (mean across 3 seeds)', bold=True)
    TBL(build_window_pivot(wm, 'sharpe').round(2))
    PIC(figs['heat'])

    H('3. Reproducibility check: main (= main experiment)', 1)
    P('Table 4  main TVM per algorithm across 3 seeds vs the original record', bold=True)
    TBL(main_repro.round(4))
    PIC(figs['main'])
    for s in ['ETF benchmark: all three seeds give 0.93186, digit-by-digit identical to the original record '
              '0.931863 -> the data pipeline, trading calendar and metric definitions are fully aligned.',
              'PPO: TVM across three seeds = 1.803 / 2.502 / 1.130 (Sharpe 5.62 / 5.53 / 3.80), the only '
              'reinforcement-learning algorithm with positive returns on all 3 seeds; consistent with the initial '
              'memory that "PPO performed well on main".',
              'Other algorithms: among the three seeds of a2c/ddpg/td3/sac, at least one seed blows up.',
              'However, compared with the original record the values of the 5 DRL algorithms differ greatly (e.g. '
              'DDPG recorded 4.279, seed42 here ~0) -> the original single-run, seedless experimental record is not '
              'reproducible.']:
        doc.add_paragraph(s, style='List Bullet')

    H('4. Pooled full-period out-of-sample results (274 trading days)', 1)
    P('Table 5  Pooled full-period metrics (3-seed mean ± std)', bold=True)
    TBL(show.round(4))
    PIC(figs['pooled'])
    PIC(figs['nav'])
    for s in ['All 5 reinforcement-learning algorithms lose heavily on all seeds, with TVM <= 0.10, far below the '
              'ETF benchmark 0.920; PPO averages 0.0001 (close to zero).',
              'Benchmarks: equal-weight option portfolio 0.0205, ETF buy-and-hold 0.9203 (identical across the '
              'three seeds).']:
        doc.add_paragraph(s, style='List Bullet')

    H('5. Statistical tests', 1)
    P('Table 6  Paired t-tests of PPO against each comparator (per seed, daily returns)', bold=True)
    TBL(ttest.round(5))
    doc.add_paragraph('Across the 3 seeds, the paired t-tests of PPO against each comparator all have p > 0.10, '
                      'i.e. no statistical significance.',
                      style='List Bullet')

    H('6. Conclusions', 1)
    for s in ['Reproducibility: checked against the main anchor, the ETF benchmark is fully reproduced; but the '
              'results of the 5 DRL algorithms are not reproducible - the original code used no random seed and '
              'trained only once, so its recorded values are highly accidental.',
              'Stability: after rolling the same window size 7 times x 3 seeds over the extended period, algorithm '
              'performance fluctuates violently across windows, most windows blow up out of sample, no algorithm '
              'beats the ETF over the pooled period, and the results are not statistically significant -> the '
              '"9-month train + 2-month test, no validation set" split is not robust.',
              'On main: PPO is profitable on all three main seeds, showing that the algorithm works in this particular '
              'window, but this is a window-specific favourable case and cannot be extrapolated into a general '
              'advantage.']:
        doc.add_paragraph(s, style='List Bullet')

    H('7. Limitations and notes', 1)
    for s in ['This rolling experiment keeps the current 67-dimensional state space (including 5 new option '
              'features), which differs from the 37 dimensions of the main experiment; the "reproduction '
              'comparison" is therefore strictly a test under "same data, same window, same training steps, but a '
              'different observation space and no seeds".',
              'No validation set means the final model at the end of training is used directly; evidence shows that '
              'this setting easily degenerates into trend-following / gambling-style strategies, which is one of the '
              'main reasons for the widespread out-of-sample blow-ups in this round.',
              'This report is a diagnostic analysis, intended to document the reproducibility and split-robustness '
              'issues of the original experiment.']:
        doc.add_paragraph(s, style='List Bullet')

    H('Appendix: list of deliverables', 1)
    for s in ['no_validation_rolling_data_summary.xlsx: 7 worksheets.',
              'figures/: fig1 pooled NAV, fig2 TVM by window, fig3 main three seeds, fig4 window heatmaps, fig5 pooled '
              'metrics.',
              'source_csv/: copies of the original summary files from the results directory.']:
        doc.add_paragraph(s, style='List Bullet')

    doc.save(path_docx)


def _md_table(df, index=True):
    df = df.copy()
    if index:
        df = df.reset_index()
    cols = [str(c) for c in df.columns]
    out = ['| ' + ' | '.join(cols) + ' |',
           '| ' + ' | '.join(['---'] * len(cols)) + ' |']
    for _, row in df.iterrows():
        vals = []
        for v in row.values:
            if isinstance(v, (float, np.floating)):
                vals.append('' if pd.isna(v) else f'{v:.4f}')
            else:
                vals.append(str(v))
        out.append('| ' + ' | '.join(vals) + ' |')
    return '\n'.join(out)


# ---------------- main ----------------
def main():
    os.makedirs(FIG, exist_ok=True)

    wm = pd.read_csv(os.path.join(SRC, 'windows_metrics.csv'))
    pooled_summary = pd.read_csv(os.path.join(SRC, 'pooled_summary.csv'))
    seed_pooled = pd.read_csv(os.path.join(SRC, 'seed_pooled_metrics.csv'))
    ttest = pd.read_csv(os.path.join(SRC, 'paired_t_test_wf.csv'))

    pooled = pooled_by_seed()

    figs = {
        'nav': fig_nav(pooled, wm),
        'tvm_window': fig_tvm_by_window(wm),
        'main': fig_main_by_seed(wm),
        'heat': fig_heatmaps(wm),
        'pooled': fig_pooled_bars(pooled_summary),
    }

    main_repro = build_main_repro(wm)
    xlsx = os.path.join(OUT, 'no_validation_rolling_data_summary.xlsx')
    write_excel(build_config_sheet(), wm, pooled_summary, seed_pooled, ttest, main_repro, xlsx)

    # copies of raw summary csvs for transparency
    rawdir = os.path.join(OUT, 'source_csv')
    os.makedirs(rawdir, exist_ok=True)
    for f in ['windows_metrics.csv', 'seed_pooled_metrics.csv', 'pooled_summary.csv',
              'paired_t_test_wf.csv', 'pooled_metrics.csv', 'summary.txt']:
        src = os.path.join(SRC, f)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(rawdir, f))

    build_report(figs,
                 (wm, pooled_summary, seed_pooled, ttest, main_repro, pooled),
                 os.path.join(OUT, 'no_validation_rolling_results_report.docx'),
                 os.path.join(OUT, 'no_validation_rolling_results_report.md'))

    print('[Done] Output ->', os.path.abspath(OUT))
    for root, _, files in os.walk(OUT):
        for f in sorted(files):
            print('  ', os.path.relpath(os.path.join(root, f), OUT))


if __name__ == '__main__':
    main()
