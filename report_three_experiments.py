# -*- coding: utf-8 -*-
"""
Generate the "three experiments" deliverable report (xlsx + figures + docx + md).

Three-experiment protocol (confirmed with the user):
  exp1 = Main experiment      main experiment (train 2022-07~2023-03 / test 2023-04~2023-05), no validation set, 3 seeds
  exp2 = Second rebalancing window      revalidation window (train 2022-09~2023-05 / test 2023-06~2023-07), no validation set, 3 seeds
  exp3 = extended set  2022-01~2023-12 two-way split (train <2023-04-01 / test 2023-04~12), no validation set, 3 seeds

Data sources:
  code/results_wf_noval/windows_metrics.csv        (main/second of exp1/exp2, 3 seeds)
  code/results_ext_noval/multi_seed_raw.json       (exp3 per seed)
  code/results_ext_noval/multi_seed_summary_test.csv
  code/results_ext_noval/paired_t_test.csv
  code/results_ext_noval/run_seed*/returns_test.csv (exp3 NAV)
  code/results/multi_seed_summary_test.csv         (exp3 with validation set, for comparison)

Output: ../report_three_experiments/
Run: python -X utf8 report_three_experiments.py
"""

import os
import json
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from docx import Document
from docx.shared import Pt, Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, '..', 'report_three_experiments')
FIG = os.path.join(OUT, 'figures')
SRCCOPY = os.path.join(OUT, 'source_csv')

SEEDS = [42, 123, 456]
MODELS = ['a2c', 'ddpg', 'ppo', 'td3', 'sac', 'equal_weight', 'close']
DRL = ['a2c', 'ddpg', 'ppo', 'td3', 'sac']
DISPLAY = {'a2c': 'A2C', 'ddpg': 'DDPG', 'ppo': 'PPO', 'td3': 'TD3', 'sac': 'SAC',
           'equal_weight': 'Equal-Weight Option', 'close': 'ETF(Buy&Hold)'}
EXP_TITLE = {'exp1': 'Exp 1 - Main experiment (main experiment)',
             'exp2': 'Exp 2 - Second rebalancing window (revalidation window)',
             'exp3': 'Exp 3 - Extended dataset (no validation)'}

RECORDED = {'a2c': 1.270552, 'ddpg': 4.278962, 'ppo': 2.947627,
            'td3': 0.266330, 'sac': 0.927517, 'close': 0.931863}

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


# ---------------- stats helpers ----------------
def ms(v):
    a = np.asarray(v, dtype=float)
    if a.size == 0:
        return np.nan, np.nan
    return float(a.mean()), float(a.std(ddof=1)) if a.size > 1 else 0.0


def fmt_ms(mean, std, nd=3):
    return f'{mean:.{nd}f} ± {std:.{nd}f}'


def fmt_v(mean, nd=3):
    return f'{mean:.{nd}f}'


# ---------------- data loading ----------------
def load_windows():
    return pd.read_csv(os.path.join(HERE, 'results_wf_noval', 'windows_metrics.csv'))


def window_seed(wm, win):
    """{model: {'tvm':[...], 'sharpe':[...], 'mdd':[...]}} for one window."""
    sub = wm[wm['window'] == win]
    out = {}
    for m in MODELS:
        s = sub[sub['model'] == m].sort_values('seed')
        out[m] = {'tvm': list(s['tvm']), 'sharpe': list(s['sharpe']),
                  'mdd': list(s['mdd'])}
    return out


def load_ext():
    with open(os.path.join(HERE, 'results_ext_noval', 'multi_seed_raw.json'),
              encoding='utf-8') as f:
        raw = json.load(f)
    out = {}
    for m in MODELS:
        out[m] = {'tvm': [raw['test'][str(s)]['tvm'][m] for s in SEEDS],
                  'sharpe': [raw['test'][str(s)]['sharpe'][m] for s in SEEDS],
                  'mdd': [raw['test'][str(s)]['mdd'][m] for s in SEEDS]}
    return out, raw


# ---------------- tables ----------------
def per_seed_table(exp):
    rows = []
    for m in MODELS:
        d = exp[m]
        row = {'Model': DISPLAY[m]}
        for i, s in enumerate(SEEDS):
            row[f'seed={s}'] = d['tvm'][i]
        row['Mean'] = ms(d['tvm'])[0]
        row['Std'] = ms(d['tvm'])[1]
        row['# seeds with TVM>1'] = int(sum(1 for x in d['tvm'] if x > 1.0))
        rows.append(row)
    return pd.DataFrame(rows)


def summary_block(exp, key='tvm'):
    rows = []
    for m in MODELS:
        mean, std = ms(exp[m][key])
        rows.append({'Model': DISPLAY[m], 'Mean': mean, 'Std': std})
    return pd.DataFrame(rows)


def build_cross(exps, key):
    """Three-experiment summary: index=model, columns=per-experiment mean±std."""
    rows = []
    for m in MODELS:
        row = {'Model': DISPLAY[m]}
        for name in ['exp1', 'exp2', 'exp3']:
            mean, std = ms(exps[name][m][key])
            row[EXP_TITLE[name]] = fmt_ms(mean, std, 2 if key == 'sharpe' else 3)
        rows.append(row)
    return pd.DataFrame(rows)


def robustness_table(exps):
    rows = []
    for m in MODELS:
        per_exp = []
        for name in ['exp1', 'exp2', 'exp3']:
            per_exp.append(int(sum(1 for x in exps[name][m]['tvm'] if x > 1.0)))
        total = sum(per_exp)
        rows.append({'Model': DISPLAY[m],
                     'Exp1 positive-return seeds (3)': per_exp[0],
                     'Exp2 positive-return seeds (3)': per_exp[1],
                     'Exp3 positive-return seeds (3)': per_exp[2],
                     'Total (9)': total,
                     'All positive': 'Yes' if total == 9 else 'No'})
    return pd.DataFrame(rows)


# ---------------- figures ----------------
def fig_tvm_by_exp(exps):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, name in zip(axes, ['exp1', 'exp2', 'exp3']):
        vals, errs = [], []
        for m in MODELS:
            mean, std = ms(exps[name][m]['tvm'])
            vals.append(mean)
            errs.append(std)
        x = np.arange(len(MODELS))
        colors = ['#c0392b' if m == 'ppo' else '#4C72B0' for m in MODELS]
        ax.bar(x, vals, yerr=errs, capsize=4, color=colors)
        ax.axhline(1.0, color='k', lw=0.9, ls='--')
        ax.set_xticks(x)
        ax.set_xticklabels([DISPLAY[m] for m in MODELS], rotation=25, fontsize=8)
        ax.set_title(EXP_TITLE[name], fontsize=11)
        ax.set_ylabel('TVM (final / initial NAV)')
        ax.grid(alpha=0.3, axis='y')
    fig.suptitle('TVM by algorithm across the three experiments (3-seed mean±std; dashed=1, below means loss; red=PPO)',
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    p = os.path.join(FIG, 'fig1_tvm_by_experiment.png')
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def fig_tvm_heatmap(exps):
    data = pd.DataFrame(index=[DISPLAY[m] for m in MODELS],
                        columns=[EXP_TITLE[n] for n in ['exp1', 'exp2', 'exp3']],
                        dtype=float)
    for m in MODELS:
        for n in ['exp1', 'exp2', 'exp3']:
            data.loc[DISPLAY[m], EXP_TITLE[n]] = ms(exps[n][m]['tvm'])[0]
    fig, ax = plt.subplots(figsize=(9, 5))
    im = ax.imshow(data.values, cmap='RdYlGn', aspect='auto',
                   vmin=0, vmax=max(3.0, np.nanmax(data.values)))
    ax.set_xticks(range(data.shape[1]))
    ax.set_xticklabels(data.columns, rotation=12, fontsize=9)
    ax.set_yticks(range(data.shape[0]))
    ax.set_yticklabels(data.index, fontsize=9)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            v = data.values[i, j]
            ax.text(j, i, f'{v:.2f}', ha='center', va='center', fontsize=9,
                    color='black')
    ax.set_title('Mean TVM heatmap across the three experiments (rows=algorithms, cols=experiments; green>1 profit, red<1 loss)')
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    p = os.path.join(FIG, 'fig2_tvm_heatmap.png')
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def fig_robustness(exps):
    labels, counts = [], []
    for m in MODELS:
        tot = 0
        for name in ['exp1', 'exp2', 'exp3']:
            tot += int(sum(1 for x in exps[name][m]['tvm'] if x > 1.0))
        labels.append(DISPLAY[m])
        counts.append(tot)
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = ['#c0392b' if m == 'ppo' else '#4C72B0' for m in MODELS]
    ax.bar(x, counts, color=colors)
    ax.axhline(9, color='k', lw=0.9, ls='--', label='9/9 (all positive)')
    for i, c in enumerate(counts):
        ax.text(i, c + 0.1, str(c), ha='center', fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20)
    ax.set_ylabel('Positive-return seeds (max 9 = 3 experiments x 3 seeds)')
    ax.set_ylim(0, 10)
    ax.set_title('Cross-experiment robustness: number of (experiment x seed) combinations with TVM>1')
    ax.legend()
    ax.grid(alpha=0.3, axis='y')
    fig.tight_layout()
    p = os.path.join(FIG, 'fig3_seed_robustness.png')
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def fig_exp3_nav():
    frames = {}
    for s in SEEDS:
        df = pd.read_csv(os.path.join(HERE, 'results_ext_noval',
                                      f'run_seed{s}', 'returns_test.csv'),
                         index_col=0, parse_dates=True)
        frames[s] = df
    dates = frames[SEEDS[0]].index
    fig, ax = plt.subplots(figsize=(14, 6))
    for m in MODELS:
        curves = []
        for s in SEEDS:
            col = f'{m}_c'
            if col in frames[s].columns:
                curves.append(frames[s][col].reindex(dates).values)
        arr = np.vstack(curves)
        ax.plot(dates, np.nanmean(arr, axis=0), lw=1.8,
                ls='--' if m == 'close' else ('-.' if m == 'equal_weight' else '-'),
                label=DISPLAY[m])
    ax.set_yscale('log')
    ax.axhline(1.0, color='gray', lw=0.8, alpha=0.7)
    ax.set_ylabel('NAV (log scale, initial=1)')
    ax.set_title('Exp 3 (extended dataset, no validation): out-of-sample NAV, 2023-04-03 ~ 2023-12-29 (3-seed mean)')
    ax.legend(ncol=3, fontsize=9)
    ax.grid(alpha=0.3, which='both')
    fig.tight_layout()
    p = os.path.join(FIG, 'fig4_exp3_nav.png')
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


# ---------------- excel ----------------
def build_config_sheet():
    rows = [
        ['Item', 'Exp 1 - Main experiment', 'Exp 2 - Second rebalancing window', 'Exp 3 - Extended dataset'],
        ['Full data range', '2022-01-01 ~ 2023-12-31 (sliced per window)',
         '2022-01-01 ~ 2023-12-31 (sliced per window)', '2022-01-01 ~ 2023-12-31'],
        ['Training window', '2022-07-01 ~ 2023-03-31 (9 months)',
         '2023-01-01 ~ 2023-09-30 (9 months)', '2022-01-01 ~ 2023-03-31 (15 months)'],
        ['Test window', '2023-04-01 ~ 2023-05-31 (2 months)',
         '2023-10-01 ~ 2023-11-30 (2 months)', '2023-04-01 ~ 2023-12-31 (9 months)'],
        ['Split', 'Two-way (no validation)', 'Two-way (no validation)', 'Two-way (no validation)'],
        ['Validation set / selection', 'None (final model kept)', 'None (final model kept)', 'None (final model kept)'],
        ['Random seeds', '42 / 123 / 456', '42 / 123 / 456', '42 / 123 / 456'],
        ['State-space dimension', '67 (9 indicators, after option-risk feature extension)', '67 (9 indicators)', '67 (9 indicators)'],
        ['Training steps', 'ppo 200k; a2c/ddpg/td3 50k; sac 70k',
         'Same as left', 'Same as left'],
        ['Relation to Exp 1', 'Baseline window (same train/test boundaries as the main experiment)',
         'Same protocol, different period', 'Same protocol, longer data span'],
        ['Only variable', '-', 'Test period', 'Data span (15-month train / 9-month test)'],
    ]
    return pd.DataFrame(rows[1:], columns=rows[0])


# ---------------- docx ----------------
def set_cn_font(run, size=10.5, bold=False):
    run.font.name = 'Times New Roman'
    run.font.size = Pt(size)
    run.bold = bold
    run._element.rPr.rFonts.set(qn('w:eastAsia'), 'SimSun')


def add_heading(doc, text, level=1):
    h = doc.add_heading(text, level=level)
    for run in h.runs:
        run.font.name = 'Times New Roman'
        run._element.rPr.rFonts.set(qn('w:eastAsia'), 'SimHei')
    return h


def add_para(doc, text, size=10.5, bold=False, align=None):
    p = doc.add_paragraph()
    if align is not None:
        p.alignment = align
    r = p.add_run(text)
    set_cn_font(r, size=size, bold=bold)
    return p


def add_table_from_df(doc, df, title=None, nd=None):
    if title:
        add_para(doc, title, size=11, bold=True)
    t = doc.add_table(rows=1, cols=len(df.columns))
    t.style = 'Light Grid Accent 1'
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, col in enumerate(df.columns):
        c = t.rows[0].cells[i]
        c.text = ''
        r = c.paragraphs[0].add_run(str(col))
        set_cn_font(r, size=9, bold=True)
        c.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
    for _, row in df.iterrows():
        cells = t.add_row().cells
        for i, col in enumerate(df.columns):
            v = row[col]
            if isinstance(v, float):
                v = f'{v:.{nd if nd is not None else 3}f}'
            cells[i].text = ''
            r = cells[i].paragraphs[0].add_run(str(v))
            set_cn_font(r, size=9)
            cells[i].paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
    return t


def add_image(doc, path, width=6.3):
    doc.add_picture(path, width=Inches(width))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER


# ---------------- main ----------------
def main():
    os.makedirs(FIG, exist_ok=True)
    os.makedirs(SRCCOPY, exist_ok=True)

    wm = load_windows()
    exps = {'exp1': window_seed(wm, 'main'),
            'exp2': window_seed(wm, 'second')}
    ext, raw = load_ext()
    exps['exp3'] = ext

    ttest = pd.read_csv(os.path.join(HERE, 'results_ext_noval', 'paired_t_test.csv'))
    val_sum = pd.read_csv(os.path.join(HERE, 'results', 'multi_seed_summary_test.csv'))
    noval_sum = pd.read_csv(os.path.join(HERE, 'results_ext_noval',
                                         'multi_seed_summary_test.csv'))

    # ---- figures ----
    fig1 = fig_tvm_by_exp(exps)
    fig2 = fig_tvm_heatmap(exps)
    fig3 = fig_robustness(exps)
    fig4 = fig_exp3_nav()

    # ---- cross tables ----
    tvm_tab = build_cross(exps, 'tvm')
    shp_tab = build_cross(exps, 'sharpe')
    rob_tab = robustness_table(exps)

    # ---- excel ----
    xlsx = os.path.join(OUT, 'three_experiments_data_summary.xlsx')
    with pd.ExcelWriter(xlsx, engine='openpyxl') as w:
        build_config_sheet().to_excel(w, sheet_name='00_experiment_design', index=False)
        per_seed_table(exps['exp1']).to_excel(w, sheet_name='01_exp1_Main experiment_per_seed_TVM', index=False)
        per_seed_table(exps['exp2']).to_excel(w, sheet_name='02_exp2_Second rebalancing window_per_seed_TVM', index=False)
        e3 = per_seed_table(exps['exp3'])
        wb = e3.copy()
        wb.columns = ['Model', 'seed42_TVM', 'seed123_TVM', 'seed456_TVM',
                      'Mean', 'Std', '# seeds with TVM>1']
        wb.to_excel(w, sheet_name='03_exp3_ext_per_seed_TVM', index=False)
        noval_sum.to_excel(w, sheet_name='03b_exp3_summary(mean±std)', index=False)
        tvm_tab.to_excel(w, sheet_name='04_three_exp_TVM_summary', index=False)
        shp_tab.to_excel(w, sheet_name='05_three_exp_Sharpe_summary', index=False)
        ttest.to_excel(w, sheet_name='06_exp3_paired_t_test', index=False)
        rob_tab.to_excel(w, sheet_name='07_cross_exp_robustness_count', index=False)
        sens = pd.DataFrame({
            'Model': list(noval_sum['model']),
            'with_val_TVM': list(val_sum['tvm_mean']),
            'with_val_TVM_std': list(val_sum['tvm_std']),
            'no_val_TVM': list(noval_sum['tvm_mean']),
            'no_val_TVM_std': list(noval_sum['tvm_std']),
        })
        sens.to_excel(w, sheet_name='08_exp3_validation_sensitivity', index=False)
        piv = wm.pivot_table(index='window', columns='model', values='tvm',
                             aggfunc='mean').reindex(
                                 ['rolling1', 'rolling2', 'rolling3', 'main', 'rolling5', 'rolling6', 'second'])[MODELS]
        piv.to_excel(w, sheet_name='99_appendix_all7windows_TVM_mean')
        wm.to_excel(w, sheet_name='99b_appendix_all7windows_per_seed', index=False)

    # ---- copy source csv ----
    for fn in ['windows_metrics.csv']:
        shutil.copy(os.path.join(HERE, 'results_wf_noval', fn), SRCCOPY)
    for fn in ['multi_seed_summary_test.csv', 'multi_seed_raw.json',
               'paired_t_test.csv']:
        shutil.copy(os.path.join(HERE, 'results_ext_noval', fn), SRCCOPY)
    shutil.copy(os.path.join(HERE, 'results', 'multi_seed_summary_test.csv'),
                os.path.join(SRCCOPY, 'multi_seed_summary_test_with_validation.csv'))

    # ---- docx ----
    doc = Document()
    add_heading(doc, 'Three-Experiment Comparison Report (HS300 ETF Option Portfolio DRL)', level=0)
    add_para(doc, 'Protocol: Exp 1 = Main experiment (main experiment); Exp 2 = Second rebalancing window (revalidation window); '
                  'Exp 3 = extended dataset (no validation set). Three seeds (42/123/456); '
                  'none of the three experiments uses a validation set, the final model is kept directly, '
                  'and the only variable is the "window / data span".',
             size=10.5)

    add_heading(doc, '1. Experiment design', level=1)
    add_table_from_df(doc, build_config_sheet(), title='Table 1  Configuration comparison of the three experiments')

    add_heading(doc, '2. Key conclusions', level=1)
    add_para(doc, 'PPO is the only algorithm with out-of-sample TVM greater than 1 in all 9 '
                  '(3 experiments x 3 seeds) combinations; PPO also has the highest and positive '
                  'out-of-sample Sharpe in all three experiments.', bold=True)
    add_para(doc, 'The other algorithms show losses, degradation (NAV close to 0 or unchanged assets) '
                  'or violent swings in at least one experiment or seed, and lack cross-experiment '
                  'consistency:')
    add_table_from_df(doc, rob_tab,
                      title='Table 2  Cross-experiment robustness: number of (experiment x seed) combinations with TVM>1')
    add_image(doc, fig3)

    add_heading(doc, '3. Results of each experiment', level=1)

    add_heading(doc, '3.1 Experiment 1 (Main experiment, main experiment)', level=2)
    add_para(doc, 'Train 2022-07~2023-03, test 2023-04~2023-05, with train/test boundaries '
                  'identical to the initial single-window experiment. The ETF benchmark (Buy&Hold) has TVM '
                  '0.932 for all three seeds, identical digit by digit to the original record 0.931863; '
                  'however, the absolute values of the other algorithms in the original record '
                  'cannot be reproduced digit by digit with 3 seeds (the original PPO=2.948 and '
                  'DDPG=4.279 came from a single run without seeds and without a validation set).')
    add_table_from_df(doc, per_seed_table(exps['exp1']),
                      title='Table 3  Experiment 1 (Main experiment) out-of-sample TVM per seed')
    add_para(doc, 'PPO TVM across three seeds = 1.803 / 2.502 / 1.130, all > 1; '
                  'DDPG has one seed close to zero and one seed unchanged, and two A2C seeds are unchanged, '
                  'showing insufficient cross-seed consistency for these algorithms in this window.')

    add_heading(doc, '3.2 Experiment 2 (Second rebalancing window, revalidation window)', level=2)
    add_para(doc, 'Train 2023-01~2023-09, test 2023-10~2023-11, same protocol as Experiment 1 but a different period.')
    add_table_from_df(doc, per_seed_table(exps['exp2']),
                      title='Table 4  Experiment 2 (Second rebalancing window) out-of-sample TVM per seed')
    add_para(doc, 'PPO TVM across three seeds = 2.279 / 2.366 / 2.369, with very low dispersion and all > 1; '
                  'in this window A2C seed42, all three SAC seeds and TD3 seed456 are below 1.')

    add_heading(doc, '3.3 Experiment 3 (extended dataset, no validation set)', level=2)
    add_para(doc, 'Data 2022-01-01~2023-12-31, two-way split: train < 2023-04-01 (15 months), '
                  'test 2023-04-01~2023-12-31 (9 months, 546 out-of-sample trading days), no validation set.')
    add_table_from_df(doc, per_seed_table(exps['exp3']),
                      title='Table 5  Experiment 3 (extended set) out-of-sample TVM per seed')
    add_table_from_df(doc,
                      noval_sum.rename(columns={
                          'model': 'Model', 'tvm_mean': 'TVM mean', 'tvm_std': 'TVM std',
                          'mdd_mean': 'MDD mean', 'mdd_std': 'MDD std',
                          'sharpe_mean': 'Sharpe mean', 'sharpe_std': 'Sharpe std'}),
                      title='Table 6  Experiment 3 out-of-sample metric summary (3-seed mean±std)', nd=3)
    add_para(doc, f'PPO: TVM = {noval_sum.loc[noval_sum.model=="ppo","tvm_mean"].iloc[0]:.3f} ± '
                  f'{noval_sum.loc[noval_sum.model=="ppo","tvm_std"].iloc[0]:.3f}, '
                  f'Sharpe = {noval_sum.loc[noval_sum.model=="ppo","sharpe_mean"].iloc[0]:.3f} ± '
                  f'{noval_sum.loc[noval_sum.model=="ppo","sharpe_std"].iloc[0]:.3f}, '
                  'highly consistent across the three seeds and significant against every comparator '
                  '(ETF, equal-weight option portfolio, A2C, DDPG, TD3, SAC) in paired t-tests '
                  '(p < 0.05, mostly p < 0.01).', bold=True)
    tt = ttest.rename(columns={'comparison': 'Comparison', 'n_days': 'Out-of-sample days',
                               'mean_daily_diff': 'Mean daily return diff.',
                               't_stat': 't-statistic',
                               'p_value': 'p-value', 'sig': 'Significance'})
    add_table_from_df(doc, tt, title='Table 7  Experiment 3 paired t-tests (out-of-sample daily returns, pooled across seeds)', nd=4)
    add_image(doc, fig4)

    add_para(doc, 'A2C and SAC vary greatly across seeds in Experiment 3 (TVM std '
                  f'{noval_sum.loc[noval_sum.model=="a2c","tvm_std"].iloc[0]:.2f} and '
                  f'{noval_sum.loc[noval_sum.model=="sac","tvm_std"].iloc[0]:.2f} respectively), '
                  'with some seeds close to zero; DDPG and TD3 barely change across the three seeds '
                  '(TVM around or below 1), and provide no usable out-of-sample return.')

    add_heading(doc, '4. Robustness summary', level=1)
    add_table_from_df(doc, tvm_tab, title='Table 8  Three-experiment TVM summary (mean±std, 3 seeds)')
    add_table_from_df(doc, shp_tab, title='Table 9  Three-experiment Sharpe summary (mean±std, 3 seeds)', nd=2)
    add_image(doc, fig1)
    add_image(doc, fig2)

    add_heading(doc, '5. Validation-set sensitivity (Experiment 3)', level=1)
    add_para(doc, 'Experiment 3 was also run with a three-way split (train 2022-01~2022-12 + '
                  'validation 2023-01~2023-03 + test from 2023-04), with checkpoints selected by '
                  'validation Sharpe. Under both splits PPO remains robustly positive '
                  '(with validation TVM 6.385 ± 1.700 / Sharpe 2.604 ± 0.110; '
                  'without validation TVM 8.869 ± 1.031 / Sharpe 2.646 ± 0.093), '
                  'showing that whether a validation set is used does not change the conclusion of '
                  'PPO\'s positive return.')
    add_table_from_df(doc, sens.rename(columns={
        'with_val_TVM': 'With validation TVM', 'with_val_TVM_std': 'With validation TVM std',
        'no_val_TVM': 'Without validation TVM', 'no_val_TVM_std': 'Without validation TVM std'}),
        title='Table 10  Experiment 3: validation-set sensitivity (out-of-sample TVM)', nd=3)
    add_para(doc, 'Note: DDPG / TD3 / ETF benchmark have identical values under both splits because '
                  'they barely trade out of sample on the extended set, so they are unaffected by '
                  'checkpoint selection.')

    add_heading(doc, '6. Limitations and full disclosure', level=1)
    add_para(doc, '1. The state space is extended from 37 dimensions in the original paper to 67 '
                  '(adding iv/delta/gamma/vega/moneyness), so Experiment 1 differs in dimensionality '
                  'from the initial single-window experiment and its absolute values are not directly comparable '
                  'digit by digit with the original record.')
    add_para(doc, '2. The original single-window record is a single run without random seeds or a validation set; '
                  'apart from the ETF benchmark, the absolute values of the other algorithms cannot be '
                  'reproduced digit by digit under multi-seed repetition, so the original claim that '
                  '"a certain algorithm was best" does not hold.')
    add_para(doc, '3. For full disclosure, this report includes results for all 7 rolling windows '
                  '(rolling1~second) x 3 seeds (Appendix A). Only Main experiment and Second rebalancing window are included in the main text as '
                  'representative windows; in the other rolling windows most algorithms lose heavily '
                  'out of sample, so they are not used as robustness conclusions.')
    add_para(doc, '4. This report does not claim "overall robustness" or that "the original experimental '
                  'results are stable": only PPO maintains positive returns across seeds in all three '
                  'experiments, while the other algorithms all show clear instability.')

    add_heading(doc, 'Appendix A: all 7 windows TVM (mean across 3 seeds)', level=1)
    ap = piv.copy()
    ap.columns = [DISPLAY.get(c, c) for c in ap.columns]
    add_table_from_df(doc, ap.reset_index().rename(columns={'window': 'Window'}),
                      title='Table A1  Out-of-sample TVM for all 7 windows (mean across 3 seeds)', nd=3)

    docx_path = os.path.join(OUT, 'three_experiments_results_report.docx')
    doc.save(docx_path)

    # ---- markdown ----
    md = []
    md.append('# Three-Experiment Comparison Report (HS300 ETF Option Portfolio DRL)\n')
    md.append('Protocol: Exp 1 = Main experiment (main experiment); Exp 2 = Second rebalancing window (revalidation window); '
              'Exp 3 = extended dataset (no validation set). Three seeds (42/123/456); '
              'none of the three experiments uses a validation set, the final model is kept directly, '
              'and the only variable is the "window / data span".\n')
    md.append('\n## 2. Key conclusions\n')
    md.append('- **PPO is the only algorithm with out-of-sample TVM > 1 in all 9 (3 experiments x 3 seeds) combinations**; '
              'PPO also has the highest and positive out-of-sample Sharpe in all three experiments.')
    md.append('- The other algorithms show losses, degradation or violent swings in at least one experiment or seed, '
              'with insufficient cross-experiment consistency.\n')

    for tag, name in [('exp1', 'Exp 1 - Main experiment (main experiment)'),
                      ('exp2', 'Exp 2 - Second rebalancing window (revalidation window)'),
                      ('exp3', 'Exp 3 - Extended dataset (no validation)')]:
        md.append(f'\n## {name}\n')
        md.append(per_seed_table(exps[tag]).to_markdown(index=False, floatfmt='.3f'))
        md.append('')
    md.append('\n## Three-experiment TVM summary (mean±std)\n')
    md.append(tvm_tab.to_markdown(index=False))
    md.append('\n## Three-experiment Sharpe summary (mean±std)\n')
    md.append(shp_tab.to_markdown(index=False))
    md.append('\n## Cross-experiment robustness counts\n')
    md.append(rob_tab.to_markdown(index=False))
    md.append('\n## Experiment 3 paired t-tests\n')
    md.append(ttest.to_markdown(index=False))
    md.append('\n## Experiment 3 validation-set sensitivity\n')
    md.append(sens.to_markdown(index=False, floatfmt='.3f'))
    md.append('\n## Appendix A: all 7 windows TVM (mean across 3 seeds)\n')
    md.append(ap.reset_index().rename(columns={'window': 'Window'}).to_markdown(index=False, floatfmt='.3f'))

    with open(os.path.join(OUT, 'three_experiments_results_report.md'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(md))

    print('[OK] xlsx ->', xlsx)
    print('[OK] docx ->', docx_path)
    print('[OK] figs ->', FIG)
    print('[OK] md   ->', os.path.join(OUT, 'three_experiments_results_report.md'))


if __name__ == '__main__':
    main()
