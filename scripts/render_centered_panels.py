"""Render corrected centered-response panels from frozen summaries (no training).

Each panel is a separate figure. Values below 1e-12 are rounded to zero for
visualization only; frozen input data are not changed.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle', type=Path, default=root / 'reproducibility')
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(a.bundle/'results/diagnostics/response_summary.csv')
    methods = [('scratch','Scratch'), ('spm_only','SPM'),
               ('spm_to_i2em',r'SPM $\to$ I$^2$EM'),
               ('risk_spm_to_i2em','Risk-shrunk'), ('two_mean','Two-channel mean')]
    specs = [('common','centered_skill','Common response: centered skill','figS02a_common_skill'),
             ('common','variance_ratio','Common response: variance ratio','figS02b_common_variance'),
             ('differential','centered_skill','Differential response: centered skill','figS02c_differential_skill'),
             ('differential','variance_ratio','Differential response: variance ratio','figS02d_differential_variance')]
    for component,metric,ylabel,name in specs:
        fig, ax = plt.subplots(figsize=(5.5,3.8))
        for i,(method,label) in enumerate(methods):
            sub = frame[(frame.component==component)&(frame.method==method)].sort_values('fraction')
            if len(sub)!=3:
                raise ValueError(f'Expected three adaptation sizes for {component}/{method}')
            y = sub[metric].to_numpy(float)
            y[np.abs(y)<1e-12]=0.
            ax.plot([2,3,6],y,marker=['o','s','^','D','x'][i],
                    linestyle=['-','--','-.',':','-'][i],label=label,linewidth=1.4,markersize=4)
        ax.set_xticks([2,3,6]); ax.set_xlabel('Planned adaptation fields')
        ax.set_ylabel(ylabel); ax.grid(True,alpha=.2)
        if component=='common': ax.set_ylim(-.01,.01)
        elif metric=='variance_ratio': ax.set_ylim(-.005,.115)
        ax.legend(fontsize=8,frameon=False,loc='best',ncol=2)
        fig.tight_layout()
        fig.savefig(a.output/f'{name}.pdf',bbox_inches='tight')
        fig.savefig(a.output/f'{name}.png',dpi=300,bbox_inches='tight')
        plt.close(fig)
    print('Rendered four panels; frozen inputs unchanged.')

if __name__=='__main__': main()
