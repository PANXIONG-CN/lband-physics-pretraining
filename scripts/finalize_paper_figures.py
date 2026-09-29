"""Format frozen evidence for manuscript review. No fitting or model selection."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', type=Path, default=Path('D:/research-pilots'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    out = args.output
    if out.exists():
        raise FileExistsError('Use a new figure directory')
    out.mkdir(parents=True)
    base = args.project_root / 'outputs/scattering/rough_ground'
    closeout = base / 'paper_submission_closeout_20260911_v1'
    audit_path = closeout / 'paper_audit_supplement_v1/summary.json'
    response_path = closeout / 'response_learning_v1/response_summary.csv'
    results = json.loads(audit_path.read_text(encoding='utf-8'))['results']
    response = pd.read_csv(response_path)
    x = np.array([r['actual_fraction'] * 100 for r in results])
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    labels = {'scratch': 'No physics pretraining', 'spm_only': 'SPM', 'spm_to_i2em': 'SPM to I2EM', 'two_mean': 'Two-channel means', 'common_offset_source_mean': 'Common offset only'}
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.4), layout='constrained')
    for method, label in labels.items():
        axes[0].plot(x, [r['rmse_db'][method] for r in results], marker='o', label=label)
    axes[0].set(title='(a) Frozen SMEX02 field-block splits', ylabel='Mean-channel RMSE (dB)', xlabel='Adaptation fields (%)')
    axes[0].legend(fontsize=8, frameon=False)
    contrast = [r['contrasts']['spm_only_minus_two_mean'] for r in results]
    y = np.array([c['delta_db'] for c in contrast])
    intervals = np.array([c['conditional_field_reweighting_interval'] for c in contrast])
    axes[1].errorbar(x, y, yerr=np.stack([y-intervals[:, 0], intervals[:, 1]-y]), fmt='o', capsize=5, color='#336699')
    axes[1].axhline(0, color='black', linewidth=1)
    axes[1].set(title='(b) SPM minus two-channel means', ylabel='RMSE difference (dB); negative favors SPM', xlabel='Adaptation fields (%)')
    for ax in axes:
        ax.set_xticks(x, ['6.67\n(2 fields)', '10\n(3 fields)', '20\n(6 fields)'])
        ax.grid(alpha=.18)
    fig.savefig(out / 'Fig02_calibration_and_contrasts.png', dpi=240)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.4), layout='constrained')
    for method, label in [('scratch', 'No physics pretraining'), ('spm_only', 'SPM'), ('spm_to_i2em', 'SPM to I2EM'), ('risk_spm_to_i2em', 'Source-risk shrinkage'), ('two_mean', 'Two-channel means')]:
        part = response[(response.method == method) & (response.component == 'differential')].sort_values('fraction')
        assert len(part) == 3
        axes[0].plot(x, part.centered_skill, marker='o', label=label)
        axes[1].plot(x, part.variance_ratio, marker='o', label=label)
    axes[0].set(title='(a) Differential response skill', ylabel='Centered skill relative to a constant')
    axes[0].axhline(0, color='black', linewidth=1)
    axes[1].set(title='(b) Differential response variance', ylabel='Predicted / observed variance')
    axes[1].legend(fontsize=8, frameon=False)
    for ax in axes:
        ax.set_xlabel('Adaptation fields (%)')
        ax.set_xticks(x, ['6.67', '10', '20'])
        ax.grid(alpha=.18)
    fig.savefig(out / 'Fig03_centered_response.png', dpi=240)
    plt.close(fig)
    copies = {
        'Fig01_synthetic_teacher_test.png': base / 'multifidelity_pretraining_20260910_v1/evaluation/02_multifidelity_test_rmse.png',
        'Fig04_campaign_diagnostics.png': closeout / 'paper_audit_supplement_v1/campaign_diagnostics.png',
        'FigS01_spm_angle_loss_scan.png': closeout / 'response_learning_v1/spm_angle_response.png',
        'FigS02_all_centered_controls.png': closeout / 'response_learning_v1/centered_skill.png',
    }
    for name, source in copies.items():
        shutil.copy2(source, out / name)
    paths = [audit_path, response_path, *copies.values()]
    manifest = {'role': 'Formatting only; posterior diagnostic figures are explicitly labeled in captions', 'sources': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}, 'figures': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(out.glob('*.png'))}}
    (out / 'figure_provenance.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print('Prepared 4 main figures and 2 supplementary figures; all existing evidence preserved.')


if __name__ == '__main__':
    main()
