"""Read-only replay of frozen paper data processing; no training or selection.

Output is a new versioned directory. Existing output directories are refused.
The replay checks numerical reproducibility, not physical footprint equivalence.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def compare(actual, replay, columns):
    keys = ['acquisition_date', 'field_id']
    left, right = actual.copy(), replay.copy()
    for frame in (left, right):
        frame['acquisition_date'] = pd.to_datetime(frame['acquisition_date']).dt.strftime('%Y-%m-%d')
        frame['field_id'] = frame['field_id'].astype(str)
        assert not frame.duplicated(keys).any(), 'Duplicate field-day'
    merged = left.merge(right[keys + columns], on=keys, how='left', suffixes=('_saved', '_replay'), validate='one_to_one', indicator=True)
    assert merged['_merge'].eq('both').all(), 'Missing replay field-day'
    errors = {}
    for column in columns:
        a = merged[column + '_saved'].to_numpy(float)
        b = merged[column + '_replay'].to_numpy(float)
        assert np.isfinite(a).all() and np.isfinite(b).all()
        errors[column] = float(np.max(np.abs(a - b)))
        assert np.allclose(a, b, rtol=1e-10, atol=1e-9), (column, errors[column])
    return {'checked_rows': len(merged), 'max_absolute_errors': errors, 'passed': True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', type=Path, default=Path('D:/research-pilots'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = args.project_root.resolve()
    if args.output.exists():
        raise FileExistsError('Use a new audit output directory')
    sys.path[:0] = [str(root / 'src'), str(root / 'scripts')]
    import collocate_smapvex_ground as source_pipeline
    import prepare_smex02_external as target_pipeline
    from research_pilots.scattering.data import read_smapvex as s12
    from research_pilots.scattering.data import read_smex02 as s02
    from research_pilots.scattering.surfaces.dielectric import topp_real_permittivity

    base = root / 'outputs/scattering/rough_ground'
    source_path = base / 'portable_source_v1/smapvex12_portable_source.csv'
    target_path = base / 'smex02_external_final_20260911_v1/smex02_field_day_model_ready.csv'
    prediction_path = base / 'smex02_preregistered_few_shot_20260911_v1/heldout_predictions.csv'
    tracked = [source_path, target_path, prediction_path,
               root / 'src/research_pilots/scattering/data/read_smapvex.py',
               root / 'src/research_pilots/scattering/data/read_smex02.py',
               root / 'scripts/collocate_smapvex_ground.py',
               root / 'scripts/prepare_smex02_external.py']
    before = {str(p): sha256(p) for p in tracked}
    source, target = pd.read_csv(source_path), pd.read_csv(target_path)
    checks = {}
    assert s12.PALS_COLUMNS[5:7] == ['sigma0_vv_db', 'sigma0_hh_db']
    assert s02.PALS_RENAME['L_HH'] == 'sigma0_hh_db'
    assert s02.PALS_RENAME['L_VV'] == 'sigma0_vv_db'
    expected = 10 * np.log10((0.1 + 0.01) / 2)
    assert abs(s02.linear_power_mean_db(pd.Series([-10., -20.])) - expected) < 1e-12
    checks['unequal_power_unit_test'] = {'result_db': float(expected), 'arithmetic_db_mean': -15., 'passed': True}
    checks['polarization_mapping'] = {'source_column6': 'VV', 'source_column7': 'HH', 'target_L_HH': 'HH', 'target_L_VV': 'VV', 'passed': True}

    sf = s12.discover_smapvex_files(root / 'data/scattering/raw/SMAPVEX12')
    soil12 = source_pipeline.add_field_id(s12.read_soil_moisture(sf['soil_moisture_data'], sf['soil_moisture_coordinates']))
    geometry12 = source_pipeline.build_field_geometry(soil12)
    soil_day12 = source_pipeline.build_soil_field_day(soil12)
    pals12, _ = source_pipeline.collocate_pals(sf['pals_files'], geometry12, soil_day12, 500., -40., 0., 'zero')
    checks['source_raw_replay'] = compare(source, pals12, ['sigma0_hh_db', 'sigma0_vv_db', 'pals_sample_count'])
    checks['source_soil_replay'] = compare(source, soil_day12, ['soil_moisture_m3_m3', 'soil_real_dielectric'])
    checks['source_saved_linear_mean'] = {p: float(np.max(np.abs(source[p] - 10 * np.log10(source[p + '_linear_mean'])))) for p in ['sigma0_hh_db', 'sigma0_vv_db']}
    assert max(checks['source_saved_linear_mean'].values()) < 1e-9

    tf = s02.discover_files(root / 'data/scattering/raw/SMEX02')
    soil02 = s02.read_soil_moisture_summary(tf['soil_moisture_summary'])
    soil_day02 = target_pipeline.aggregate_soil(soil02)
    nodes02 = target_pipeline.sampling_nodes(soil02)
    dates = set(pd.to_datetime(soil_day02['acquisition_date']))
    nearest02, inventory02 = target_pipeline.match_pals_to_nodes(tf['pals_files'], nodes02, dates, 500.)
    pals02 = target_pipeline.aggregate_pals(nearest02, 500.)
    checks['target_raw_replay'] = compare(target, pals02, ['sigma0_hh_db', 'sigma0_vv_db', 'pals_sample_count', 'incidence_angle_deg'])
    checks['target_soil_replay'] = compare(target, soil_day02, ['soil_moisture_m3_m3'])
    eps = topp_real_permittivity(target['soil_moisture_m3_m3'].to_numpy(float))
    checks['target_dielectric_topp_max_error'] = float(np.max(np.abs(eps - target['soil_real_dielectric'])))
    assert checks['target_dielectric_topp_max_error'] < 1e-9
    checks['source_moisture_fraction_range'] = [float(source.soil_moisture_m3_m3.min()), float(source.soil_moisture_m3_m3.max())]
    checks['target_moisture_fraction_range'] = [float(target.soil_moisture_m3_m3.min()), float(target.soil_moisture_m3_m3.max())]
    checks['july06_retained_field_days'] = int(target.acquisition_date.eq('2002-07-06').sum())
    checks['target_files_inventoried'] = len(inventory02)
    checks['target_rows_read'] = int(inventory02.raw_rows.sum())
    after = {str(p): sha256(p) for p in tracked}
    assert before == after, 'A frozen input or audited module changed'
    report = {
        'status': 'NUMERICAL_REPLAY_PASSED_WITH_SEMANTIC_LIMITATIONS',
        'scope': 'Frozen-table reproducibility, not equal-footprint validation or new model evidence',
        'checks': checks, 'input_sha256': before,
        'frozen_inputs_unchanged': True,
        'source_cohort': {'rows': len(source), 'fields': int(source.field_id.nunique()), 'dates': int(source.acquisition_date.nunique())},
        'target_cohort': {'rows': len(target), 'fields': int(target.field_id.nunique()), 'dates': int(target.acquisition_date.nunique())},
        'semantic_findings': [
            'Source matches daily eligible field centroids; target matches all available sampling nodes, then joins by date.',
            'Both use 500 m thresholds but not identical support operators or antenna weighting.',
            'Source QC uses all four polarizations in [-40,0] dB and heading flag zero; target thresholds HH/VV in [-50,10] dB and angle/geolocation bounds.',
            'Source dielectric is field-aggregated probe measurement; target dielectric is a deterministic moisture polynomial.',
            'Source roughness is PALS-direction; target pools grid and slope profiles. Physical equivalence is not established.',
            'SMEX02 polar_angle is polarization rotation, but the legacy reader labels it antenna_roll_deg; this field is not selected by the matching or training feature pipeline.',
            'Legacy source summary note mentions 600 m although recorded parameters and replay use 500 m.',
            'Daily temporal association is not synchronous footprint matching; flight remarks are not an exhaustive QA mask.'
        ]
    }
    args.output.mkdir(parents=True)
    (args.output / 'data_contract_audit.json').write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    print(json.dumps({'status': report['status'], 'checks': checks}, indent=2))


if __name__ == '__main__':
    main()
