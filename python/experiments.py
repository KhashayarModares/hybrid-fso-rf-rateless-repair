"""Shared reproducible, resumable experiment IO and paired block workers."""
from concurrent.futures import ProcessPoolExecutor,as_completed
from dataclasses import replace
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import sys
import tempfile
import time
import numpy as np

from channel import trial_inputs
from model import simulate
from statistics_tools import finite_json

ROOT = Path(__file__).resolve().parents[1]
COLUMNS = ('seed','success','completion_slots','observation_s','launched','completed',
           'received','erased','in_flight','bitmap_payload_bits','reports_launched',
           'reports_completed','bitmap_airtime_observed_s','targeted_launched',
           'targeted_received','targeted_without_recovery','fso_estimated_windows',
           'bad_fso_windows','state_launched','state_received','state_productive',
           'feedback_frame_bits','reports_generated','feedback_max_queue_s',
           'feedback_mean_state_age_s','feedback_airtime_allocated_s','rf_forward_silenced_s')


CONTROLLER_FIELDS = {'window','state_threshold','state_repair','feedback_link','feedback_ideal',
                     'feedback_rate_bps','feedback_propagation_s','feedback_processing_s',
                     'feedback_loss_probability','turnaround_s'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sources():
    # Simulation code only: manuscript or figure-style edits must not invalidate samples.
    paths = [p for p in (ROOT/'python').glob('*.py') if p.name != 'plot_results.py']
    return {p.relative_to(ROOT).as_posix():sha(p) for p in sorted(paths)}


def runtime():
    return {'python':sys.version,'platform':platform.platform(),
            'dependencies':{name:importlib.metadata.version(name) for name in ('numpy','scipy','matplotlib')}}


def write_json(path,value):
    path = Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='w',encoding='utf-8',dir=path.parent,
                                     suffix='.tmp',delete=False) as stream:
        json.dump(finite_json(value),stream,indent=2,allow_nan=False)
        temp = Path(stream.name)
    temp.replace(path)


def compatible_manifest(path,value):
    normalized = json.loads(json.dumps(value))
    path = Path(path)
    if path.exists() and json.loads(path.read_text()) != normalized:
        raise ValueError('Different code/configuration/runtime: choose a NEW output directory')
    write_json(path,normalized)


def block_row(r):
    return (r['seed'],float(r['success']),r['receiver_completion_slots'] if r['success'] else np.inf,
            r['observation_end_s'],sum(r['launched']),sum(r['completed']),sum(r['received']),
            sum(r['erased']),sum(r['in_flight']),r['bitmap_payload_bits_launched'],
            r['bitmap_reports_launched'],r['bitmap_reports_completed'],
            r['bitmap_airtime_to_observation_s'],r['targeted_launched'],
            r['targeted_received'],r['received_targeted_without_recovery'],
            r['fso_estimated_windows'],r['bad_fso_windows'],r['state_launched'],
            r['state_received'],r['state_productive'],
            r['feedback_frame_bits_launched'],r['bitmap_reports_generated'],
            r['feedback_max_queue_s'],r['feedback_mean_state_age_s'],r['bitmap_airtime_allocated_s'],
            r['rf_forward_silenced_s'])


def variant_config(cfg, variant):
    """Variant: (key, method, overrides). Overrides change only controller/feedback
    fields, never the channel, so every variant shares the block's channel bank."""
    if len(variant) != 3 or not isinstance(variant[2], dict):
        raise ValueError('Variant must specify key, method and an override dictionary')
    if set(variant[2]) - CONTROLLER_FIELDS:
        raise ValueError('Variants may override only controller and feedback fields')
    return replace(cfg, **variant[2])


def block_job(cfg,start,count,variants,path):
    """Same immutable physical-time channel bank and public graph seed per block."""
    rows = {v[0]:[] for v in variants}
    for seed in range(start,start+count):
        bank = trial_inputs(cfg,seed)
        for v in variants:
            key, method = v[:2]
            result = simulate(variant_config(cfg, v),method,seed,bank=bank)
            if result['fingerprint'] != bank.fingerprint:
                raise RuntimeError('Common channel mismatch')
            rows[key].append(block_row(result))
    path = Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent,suffix='.npz',delete=False) as stream:
        np.savez_compressed(stream,**{k:np.asarray(v) for k,v in rows.items()})
        temp = Path(stream.name)
    temp.replace(path)
    return str(path)


def run_jobs(specs,out,workers,chunk_size):
    """Specs: (cell name, Config, first seed, n, variants)."""
    jobs = []
    out = Path(out)
    for cell,cfg,first,n,variants in specs:
        for offset in range(0,n,chunk_size):
            count = min(chunk_size,n-offset)
            path = out/'samples'/cell/f'seed_{first+offset}_n{count}.npz'
            if not path.exists():
                jobs.append((cfg,first+offset,count,variants,path))
    start = time.perf_counter()
    print(f'{len(jobs)} new paired chunks',flush=True)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(block_job,*job) for job in jobs]
        for i,future in enumerate(as_completed(futures),1):
            future.result()
            print(f'{i}/{len(jobs)} chunks; elapsed {time.perf_counter()-start:.1f}s',flush=True)


def load_samples(out,cell,first,n,keys,windows):
    pieces = {key:[] for key in keys}
    for path in sorted((Path(out)/'samples'/cell).glob('*.npz')):
        with np.load(path,allow_pickle=False) as data:
            if set(data.files) != set(keys):
                raise ValueError('Unexpected sample variants')
            for key in keys:
                pieces[key].append(data[key])
    arrays = {}
    for key in keys:
        if not pieces[key]:
            raise ValueError(f'No samples for {cell}/{key}')
        a = np.concatenate(pieces[key])
        a = a[np.argsort(a[:,0])]
        if a.shape != (n,len(COLUMNS)) or not np.array_equal(a[:,0],np.arange(first,first+n)):
            raise ValueError('Incomplete, duplicated or wrong seed blocks')
        if not np.array_equal(np.isfinite(a[:,2]),a[:,1].astype(bool)):
            raise ValueError('Failed blocks must have infinite completion times')
        if not np.array_equal(a[:,4],a[:,5]+a[:,8]) or not np.array_equal(a[:,5],a[:,6]+a[:,7]):
            raise ValueError('Packet accounting mismatch')
        if not np.array_equal(a[:,9],windows[key]*a[:,10]):
            raise ValueError('Feedback accounting mismatch')
        if np.any(a[:,16:23] < 0) or np.any(a[:,16:23] != np.floor(a[:,16:23])):
            raise ValueError('Invalid state/report event counts')
        for low, high in ((17,16),(18,13),(19,18),(19,14),(20,19)):
            if np.any(a[:,low] > a[:,high]):
                raise ValueError('State/repair accounting mismatch')
        if np.any(a[:,11] > a[:,10]) or np.any(a[:,10] > a[:,22]) or np.any(a[:,21] < a[:,9]):
            raise ValueError('Feedback generation/start/delivery accounting mismatch')
        if np.any(a[:,23:] < 0) or np.any(~np.isfinite(a[:,23:])):
            raise ValueError('Invalid feedback delay/airtime measurement')
        arrays[key] = a
    return arrays
