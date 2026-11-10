"""Export the campaign's own raw samples to CSV, without creating or changing any value.

usage: python -B tools/export_csv.py <outputs folder> [--dest <csv folder>]

For every results folder under <outputs> (publication, optimization_*):
  1. every samples/*/*.npz file is checked against the SHA-256 the run itself recorded
     (completed.json for the evaluation, optimization.json for each training); any
     missing, extra or changed file stops the export;
  2. each variant's rows are concatenated, sorted by seed and checked for duplicate or
     missing seeds;
  3. rows are written with the run's own column names and 17 significant digits, so
     every float is reproduced exactly (failed blocks keep completion_slots = inf).
Figure data files (figure_data/*.dat) are copied as CSV unchanged in value.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def recorded_hashes(folder):
    for name in ('completed.json', 'optimization.json'):
        record = folder/name
        if record.exists():
            return json.loads(record.read_text())['raw_samples_sha256'], name
    raise ValueError(f'{folder}: no completed.json/optimization.json; the run is not finished')


def verify(folder):
    hashes, record = recorded_hashes(folder)
    files = {p.relative_to(folder).as_posix(): p for p in sorted((folder/'samples').rglob('*.npz'))}
    if set(files) != set(hashes):
        raise ValueError(f'{folder}: sample files differ from {record} '
                         f'(missing {sorted(set(hashes)-set(files))[:3]}, extra {sorted(set(files)-set(hashes))[:3]})')
    for key, path in files.items():
        if sha(path) != hashes[key]:
            raise ValueError(f'{folder}: {key} does not match the SHA-256 recorded in {record}')
    return files, record


def export_folder(folder, dest):
    files, record = verify(folder)
    columns = json.loads((folder/'manifest.json').read_text())['columns']
    cells = sorted({Path(k).parts[1] for k in files})
    written = []
    for cell in cells:
        pieces = {}
        for key, path in files.items():
            if Path(key).parts[1] != cell:
                continue
            with np.load(path, allow_pickle=False) as data:
                for variant in data.files:
                    pieces.setdefault(variant, []).append(data[variant])
        for variant, parts in sorted(pieces.items()):
            rows = np.concatenate(parts)
            rows = rows[np.argsort(rows[:, 0], kind='stable')]
            seeds = rows[:, 0].astype(np.int64)
            if rows.shape[1] != len(columns) or np.any(seeds != rows[:, 0]):
                raise ValueError(f'{folder}/{cell}/{variant}: unexpected column layout')
            if len(np.unique(seeds)) != len(seeds) or seeds[-1]-seeds[0]+1 != len(seeds):
                raise ValueError(f'{folder}/{cell}/{variant}: duplicate or missing seeds')
            out = dest/folder.name/cell/f'{variant}.csv'
            out.parent.mkdir(parents=True, exist_ok=True)
            with out.open('w', newline='') as stream:
                writer = csv.writer(stream)
                writer.writerow(columns)
                for seed, row in zip(seeds, rows):
                    writer.writerow([int(seed), *(repr(float(x)) for x in row[1:])])
            written.append((out, len(rows)))
    for dat in sorted((folder/'figure_data').glob('*.dat')):
        lines = dat.read_text().split('\n')
        out = dest/folder.name/'figure_data'/f'{dat.stem}.csv'
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text('\n'.join(','.join(line.split()) for line in lines if line.strip()) + '\n')
        written.append((out, len([l for l in lines if l.strip()]) - 1))
    print(f'{folder.name}: {len(files)} sample files verified against {record}; '
          f'{len(written)} CSV files written', flush=True)
    return written


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('outputs', type=Path)
    p.add_argument('--dest', type=Path, default=None)
    p.add_argument('--folders', nargs='+', default=None,
                   help='only these results folders (e.g. publication_lossy optimization_ded_lossy)')
    args = p.parse_args()
    outputs = args.outputs.resolve()
    dest = (args.dest or outputs/'csv').resolve()
    folders = [f for f in sorted(outputs.iterdir()) if (f/'samples').is_dir()
               and (args.folders is None or f.name in args.folders)]
    if args.folders and {f.name for f in folders} != set(args.folders):
        raise SystemExit(f'Missing results folders: {sorted(set(args.folders)-{f.name for f in folders})}')
    if not folders:
        raise SystemExit('No results folders with samples found')
    for folder in folders:
        for out, n in export_folder(folder, dest):
            print(f'  {out.relative_to(dest)}  ({n} rows)', flush=True)
    print('CSV folder:', dest, flush=True)


if __name__ == '__main__':
    main()
