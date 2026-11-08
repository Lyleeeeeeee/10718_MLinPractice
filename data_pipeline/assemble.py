"""Assemble verified numeric, historical-text and encoder stages into model inputs."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
from data_analysis.io import load, save, sha
from data_pipeline.preparation import fresh_output

NUMERIC_FILES = ('panel.json.gz', 'evaluation_rows.json.gz', 'evaluation_labels.json.gz',
                 'source_monthly_counts.json.gz', 'title_mapping.csv', 'catalogue.csv', 'provenance.json')
TEXT_FILES = ('snapshots.json.gz', 'row_snapshots.json.gz', 'ages.json.gz')


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def assemble(numeric, text, encoding, output):
    numeric, text, encoding = map(Path, (numeric, text, encoding))
    output = fresh_output(output)
    for base, names in ((numeric, NUMERIC_FILES), (text, TEXT_FILES),
                        (encoding, ('minilm_embeddings.npz', 'encoder.json'))):
        for name in names:
            if not (base / name).is_file():
                raise ValueError('Missing stage input: ' + str(base / name))
    snapshots = load(text / 'snapshots.json.gz')
    mapping = load(text / 'row_snapshots.json.gz')
    manifest = load(encoding / 'encoder.json')
    byte_match = manifest.get('snapshots_sha256') == sha(text / 'snapshots.json.gz')
    semantic_match = manifest.get('snapshots_canonical_sha256') == canonical_sha(snapshots)
    if not byte_match and not semantic_match:
        raise ValueError('Encoder input does not match the supplied historical text')
    if manifest.get('embeddings_sha256') != sha(encoding / 'minilm_embeddings.npz'):
        raise ValueError('Encoder vectors changed after encoding')
    panel = load(numeric / 'panel.json.gz')
    evaluation = load(numeric / 'evaluation_rows.json.gz')
    rows = {r['key']: r for r in panel['rows']}
    for group in evaluation.values():
        rows.update({r['key']: r for r in group})
    if set(mapping) != set(rows) or set(mapping.values()) != set(snapshots):
        raise ValueError('Numeric/text row or snapshot membership differs')
    with np.load(encoding / 'minilm_embeddings.npz', allow_pickle=False) as arrays:
        ids, vectors = arrays['snapshot_ids'].tolist(), arrays['chunked']
        if len(set(ids)) != len(ids) or set(ids) != set(snapshots) or vectors.shape != (len(ids), 384) or not np.isfinite(vectors).all():
            raise ValueError('Encoder membership/dimensions/values differ')
    output.mkdir(parents=True, exist_ok=False)
    (output / 'text').mkdir()
    for name in NUMERIC_FILES:
        shutil.copyfile(numeric / name, output / name)
    for name in TEXT_FILES:
        shutil.copyfile(text / name, output / 'text' / name)
    for name in ('minilm_embeddings.npz', 'encoder.json'):
        shutil.copyfile(encoding / name, output / 'text' / name)
    if (text / 'text_manifest.json').is_file():
        shutil.copyfile(text / 'text_manifest.json', output / 'text/text_manifest.json')
    save(output / 'assembly.json', dict(schema='wikiflow.assembled-inputs.v1',
        source_files_sha256={str(p.relative_to(output)): sha(p) for p in output.rglob('*') if p.is_file()},
        snapshots_canonical_sha256=canonical_sha(snapshots), encoder_text_match=True))
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('numeric', 'text', 'encoding', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args(argv)
    output = assemble(args.numeric, args.text, args.encoding, args.output)
    print('Assembled data:', output)


if __name__ == '__main__':
    main()
