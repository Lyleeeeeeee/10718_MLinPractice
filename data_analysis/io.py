"""Portable JSON/CSV I/O; outputs must be new files."""
import csv
import gzip
import hashlib
import json
from pathlib import Path

def load(path):
    path = Path(path)
    body = path.read_bytes()
    return json.loads(gzip.decompress(body) if path.suffix == '.gz' else body)

def save(path, value):
    path = Path(path)
    body = (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode()
    with path.open('xb') as stream:
        stream.write(gzip.compress(body, mtime=0) if path.suffix == '.gz' else body)

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def digest(value):
    return hashlib.sha256(json.dumps(value, separators=(',', ':')).encode()).hexdigest()

def write_csv(path, rows):
    with Path(path).open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
