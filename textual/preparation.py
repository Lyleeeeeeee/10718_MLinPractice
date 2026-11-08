"""Historical article-month text and evidence-qualified visible-history age."""
import datetime as dt
import hashlib
import math
import re

from pathlib import Path
import gzip
import json
import argparse


def utc_timestamp(value):
    parsed = dt.datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ")
    if parsed.strftime("%Y-%m-%dT%H:%M:%SZ") != value:
        raise ValueError("Expected canonical UTC timestamp")
    return parsed.replace(tzinfo=dt.timezone.utc)


def covered(intervals, start, end):
    for value in [start, end] + [v[k] for v in intervals for k in ("start", "end_exclusive")]:
        utc_timestamp(value)
    if start >= end or any(v["start"] >= v["end_exclusive"] for v in intervals):
        raise ValueError("Invalid coverage interval")
    cursor = start
    for a, b in sorted((v["start"], v["end_exclusive"]) for v in intervals):
        if b <= cursor:
            continue
        if a > cursor:
            return False
        cursor = max(cursor, b)
        if cursor >= end:
            return True
    return False


def latest_revision(row, revisions, coverage):
    unique = {}
    for revision in revisions:
        utc_timestamp(revision["timestamp"])
        if unique.setdefault(revision["revid"], revision) != revision:
            raise ValueError("Conflicting revision metadata")
    window = row["editor_activity_asof"]
    start, end = window["window_start"], row["target_month"] + "-01T00:00:00Z"
    if end != window["window_end_exclusive"] or not covered(coverage, start, end):
        raise ValueError("Incomplete revision query coverage")
    records = sorted({r["revid"]: r for r in revisions if r["timestamp"] < end}.values(),
                     key=lambda r: (r["timestamp"], r["revid"]))
    before = [r for r in records if r["timestamp"] < start]
    if not before:
        raise ValueError("Missing pre-window baseline; nonexistence not certified")
    chain = before[-1:] + [r for r in records if r["timestamp"] >= start]
    if any(b["parentid"] != a["revid"] for a, b in zip(chain, chain[1:])):
        raise ValueError("Broken revision parent chain")
    return dict(chain[-1])


def extract_document(text):
    # Match the original fixed-snapshot parser, isolating the temporal change.
    import mwparserfromhell
    categories = sorted(set(m.group(1).strip() for m in re.finditer(
        r"\[\[Category:([^\]|]+)(?:\|[^\]]*)?\]\]", text, re.I)))
    lead = re.split(r"^\s*==", text, maxsplit=1, flags=re.M)[0]
    code = mwparserfromhell.parse(lead)
    for template in code.filter_templates(recursive=False):
        code.remove(template)
    for tag in code.filter_tags():
        if str(tag.tag).lower() in ["ref", "gallery", "table", "math", "timeline"]:
            try:
                code.remove(tag)
            except ValueError:
                pass
    intro = re.sub(r"\s+", " ", code.strip_code(normalize=True, collapse=True)).strip()[:4000]
    return intro, categories


def parse_batch(body, expected):
    if "error" in body:
        raise ValueError("API error: " + str(body["error"]))
    result = {}
    for page in body.get("query", {}).get("pages", []):
        for rev in page.get("revisions", []):
            rid = rev["revid"]
            if rid not in expected or page["pageid"] != expected[rid]["pageid"]:
                raise ValueError("Revision/page identity mismatch")
            meta = expected[rid]
            if rev.get("timestamp") != meta["timestamp"]:
                raise ValueError("Revision timestamp mismatch")
            record = dict(pageid=meta["pageid"], revision_id=rid, timestamp=rev["timestamp"],
                          intro="", categories=[])
            text = rev.get("slots", {}).get("main", {}).get("content")
            if not isinstance(text, str):
                record["status"] = "hidden_content"
            else:
                raw = text.encode()
                if meta.get("size") is not None and meta["size"] != len(raw):
                    raise ValueError("Revision size mismatch")
                if meta.get("sha1") and meta["sha1"] != hashlib.sha1(raw).hexdigest():
                    raise ValueError("Revision content hash mismatch")
                record["wikitext_sha256"] = hashlib.sha256(raw).hexdigest()
                if text.lstrip().lower().startswith("#redirect"):
                    record["status"] = "historical_redirect"
                else:
                    record["intro"], record["categories"] = extract_document(text)
                    record["status"] = "usable" if record["intro"] else "parser_empty"
            if rid in result:
                raise ValueError("Duplicate returned revision")
            result[rid] = record
    for rid, meta in expected.items():
        result.setdefault(rid, dict(pageid=meta["pageid"], revision_id=rid,
            timestamp=meta["timestamp"], status="revision_unavailable", intro="", categories=[]))
    return result


def creation_evidence(pageid, first, known_revisions):
    """First visible revision is a qualified proxy, never certified original creation."""
    roots = {r["revid"] for r in known_revisions if r.get("parentid") == 0}
    if first is None:
        status = "fetch_unavailable"
    elif any(r["timestamp"] < first["timestamp"] for r in known_revisions):
        status = "earlier_cached_history_conflict"
    elif first.get("parentid") != 0:
        status = "earliest_visible_has_missing_parent"
    elif roots - {first["revid"]}:
        status = "multiple_roots_uncertain"
    else:
        status = "earliest_visible_root"
    return dict(pageid=pageid, status=status,
                earliest_visible_revision=first.get("revid") if first else None,
                earliest_visible_at=first.get("timestamp") if first else None,
                observed_root_ids=sorted(roots), original_creation_certified=False,
                observed_roots=sorted([dict(revid=r["revid"], timestamp=r["timestamp"])
                    for r in known_revisions if r.get("parentid") == 0],
                    key=lambda r: (r["timestamp"], r["revid"])),
                source="MediaWiki revisions rvdir=newer rvlimit=1; retrospective visible history")



ROOT = Path(__file__).resolve().parents[1]

def load(path):
    path = Path(path)
    return json.loads(gzip.decompress(path.read_bytes()) if path.suffix == ".gz" else path.read_bytes())

def guarded(path):
    path = Path(path).resolve()
    if not any(path.is_relative_to(ROOT / name) and path != ROOT / name for name in ("runs", "local_data")):
        raise ValueError("Outputs must be beneath this repository's runs/ or local_data/")
    return path

def save(path, value):
    path = guarded(path)
    if path.exists():
        raise ValueError("Refusing to overwrite existing output: " + str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n").encode()
    with path.open("xb") as stream:
        stream.write(gzip.compress(raw, mtime=0) if path.suffix == ".gz" else raw)

def prepared_rows(data):
    data = Path(data)
    panel, evaluation = load(data / "panel.json.gz"), load(data / "evaluation_rows.json.gz")
    rows = list(panel["rows"])
    if any(r["target_month"] >= "2026-10" for r in rows):
        raise ValueError("Training panel contains sealed targets")
    for month, group in evaluation.items():
        if any(r["target_month"] != month for r in group):
            raise ValueError("Evaluation month mismatch")
        rows.extend(group)
    seen = {}
    for row in rows:
        if row["key"] != f"{row['target_month']}:{row['pageid']}":
            raise ValueError("Row identity mismatch")
        previous = seen.setdefault(row["key"], row)
        if previous != row:
            raise ValueError("Conflicting duplicate row")
    return list(seen.values())

def build_plan(data, metadata_path, coverage_path, output, metadata_by_target=None):
    vintage = load(metadata_by_target) if metadata_by_target else None
    base = Path(metadata_by_target).resolve().parent if vintage else None
    metadata, coverage = (load(metadata_path), load(coverage_path)) if not vintage else ({}, {})
    vintage_hashes, source_fingerprints = {}, {}
    cache = {}
    selected, mapping = {}, {}
    for row in prepared_rows(data):
        pid = row["pageid"]
        if vintage is not None:
            entry = vintage[row["target_month"]]
            paths = {key: (base / entry[key]).resolve() for key in ("metadata", "coverage")}
            for path in paths.values():
                if str(path) not in cache:
                    cache[str(path)] = load(path)
            metadata, coverage = cache[str(paths["metadata"])], cache[str(paths["coverage"])]
            if row["target_month"] not in vintage_hashes:
                for path in paths.values():
                    if str(path) not in source_fingerprints:
                        source_fingerprints[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
                vintage_hashes[row["target_month"]] = {key: source_fingerprints[str(path)] for key, path in paths.items()}
        record = latest_revision(row, metadata[str(pid)], coverage[str(pid)])
        rid = str(record["revid"])
        item = dict(pageid=pid, **{k: v for k, v in record.items() if k in ("revid", "parentid", "timestamp", "size", "sha1")})
        if selected.setdefault(rid, item) != item:
            raise ValueError("Revision identity/content collision")
        mapping[row["key"]] = record["revid"]
    sources = {name: hashlib.sha256(Path(path).read_bytes()).hexdigest() for name, path in
        (("panel.json.gz", Path(data)/"panel.json.gz"), ("evaluation_rows.json.gz", Path(data)/"evaluation_rows.json.gz"),
         ("metadata", metadata_path), ("coverage", coverage_path)) if path is not None}
    plan = dict(schema="wikiflow.text.plan.v1", selected=selected, row_revision=mapping, source_hashes=sources,
                metadata_by_target_sha256=vintage_hashes,
                selection="Latest revision strictly before target boundary; complete coverage and baseline parent chain required")
    save(output, plan)
    return plan

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    for name in ("metadata", "coverage", "metadata-by-target"):
        parser.add_argument("--"+name, type=Path)
    args = parser.parse_args(argv)
    if bool(args.metadata_by_target) == bool(args.metadata or args.coverage) or (not args.metadata_by_target and not (args.metadata and args.coverage)):
        parser.error("Supply --metadata and --coverage, or --metadata-by-target")
    plan = build_plan(args.data, args.metadata, args.coverage, args.output, args.metadata_by_target)
    print(f"Planned {len(plan['row_revision'])} rows and {len(plan['selected'])} revisions")

if __name__ == "__main__":
    main()
