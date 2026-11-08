"""Offline raw-cache preparation of the fixed-eligibility, loose-label panel.

No target counts participate in row construction. Unpublished counts stay unknown.
Metadata uses explicit boolean minor flags and contiguous parent chains. Inputs are
immutable evidence: a refreshed mapping is a new experiment, not a reference fix.
"""
import argparse
import collections
import csv
import datetime as dt
import gzip
import hashlib
import json
import math
import re
from pathlib import Path

from features.numeric import FEATURES, article_features, days, shift, historical_bound, label_bound
from features.assembly import make_label

# Predeclared protocol exclusions, not inferred from target outcomes.
IDENTITY_EXCLUSIONS = frozenset({1164, 72417803, 79745475})
ROOT = Path(__file__).resolve().parents[1]


def months(start, end):
    for value in (start, end):
        if not re.fullmatch(r"[0-9]{4}-[0-9]{2}", value):
            raise ValueError("Expected YYYY-MM")
        days(value)
    if start > end:
        raise ValueError("Reversed month interval")
    current = start
    while current <= end:
        yield current
        current = shift(current, 1)


def metadata_activity(records,coverage,target):
    """Two full calendar months; exact parent net bytes, including reverts."""
    start=shift(target,-2)+'-01T00:00:00Z';end=target+'-01T00:00:00Z';left=start
    for interval in sorted(coverage,key=lambda q:q['start']):
        if interval['end_exclusive']<=left:continue
        if interval['start']>left:break
        left=max(left,interval['end_exclusive'])
        if left>=end:break
    unknown=dict(coverage_complete=False,max_abs_nonminor=None,cumulative_abs_nonminor=None,window_start=start,window_end_exclusive=end)
    if left<end:return unknown
    byid={}
    for r in records:
        if r['timestamp']>=end:continue
        if r['revid'] in byid and byid[r['revid']]!=r:raise ValueError('Conflicting revision ID')
        byid[r['revid']]=r
    ordered=sorted(byid.values(),key=lambda r:(r['timestamp'],r['revid']))
    prior=[r for r in ordered if r['timestamp']<start]
    inside=[r for r in ordered if r['timestamp']>=start]
    if not prior:return unknown
    chain=[prior[-1]]+inside
    if any(not r.get('sha1') or type(r.get('size')) is not int or any(r.get(k) for k in ['sha1hidden','texthidden','commenthidden','suppressed']) for r in chain):return unknown
    if any(child['parentid']!=parent['revid'] for parent,child in zip(chain,chain[1:])):return unknown
    if any(type(r.get('minor')) is not bool for r in inside):return unknown
    deltas=[abs(r['size']-byid[r['parentid']]['size']) for r in inside if not r['minor']]
    return dict(coverage_complete=True,max_abs_nonminor=max(deltas,default=0),cumulative_abs_nonminor=sum(deltas),window_start=start,window_end_exclusive=end)



def build_panel(mapping, volumes, pageviews, revisions, coverage,
                start="2023-10", end="2026-10", identity=None, metadata_by_target=None):
    """Return (training panel, month→eligible rows, key→evaluation label).

    Mapping is canonical title→integer page ID. Metadata/coverage keys are string
    IDs. Identity is an optional frozen acquisition manifest of page ID→
    {expected_title, returned_title}; mismatches disqualify October only. Historical
    identity exclusions are fixed above. Nothing resolves or rewrites titles live.
    """
    if start < "2023-10" or end > "2026-10":
        raise ValueError("Only the declared 2023-10 through 2026-10 protocol")
    if len(set(mapping.values())) != len(mapping):
        raise ValueError("Duplicate canonical page identity")
    rows, labels, evaluation, evaluation_labels = [], {}, {}, {}
    unknown_quality = 0
    identity = identity or {}
    metadata_by_target = metadata_by_target or {}
    for target in months(start, end):
        if target >= "2024-09":
            evaluation[target] = []
        target_revisions, target_coverage = metadata_by_target.get(target, (revisions, coverage))
        cutoff = shift(target, -1)
        history = [shift(cutoff, k) for k in range(-6, 1)]
        for article, pid in sorted(mapping.items()):
            if pid in IDENTITY_EXCLUSIONS:
                continue
            if not all(article in volumes.get(m, {}) for m in history):
                continue
            current = volumes[cutoff][article]
            if current < 100:
                continue
            features, bound = article_features(article, cutoff, volumes, pageviews)
            previous_bound = historical_bound([volumes[m][article] / days(m) for m in history[:-1]], days(cutoff))
            if current / days(cutoff) > bound / days(target) or current > previous_bound:
                continue
            activity = metadata_activity(target_revisions.get(str(pid), []), target_coverage.get(str(pid), []), target)
            ident = identity.get(str(pid), {})
            mismatch = target == "2026-10" and ident and (
                ident.get("expected_title") != article or ident.get("returned_title") != article
                or ident.get("status") == "excluded")
            if not activity["coverage_complete"] or mismatch:
                unknown_quality += 1
                continue
            eligible = activity["max_abs_nonminor"] < 500 and activity["cumulative_abs_nonminor"] < 1000
            loose = label_bound([volumes[m][article] / days(m) for m in history[1:]], days(target))
            threshold = max(loose, 1.10 * current * days(target) / days(cutoff))
            # Frozen final protocol changes the margin but keeps the original bands.
            features[11] = math.log1p(current * days(target) / days(cutoff)) - math.log1p(loose)
            row = dict(key=f"{target}:{pid}", pageid=pid, article=article,
                       target_month=target, cutoff_month=cutoff, current_count=current,
                       U_count=bound, current_precurrent_U=previous_bound, features=features,
                       metadata_quality_complete=True, editor_activity_asof=activity,
                       effective_event_U_asof=threshold,
                       max_feature_date_declared=f"{cutoff}-{days(cutoff):02d}",
                       evaluation_eligible_asof=eligible, label_U_count=loose)
            label = make_label(volumes.get(target, {}).get(article), threshold)
            if target < "2026-10":
                rows.append(row)
                labels[row["key"]] = label
            if target in evaluation and eligible:
                evaluation[target].append(row)
                evaluation_labels[row["key"]] = label
    panel = dict(schema="wikiflow.final.binary.v1", rows=rows, labels=labels,
                 feature_names=FEATURES, preparation=dict(
                     unknown_quality_rows_excluded=unknown_quality, catalogue_count=len(mapping),
                     canonical_only=True, label_release_delay_verified=False,
                     identity_exclusions=sorted(IDENTITY_EXCLUSIONS),
                     october_identity_policy="frozen acquisition title equality; never live repair"))
    return panel, evaluation, evaluation_labels


def open_text(path):
    return gzip.open(path, "rt", encoding="utf-8", newline="") if str(path).endswith(".gz") else open(path, encoding="utf-8", newline="")


def read_json(path):
    with open_text(path) as stream:
        return json.load(stream)


def read_mapping(path):
    mapping, ids = {}, set()
    with open_text(path) as stream:
        for row in csv.DictReader(stream):
            article, pid = row["article"], int(row["pageid"])
            if article in mapping:
                if mapping[article] != pid:
                    raise ValueError("Conflicting title page ID")
                continue  # Multiple original titles may map to one canonical title.
            if pid in ids or pid <= 0 or not article:
                raise ValueError("Invalid or duplicate canonical page identity")
            mapping[article] = pid
            ids.add(pid)
    if not mapping:
        raise ValueError("Empty fixed catalogue")
    return mapping


def read_incoming(directory, start, end, catalogue):
    volumes = {}
    for month in months(start, end):
        path = Path(directory) / f"incoming_{month}.csv.gz"
        if not path.exists():
            continue
        counts, seen = collections.Counter(), set()
        with open_text(path) as stream:
            for row in csv.DictReader(stream):
                count = int(row["count"])
                pair = row["source"], row["destination"]
                if row["month"] != month or pair[1] not in catalogue or count < 10 or pair in seen:
                    raise ValueError("Invalid cached clickstream extract")
                seen.add(pair)
                counts[pair[1]] += count
        # Missing title is suppressed/unpublished, never a synthetic zero.
        volumes[month] = dict(counts)
    return volumes


def read_pageviews(path, max_date):
    pageviews = collections.defaultdict(dict)
    with open_text(path) as stream:
        for row in csv.DictReader(stream):
            date = row["date"]
            if date > max_date:
                continue
            dt.date.fromisoformat(date)
            views, title = int(row["views"]), row["destination"]
            if views < 0 or date in pageviews[title]:
                raise ValueError("Invalid or duplicate daily cached PV")
            pageviews[title][date] = views
    return dict(pageviews)


def fresh_output(path):
    path = Path(path).resolve()
    runs = (ROOT / "runs").resolve()
    local = (ROOT / "local_data").resolve()
    if not any(parent in path.parents for parent in (runs, local)):
        raise ValueError("Output must be a fresh directory under runs/ or local_data/")
    if path.exists():
        raise FileExistsError("Refusing to replace an existing output")
    return path


def save_json(path, value):
    raw = json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode("utf-8")
    with Path(path).open("xb") as stream:
        stream.write(gzip.compress(raw, mtime=0) if str(path).endswith(".gz") else raw)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("mapping", "incoming", "pageviews", "metadata", "coverage", "output"):
        parser.add_argument("--" + option, required=True)
    parser.add_argument("--identity", help="Frozen page ID to expected/returned title manifest")
    parser.add_argument("--metadata-by-target", help="JSON target month to {metadata,coverage} file paths; relative to config")
    parser.add_argument("--start", default="2023-10")
    parser.add_argument("--end", default="2026-10")
    args = parser.parse_args(argv)
    output = fresh_output(args.output)
    mapping = read_mapping(args.mapping)
    volumes = read_incoming(args.incoming, "2022-09", args.end, set(mapping))
    last = shift(args.end, -1)
    pageviews = read_pageviews(args.pageviews, f"{last}-{days(last):02d}")
    identity_path = Path(args.identity) if args.identity else Path(args.metadata).parent / "identity.json"
    identity = read_json(identity_path) if identity_path.exists() else {}
    target_sources, extra_inputs, json_cache = {}, [], {}

    def cached_json(path):
        resolved = Path(path).resolve()
        if resolved not in json_cache:
            json_cache[resolved] = read_json(resolved)
        return json_cache[resolved]

    if args.metadata_by_target:
        config_path = Path(args.metadata_by_target)
        extra_inputs.append(config_path)
        for target, sources in read_json(config_path).items():
            list(months(target, target))
            paths = [Path(sources[key]) for key in ("metadata", "coverage")]
            paths = [p if p.is_absolute() else config_path.parent / p for p in paths]
            target_sources[target] = tuple(cached_json(p) for p in paths)
            extra_inputs.extend(paths)
    panel, evaluation, labels = build_panel(mapping, volumes, pageviews,
        cached_json(args.metadata), cached_json(args.coverage), args.start, args.end, identity, target_sources)
    output.mkdir(parents=True, exist_ok=False)
    for name, value in (("panel", panel), ("evaluation_rows", evaluation),
                        ("evaluation_labels", labels), ("source_monthly_counts", volumes)):
        save_json(output / (name + ".json.gz"), value)
    # Preserve the exact original-to-canonical mapping, including normalization history.
    with open_text(args.mapping) as stream:
        (output / "title_mapping.csv").write_text(stream.read(), encoding="utf-8")
    with (output / "catalogue.csv").open("x", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream); writer.writerow(["article", "pageid"])
        writer.writerows(sorted(mapping.items()))
    inputs = [Path(args.mapping), Path(args.pageviews), Path(args.metadata), Path(args.coverage)]
    inputs += sorted(Path(args.incoming).glob("incoming_*.csv.gz")) + extra_inputs
    if identity_path.exists():
        inputs.append(identity_path)
    inputs = list(dict.fromkeys(p.resolve() for p in inputs))
    provenance = dict(schema="wikiflow.raw-preparation.v1", start=args.start, end=args.end,
        input_files=[dict(name=p.name, sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in inputs],
        fixed_reference_modified=False, metadata_identity_manifest_present=bool(identity),
        actual_release_availability_certified=False, no_network_requests=True)
    save_json(output / "provenance.json", provenance)
    print(json.dumps(dict(training_rows=len(panel["rows"]), evaluation_rows=sum(map(len, evaluation.values())), output=output.name)))


if __name__ == "__main__":
    main()
