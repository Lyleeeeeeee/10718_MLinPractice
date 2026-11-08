"""Resumable, serial acquisition of exact historical revisions and first visible history."""
import argparse
from collections import Counter, defaultdict
import datetime as dt
import email.utils
import gzip
import hashlib
import json
from pathlib import Path
import time
import urllib.error
import urllib.parse
import urllib.request

from .preparation import load, save, guarded, creation_evidence, parse_batch, prepared_rows

USER_AGENT = "WikiFlow-course-research/2.0 (https://github.com/Lyleeeeeeee/10718_MLinPractice)"


def retry_request(operation, wait=time.sleep):
    for attempt in range(3):
        try:
            return operation()
        except (urllib.error.URLError, TimeoutError) as error:
            status = getattr(error, "code", None)
            if attempt == 2 or (status is not None and status != 429 and status < 500):
                raise
            value = (getattr(error, "headers", None) or {}).get("Retry-After")
            delay = float(30 * 2 ** attempt)
            if value:
                try:
                    delay = max(delay, float(value))
                except ValueError:
                    target = email.utils.parsedate_to_datetime(value)
                    delay = max(delay, (target - dt.datetime.now(dt.timezone.utc)).total_seconds())
            if delay > 300:
                raise  # Do not bypass a long server-requested cooldown.
            wait(delay)


def wait_seconds(seconds):
    while seconds > 0:
        part = min(seconds, 30)
        time.sleep(part)
        seconds -= part


class PublicClient:
    def __init__(self, cache, delay=7., max_requests=1000):
        self.cache = guarded(cache)
        (self.cache / "responses").mkdir(parents=True, exist_ok=True)
        if delay < 7 or max_requests < 1:
            raise ValueError("Require delay >= 7 seconds and positive request budget")
        self.delay, self.max_requests = delay, max_requests
        self.requests = 0
        self.last_request = 0.

    def log(self, record):
        record["utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
        with (self.cache / "requests.jsonl").open("a") as stream:
            stream.write(json.dumps(record, allow_nan=False) + "\n")

    def get(self, params, name):
        path = self.cache / "responses" / (name + ".json.gz")
        if path.exists():
            return load(path), path
        query = dict(action="query", format="json", formatversion=2, maxlag=5, **params)
        url = "https://en.wikipedia.org/w/api.php?" + urllib.parse.urlencode(query)

        def operation():
            if self.requests >= self.max_requests:
                raise RuntimeError("Request budget exhausted; resume the same cache later")
            wait_seconds(max(0., self.delay - (time.monotonic() - self.last_request)))
            self.last_request = time.monotonic()
            self.requests += 1
            request = urllib.request.Request(url, headers={
                "User-Agent": USER_AGENT, "Accept-Encoding": "gzip"})
            try:
                with urllib.request.urlopen(request, timeout=60) as response:
                    body = response.read()
                    if response.headers.get("Content-Encoding") == "gzip":
                        body = gzip.decompress(body)
                    obj = json.loads(body)
                    if obj.get("error", {}).get("code") in ("maxlag", "ratelimited", "readonly"):
                        raise urllib.error.HTTPError(url, 503, str(obj["error"]),
                                                     {"Retry-After": "30"}, None)
                    if "error" in obj:
                        raise ValueError("API error: " + str(obj["error"]))
                    path.write_bytes(gzip.compress(body, mtime=0))
                    self.log(dict(url=url, http_status=response.status,
                        response_file=str(path), raw_sha256=hashlib.sha256(body).hexdigest(),
                        bytes=len(body)))
                    return obj, path
            except Exception as error:
                self.log(dict(url=url, error=repr(error),
                    http_status=getattr(error, "code", None),
                    retry_after=(getattr(error, "headers", None) or {}).get("Retry-After")))
                raise
        return retry_request(operation, wait_seconds)


def acquire_revisions(plan, client):
    expected = {int(k): v for k, v in plan["selected"].items()}
    records_dir = client.cache / "records"
    records_dir.mkdir(exist_ok=True)
    remaining = [rid for rid in sorted(expected) if not (records_dir / f"{rid}.json.gz").exists()]
    print(f"historical content: {len(expected)-len(remaining)} cached, {len(remaining)} remaining", flush=True)
    for start in range(0, len(remaining), 25):
        ids = remaining[start:start + 25]
        name = "revisions_" + hashlib.sha256(json.dumps(ids).encode()).hexdigest()[:20]
        body, path = client.get(dict(prop="revisions", revids="|".join(map(str, ids)),
            rvprop="ids|timestamp|size|sha1|content", rvslots="main"), name)
        result = parse_batch(body, {rid: expected[rid] for rid in ids})
        for rid, record in result.items():
            record["raw_response_file"] = str(path)
            save(records_dir / f"{rid}.json.gz", record)
        print(f"content batch {start//25+1}: {min(start+25,len(remaining))}/{len(remaining)}; network requests {client.requests}", flush=True)


def acquire_age(plan, client, metadata):
    directory = client.cache / "age"
    directory.mkdir(exist_ok=True)
    pageids = sorted({v["pageid"] for v in plan["selected"].values()})
    for index, pid in enumerate(pageids):
        output = directory / f"{pid}.json"
        if output.exists():
            continue
        body, path = client.get(dict(prop="revisions", pageids=pid,
            rvprop="ids|timestamp", rvdir="newer", rvlimit=1,
            rvstart="1970-01-01T00:00:00Z"), f"first_{pid}")
        pages = body.get("query", {}).get("pages", [])
        if len(pages) != 1 or pages[0].get("pageid") != pid:
            raise ValueError("Creation query identity mismatch")
        rs = pages[0].get("revisions", [])
        record = creation_evidence(pid, rs[0] if rs else None, metadata[str(pid)])
        record["raw_response_file"] = str(path)
        save(output, record)
        print(f"visible history {index+1}/{len(pageids)}; network requests {client.requests}", flush=True)



def verify_metadata(plan, metadata):
    if plan.get("metadata_by_target_sha256"):
        config = load(metadata)
        combined, fingerprints = {}, {}
        for target, hashes in sorted(plan["metadata_by_target_sha256"].items()):
            path = (Path(metadata).resolve().parent / config[target]["metadata"]).resolve()
            if path in fingerprints:
                if fingerprints[path] != hashes["metadata"]:
                    raise ValueError("Target metadata changed after plan freeze")
                continue
            fingerprints[path] = hashlib.sha256(path.read_bytes()).hexdigest()
            if fingerprints[path] != hashes["metadata"]:
                raise ValueError("Target metadata changed after plan freeze")
            for pid, revisions in load(path).items():
                # Earlier source vintage wins for repeated revision IDs; retain all roots.
                records = combined.setdefault(pid, {})
                for revision in revisions:
                    records.setdefault(revision["revid"], revision)
        return {pid: list(records.values()) for pid, records in combined.items()}
    if hashlib.sha256(Path(metadata).read_bytes()).hexdigest() != plan["source_hashes"]["metadata"]:
        raise ValueError("Metadata changed after plan freeze")
    return load(metadata)

def age_metadata_by_page(metadata, config_path, pageids):
    """Select a declared retained metadata vintage per page, never infer precedence."""
    if config_path is None:
        return metadata, {}
    config_path = Path(config_path)
    config, cached, selected, hashes = load(config_path), {}, {}, {}
    for pid in pageids:
        key = str(pid)
        if key not in config:
            raise ValueError("Age metadata vintage missing page: " + key)
        path = (config_path.resolve().parent / config[key]).resolve()
        if path not in cached:
            cached[path] = load(path)
            hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        selected[key] = cached[path][key]
    return selected, {"config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
                      "source_sha256": hashes}


def build_dataset(plan_path, cache, data, metadata_path, output, age_metadata_by_page_path=None):
    plan, cache, data = load(plan_path), Path(cache), Path(data)
    metadata = verify_metadata(plan, metadata_path)
    for name in ("panel.json.gz", "evaluation_rows.json.gz"):
        if hashlib.sha256((data/name).read_bytes()).hexdigest() != plan["source_hashes"][name]:
            raise ValueError("Prepared rows changed after plan freeze")
    output = guarded(output)
    if output.exists():
        raise ValueError("Use a fresh output directory")
    revisions, raw_hashes, groups = {}, {}, defaultdict(dict)
    for rid in plan["selected"]:
        stored = load(cache / "records" / f"{rid}.json.gz")
        path = cache / "responses" / Path(stored["raw_response_file"]).name
        groups[path][int(rid)] = stored
    for path, stored_records in groups.items():
        raw = load(path)
        pages = [dict(p, revisions=[r for r in p.get("revisions", []) if r["revid"] in stored_records])
                 for p in raw.get("query", {}).get("pages", [])]
        expected = {rid: plan["selected"][str(rid)] for rid in stored_records}
        parsed = parse_batch({"query": {"pages": pages}}, expected)
        for rid, stored in stored_records.items():
            if parsed[rid] != {k: v for k, v in stored.items() if k != "raw_response_file"}:
                raise ValueError("Cached content differs from raw response")
            revisions[str(rid)] = parsed[rid]
        raw_hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    ages = {}
    pageids = sorted({r["pageid"] for r in revisions.values()})
    age_metadata, age_sources = age_metadata_by_page(metadata, age_metadata_by_page_path, pageids)
    for pid in pageids:
        stored = load(cache / "age" / f"{pid}.json")
        path = cache / "responses" / Path(stored["raw_response_file"]).name
        pages = load(path).get("query", {}).get("pages", [])
        if len(pages) != 1 or pages[0].get("pageid") != pid:
            raise ValueError("First-visible query identity mismatch")
        first = pages[0].get("revisions", [])
        if len(first) > 1:
            raise ValueError("Expected one earliest-visible revision")
        parsed = creation_evidence(pid, first[0] if first else None, age_metadata[str(pid)])
        if parsed != {k: v for k, v in stored.items() if k != "raw_response_file"}:
            raise ValueError("Cached age differs from raw evidence")
        ages[str(pid)] = parsed
        raw_hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    snapshots, mapping = {}, {}
    rows = prepared_rows(data)
    if set(plan["row_revision"]) != {r["key"] for r in rows}:
        raise ValueError("Plan row support differs")
    for row in rows:
        rid = plan["row_revision"][row["key"]]
        rec = revisions[str(rid)]
        if rec["pageid"] != row["pageid"] or rec["timestamp"] >= row["target_month"]+"-01T00:00:00Z":
            raise ValueError("Historical snapshot identity or boundary mismatch")
        sid = f"{row['pageid']}:{rid}"
        missing = rec["status"] != "usable"
        item = dict(snapshot_id=sid, pageid=rec["pageid"], revision_id=rid,
                    status=rec["status"], timestamp=rec["timestamp"],
                    source_url=f"https://en.wikipedia.org/w/index.php?oldid={rid}", missing=missing,
                    text="" if missing else rec["intro"]+" Categories: "+" ".join(rec["categories"]))
        if snapshots.setdefault(sid, item) != item:
            raise ValueError("Snapshot collision")
        mapping[row["key"]] = sid
    for name, value in (("snapshots", snapshots), ("row_snapshots", mapping), ("ages", ages)):
        save(output / (name+".json.gz"), value)
    save(output / "text_manifest.json", dict(plan_sha256=hashlib.sha256(Path(plan_path).read_bytes()).hexdigest(),
         raw_response_sha256=raw_hashes, age_metadata_sources=age_sources, original_creation_certified=False,
         limitations=["Retrospective visible history, not certified original creation or historical first release",
                      "Historical redirect is missing; no current redirect resolution or template expansion"]))
    return snapshots

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("acquire", "age", "build"):
        sub = commands.add_parser(name)
        sub.add_argument("--plan", required=True, type=Path)
        sub.add_argument("--cache", required=True, type=Path)
        if name in ("age", "build"):
            sub.add_argument("--metadata", required=True, type=Path)
            sub.add_argument("--age-metadata-by-page", type=Path, help="Explicit per-page retained metadata vintage for age evidence")
        if name == "build":
            sub.add_argument("--data", required=True, type=Path)
            sub.add_argument("--output", required=True, type=Path)
        else:
            sub.add_argument("--allow-network", action="store_true")
            sub.add_argument("--delay", type=float, default=7.)
            sub.add_argument("--max-requests", type=int, default=1000)
    args = parser.parse_args(argv)
    if args.command == "build":
        build_dataset(args.plan, args.cache, args.data, args.metadata, args.output, args.age_metadata_by_page)
    else:
        if not args.allow_network:
            parser.error("Public API acquisition requires --allow-network")
        plan = load(args.plan)
        metadata = verify_metadata(plan, args.metadata) if args.command == "age" else None
        client = PublicClient(args.cache, args.delay, args.max_requests)
        if args.command == "acquire":
            acquire_revisions(plan, client)
        else:
            pageids = sorted({r["pageid"] for r in plan["selected"].values()})
            metadata, _ = age_metadata_by_page(metadata, args.age_metadata_by_page, pageids)
            acquire_age(plan, client, metadata)

if __name__ == "__main__":
    main()
