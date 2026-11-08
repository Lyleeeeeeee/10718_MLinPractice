"""Synthetic historical selection, parsing, evidence and pooling checks."""
import copy
import hashlib
import unittest
from unittest.mock import patch
import tempfile
from pathlib import Path
import numpy as np
from textual.preparation import covered, latest_revision, parse_batch, creation_evidence, extract_document, guarded
from textual.acquisition import build_dataset, verify_metadata, age_metadata_by_page
from textual.preparation import ROOT, save, load, build_plan
from textual.encoding import split_tokens, pool_chunks, encode, snapshots_canonical_sha256
from textual.representations import age_features

class TextPipelineTests(unittest.TestCase):
    def setUp(self):
        self.row = dict(target_month="2024-02", editor_activity_asof=dict(
            window_start="2024-01-01T00:00:00Z", window_end_exclusive="2024-02-01T00:00:00Z"))
        self.revs = [dict(revid=1,parentid=0,timestamp="2023-12-01T00:00:00Z"),
                     dict(revid=2,parentid=1,timestamp="2024-01-31T23:59:59Z"),
                     dict(revid=3,parentid=2,timestamp="2024-02-01T00:00:00Z")]
        self.coverage = [dict(start="2024-01-01T00:00:00Z",end_exclusive="2024-02-01T00:00:00Z")]
    def test_strict_boundary_and_parent_chain(self):
        self.assertEqual(latest_revision(self.row,self.revs,self.coverage)["revid"],2)
        broken = copy.deepcopy(self.revs); broken[1]["parentid"]=9
        with self.assertRaises(ValueError): latest_revision(self.row,broken,self.coverage)
        with self.assertRaises(ValueError): latest_revision(self.row,self.revs[1:],self.coverage)
    def test_coverage_union_gap_and_timestamp(self):
        intervals=[dict(start="2024-01-01T00:00:00Z",end_exclusive="2024-01-15T00:00:00Z"),
                   dict(start="2024-01-15T00:00:00Z",end_exclusive="2024-02-01T00:00:00Z")]
        self.assertEqual(latest_revision(self.row,self.revs,intervals)["revid"],2)
        intervals[1]["start"]="2024-01-16T00:00:00Z"
        with self.assertRaises(ValueError): latest_revision(self.row,self.revs,intervals)
        revs=copy.deepcopy(self.revs); revs[0]["timestamp"]="2023-12-01"
        with self.assertRaises(ValueError): latest_revision(self.row,revs,self.coverage)
    def test_revision_collision(self):
        with self.assertRaises(ValueError):
            latest_revision(self.row,self.revs+[dict(self.revs[0],parentid=100)],self.coverage)
    def test_parser_and_identity(self):
        text="{{box|ignored}} An [[Article|example]] <ref>citation</ref> lead.\n== Body ==\nBody [[Category:Science|S]] [[Category:Science]]"
        intro,categories=extract_document(text)
        self.assertEqual(intro,"An example lead.")
        self.assertEqual(categories,["Science"])
        raw=text.encode(); meta=dict(pageid=42,revid=2,timestamp="2024-01-31T23:59:59Z",size=len(raw),sha1=hashlib.sha1(raw).hexdigest())
        rev=dict(revid=2,timestamp=meta["timestamp"],slots={"main":{"content":text}})
        body={"query":{"pages":[dict(pageid=42,revisions=[rev])]}}
        self.assertEqual(parse_batch(body,{2:meta})[2]["status"],"usable")
        body["query"]["pages"][0]["pageid"]=43
        with self.assertRaises(ValueError): parse_batch(body,{2:meta})
        self.assertEqual(parse_batch({},{2:meta})[2]["status"],"revision_unavailable")
    def test_lead_cap_and_missing_states(self):
        self.assertEqual(len(extract_document("x"*5000)[0]),4000)
        meta=dict(pageid=1,revid=1,timestamp="2024-01-01T00:00:00Z")
        for content,status in (("#REDIRECT [[Other]]","historical_redirect"),("{{box}}","parser_empty"),(None,"hidden_content")):
            body={"query":{"pages":[dict(pageid=1,revisions=[dict(revid=1,timestamp=meta["timestamp"],slots={"main":{"content":content}})])]}}
            self.assertEqual(parse_batch(body,{1:meta})[1]["status"],status)
    def test_age_retains_future_root_semantics(self):
        roots=[self.revs[0],dict(revid=8,parentid=0,timestamp="2025-01-01T00:00:00Z")]
        evidence=creation_evidence(1,roots[0],roots)
        self.assertEqual(evidence["status"],"multiple_roots_uncertain")
        self.assertEqual(age_features(evidence,"2024-02-01T00:00:00Z")[1],0)
        self.assertEqual(age_features(evidence,"2025-02-01T00:00:00Z"),[None,1.])
        self.assertFalse(evidence["original_creation_certified"])
    def test_chunk_coverage_and_pooling(self):
        ids=list(range(509)); chunks=split_tokens(ids)
        self.assertEqual([len(c["ids"]) for c in chunks],[254,254,1])
        self.assertEqual([i for c in chunks for i in c["ids"]],ids)
        result=pool_chunks([[2.,0.],[0.,3.]],[2,1])
        np.testing.assert_allclose(result,np.array([2.,1.])/np.sqrt(5))
        for vectors,weights in (([[0,0]],[1]),([[1,0]],[0]),([[1,0],[-1,0]],[1,1])):
            with self.assertRaises(ValueError): pool_chunks(vectors,weights)
        with self.assertRaises(ValueError): split_tokens([])
    def test_offline_end_to_end(self):
        with tempfile.TemporaryDirectory(dir=ROOT/"runs") as directory:
            root = Path(directory)
            row = dict(self.row, key="2024-02:42", pageid=42)
            save(root/"data/panel.json.gz", {"rows":[row], "labels":{}})
            save(root/"data/evaluation_rows.json.gz", {})
            metadata = {"42":self.revs}
            save(root/"metadata.json", metadata)
            save(root/"coverage.json", {"42":self.coverage})
            plan = build_plan(root/"data", root/"metadata.json", root/"coverage.json", root/"plan.json")
            body = {"query":{"pages":[dict(pageid=42,revisions=[dict(self.revs[1],slots={"main":{"content":"Historical lead. [[Category:Test]]"}})])]}}
            response = root/"cache/responses/content.json.gz"
            save(response,body)
            parsed = parse_batch(body,{2:plan["selected"]["2"]})[2]
            save(root/"cache/records/2.json.gz",dict(parsed,raw_response_file=str(response)))
            first = root/"cache/responses/first.json.gz"
            save(first,{"query":{"pages":[dict(pageid=42,revisions=[self.revs[0]])]}})
            save(root/"cache/age/42.json",dict(creation_evidence(42,self.revs[0],self.revs),raw_response_file=str(first)))
            snapshots = build_dataset(root/"plan.json",root/"cache",root/"data",root/"metadata.json",root/"built/text")
            self.assertEqual(set(snapshots["42:2"]),{"snapshot_id","pageid","revision_id","missing","text","status","timestamp","source_url"})
            self.assertEqual(load(root/"built/text/row_snapshots.json.gz"),{"2024-02:42":"42:2"})
            self.assertEqual(load(root/"built/text/ages.json.gz")["42"]["status"],"earliest_visible_root")
            with self.assertRaises(ValueError):
                build_dataset(root/"plan.json",root/"cache",root/"data",root/"metadata.json",root/"built/text")
            save(root/"vintages.json",{"2024-02":{"metadata":"metadata.json","coverage":"coverage.json"}})
            vintage = build_plan(root/"data",None,None,root/"vintage-plan.json",root/"vintages.json")
            self.assertEqual(vintage["row_revision"],plan["row_revision"])
            self.assertEqual(verify_metadata(vintage,root/"vintages.json"),metadata)
            save(root/"age-vintages.json",{"42":"metadata.json"})
            ages, hashes = age_metadata_by_page({},root/"age-vintages.json",[42])
            self.assertEqual(ages,metadata)
            self.assertEqual(len(hashes["source_sha256"]),1)
            with self.assertRaises(ValueError): age_metadata_by_page({},root/"age-vintages.json",[43])
            config = {target:{"metadata":"metadata.json","coverage":"coverage.json"}
                      for target in ("2024-02","2024-03")}
            save(root/"repeated-vintages.json",config)
            vintage["metadata_by_target_sha256"]["2024-03"] = dict(vintage["metadata_by_target_sha256"]["2024-02"])
            with patch("textual.acquisition.load", wraps=load) as reader:
                self.assertEqual(verify_metadata(vintage,root/"repeated-vintages.json"),metadata)
                self.assertEqual(reader.call_count,2)  # Configuration plus one metadata parse.
            vintage["metadata_by_target_sha256"]["2024-03"]["metadata"] = "bad"
            with self.assertRaises(ValueError): verify_metadata(vintage,root/"repeated-vintages.json")

    def test_semantic_snapshot_fingerprint(self):
        self.assertEqual(snapshots_canonical_sha256({"b":2,"a":"é"}),
                         snapshots_canonical_sha256({"a":"é","b":2}))
        self.assertNotEqual(snapshots_canonical_sha256({"a":"lead"}),
                            snapshots_canonical_sha256({"a":"changed"}))

    def test_opt_in_and_output_guard(self):
        with self.assertRaises(ValueError): encode("unused","unused","unused")
        with self.assertRaises(ValueError): guarded("data/text/forbidden.json")

if __name__ == "__main__": unittest.main()
