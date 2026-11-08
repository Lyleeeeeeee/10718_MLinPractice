"""Real stage consistency and immutable-output contracts for data assembly."""
import gzip
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from data_pipeline.assemble import assemble, canonical_sha
from data_analysis.io import sha

class AssemblyContracts(unittest.TestCase):
    def fixture(self, root):
        numeric, text, encoding = [root/name for name in ('numeric','text','encoding')]
        for p in (numeric,text,encoding): p.mkdir()
        def write(path,obj):
            data=json.dumps(obj).encode();path.write_bytes(gzip.compress(data) if path.suffix=='.gz' else data)
        row={'key':'2026-05:1','pageid':1,'target_month':'2026-05'}
        write(numeric/'panel.json.gz',{'rows':[row]})
        write(numeric/'evaluation_rows.json.gz',{'2026-05':[row]})
        for name in ('evaluation_labels.json.gz','source_monthly_counts.json.gz','provenance.json'):write(numeric/name,{})
        for name in ('catalogue.csv','title_mapping.csv'):(numeric/name).write_text('article,pageid\nExample,1\n')
        snapshots={'1:10':{'snapshot_id':'1:10','text':'Example'}}
        write(text/'snapshots.json.gz',snapshots)
        write(text/'row_snapshots.json.gz',{'2026-05:1':'1:10'})
        write(text/'ages.json.gz',{})
        np.savez_compressed(encoding/'minilm_embeddings.npz',snapshot_ids=np.array(['1:10']),chunked=np.zeros((1,384)))
        manifest={'snapshots_canonical_sha256':canonical_sha(snapshots),'embeddings_sha256':sha(encoding/'minilm_embeddings.npz')}
        write(encoding/'encoder.json',manifest)
        return numeric,text,encoding

    def test_semantic_match_and_existing_output_preservation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);args=self.fixture(root);output=root/'runs/result'
            with patch('data_pipeline.preparation.ROOT',root):
                assemble(*args,output)
                self.assertTrue((output/'text/minilm_embeddings.npz').is_file())
                self.assertTrue((output/'assembly.json').is_file())
                original=(output/'text/snapshots.json.gz').read_bytes()
                with self.assertRaises(FileExistsError):assemble(*args,output)
                self.assertEqual(original,(output/'text/snapshots.json.gz').read_bytes())

    def test_encoder_text_mismatch_rejected_before_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);args=self.fixture(root);output=root/'runs/result'
            (args[2]/'encoder.json').write_text(json.dumps({'snapshots_canonical_sha256':'wrong'}))
            with patch('data_pipeline.preparation.ROOT',root),self.assertRaisesRegex(ValueError,'Encoder input'):
                assemble(*args,output)
            self.assertFalse(output.exists())

    def test_row_membership_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);args=self.fixture(root)
            (args[1]/'row_snapshots.json.gz').write_bytes(gzip.compress(json.dumps({'other':'1:10'}).encode()))
            with patch('data_pipeline.preparation.ROOT',root),self.assertRaisesRegex(ValueError,'membership'):
                assemble(*args,root/'runs/result')

    def test_modified_embeddings_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);args=self.fixture(root)
            (args[2]/'minilm_embeddings.npz').write_bytes(b'changed')
            with patch('data_pipeline.preparation.ROOT',root),self.assertRaisesRegex(ValueError,'vectors changed'):
                assemble(*args,root/'runs/result')
