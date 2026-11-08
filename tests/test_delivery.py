"""Contracts for the deliberately small final delivery."""
import pathlib
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]

class DeliveryContracts(unittest.TestCase):
    def test_required_top_level_packages_exist(self):
        for name in ('models', 'data_analysis', 'data_pipeline', 'features', 'textual', 'data'):
            self.assertTrue((ROOT / name).is_dir(), name)

    def test_only_final_scope_is_exposed(self):
        self.assertFalse((ROOT / 'experiments').exists())
        self.assertFalse((ROOT / 'src').exists())
        self.assertTrue((ROOT / 'run.py').is_file())


class OutputAndIntegrityContracts(unittest.TestCase):
    def test_training_artifacts_are_not_delivery_files(self):
        from data_analysis.verify import package_files
        files = package_files(ROOT)
        self.assertEqual([p for p in files if p.endswith('.npz')], ['data/text/minilm_embeddings.npz'])
        self.assertFalse(any(p.endswith(('.joblib', '.pkl', '.pickle')) or '_booster' in p or '_matrix' in p for p in files))

    def test_no_archived_verification_or_fit_logs(self):
        for name in ('verification.json', 'fit_metadata.json.gz'):
            self.assertFalse((ROOT / 'results' / name).exists())

    def test_output_guard_and_no_overwrite(self):
        import tempfile
        from data_analysis.pipeline import fresh_output
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            with self.assertRaises(ValueError):
                fresh_output(root / 'data' / 'bad', root)
            with self.assertRaises(ValueError):
                fresh_output(root / 'runs', root)
            output = fresh_output(root / 'runs' / 'one', root)
            (output / 'sentinel').write_text('preserve')
            with self.assertRaises(FileExistsError):
                fresh_output(output, root)
            self.assertEqual((output / 'sentinel').read_text(), 'preserve')

    def test_manifest_detects_modification_and_extra_file(self):
        import json
        import tempfile
        from data_analysis.io import sha
        from data_analysis.verify import check_manifest
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            path = root / 'input.txt'
            path.write_text('original')
            (root / 'manifest.json').write_text(json.dumps({'files_sha256': {'input.txt': sha(path)}}))
            self.assertEqual(check_manifest(root), 1)
            path.write_text('changed')
            with self.assertRaises(ValueError):
                check_manifest(root)
            path.write_text('original')
            (root / 'unexpected.txt').write_text('unapproved')
            with self.assertRaises(ValueError):
                check_manifest(root)

if __name__ == '__main__':
    unittest.main()

class GeneratedDataContracts(unittest.TestCase):
    def test_package_inventory_excludes_local_data(self):
        import tempfile
        from data_analysis.verify import package_files
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            (root / 'local_data').mkdir()
            (root / 'local_data/raw.json').write_text('{}')
            (root / 'source.py').write_text('pass\n')
            self.assertEqual(package_files(root), ['source.py'])
