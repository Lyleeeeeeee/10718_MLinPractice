import csv
import gzip
import json
import tempfile
from pathlib import Path
from unittest.mock import patch
from data_pipeline import preparation
import math
import unittest
from features.numeric import article_features, days, historical_bound, shift
from data_pipeline.preparation import build_panel, metadata_activity, main, ROOT, fresh_output


class NumericPipelineTests(unittest.TestCase):
    def inputs(self, target='2024-09', current_rate=10):
        cut=shift(target,-1)
        volumes={shift(cut,i):{'A':10*days(shift(cut,i))} for i in range(-6,1)}
        volumes[cut]['A']=current_rate*days(cut)
        records={'1':[dict(revid=1,parentid=0,timestamp=shift(target,-3)+'-01T00:00:00Z',size=100,sha1='x',minor=False)]}
        coverage={'1':[dict(start=shift(target,-2)+'-01T00:00:00Z',end_exclusive=target+'-01T00:00:00Z')]}
        return {'A':1},volumes,{},records,coverage

    def test_twelve_raw_features(self):
        _,v,p,_,_=self.inputs()
        p={'A':{f'2024-08-{d:02d}':100 for d in range(1,32)}}
        f,u=article_features('A','2024-08',v,p)
        self.assertEqual(len(f),12)
        self.assertEqual(u,450)
        for i in [0,1,2,3,5]:self.assertAlmostEqual(f[i],math.log1p(10))
        self.assertEqual(f[4],0);self.assertEqual(f[6],1)
        self.assertAlmostEqual(f[7],math.log1p(100));self.assertEqual(f[8:11],[0,0,0])
        self.assertAlmostEqual(f[11],math.log1p(300)-math.log1p(450))

    def test_unknown_and_fixed_not_loose_pool(self):
        args=self.inputs(current_rate=14)
        panel,evaluation,labels=build_panel(*args,start='2024-09',end='2024-09')
        row=panel['rows'][0]
        self.assertTrue(row['evaluation_eligible_asof'])
        self.assertEqual(row['U_count'],450)
        self.assertEqual(row['label_U_count'],400)
        self.assertFalse(labels[row['key']]['known'])
        self.assertIsNone(labels[row['key']]['event'])
        self.assertEqual(evaluation['2024-09'],[row])

    def test_future_data_cannot_change_features(self):
        args=self.inputs();first=build_panel(*args,start='2024-09',end='2024-09')
        args[1]['2024-09']={'A':9999};args[2]['A']={'2024-09-01':999999}
        args[3]['1'].append(dict(revid=2,parentid=1,timestamp='2024-09-01T00:00:00Z',size=9999,sha1='z',minor=False))
        second=build_panel(*args,start='2024-09',end='2024-09')
        self.assertEqual(first[0]['rows'],second[0]['rows'])
        self.assertTrue(second[2]['2024-09:1']['event'])

    def test_metadata_gates_and_parent_chain(self):
        args=self.inputs();r=args[3]['1'];c=args[4]['1']
        r.append(dict(revid=2,parentid=1,timestamp='2024-08-01T00:00:00Z',size=700,sha1='y',minor=False))
        activity=metadata_activity(r,c,'2024-09')
        self.assertEqual(activity['max_abs_nonminor'],600)
        panel,ev,_=build_panel(*args,start='2024-09',end='2024-09')
        self.assertEqual(len(panel['rows']),1);self.assertEqual(ev['2024-09'],[])
        r[-1]['parentid']=100
        self.assertFalse(metadata_activity(r,c,'2024-09')['coverage_complete'])
        self.assertEqual(build_panel(*args,start='2024-09',end='2024-09')[0]['rows'],[])

    def test_cli_raw_fixture_and_fresh_output_guard(self):
        args=self.inputs()
        with tempfile.TemporaryDirectory(dir=ROOT / 'runs', prefix='numeric_test_') as tmp:
            base=Path(tmp); incoming=base/'incoming';incoming.mkdir()
            (base/'mapping.csv').write_text('original_title,article,pageid\nA,A,1\n')
            for month,counts in args[1].items():
                with gzip.open(incoming/f'incoming_{month}.csv.gz','wt') as f:
                    f.write(f'month,source,destination,count\n{month},other-search,A,{counts["A"]}\n')
            (base/'pv.csv').write_text('destination,date,views\n')
            (base/'meta.json').write_text(json.dumps(args[3]))
            (base/'coverage.json').write_text(json.dumps(args[4]))
            output=base/'data'
            command=['--mapping',str(base/'mapping.csv'),'--incoming',str(incoming),
                '--pageviews',str(base/'pv.csv'),'--metadata',str(base/'meta.json'),
                '--coverage',str(base/'coverage.json'),'--output',str(output),
                '--start','2024-09','--end','2024-09']
            (base/'vintages.json').write_text(json.dumps({m:{'metadata':'meta.json','coverage':'coverage.json'} for m in ['2024-08','2024-09']}))
            command.extend(['--metadata-by-target',str(base/'vintages.json')])
            with patch.object(preparation, 'read_json', wraps=preparation.read_json) as reads:
                main(command)
                loaded=[Path(call.args[0]).resolve() for call in reads.call_args_list]
                self.assertEqual(loaded.count((base/'meta.json').resolve()),1)
                self.assertEqual(loaded.count((base/'coverage.json').resolve()),1)
            provenance=json.loads((output/'provenance.json').read_text())
            names=[entry['name'] for entry in provenance['input_files']]
            self.assertEqual(names.count('meta.json'),1)
            self.assertEqual(names.count('coverage.json'),1)
            with gzip.open(output/'evaluation_labels.json.gz','rt') as f:
                self.assertFalse(json.load(f)['2024-09:1']['known'])
            self.assertNotIn(str(base),(output/'provenance.json').read_text())
            with self.assertRaises(FileExistsError):main(command)
            with self.assertRaises(ValueError):fresh_output(ROOT/'data'/'new')

    def test_target_specific_metadata_vintage(self):
        args=self.inputs()
        panel,_,_=build_panel(*args,start='2024-09',end='2024-09',
            metadata_by_target={'2024-09':({}, {})})
        self.assertEqual(panel['rows'],[])

    def test_october_training_excluded_and_identity_mismatch(self):
        args=self.inputs('2026-10')
        panel,ev,_=build_panel(*args,start='2026-10',end='2026-10')
        self.assertEqual(panel['rows'],[]);self.assertEqual(len(ev['2026-10']),1)
        _,ev,_=build_panel(*args,start='2026-10',end='2026-10',identity={'1':{'expected_title':'A','returned_title':'B'}})
        self.assertEqual(ev['2026-10'],[])

    def test_october_excluded_identity_even_when_titles_match(self):
        args=self.inputs('2026-10')
        _,ev,_=build_panel(*args,start='2026-10',end='2026-10',identity={'1':{
            'expected_title':'A','returned_title':'A','status':'excluded','ns':1}})
        self.assertEqual(ev['2026-10'],[])

if __name__=='__main__':unittest.main()
