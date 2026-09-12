import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from pebby.level_banks import LevelBanks,canonical_hash,JOB_FORMAT

ROOT=Path(__file__).resolve().parents[1]


def fixture_row():
    # Stored verified generated pilot, no generation/search/official input.
    path=ROOT/'data/ls20-reference-calibration-v1/train.jsonl'
    with path.open() as stream:
        return next(json.loads(line) for line in stream if json.loads(line)['difficulty']==1)


def write(path,value):path.write_text(json.dumps(value))
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


class BankTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        (self.root/'jobs').mkdir()
        source=self.root/'source.txt';source.write_text('fixed source')
        self.row=fixture_row(); low=self.row['seed']
        self.manifest={'config':{'profile':'ls20-reference-v1','train_count':7,'validation_count':7,
              'attempts':400,'limit':None,'train_seed_range':[low,low+1000],
              'validation_seed_range':[1000000,1001000]},'code_hashes':{str(source):sha(source)},
              'source_hashes':{str(source):sha(source)}}
        write(self.root/'manifest.json',self.manifest)
        self.job={'id':'train-000000','split':'train','seed':low,'mode':1,
                  'profile':'ls20-reference-v1','attempts':400,'limit':None}
        self.payload={'format':JOB_FORMAT,'binding':canonical_hash(self.manifest),'job':self.job,
                      'state':{'status':'accepted','next_seed':low+1,'failures':[],'row':self.row}}
        self.publish()
        self.catalogue=LevelBanks({'fixture':{'label':'Fixture','path':self.root}})

    def tearDown(self):self.temp.cleanup()
    def publish(self):
        write(self.root/'jobs/train-000000.json',{'payload':self.payload,'sha256':canonical_hash(self.payload)})

    def test_exact_stored_level_filter_pagination_and_incremental_cache(self):
        with patch('pebby.ls20.generate.generate_level',side_effect=AssertionError('no regeneration')):
            info=self.catalogue.catalogue()['banks'][0]
            self.assertEqual(info['status'],'partial');self.assertEqual(info['accepted']['train'],1)
            with patch.object(self.catalogue,'_job',wraps=self.catalogue._job) as reader:
                page=self.catalogue.levels('fixture',difficulty=1)
                self.assertEqual(len(page['levels']),1);self.assertEqual(reader.call_count,0)
                self.assertEqual(self.catalogue.levels('fixture',difficulty=2)['total'],0)
                self.assertEqual(self.catalogue.levels('fixture',offset=1)['levels'],[])
                row,metadata=self.catalogue.level('fixture','train-000000')
                self.assertEqual(reader.call_count,1)
            self.assertEqual(row,self.row);self.assertEqual(metadata['bank_status'],'partial')
        for options in ({'limit':101},{'offset':True},{'difficulty':8},{'split':'../'}):
            with self.assertRaises(ValueError):self.catalogue.levels('fixture',**options)
        with self.assertRaises(ValueError):self.catalogue.level('../../etc','train-000000')

    def test_tampering_and_source_change_fail_closed(self):
        self.catalogue.catalogue()
        self.payload['state']['row']['step_counter']=41
        # Even an updated envelope checksum cannot bypass profile validation.
        self.publish()
        with self.assertRaises(ValueError):self.catalogue.levels('fixture')
        self.payload['state']['row']['step_counter']=42;self.publish()
        (self.root/'source.txt').write_text('changed source')
        with self.assertRaises(ValueError):self.catalogue.catalogue()

    def test_binding_checksum_and_pending(self):
        self.payload['state']['status']='rejected';self.publish()
        self.assertEqual(self.catalogue.levels('fixture')['total'],0)
        self.payload['binding']='0'*64;self.publish()
        with self.assertRaises(ValueError):self.catalogue.catalogue()
        self.payload['binding']=canonical_hash(self.manifest);self.publish()
        value=json.loads((self.root/'jobs/train-000000.json').read_text());value['sha256']='0'*64
        write(self.root/'jobs/train-000000.json',value)
        with self.assertRaises(ValueError):self.catalogue.catalogue()

    def test_resealed_job_cannot_change_assigned_seed_block(self):
        self.payload['job']['seed']+=64
        self.payload['state']['next_seed']+=64
        self.publish()
        with self.assertRaises(ValueError):self.catalogue.catalogue()

    def test_false_complete_report_rejected(self):
        write(self.root/'generation-report.json',{'status':'complete','manifest_sha256':canonical_hash(self.manifest)})
        (self.root/'train.jsonl').write_text(json.dumps(self.row)+'\n')
        (self.root/'validation.jsonl').write_text('')
        with self.assertRaises(ValueError):self.catalogue.catalogue()

    def test_complete_publication_exact_match_and_corruption(self):
        path=ROOT/'data/ls20-reference-calibration-v1/validation.jsonl'
        with path.open() as stream:
            other=next(json.loads(line) for line in stream if json.loads(line)['difficulty']==1)
        self.manifest['config'].update(train_count=1,validation_count=1,
            validation_seed_range=[other['seed'],other['seed']+1000])
        write(self.root/'manifest.json',self.manifest)
        self.payload['binding']=canonical_hash(self.manifest);self.publish()
        val=copy.deepcopy(self.payload)
        val['job'].update(id='validation-000000',split='validation',seed=other['seed'])
        val['state'].update(row=other,next_seed=other['seed']+1)
        write(self.root/'jobs/validation-000000.json',{'payload':val,'sha256':canonical_hash(val)})
        banks={}
        for split,row in [('train',self.row),('validation',other)]:
            path=self.root/(split+'.jsonl');path.write_text(json.dumps(row)+'\n')
            banks[split]={'levels':1,'sha256':sha(path)}
        write(self.root/'generation-report.json',{'status':'complete',
            'manifest_sha256':canonical_hash(self.manifest),'banks':banks})
        self.assertEqual(self.catalogue.catalogue()['banks'][0]['status'],'complete')
        (self.root/'train.jsonl').write_text('{}\n')
        with self.assertRaises(ValueError):self.catalogue.catalogue()

    def test_persisted_index_lets_a_new_process_skip_revalidation(self):
        self.catalogue.catalogue()
        index=self.root/'row-index.json'
        self.assertTrue(index.is_file())
        # Never *.jsonl: the generator rglobs data/**/*.jsonl as seed inventory.
        self.assertEqual(index.suffix,'.json')
        fresh=LevelBanks({'fixture':{'label':'Fixture','path':self.root}})
        with patch('pebby.ls20.generate.generate_level',side_effect=AssertionError('no regeneration')), \
             patch.object(fresh,'_job',wraps=fresh._job) as reader:
            info=fresh.catalogue()['banks'][0]
            self.assertEqual(info['accepted']['train'],1)
            self.assertEqual(reader.call_count,0)
            self.assertEqual(fresh.levels('fixture',difficulty=1)['total'],1)
            self.assertEqual(reader.call_count,0)
            row,metadata=fresh.level('fixture','train-000000')
            self.assertEqual(reader.call_count,1)
        self.assertEqual(row,self.row);self.assertEqual(metadata['bank_status'],'partial')

    def test_persisted_index_does_not_mask_a_tampered_job(self):
        self.catalogue.catalogue()
        self.payload['state']['row']=copy.deepcopy(self.row)
        self.payload['state']['row']['step_counter']=41
        self.publish()
        fresh=LevelBanks({'fixture':{'label':'Fixture','path':self.root}})
        with self.assertRaises(ValueError):fresh.catalogue()

    def test_unusable_index_is_rebuilt_rather_than_trusted(self):
        self.catalogue.catalogue();index=self.root/'row-index.json'
        self.assertIsNone(self.catalogue._load_index(self.root,'a-different-binding'))
        forged={'binding':canonical_hash(self.manifest),'jobs':{},'summaries':{},'final':None}
        for damaged in ('not json at all',
                        json.dumps({'format':'pebby.bank-read-index.v1','sha256':'0'*64,'index':forged}),
                        json.dumps({'format':'wrong.format.v1','sha256':canonical_hash(forged),'index':forged})):
            index.write_text(damaged)
            fresh=LevelBanks({'fixture':{'label':'Fixture','path':self.root}})
            self.assertEqual(fresh.catalogue()['banks'][0]['accepted']['train'],1)
            self.assertNotEqual(index.read_text(),damaged)

    def test_index_survives_an_unwritable_bank_directory(self):
        self.root.chmod(0o555)
        try:
            fresh=LevelBanks({'fixture':{'label':'Fixture','path':self.root}})
            self.assertEqual(fresh.catalogue()['banks'][0]['accepted']['train'],1)
            self.assertFalse((self.root/'row-index.json').exists())
        finally:
            self.root.chmod(0o755)

    def test_hostai_schema_covers_bank_operations(self):
        from pebby.hostai_provider import interaction
        import jsonschema
        schema=interaction().input_schema
        for request in ({'op':'banks'}, {'op':'bank_levels','bank':'fixture','split':'train','limit':100},
                        {'op':'bank_level','bank':'fixture','id':'train-000000'},
                        {'op':'boot'}, {'op':'boot','limit':100}):
            jsonschema.validate(request,schema)
        for request in ({'op':'bank_levels','bank':'fixture','limit':101},
                        {'op':'bank_level','bank':'fixture'}, {'op':'banks','path':'/etc'},
                        {'op':'boot','limit':101}, {'op':'boot','bank':'fixture'}):
            with self.assertRaises(jsonschema.ValidationError):jsonschema.validate(request,schema)

    def test_engine_load_replays_exact_record_without_search(self):
        from inference import Engine
        engine=Engine(banks=self.catalogue)
        with patch('pebby.ls20.generate.generate_level',side_effect=AssertionError('no regeneration')), \
             patch('inference.planned',side_effect=AssertionError('no search')):
            result=engine.dispatch({'op':'bank_level','bank':'fixture','id':'train-000000'})
        self.assertEqual(result['level']['seed'],self.row['seed'])
        self.assertEqual(result['level']['walls'],self.row['walls'])
        self.assertEqual(len(result['frame']),64)
        self.assertIsInstance(result['status'],dict)
        self.assertEqual(result['bank_status'],'partial')

if __name__=='__main__':unittest.main()
