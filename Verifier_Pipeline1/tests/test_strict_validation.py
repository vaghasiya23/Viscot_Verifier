import ast
import importlib.util
import json
from pathlib import Path
import sys
import types
import tempfile
from PIL import Image
import unittest
from unittest.mock import patch

from generation.skeleton_gen import build_scaffold, canonical_code
from filters import coverage_filter, outcome_filter
from verification_utils import answer_consensus, deduplicate_boxes, select_position, annotated_regions

ROOT = Path(__file__).resolve().parents[1]
SAMPLES = json.loads((ROOT / 'data/raw/gqa_300_samples.json').read_text())


class StrictValidationTests(unittest.TestCase):
    def test_role_order(self):
        code = canonical_code(build_scaffold(SAMPLES[1]['reasoning'], SAMPLES[1]))
        self.assertIn("vlm_relation(img_path, reference, c, 'hanging on', 'cups', 'furniture')", code)
        right = canonical_code(build_scaffold(SAMPLES[0]['reasoning'], SAMPLES[0]))
        self.assertIn("check_spatial_relation(c, reference, 'right')", right)

    def test_target_not_in_program(self):
        sample = dict(SAMPLES[0], answer='SECRET_TARGET')
        self.assertNotIn('SECRET_TARGET', canonical_code(build_scaffold(sample['reasoning'], sample)))

    def test_all_supported_programs_compile(self):
        for sample in SAMPLES:
            scaffold = build_scaffold(sample['reasoning'], sample)
            ast.parse(canonical_code(scaffold))
            if not any(s['needs_review'] for s in scaffold):
                self.assertTrue(coverage_filter.check(canonical_code(scaffold), scaffold)['passed'])

    def test_unknown_entity_and_operations_deferred(self):
        self.assertFalse(any(x['needs_review'] for x in build_scaffold(SAMPLES[3]['reasoning'], SAMPLES[3])))
        scaffold = build_scaffold([{'operation':'choose name','argument':'dog|cat','dependencies':[]}])
        self.assertFalse(coverage_filter.check('final_answer="dog"', scaffold)['passed'])

    def test_mutations_cannot_pass(self):
        scaffold = build_scaffold(SAMPLES[0]['reasoning'], SAMPLES[0])
        code = canonical_code(scaffold)
        mutations = [code.replace("'right'", "'near'"), code+'\nfinal_answer="sofa"',
                     code.replace('assert len(e0) > 0', 'assert True'),
                     code.replace('c, reference', 'reference, c'),
                     code.replace("final_answer = e2", 'final_answer = "sofa"'),
                     'if False:\n'+''.join('    '+line+'\n' for line in code.splitlines()),
                     code.replace('if any(', 'if True or any(')]
        for mutated in mutations:
            self.assertFalse(coverage_filter.check(mutated, scaffold)['passed'])
        self.assertTrue(coverage_filter.check('# comment\n'+code, scaffold)['passed'])

    def test_answer_matching_and_unproven_invalid(self):
        for a,b in [('', 'bird'), ('bird','animal'), ('person','children'), ('no','not'), ('shelf','cabinet')]:
            self.assertFalse(outcome_filter.are_answers_compatible(a,b))
        self.assertTrue(outcome_filter.are_answers_compatible('Couch.', 'sofa'))
        result={'ok':True,'local_scope':{'final_answer':'sofa'},'evidence':[]}
        self.assertEqual(outcome_filter.check(result, {'answer':'sofa'}, 'INVALID')['status'],'failed')
        self.assertEqual(outcome_filter.check(result, {'answer':'sofa'}, 'VALID')['status'],'failed')

    def test_runtime_evidence_and_failure(self):
        registry = {'detect_and_crop':lambda *a:[[0,0,10,10]],
                    'check_spatial_relation':lambda *a: True,
                    'vlm_query':lambda *a: 'sofa', 'vlm_related_name':lambda *a:'sofa', 'annotated_regions':lambda *a:[[0,0,10,10]], 'answer_consensus':answer_consensus}
        spec=importlib.util.spec_from_file_location('isolated_executor',ROOT/'filters/execution_filter.py')
        module=importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {'tools':types.SimpleNamespace(TOOL_REGISTRY=registry)}):
            spec.loader.exec_module(module)
        code=canonical_code(build_scaffold(SAMPLES[0]['reasoning'],SAMPLES[0]))
        result=module.run(code,'test.jpg')
        self.assertTrue(result['ok'])
        self.assertEqual(len(result['evidence']),5)
        self.assertEqual(outcome_filter.check(result, SAMPLES[0], 'VALID')['status'],'valid')
        registry['detect_and_crop']=lambda *a: []
        result=module.run(code,'test.jpg')
        self.assertFalse(result['ok'])
        self.assertEqual(outcome_filter.check(result,SAMPLES[0],'VALID')['status'],'failed')

    def test_visual_pair_binding_and_uncertainty(self):
        fake_loader = types.SimpleNamespace(from_pretrained=lambda *a, **k: None)
        fake_transformers = types.SimpleNamespace(Qwen2VLForConditionalGeneration=fake_loader,
                                                  AutoProcessor=fake_loader)
        spec = importlib.util.spec_from_file_location('isolated_probe', ROOT/'tools/probe.py')
        probe = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {'transformers': fake_transformers}):
            spec.loader.exec_module(probe)
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp)/'image.png')
            Image.new('RGB', (50, 50), 'white').save(path)
            with patch.object(probe, '_ask', return_value='yes') as ask:
                self.assertTrue(probe.vlm_relation(path, [2,2,10,10], [20,20,35,35],
                                                   'hanging on', 'cups', 'cabinet'))
                marked, question = ask.call_args.args
                self.assertEqual(marked.getpixel((2,2)), (255,0,0))
                self.assertEqual(marked.getpixel((20,20)), (0,0,255))
                self.assertIn('cups in the RED box hanging on the cabinet in the BLUE box', question)
            with patch.object(probe, '_ask', return_value='probably yes'):
                with self.assertRaises(probe.UncertainVisualEvidence):
                    probe.vlm_probe(path, [2,2,10,10], 'Is this red?')
            self.assertFalse(probe.vlm_relation(path, [2,2,10,10], [2,2,10,10], 'on', 'cup', 'cup'))

    def test_multiple_objects_require_consensus(self):
        self.assertEqual(answer_consensus(['couch', 'sofa']), 'couch')
        with self.assertRaises(AssertionError):
            answer_consensus(['sofa', 'table'])
        with self.assertRaises(AssertionError):
            answer_consensus([])

    def test_duplicate_suppression_and_position(self):
        boxes = [[1,1,20,20], [2,2,21,21], [70,1,90,20]]
        self.assertEqual(deduplicate_boxes(boxes), [boxes[0], boxes[2]])
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp)/'test.png')
            Image.new('RGB', (100,50)).save(path)
            self.assertEqual(select_position(path, boxes, 'horizontal', 'right'), [boxes[2]])

    def test_multiple_references_can_verify_relation(self):
        code=canonical_code(build_scaffold(SAMPLES[0]['reasoning'], SAMPLES[0]))
        references=[[20,0,30,10], [0,0,10,10]]
        sofa=[15,0,20,10]
        scope={'img_path':'test', 'detect_and_crop':lambda p,label: references if label=='chair' else [sofa],
               'check_spatial_relation':lambda a,b,r:a[0]>b[0],
               'vlm_query':lambda *a:'sofa', 'vlm_related_name':lambda *a:'sofa', 'annotated_regions':lambda *a:[sofa], 'answer_consensus':answer_consensus}
        exec(code,scope,scope)
        self.assertEqual(scope['final_answer'], 'sofa')
        self.assertEqual(scope['e1'], [sofa])

    def test_annotation_is_localization_not_target_answer(self):
        sample = dict(SAMPLES[3], answer='SECRET_TARGET')
        code = canonical_code(build_scaffold(sample['reasoning'], sample))
        self.assertIn('annotated_regions', code)
        self.assertIn('vlm_query', code)
        self.assertIn('vlm_relation(img_path, reference, c', code)
        self.assertNotIn('SECRET_TARGET', code)
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp)/'test.png')
            Image.new('RGB',(100,100)).save(path)
            self.assertEqual(annotated_regions(path,[[1,2,10,20]]), [[1,2,10,20]])
            with self.assertRaises(ValueError):
                annotated_regions(path,[[1,2,150,20]])


if __name__ == '__main__':
    unittest.main()
