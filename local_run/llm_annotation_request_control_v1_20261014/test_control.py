import copy
from pathlib import Path
from types import SimpleNamespace
import unittest
import control as c


def fixture():
    binding = {'bytes': 20, 'sha256': 'a'*64}
    result = {'schema': 'third-llm-selection-sample-v1', 'mode': 'selection', 'passed': True,
              'stage': 'complete', 'production_sampling_performed': True, 'selected_frame_rows': 20996,
              'unordered_pair_count': 220405510, 'inference_activation': False, 'model_inference_calls': 0}
    review = {'schema': 'third-llm-selection-metadata-identity-review-v1', 'passed': True, 'selection_result': binding}
    return result, review, binding


class RequestControlTests(unittest.TestCase):
    def test_engineering_qualification_cannot_launch_requests(self):
        result, review, binding = fixture()
        c.check_result_review(result, review, binding)
        for field, value in [('schema', 'third-llm-selection-qualification-v1'), ('mode', 'qualification'),
                             ('production_sampling_performed', False), ('selected_frame_rows', 512),
                             ('unordered_pair_count', 130816), ('stage', 'partial'), ('passed', False)]:
            bad = dict(result, **{field: value})
            with self.assertRaises(ValueError): c.check_result_review(bad, review, binding)

    def test_review_must_bind_exact_result(self):
        result, review, binding = fixture()
        for change in [{'passed': False}, {'schema': 'a-different-review'},
                       {'selection_result': {'bytes': 20, 'sha256': 'b'*64}}]:
            with self.assertRaises(ValueError): c.check_result_review(result, dict(review, **change), binding)

    def test_prior_activation_or_calls_refused(self):
        result, review, binding = fixture()
        for change in [{'inference_activation': True}, {'model_inference_calls': 1}]:
            with self.assertRaises(ValueError): c.check_result_review(dict(result, **change), review, binding)

    def test_queue_reconciles_before_preparation(self):
        c.queue_clear('999|unrelated\n')
        for queue in ['13571|different-name\n', '999|third-llm-requests-selection\n',
                      '999|third-llm-sampler-selection\n', '999|third-llm-infer-qwen\n', 'bad-row']:
            with self.assertRaises(ValueError): c.queue_clear(queue)

    def test_command_keeps_evidence_source_assets_and_lock_distinct(self):
        args = SimpleNamespace(preparation_package_sha256='a'*64, selection_result_sha256='b'*64,
                               selection_review_sha256='c'*64)
        command = c.child_command(args)
        fields = dict(zip(command[3::2], command[4::2]))
        self.assertEqual(command[0], str(c.W/'envs/judge-qwen-v3-20261011/bin/python'))
        self.assertEqual(fields['--sampling-lock-sha256'], c.LOCK_SHA)
        self.assertEqual(fields['--selection-result'], str(c.CENSUS/'run/output/result.json'))
        self.assertEqual(fields['--repo'], str(c.REPO))
        self.assertNotEqual(fields['--repo'], fields['--protocol-evidence'])
        self.assertEqual(fields['--out'], str(c.ROOT/'run/output'))
        self.assertNotIn('--seed', command)
        self.assertNotIn('--activate', command)


if __name__ == '__main__':
    unittest.main()
