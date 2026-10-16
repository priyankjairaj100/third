"""Create one immutable private bundle only after terminal positive calibration."""
import argparse
import base64
import datetime
import os
from pathlib import Path
import admission
import common as c


def prepare(controller, exported_path, output):
    c.require(not os.path.lexists(output), 'occupied_admission_no_overwrite')
    c.require(not (controller/'stopped.json').exists(), 'controller_stopped')
    # Missing completion fails before reading an export or creating any output.
    completed = (controller/'completion.json').read_bytes()
    done = c.strict(completed)
    c.require(done['selection_pipeline_complete'] is True and done['selection_passed'] is True
              and done['validation_run'] is True, 'terminal_positive_calibration_required')
    raw = {'controller_completion.json': completed,
           'selection_review.json': (controller/'selection_completion.json').read_bytes(),
           'validation_review.json': (controller/'independent_validation_review.json').read_bytes()}
    exported = c.strict(exported_path.read_bytes())
    c.require(set(exported) == set(c.REMOTE_FILES), 'exact_export_set')
    for name, item in exported.items():
        c.require(item['bytes'] <= 2 << 20, 'export_cap')
        raw[name] = base64.b64decode(item['base64'], validate=True)
        c.require(c.byte_binding(raw[name]) == {k: item[k] for k in ('bytes', 'sha256')}, 'export_binding')
    bindings = {name: c.byte_binding(value) for name, value in raw.items()}
    data = {name: [c.strict(line) for line in value.splitlines()] if name.endswith('.jsonl') else c.strict(value)
            for name, value in raw.items()}
    reviewed = admission.check_data(data, bindings)
    manifest = {'schema': 'third-full-development-guard-admission-v1',
                'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                'files': bindings, 'review': reviewed, 'human_annotation_calls': 0,
                'source_text_or_task_labels_included': False, 'automatic_launch': False}
    os.umask(0o077); output.mkdir(mode=0o700, exist_ok=False)
    for name, value in raw.items():
        c.write_bytes(output/name, value, cap=32 << 20)
    c.publish(output/'admission.json', manifest)
    pin = c.binding(output/'admission.json')['sha256']
    admission.check_bundle(output, pin)
    return {'prepared': True, 'admission_sha256': pin, 'review': reviewed}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('controller', 'export', 'output'):
        parser.add_argument('--'+name, required=True, type=Path)
    a = parser.parse_args()
    print(c.canonical(prepare(a.controller.resolve(), a.export.resolve(), a.output.resolve())).decode())
