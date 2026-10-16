"""Read-only export of completed calibration evidence; stdout is private."""
import base64
import json
import common as c


def main():
    files = {}
    for name, path in c.REMOTE_FILES.items():
        bound = c.binding(path)
        c.require(bound['bytes'] <= 2 << 20, 'export_file_cap')
        raw = path.read_bytes(); c.require(c.byte_binding(raw) == bound, 'export_read_race')
        files[name] = {**bound, 'base64': base64.b64encode(raw).decode()}
    print(json.dumps(files))


if __name__ == '__main__':
    main()
