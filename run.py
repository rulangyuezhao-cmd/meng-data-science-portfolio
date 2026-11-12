"""Execute reproducible coursework adaptations from any working directory."""
import argparse
import json
import platform
import time
import traceback
from pathlib import Path
from projects import mie1624, aps1070, che1148, aps1052
from projects.common import ROOT

PROJECTS = {**mie1624.PROJECTS, **aps1070.PROJECTS, **che1148.PROJECTS, **aps1052.PROJECTS}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('project', nargs='?', default='list', choices=['list', 'all', *PROJECTS])
    args = parser.parse_args()
    if args.project == 'list':
        for name in PROJECTS: print(name)
        return 0
    names = list(PROJECTS) if args.project == 'all' else [args.project]
    records = []
    for name in names:
        start = time.perf_counter()
        try:
            PROJECTS[name]()
            record = {'project':name, 'status':'passed', 'seconds':round(time.perf_counter()-start, 2)}
        except Exception as exc:
            traceback.print_exc()
            record = {'project':name, 'status':'failed', 'error':str(exc)}
        records.append(record)
        print(json.dumps(record), flush=True)
    (ROOT/'generated').mkdir(exist_ok=True)
    (ROOT/'generated'/'validation.json').write_text(json.dumps({'python':platform.python_version(), 'platform':platform.system(), 'runs':records}, indent=2)+'\n', encoding='utf-8')
    return int(any(r['status']!='passed' for r in records))

if __name__ == '__main__':
    raise SystemExit(main())
