"""Execute each public course notebook; keep executed copies in generated/."""
import json
import time
from pathlib import Path
import nbformat
from nbclient import NotebookClient

ROOT=Path(__file__).resolve().parents[1]
records=[]
dest=ROOT/'generated/notebooks';dest.mkdir(parents=True,exist_ok=True)
for path in sorted((ROOT/'notebooks').glob('*.ipynb')):
    notebook=nbformat.read(path,as_version=4)
    start=time.perf_counter()
    NotebookClient(notebook,timeout=180,startup_timeout=60,kernel_name='python3',
                   resources={'metadata':{'path':str(ROOT)}}).execute()
    nbformat.write(notebook,dest/path.name)
    record={'notebook':path.name,'status':'passed','seconds':round(time.perf_counter()-start,2),
            'code_cells':sum(cell.cell_type=='code' for cell in notebook.cells)}
    records.append(record)
    print(json.dumps(record),flush=True)
(ROOT/'generated/notebook-validation.json').write_text(json.dumps(records,indent=2)+'\n',encoding='utf-8')
