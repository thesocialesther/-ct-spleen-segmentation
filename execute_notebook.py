"""Execute and save ct.ipynb with the isolated project Python kernel."""
import json
import os
import sys
import time
from pathlib import Path

root = Path(__file__).resolve().parent
os.chdir(root)
os.environ['MPLBACKEND'] = 'Agg'
os.environ['JUPYTER_PATH'] = str(root / '.jupyter')
os.environ['IPYTHONDIR'] = str(root / '.ipython')
os.environ['JUPYTER_RUNTIME_DIR'] = str(root / '.jupyter' / 'runtime')
kernel_dir = root / '.jupyter' / 'kernels' / 'ct-project'
kernel_dir.mkdir(parents=True, exist_ok=True)
(kernel_dir / 'kernel.json').write_text(json.dumps({
    'argv': [sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}'],
    'display_name': 'CT Imaging (.venv)', 'language': 'python'}), encoding='utf-8')

import nbformat
from nbclient import NotebookClient

path = root / 'ct.ipynb'
notebook = nbformat.read(path, as_version=4)
notebook.metadata.kernelspec = {'name':'ct-project', 'display_name':'CT Imaging (.venv)', 'language':'python'}
# Libraries were installed into this exact environment before execution.
for cell in notebook.cells:
    if cell.cell_type == 'code' and cell.source.startswith('%pip'):
        cell.source = "import importlib.metadata as package_metadata\nfor package in ['torch', 'monai', 'nibabel', 'matplotlib', 'pandas']:\n    print(package, package_metadata.version(package))"

started = time.perf_counter()
def save_progress(cell, cell_index, execute_reply=None, **kwargs):
    nbformat.write(notebook, path)
    print(f'Completed cell {cell_index + 1}/{len(notebook.cells)} at {(time.perf_counter()-started)/60:.1f} min', flush=True)

client = NotebookClient(notebook, timeout=None, kernel_name='ct-project',
    resources={'metadata': {'path':str(root)}}, on_cell_executed=save_progress)
try:
    client.execute()
finally:
    nbformat.write(notebook, path)
print('Notebook execution completed and saved.', flush=True)
