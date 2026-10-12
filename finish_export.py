"""Repair the prediction export without repeating model training."""
import json, os, sys
from pathlib import Path
root = Path(__file__).resolve().parent
os.chdir(root)
os.environ['JUPYTER_PATH'] = str(root / '.jupyter')
os.environ['IPYTHONDIR'] = str(root / '.ipython')
os.environ['JUPYTER_RUNTIME_DIR'] = str(root / '.jupyter' / 'runtime')
import nbformat
from nbclient import NotebookClient
path = root / 'ct.ipynb'
original = nbformat.read(path, as_version=4)
original.cells[21].source = original.cells[21].source.replace(
    "sample = test_ds[next(i for i, f in enumerate(test_files) if f['case'] == worst)]",
    "# Reload transforms directly: cached tensors may omit affine metadata.\nsample = eval_transform(test_files[next(i for i, f in enumerate(test_files) if f['case'] == worst)])")
sources = [original.cells[3].source, original.cells[5].source,
    "splits = json.loads((RUN / 'splits.json').read_text())\ntrain_files = splits['train']\nval_files = splits['validation']\ntest_files = splits['test']",
    original.cells[13].source.split('patch =')[0], original.cells[15].source,
    "checkpoint = torch.load(RUN / 'best_model.pt', map_location=device, weights_only=True)\nmodel.load_state_dict(checkpoint['model'])\ntest_results = pd.read_csv(RUN / 'test_metrics.csv')",
    original.cells[9].source.split('example =')[0], original.cells[21].source]
mini = nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell(s) for s in sources])
NotebookClient(mini, timeout=None, kernel_name='ct-project', resources={'metadata':{'path':str(root)}}).execute()
original.cells[21].outputs = mini.cells[-1].outputs
original.cells[21].execution_count = 11
nbformat.write(original, path)
print('Prediction export repaired; notebook contains no cell errors.')
