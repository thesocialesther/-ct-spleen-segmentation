import json
from pathlib import Path
import nbformat
root = Path(__file__).resolve().parent
run = root / 'artifacts' / 'cpu_baseline'
summary = json.loads((run / 'test_summary.json').read_text())
nb = nbformat.read(root / 'ct.ipynb', as_version=4)
heading = '## Measured results from this run'
nb.cells = [c for c in nb.cells if not (c.cell_type == 'markdown' and c.source.startswith(heading))]
nb.cells.append(nbformat.v4.new_markdown_cell(f'''{heading}

The notebook was executed with the project `.venv`: **30 CPU epochs**, 25 training volumes, 8 validation volumes and 8 held-out test volumes. All 41 labeled scans passed the scan/mask audit.

The best checkpoint was epoch **{summary['best_epoch']}**, chosen using validation Dice **0.231**. Held-out mean Dice was **{summary['mean_dice']:.3f}**, with standard deviation **{summary['std_dice']:.3f}** across **{summary['n_test']}** cases. This is a trained baseline with poor segmentation overlap. The displayed worst-case prediction includes substantial false-positive regions outside the spleen; successful training does not imply accurate organ outlining.

Weights, patient splits, configuration, audit, curves, per-case scores and a resampled prediction mask are saved in `artifacts/cpu_baseline`. To improve this baseline, investigate larger training context, finer voxel spacing and longer training using validation data; keep test cases separate.
'''))
nbformat.validate(nb)
assert not [o for c in nb.cells if c.cell_type == 'code' for o in c.outputs if o.output_type == 'error']
nbformat.write(nb, root / 'ct.ipynb')
print('Saved measured results; notebook schema passes and no cells contain errors.')
