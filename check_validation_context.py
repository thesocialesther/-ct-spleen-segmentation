"""Compare inference window sizes on validation data only."""
import json
from pathlib import Path
nb = json.loads(Path('ct.ipynb').read_text(encoding='utf-8'))
namespace = {}
imports = ''.join(nb['cells'][3]['source'])
imports = imports.replace("get_ipython().run_line_magic('matplotlib', 'inline')", '')
exec(imports, namespace)
exec(''.join(nb['cells'][5]['source']), namespace)
splits = json.loads((namespace['RUN'] / 'splits.json').read_text())
namespace.update(train_files=splits['train'], val_files=splits['validation'], test_files=splits['test'])
transforms = ''.join(nb['cells'][13]['source']).split('patch =')[0]
exec(transforms, namespace)
exec(''.join(nb['cells'][15]['source']), namespace)
torch = namespace['torch']
checkpoint = torch.load(namespace['RUN'] / 'best_model.pt', map_location='cpu', weights_only=True)
namespace['model'].load_state_dict(checkpoint['model'])
loader = namespace['DataLoader'](namespace['val_ds'][:2], batch_size=1, num_workers=0)
results = []
for roi in [(48, 48, 48), (96, 96, 96)]:
    namespace['INFER_ROI'] = roi
    frame = namespace['evaluate'](loader)
    results.append({'roi':roi, 'checkpoint_epoch':checkpoint['epoch'], 'mean_dice':float(frame.dice.mean()),
                    'cases':frame.to_dict('records')})
    print(results[-1], flush=True)
(namespace['RUN'] / 'validation_window_comparison.json').write_text(json.dumps(results, indent=2))
