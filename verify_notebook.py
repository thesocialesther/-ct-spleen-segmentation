"""Check notebook syntax and archived labels without installing ML libraries."""
import ast
import gzip
import json
import struct
import tarfile
from pathlib import Path
import numpy as np

nb = json.loads(Path('ct.ipynb').read_text(encoding='utf-8'))
count = 0
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = ''.join(cell['source'])
        if source.startswith('%pip'): continue
        ast.parse(source)
        count += 1

def read_nifti(archive, name, voxels=False):
    with gzip.GzipFile(fileobj=archive.extractfile(name)) as stream:
        header = stream.read(348)
        endian = '<' if struct.unpack('<i', header[:4])[0] == 348 else '>'
        dims = struct.unpack(endian + '8h', header[40:56])
        shape = dims[1:1+dims[0]]
        offset = int(struct.unpack(endian + 'f', header[108:112])[0])
        datatype = struct.unpack(endian + 'h', header[70:72])[0]
        result = {'shape': shape, 'datatype': datatype}
        if voxels:
            formats = {2:'u1', 4:'i2', 8:'i4', 16:'f4', 64:'f8', 256:'i1', 512:'u2', 768:'u4'}
            stream.read(offset - 348)
            data = np.frombuffer(stream.read(), dtype=np.dtype(endian + formats[datatype]))
            assert data.size == np.prod(shape), f'{name}: truncated voxel data'
            slope, intercept = struct.unpack(endian + '2f', header[112:120])
            if slope != 0 and np.isfinite(slope): data = data * slope + intercept
            assert np.isfinite(data).all(), f'{name}: nonfinite mask'
            assert np.isin(data, [0, 1]).all(), f'{name}: nonbinary mask'
            assert (data == 1).any(), f'{name}: empty mask'
            result['spleen_voxels'] = int((data == 1).sum())
        return result

with tarfile.open('Task09_Spleen.tar') as archive:
    names = archive.getnames()
    def index(folder):
        return {Path(n).name:n for n in names if n.startswith('Task09_Spleen/' + folder + '/')
                and n.endswith('.nii.gz') and not Path(n).name.startswith('._')}
    images, labels, test = index('imagesTr'), index('labelsTr'), index('imagesTs')
    missing, orphan = sorted(images.keys()-labels.keys()), sorted(labels.keys()-images.keys())
    assert not missing and not orphan, (missing, orphan)
    audited = []
    for name in sorted(images):
        image = read_nifti(archive, images[name])
        label = read_nifti(archive, labels[name], voxels=True)
        assert len(image['shape']) == 3 and image['shape'] == label['shape'], name
        audited.append({'case':name, **label})

report = {'syntax_checked_code_cells':count, 'training_scans':len(images), 'masks':len(labels),
    'unlabeled_test_scans':len(test), 'missing_labels':missing, 'orphan_labels':orphan,
    'archived_mask_checks':'All masks readable, binary, nonempty, and shape-matched.',
    'runtime_training':'Not executed: torch and MONAI unavailable; PyTorch installation failed.',
    'note':'Notebook performs additional affine and full image checks when executed.', 'cases':audited}
summary_path = Path('artifacts/cpu_baseline/test_summary.json')
if summary_path.exists():
    report['runtime_training'] = 'Completed: see executed ct.ipynb and artifacts/cpu_baseline.'
    report['test_summary'] = json.loads(summary_path.read_text())
    report['note'] = 'The executed notebook also checks image finiteness and affine alignment.'
Path('notebook_verification.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
print({k:v for k,v in report.items() if k != 'cases'})
