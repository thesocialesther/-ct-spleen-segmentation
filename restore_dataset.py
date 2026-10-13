"""Reassemble Git LFS dataset parts and verify the original archive checksum."""
import hashlib
import json
from pathlib import Path

def restore_archive(root=None):
    root = Path(root or Path(__file__).resolve().parent).resolve()
    archive = root / 'Task09_Spleen.tar'
    if archive.exists():
        return archive
    folder = root / 'dataset_parts'
    manifest = json.loads((folder / 'manifest.json').read_text(encoding='utf-8'))
    temporary = root / 'Task09_Spleen.tar.assembling'
    total_hash, total_size = hashlib.sha256(), 0
    try:
        with temporary.open('wb') as output:
            for item in manifest['parts']:
                part = (folder / item['file']).resolve()
                if part.parent != folder.resolve():
                    raise ValueError('Invalid dataset part path')
                if not part.exists() or part.stat().st_size != item['size']:
                    raise ValueError(f'Missing or incomplete {part.name}; run git lfs pull first.')
                part_hash = hashlib.sha256()
                with part.open('rb') as source:
                    for data in iter(lambda: source.read(8*1024*1024), b''):
                        output.write(data)
                        part_hash.update(data)
                        total_hash.update(data)
                        total_size += len(data)
                if part_hash.hexdigest() != item['sha256']:
                    raise ValueError(f'Checksum mismatch: {part.name}')
                print('Verified', part.name, flush=True)
        if total_size != manifest['size'] or total_hash.hexdigest() != manifest['sha256']:
            raise ValueError('Reassembled archive does not match the original dataset')
        temporary.replace(archive)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return archive

if __name__ == '__main__':
    print('Dataset archive ready:', restore_archive())
