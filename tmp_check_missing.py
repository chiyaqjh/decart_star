import json, glob, sys
from pathlib import Path
folder, model, record = sys.argv[1], sys.argv[2], int(sys.argv[3])
for fp in sorted(glob.glob(str(Path('experiments/results') / folder / '*.json')), reverse=True):
    try:
        with open(fp, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception:
        continue
    cfg = data.get('config', {})
    block = data.get('models', {}).get(model, {})
    if cfg.get('N') == 10000 and cfg.get('n') == 128 and cfg.get('num_records') == record and cfg.get('record_dim') == record and cfg.get('policy_size') == 32 and cfg.get('num_runs') == 1 and (block.get('query_times') or block.get('encrypt_times') or block.get('decrypt_times')):
        print('present')
        sys.exit(0)
print('missing')
