import json, sys, glob, os
def check(dir_path, label):
    files = sorted(glob.glob(os.path.join(dir_path, '*.json')), key=os.path.getmtime, reverse=True)
    if not files:
        print(f'{label}: No JSON found')
        return
    with open(files[0], 'r', encoding='utf-8') as f:
        data = json.load(f)
    comm = data.get('models', {}).get('decision_tree', {}).get('comm_check_sizes')
    if comm is None:
        print(f'{label}: comm_check_sizes is missing')
    else:
        first = comm[0] if len(comm) > 0 else 'empty'
        print(f'{label} comm_check_sizes: {comm} (First: {first})')

if len(sys.argv) > 1:
    if sys.argv[1] == 'secpq': check('experiments/results/tmp_check_comm_validate_secpq', 'SecPQ')
    elif sys.argv[1] == 'naive': check('experiments/results/tmp_check_comm_validate_naive', 'Naive')
