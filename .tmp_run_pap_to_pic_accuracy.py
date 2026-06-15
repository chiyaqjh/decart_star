from pathlib import Path
import importlib.util
import io
from contextlib import redirect_stdout

mod_path = Path('E:/decart/experiments/compare/pap_size_vs_n_benchmark.py')
spec = importlib.util.spec_from_file_location('pap_mod', str(mod_path))
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

bench = mod.PapSizeVsNBenchmark()
bench.results_dir = Path('E:/decart/experiments/results/pic_new/size')
bench.results_dir.mkdir(parents=True, exist_ok=True)

n_values = mod.parse_n_values('8,16,32,64,128,256')

buffer = io.StringIO()
with redirect_stdout(buffer):
    bench.run(
        n_values=n_values,
        n_mode='fixed',
        n_factor=4,
        n_fixed=256,
        sample_users=0,
        show_plot=False,
    )

print('Summary (KB)')
print(f"{'n':>6} {'N':>6} {'DeCart':>12} {'DeCart*':>12} {'Ratio':>10}")
print('-' * 52)
for i, n in enumerate(bench.results['decart']['n']):
    N = bench.results['decart']['N'][i]
    d = bench.results['decart']['pap_avg'][i]
    s = bench.results['decart_star']['pap_avg'][i]
    ratio = d / s if s > 0 else 0.0
    print(f'{n:6d} {N:6d} {d:12.3f} {s:12.3f} {ratio:10.2f}x')
