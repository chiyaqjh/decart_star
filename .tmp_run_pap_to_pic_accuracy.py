from pathlib import Path
import importlib.util

mod_path = Path('E:/decart/experiments/compare/pap_size_vs_n_benchmark.py')
spec = importlib.util.spec_from_file_location('pap_mod', str(mod_path))
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

bench = mod.PapSizeVsNBenchmark()
bench.results_dir = Path('E:/decart/experiments/results/pic_accuracy')
bench.results_dir.mkdir(parents=True, exist_ok=True)

n_values = mod.parse_n_values('8,16,24,32,48,64')
bench.run(
    n_values=n_values,
    n_mode='fixed',
    n_factor=4,
    n_fixed=256,
    sample_users=0,
    show_plot=False,
)
