"""Run any saved feature experiment through one interactive or CLI entry."""
import argparse
import atexit
import faulthandler
import importlib
import json
import os
from pathlib import Path
import sys

from utils.sop_config import discover_experiments, load_config, validate_config
from utils.sop_reporting import update_comparisons

ROOT = Path(__file__).resolve().parent
EXIT_WATCHDOG_SECONDS = 120


def select_groups(groups, selection):
    names = list(groups)
    if selection.lower() == 'all':
        return names
    chosen = []
    for token in selection.split(','):
        token = token.strip()
        token = 'F2.1' if token == 'F1.2' else token  # Accept the former command spelling.
        if token.isdigit() and 1 <= int(token) <= len(names):
            matches = [names[int(token) - 1]]
        else:
            matches = [name for name in names if name == token or name.split('_')[0] == token]
        if len(matches) != 1:
            raise ValueError('Unknown or ambiguous group: ' + token)
        if matches[0] not in chosen:
            chosen.append(matches[0])
    return chosen


def prepare_runtime(configs, method):
    """Load PyTorch before fitting or plotting, and check requested devices."""
    uses_supcon = method in ('supcon', 'all')
    uses_resnet = any(
        component['extractor'].startswith('resnet')
        for config in configs
        for feature in config['feature_sets'].values()
        for component in feature.get('components', [feature])
    )
    if not uses_supcon and not uses_resnet:
        return
    # Set this before any CUDA context is created, including ResNet extraction.
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    try:
        torch = importlib.import_module('torch')
        if uses_resnet:
            importlib.import_module('torchvision')
        from utils.sop_feature_bank import choose_device
        devices = set()
        for config in configs:
            if uses_supcon:
                devices.add(choose_device(config['supcon']['device'], torch))
            if any(component['extractor'].startswith('resnet')
                   for feature in config['feature_sets'].values()
                   for component in feature.get('components', [feature])):
                devices.add(choose_device(config['extraction']['device'], torch))
        # CUDA detection alone does not check that kernels can execute.
        for device in sorted(devices):
            probe = torch.ones((2, 2), device=device)
            if (probe @ probe).sum().item() != 8.0:
                raise RuntimeError('Unexpected tensor calculation on ' + device)
        print('PyTorch ready:', torch.__version__, '/ devices:', ', '.join(sorted(devices)),
              '/ Python:', sys.executable, flush=True)
    except (ImportError, OSError, RuntimeError) as error:
        raise RuntimeError(
            'PyTorch/torchvision initialization failed before experiment execution. '
            'Python: ' + sys.executable + '. Original error: ' + str(error)
        ) from error


def arm_exit_watchdog(run_directory, seconds=EXIT_WATCHDOG_SECONDS):
    """Record all thread stacks if the process has not exited `seconds` after the work.

    Called only after every requested run and comparison has been saved. A normal
    shutdown removes the file; it remains only when the interpreter is still waiting
    (for example joining threads at exit) and contains the stacks for diagnosis.
    """
    run_directory = Path(run_directory)
    if not run_directory.is_dir():
        return None
    path = run_directory / 'exit_watchdog_traceback.txt'
    stream = path.open('w', encoding='utf-8')
    stream.write(f'All requested runs had finished. Stack traces below mean the process '
                 f'had not exited {seconds} s later; saved results are unaffected.\n')
    stream.flush()
    faulthandler.dump_traceback_later(seconds, repeat=False, file=stream, exit=False)

    def disarm():
        faulthandler.cancel_dump_traceback_later()
        stream.close()
        if 'File "' not in path.read_text(encoding='utf-8', errors='replace'):
            path.unlink(missing_ok=True)
    atexit.register(disarm)
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'config.json')
    parser.add_argument('--group', help='Group name, F1,F2, menu numbers, or all.')
    parser.add_argument('--method', choices=('baselines', 'supcon', 'all'))
    parser.add_argument('--run-name')
    parser.add_argument('--list', action='store_true', help='List groups without running.')
    parser.add_argument('--compare-only', action='store_true', help='Update comparisons without training.')
    parser.add_argument('--smoke-test', action='store_true')
    parser.add_argument('--all-train', action='store_true')
    parser.add_argument('--skip-official-test', action='store_true')
    for name in ('data-dir', 'results-dir', 'protocol-dir'):
        parser.add_argument('--' + name, type=Path)
    args = parser.parse_args(argv)
    common = json.loads(args.config.read_text(encoding='utf-8'))
    directory = Path(common.get('experiment_dir', 'experiment_configs'))
    if not directory.is_absolute():
        directory = args.config.resolve().parent / directory
    groups = discover_experiments(directory)
    if not groups:
        parser.error('No experiment configurations found in ' + str(directory))
    for number, (name, (_, group)) in enumerate(groups.items(), 1):
        print(f'{number}. {name}: {group.get("description", name)}')
    if args.list:
        return
    if args.compare_only:
        config = load_config(args.config, args.smoke_test, next(iter(groups.values()))[0])
        config['run_name'] = args.run_name or config['run_name']
        if args.results_dir:
            config['results_dir'] = str(args.results_dir.resolve())
        update_comparisons(Path(config['results_dir']) / config['run_name'])
        return
    try:
        selected = select_groups(groups, args.group or input('Select groups (comma separated) or all: '))
    except ValueError as error:
        parser.error(str(error))
    method = args.method
    if method is None:
        print('1. Logistic Regression + SVM\n2. SupCon\n3. All three models')
        method = {'1': 'baselines', '2': 'supcon', '3': 'all'}.get(input('Select models: ').strip())
        if method is None:
            parser.error('Select 1, 2 or 3.')
    run_name = args.run_name
    if run_name is None and args.group is None:
        run_name = input(f'Run name [{common["run_name"]}]: ').strip() or common['run_name']
    configs = []
    for name in selected:
        config = load_config(args.config, args.smoke_test, groups[name][0])
        if run_name:
            config['run_name'] = run_name
        for field in ('data_dir', 'results_dir', 'protocol_dir'):
            if getattr(args, field):
                config[field] = str(getattr(args, field).resolve())
        if args.all_train:
            config['use_all_official_train'] = True
        if args.skip_official_test:
            config['run_official_test'] = False
        validate_config(config)
        configs.append(config)
    prepare_runtime(configs, method)
    from utils.sop_experiment import Experiment
    from utils.sop_reporting import combine_results, print_tables
    for name, config in zip(selected, configs):
        print('Running:', name, '/', method, '/', config['mode'])
        experiment = Experiment(config)
        for family in (('baselines', 'supcon') if method == 'all' else (method,)):
            if family == 'baselines':
                from utils.sop_baselines import run_experiment
            else:
                from utils.sop_supcon import run_experiment
            output = run_experiment(experiment)
            print_tables(output)
            combine_results(experiment)
        update_comparisons(Path(config['results_dir']) / config['run_name'])
    run_directory = Path(configs[-1]['results_dir']) / configs[-1]['run_name']
    print('All requested runs finished. Results:', run_directory, flush=True)
    watchdog = arm_exit_watchdog(run_directory)
    if watchdog is not None:
        print(f'If this process has not exited after {EXIT_WATCHDOG_SECONDS} s, thread stacks are saved to:',
              watchdog, flush=True)


if __name__ == '__main__':
    main()
