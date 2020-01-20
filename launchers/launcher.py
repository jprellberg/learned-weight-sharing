import itertools
import subprocess
import sys


def config_product(**kwargs):
    keys = list(kwargs.keys())
    vals = list(kwargs.values())
    vals = [x if isinstance(x, list) else [x] for x in vals]
    for item in itertools.product(*vals):
        yield dict(zip(keys, item))


def launch_call(calls, id):
    call = list(calls)[id]
    print("Launching:", ' '.join(call))
    subprocess.run(call)


def list_calls(calls):
    for i, call in enumerate(calls):
        print(i, ' '.join(call), sep='\t')


def to_call(cfg):
    call = ['python3.6', '-u', 'run_training.py']
    for k, v in cfg.items():
        k = k.replace('_', '-')
        call += [f'--{k}', str(v)]
    return call


def launcher(configs):
    calls = [to_call(cfg) for cfg in configs]
    if len(sys.argv) > 1:
        id = int(sys.argv[1])
        launch_call(calls, id)
    else:
        print("Expected task ID as argument")
        list_calls(calls)
