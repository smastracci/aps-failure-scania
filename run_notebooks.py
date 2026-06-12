import nbformat
from nbclient import NotebookClient
from pathlib import Path
import datetime

def log(msg):
    ts = datetime.datetime.now().strftime('%H:%M:%S')
    print(f'[{ts}] {msg}', flush=True)

for name in ['01-eda-preprocessing', '02-modelling', '03-evaluation', '04-new-models']:
    path = Path('notebooks') / f'{name}.ipynb'
    nb = nbformat.read(path, as_version=4)
    log(f'Starting {name}...')
    NotebookClient(nb, timeout=28800, kernel_name='python3').execute()
    nbformat.write(nb, path)
    log(f'Done: {name}')

log('All notebooks complete.')
