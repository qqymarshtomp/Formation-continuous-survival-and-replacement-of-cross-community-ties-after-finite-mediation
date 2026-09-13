"""Bounded four-process execution of the frozen experiment stages."""
import subprocess
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
(root / 'logs').mkdir(exist_ok=True)
for stage in ['baseline', 'shuffle', 'stress', 'numerics', 'futures']:
    processes = []
    streams = []
    for partition in range(4):
        stream = (root/'logs'/f'{stage}_{partition}.log').open('w')
        streams.append(stream)
        processes.append(subprocess.Popen([sys.executable, str(root/'scripts/run_revision.py'), stage, str(partition)],
                                          stdout=stream, stderr=subprocess.STDOUT))
    codes = [process.wait() for process in processes]
    for stream in streams:
        stream.close()
    if any(codes):
        raise RuntimeError(f'{stage} process exit codes: {codes}; see logs')
    print('COMPLETED STAGE', stage, flush=True)
