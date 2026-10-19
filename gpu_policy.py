"""Driver snapshots and eligibility for exclusive or memory-based GPU use."""
import csv
import subprocess


def snapshot():
    text = subprocess.check_output(
        ['nvidia-smi', '--query-gpu=index,uuid,memory.used,memory.free,memory.total,utilization.gpu',
         '--format=csv,noheader,nounits'], text=True, timeout=20)
    processes = subprocess.check_output(
        ['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader,nounits'],
        text=True, timeout=20)
    pids = {}
    for row in csv.reader(processes.splitlines()):
        if row:
            pids.setdefault(row[0].strip(), []).append(int(row[1]))
    return [{'index': int(row[0]), 'uuid': row[1].strip(), 'memory_mib': int(row[2]),
             'free_mib': int(row[3]), 'total_mib': int(row[4]), 'utilization': int(row[5]),
             'process_pids': pids.get(row[1].strip(), [])}
            for row in csv.reader(text.splitlines()) if row]


def eligible(gpu, policy, required_mib):
    if policy == 'shared':
        return gpu['free_mib'] >= required_mib
    if policy == 'exclusive':
        return (not gpu['process_pids'] and gpu['memory_mib'] <= 512
                and gpu['utilization'] <= 3 and gpu['free_mib'] >= required_mib)
    raise ValueError(f'Unknown GPU policy: {policy}')
