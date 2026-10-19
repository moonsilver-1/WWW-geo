"""One coordinator for single-card, sharded-model and disjoint page tasks."""
import argparse,fcntl,json,os,subprocess,sys,time
from pathlib import Path
from gpu_policy import snapshot
from model_registry import load_registry
from worker_handle import ExistingWorker
from parallel_tasks import candidate_groups,weight_budgets,output_stem
ROOT=Path(__file__).resolve().parent

def task_specs(registry):
    specs={key:{'task_id':key,'model':key,'allow_shard':key=='phi4vision15b'} for key in registry}
    plan=ROOT/'queue_tasks.json'
    if plan.exists():
        for task in json.loads(plan.read_text())['tasks']:
            key=output_stem(task['model'],task['task_id'])
            assert key not in specs and task['model'] in registry and task['pages']
            specs[key]=task
    return specs

def devices(job):
    return job.get('gpu_indices',[job['gpu']]),job.get('gpu_uuids',[job['gpu_uuid']])

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--poll-seconds',type=int,default=10)
    ap.add_argument('--idle-samples',type=int,default=2)
    ap.add_argument('--gpu-policy',choices=['exclusive','shared'],default='shared')
    ap.add_argument('--min-free-gib',type=float,default=30)
    ap.add_argument('--reserve-gib',type=float,default=4)
    args=ap.parse_args();assert args.min_free_gib>args.reserve_gib>=0
    os.chdir(ROOT);(ROOT/'locks').mkdir(exist_ok=True);(ROOT/'logs').mkdir(exist_ok=True)
    coordinator=(ROOT/'locks/queue.lock').open('w')
    try:fcntl.flock(coordinator,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:print('A project queue is already running',flush=True);return
    path=ROOT/'results/www/queue_state.json'
    state=json.loads(path.read_text()) if path.exists() else {}
    running,stable={},{}
    for key,job in state.items():
        if job.get('status')!='running':continue
        process=ExistingWorker(ROOT,key,job);code=process.poll()
        if code is not None:job['status']='complete' if code==0 else 'interrupted';continue
        locks=[]
        for index in devices(job)[0]:
            lock=(ROOT/'locks'/f'gpu_{index}.lock').open('w')
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(lock)
        running[key]={'process':process,'locks':locks,'log':(ROOT/'logs'/f'{key}.log').open('a')}
        print(f'Adopted {key} PID{job["pid"]} on GPUs{devices(job)[0]}',flush=True)
    reserve=int(args.reserve_gib*1024)
    try:
        while True:
            registry=load_registry();specs=task_specs(registry);observed=snapshot()
            for key,item in list(running.items()):
                code=item['process'].poll()
                if code is None:continue
                state[key].update(status='complete' if code==0 else 'failed',exit_code=code)
                if code in (75,76):
                    state[key]['attempts']-=1;state[key]['status']='waiting_memory'
                    if code==76:state[key]['required_free_mib']+=4096
                item['log'].close()
                for lock in item['locks']:lock.close()
                del running[key]
            unfinished=[key for key in specs if key not in running and state.get(key,{}).get('status')!='complete' and state.get(key,{}).get('attempts',0)<2]
            def processor_check(key):
                task=specs[key];radius=task.get('neutral_radius',12)
                suffix='' if radius==12 else f'_r{radius}'
                return ROOT/'results/www'/f'{task["model"]}{suffix}_processor_check.json'
            pending=[key for key in unfinished if (ROOT/'models'/specs[key]['model']/'download_complete.json').exists() and processor_check(key).exists() and json.loads(processor_check(key).read_text()).get('passed')]
            def requirement(key):
                task=specs[key];value=task.get('min_free_gib',registry[task['model']].get('min_free_gib',args.min_free_gib))
                return max(int(value*1024),state.get(key,{}).get('required_free_mib',0))
            def occupied():return {uuid for key in running for uuid in devices(state[key])[1]}
            def groups(key,gpus):
                task=specs[key]
                usable=gpus if args.gpu_policy=='shared' else [g for g in gpus if not g['process_pids'] and g['memory_mib']<=512 and g['utilization']<=3]
                return candidate_groups(usable,requirement(key),reserve,occupied(),task.get('allow_shard',False),task.get('preferred_gpu'))
            for key in pending:
                valid={(key,tuple(g['uuid'] for g in group)) for group in groups(key,observed)}
                for token in list(stable):
                    if token[0]==key and token not in valid:stable[token]=0
                for token in valid:stable[token]=stable.get(token,0)+1
            for key in pending:
                choices=[group for group in groups(key,observed) if stable.get((key,tuple(g['uuid'] for g in group)),0)>=args.idle_samples]
                if not choices:continue
                group=choices[0];locks=[]
                try:
                    for gpu in sorted(group,key=lambda g:g['index']):
                        lock=(ROOT/'locks'/f'gpu_{gpu["index"]}.lock').open('w')
                        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(lock)
                    uuids={g['uuid'] for g in group};fresh=snapshot()
                    checked=next((item for item in groups(key,fresh) if {g['uuid'] for g in item}==uuids),None)
                    if checked is None:
                        for lock in locks:lock.close()
                        continue
                    group=checked
                except BlockingIOError:
                    for lock in locks:lock.close()
                    continue
                with (ROOT/'logs'/f'nvitop_before_{key}.txt').open('w') as capture:
                    subprocess.run([sys.executable,'-m','nvitop','--once','--ascii','--readonly'],stdout=capture,check=True,timeout=30)
                task=specs[key];indices=[g['index'] for g in group];uuid_list=[g['uuid'] for g in group]
                env=dict(os.environ,CUDA_DEVICE_ORDER='PCI_BUS_ID',CUDA_VISIBLE_DEVICES=','.join(uuid_list),TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='4',WWW_GPU_POLICY=args.gpu_policy,WWW_MIN_FREE_MIB=str(requirement(key)),WWW_RESERVE_FREE_MIB=str(reserve))
                env.pop('WWW_MAX_MEMORY_MIB',None)
                if len(group)>1:env['WWW_MAX_MEMORY_MIB']=json.dumps(weight_budgets(group,reserve))
                command=['python','-u','run_web_eval.py','--model',task['model'],'--device',','.join(f'cuda:{i}' for i in indices),'--workers','4']
                if key!=task['model']:command+=['--task-id',key,'--pages',','.join(task['pages']),'--neutral-radius',str(task.get('neutral_radius',12))]
                log=(ROOT/'logs'/f'{key}.log').open('a');process=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
                state[key]={'status':'running','pid':process.pid,'gpu':indices[0],'gpu_uuid':uuid_list[0],'gpu_indices':indices,'gpu_uuids':uuid_list,'command':' '.join(command),'model':task['model'],'task':task,'gpu_policy':args.gpu_policy,'required_free_mib':requirement(key),'prelaunch_free_mib':group[0]['free_mib'],'prelaunch_gpus':group,'max_memory_mib':json.loads(env.get('WWW_MAX_MEMORY_MIB','{}')),'reserve_mib':reserve,'attempts':state.get(key,{}).get('attempts',0)+1,'started_at_unix':time.time()}
                running[key]={'process':process,'locks':locks,'log':log};print(f'Launched {key} PID{process.pid} on GPUs{indices}',flush=True)
            tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(state,indent=2)+'\n');tmp.replace(path)
            (ROOT/'results/www/gpu_status.json').write_text(json.dumps({'jobs':state,'gpus':observed,'gpu_policy':args.gpu_policy,'observed_at_unix':time.time()},indent=2)+'\n')
            (ROOT/'results/www/per_card_tasks.json').write_text(json.dumps({str(g['index']):{'free_mib':g['free_mib'],'running':[key for key in running if g['index'] in devices(state[key])[0]],'queued':[key for key in unfinished if specs[key].get('preferred_gpu')==g['index']]} for g in observed},indent=2)+'\n')
            print('GPU check: '+', '.join(f"{g['index']}={g['free_mib']}MiB free" for g in observed),flush=True)
            if not unfinished and not running:break
            time.sleep(args.poll_seconds)
    finally:
        for item in running.values():
            item['log'].close()
            for lock in item['locks']:lock.close()
        coordinator.close()

if __name__=='__main__':main()
