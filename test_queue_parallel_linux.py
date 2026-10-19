"""Coordinator integration with a live worker and two-card admission."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


@unittest.skipUnless(Path('/proc/self').exists(), 'Linux GPU lock integration')
class QueueParallel(unittest.TestCase):
    def test_adopt_then_launch_disjoint_gpu_groups_after_two_samples(self):
        import parallel_coordinator as queue
        from parallel_tasks import make_control_tasks
        gpus=[{'index':i,'uuid':f'g{i}','free_mib':free,'utilization':100,
               'process_pids':[123],'memory_mib':49140-free}
              for i,free in enumerate([15137,5489,21949,1505,9209,11282,16361,20505])]
        registry={key:{'min_free_gib':value} for key,value in
                  [('pixtral12b',34),('phi4vision15b',36),('smolvlm2_22b',12)]}
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'results/www').mkdir(parents=True);(root/'logs').mkdir()
            (root/'queue_tasks.json').write_text(json.dumps({'tasks':make_control_tasks([f'p{i}' for i in range(18)])}))
            for key in registry:
                (root/'models'/key).mkdir(parents=True)
                (root/'models'/key/'download_complete.json').write_text('{}')
                (root/'results/www'/f'{key}_processor_check.json').write_text('{"passed":true}')
            (root/'results/www/smolvlm2_22b_r24_processor_check.json').write_text('{"passed":true}')
            (root/'results/www/queue_state.json').write_text(json.dumps({
                'pixtral12b':{'status':'running','pid':99,'gpu':0,'gpu_uuid':'g0'},
                'smolvlm2_22b':{'status':'complete'}}))
            launched=[];ticks=[]
            def popen(command,**kwargs):
                launched.append((command,kwargs['env']))
                return SimpleNamespace(pid=100+len(launched),poll=lambda:None)
            def stop_after_two(t):
                ticks.append(t)
                if len(ticks)==2:raise RuntimeError('fixture_finished')
            import os
            previous=os.getcwd()
            try:
                with patch.object(queue,'ROOT',root),patch.object(queue,'load_registry',return_value=registry),\
                     patch.object(queue,'snapshot',return_value=gpus),\
                     patch.object(queue,'ExistingWorker',return_value=SimpleNamespace(poll=lambda:None)),\
                     patch.object(queue.subprocess,'run'),patch.object(queue.subprocess,'Popen',side_effect=popen),\
                     patch.object(queue.time,'sleep',side_effect=stop_after_two),\
                     patch('sys.argv',['wait_for_gpu.py','--gpu-policy','shared','--idle-samples','2']):
                    with self.assertRaisesRegex(RuntimeError,'fixture_finished'):queue.main()
            finally:os.chdir(previous)
            state=json.loads((root/'results/www/queue_state.json').read_text())
            self.assertEqual(state['pixtral12b']['pid'],99)
            self.assertEqual(len(launched),3)
            self.assertEqual(state['phi4vision15b']['gpu_indices'],[2,7])
            self.assertEqual({state['smol_blur24_g0']['gpu'],state['smol_blur24_g1']['gpu']},{5,6})
            cards=[g for key,job in state.items() if job.get('status')=='running'
                   for g in job.get('gpu_indices',[job['gpu']])]
            self.assertEqual(len(cards),len(set(cards)))
            self.assertEqual(len(json.loads((root/'results/www/per_card_tasks.json').read_text())),8)
            self.assertTrue(all(env['WWW_RESERVE_FREE_MIB']=='4096' for _,env in launched))


if __name__=='__main__':unittest.main()
