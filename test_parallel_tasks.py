"""Resource and archive contracts for parallel page tasks."""
import unittest

try:
    from parallel_tasks import candidate_groups, weight_budgets, make_control_tasks, output_stem, support_keys
except ModuleNotFoundError:
    candidate_groups = weight_budgets = make_control_tasks = output_stem = support_keys = None


class ParallelTasks(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(candidate_groups)
        self.gpus = [{'index': i, 'uuid': f'g{i}', 'free_mib': free, 'utilization': 100}
                     for i, free in enumerate([15137, 5489, 21949, 1505, 9209, 11282, 16361, 20505])]

    def test_phi_uses_two_busy_cards_with_individual_reserves(self):
        groups = candidate_groups(self.gpus, 36864, 4096, {'g0'}, allow_shard=True)
        self.assertEqual([[g['index'] for g in group] for group in groups], [[2, 7]])

    def test_single_card_is_preferred_when_available(self):
        self.gpus[6]['free_mib'] = 40000
        self.assertEqual(candidate_groups(self.gpus, 36864, 4096, {'g0'}, True)[0][0]['index'], 6)

    def test_memory_drop_rejects_two_card_plan(self):
        self.gpus[7]['free_mib'] = 15000
        self.gpus[6]['free_mib'] = 8000
        self.assertEqual(candidate_groups(self.gpus, 36864, 4096, {'g0'}, True), [])

    def test_three_cards_are_used_when_two_card_headroom_drops(self):
        self.gpus[7]['free_mib']=15000
        group=candidate_groups(self.gpus,36864,4096,{'g0'},True)[0]
        self.assertEqual({g['index'] for g in group},{2,6,7})

    def test_project_cards_are_reserved_as_a_group(self):
        self.assertEqual(candidate_groups(self.gpus, 36864, 4096, {'g0', 'g2'}, True), [])

    def test_weight_budgets_leave_workspace_and_vision_headroom(self):
        budgets = weight_budgets([self.gpus[2], self.gpus[7]], 4096)
        self.assertEqual(budgets, {0: 21949-4096-1536-2048, 1: 20505-4096-1536})
        self.assertGreater(sum(budgets.values()), 27*1024)

    def test_eight_page_tasks_are_disjoint_and_complete(self):
        pages = [f'p{i}' for i in range(18)]
        tasks = make_control_tasks(pages)
        self.assertEqual(len(tasks), 8)
        self.assertEqual(sorted(p for t in tasks for p in t['pages']), sorted(pages))
        self.assertEqual({t['preferred_gpu'] for t in tasks}, set(range(8)))
        self.assertEqual(len({t['task_id'] for t in tasks}), 8)

    def test_output_paths_are_separate_and_project_local(self):
        self.assertEqual(output_stem('smolvlm2_22b', 'smol_blur24_g6'), 'smol_blur24_g6')
        self.assertEqual(output_stem('pixtral12b', None), 'pixtral12b')
        with self.assertRaises(ValueError): output_stem('smolvlm2_22b', '../outside')

    def test_format_abstentions_are_successful_support_observations(self):
        rows = [{'page': 'p0', 'state': [-1]*6, 'parse_ok': False,
                 'A_j': -1., 'A_reference': -1.2}]
        self.assertEqual(support_keys(rows), {('p0', (-1,)*6)})
        rows.append(dict(rows[0]))
        with self.assertRaises(ValueError): support_keys(rows)


if __name__ == '__main__': unittest.main()
