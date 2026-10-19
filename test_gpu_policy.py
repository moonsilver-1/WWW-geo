"""Allocation rules for coexisting GPU jobs and memory-race deferral."""
import unittest

try:
    import gpu_policy
except ModuleNotFoundError:
    gpu_policy = None


class GpuPolicyTests(unittest.TestCase):
    def check(self, free, pids, utilization, policy, expected):
        self.assertIsNotNone(gpu_policy, 'Shared GPU eligibility is not implemented')
        gpu = {'free_mib': free, 'process_pids': pids, 'utilization': utilization,
               'memory_mib': 49140 - free}
        self.assertEqual(gpu_policy.eligible(gpu, policy, 30720), expected)

    def test_busy_gpu_with_headroom_can_run(self):
        self.check(35000, [100, 200], 100, 'shared', True)

    def test_busy_gpu_below_threshold_waits(self):
        self.check(29000, [100], 100, 'shared', False)

    def test_zero_utilization_does_not_override_memory(self):
        self.check(8000, [100], 0, 'shared', False)

    def test_launch_threshold_is_inclusive(self):
        self.check(30720, [100], 100, 'shared', True)

    def test_memory_drop_blocks_fresh_launch(self):
        self.check(30719, [100], 100, 'shared', False)

    def test_exclusive_still_requires_no_process(self):
        self.check(48900, [100], 0, 'exclusive', False)

    def test_clean_gpu_can_run_exclusively(self):
        self.check(48900, [], 0, 'exclusive', True)


if __name__ == '__main__':
    unittest.main()
