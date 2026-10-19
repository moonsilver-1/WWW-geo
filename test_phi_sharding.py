import unittest
from phi_sharding import device_map


class PhiPlacement(unittest.TestCase):
    def sizes(self):
        sizes = {'model.embed_tokens': 980*2**20, 'model.mm_projector': 62*2**20,
                 'model.vision_tower': 817*2**20, 'model.norm': 10240, 'lm_head': 980*2**20}
        sizes.update({f'model.layers.{i}': 650*2**20+20480 for i in range(40)})
        return sizes

    def test_two_card_checkpoint_fits_without_offload(self):
        mapping, used = device_map(self.sizes(), {'0':14269, '1':14873})
        self.assertEqual(sum(mapping[f'model.layers.{i}']==0 for i in range(40)), 19)
        self.assertEqual(set(mapping.values()), {0,1})
        self.assertLess(used[0], 14269)
        self.assertLess(used[1], 14873)

    def test_insufficient_weight_capacity_is_requeued(self):
        with self.assertRaisesRegex(ValueError, 'GPU'):
            device_map(self.sizes(), {0:13000, 1:13000})

    def test_unknown_weights_fail_instead_of_implicit_offload(self):
        sizes=self.sizes();sizes['unknown.weight']=100
        with self.assertRaisesRegex(ValueError, 'layout'):
            device_map(sizes, {0:20000,1:20000})


if __name__=='__main__':unittest.main()
