import unittest
from executors import ClevrExecutor


class TestStringMethods(unittest.TestCase):

    def setUp(self):
        scene_json = 'val-scenes.json'
        vocab_json = 'vocab.json'
        self.executor = ClevrExecutor(scene_json, scene_json, vocab_json)

    def test_execute(self):
        # count, subtraction_set, filter_shape[cube], filter_color[cyan],
        # subtraction_set, filter_shape[sphere], filter_color[gray], scene, end
        x = [4, 17, 13, 7, 17, 15, 8, 16, 2]
        index = 0
        split = 'val'
        ans = self.executor.run(x,index,split)
        self.assertEquals(ans, '5')


if __name__ == '__main__':
    unittest.main()
