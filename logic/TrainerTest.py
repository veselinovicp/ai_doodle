import unittest

import logic.Trainer as lg


class TestMethods(unittest.TestCase):

    def test_simple(self):
        trainer = lg.Trainer(train_size=10)
        trainer.predict_test_value("../output/weights.10.hdf5", image_path="../data/input_image2.png")
        # trainer.train()


if __name__ == '__main__':
    unittest.main()
