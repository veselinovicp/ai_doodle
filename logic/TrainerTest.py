import unittest

import logic.Trainer as lg


class TestMethods(unittest.TestCase):

    # def test_simple(self):
    #     trainer = lg.Trainer(train_size=2, epochs=5)
    #     trainer.predict_test_value("../output/weights.hdf5", image_path="../data/input_image3.png")
    #     # trainer.train()

    def test_vgg(self):
        trainer = lg.VGGTrainer(train_size=2, epochs=5, batch_size=2)
        trainer.predict_test_value("../output/weights.hdf5", image_path="../data/input_image3.png")
        # trainer.train()


if __name__ == '__main__':
    unittest.main()
