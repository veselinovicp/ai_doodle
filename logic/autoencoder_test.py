import unittest

import logic.autoencoder as lg


class TestMethods(unittest.TestCase):

    # def test_simple(self):
    #     trainer = lg.Trainer(train_size=2, epochs=5)
    #     trainer.predict_test_value("../output/weights.hdf5", image_path="../data/input_image3.png")
    #     # trainer.train()

    def test_autoencoder(self):
        autoencoder = lg.AutoEncoder(train_size=100, batch_size=5, input_images = '../data/michelangelo_2')

        autoencoder.train()



if __name__ == '__main__':
    unittest.main()
