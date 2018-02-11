import unittest

import logic.StyleTransfer as lg


class TestMethods(unittest.TestCase):

    # def test_simple(self):
    #     trainer = lg.Trainer(train_size=2, epochs=5)
    #     trainer.predict_test_value("../output/weights.hdf5", image_path="../data/input_image3.png")
    #     # trainer.train()

    def test_style_transfer(self):
        style_transfer = lg.StyleTransfer(width=200, height=200, content_image_path="../data/faca.jpg", style_image_path="../data/Satyrs_HeadBrown_InkLouvre.jpg", iterations=5)

        style_transfer.transfer('../output/1.png')
        # trainer.train()


if __name__ == '__main__':
    unittest.main()
