import logic.StyleTransfer as lg


def train():

    # trainer = lg.Trainer(train_size=100, batch_size=1, epochs=5)
    # trainer.train()
    style_transfer_1 = lg.StyleTransfer(width=500, height=500, content_image_path="../data/faca.jpg",
                                      style_image_path="../data/picasso.jpg", iterations=10, style_weight=10)

    style_transfer_1.transfer('../output/1.png')

    style_transfer_2 = lg.StyleTransfer(width=500, height=500, content_image_path="../data/faca.jpg",
                                      style_image_path="../data/wave.jpg", iterations=10, style_weight=10)

    style_transfer_2.transfer('../output/2.png')

    style_transfer_3 = lg.StyleTransfer(width=500, height=500, content_image_path="../data/faca.jpg",
                                      style_image_path="../data/van_gough.jpg", iterations=10, style_weight=10)

    style_transfer_3.transfer('../output/3.png')

    style_transfer_4 = lg.StyleTransfer(width=500, height=500, content_image_path="../data/faca.jpg",
                                        style_image_path="../data/block.jpg", iterations=10, style_weight=10)

    style_transfer_4.transfer('../output/4.png')

    style_transfer_5 = lg.StyleTransfer(width=500, height=500, content_image_path="../data/faca.jpg",
                                        style_image_path="../data/marilyn.jpg", iterations=10, style_weight=10)

    style_transfer_5.transfer('../output/5.png')


if __name__ == '__main__':
    train()
