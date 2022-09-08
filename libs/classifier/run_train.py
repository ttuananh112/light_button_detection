import pytorch_lightning as pl
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint

import libs.configs.infer as config_infer
from classifier.utils.dataset import ButtonDataset
from classifier.model.efficientnetv2 import effnetv2_tiny

if __name__ == "__main__":
    # data
    data_folder = "/home/anhtt163/PycharmProjects/outsource/dataset/phone/classifier/data"
    dataset = ButtonDataset(data_folder=data_folder)

    # try to over-fit first
    train_data = dataset
    val_data = dataset

    train_loader = DataLoader(train_data, batch_size=32)
    val_loader = DataLoader(val_data, batch_size=32)

    # model
    # model = ButtonClassifier()
    model = effnetv2_tiny(num_classes=config_infer.Recognition.NUM_CLASS)

    checkpoint_callback = ModelCheckpoint(
        monitor='val_f1',
        dirpath='weights/efficientnetv2_tiny_poc',
        filename='button-{epoch:02d}-{val_f1:.2f}',
        mode='max'
    )

    # training
    trainer = pl.Trainer(gpus=1, precision=32,
                         limit_train_batches=0.5, max_epochs=100,
                         callbacks=[checkpoint_callback])
    trainer.fit(model, train_loader, val_loader)
    print(checkpoint_callback.best_model_path)
