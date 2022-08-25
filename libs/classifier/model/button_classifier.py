import torch
from torch import nn
import pytorch_lightning as pl
import torchmetrics as tm

import classifier.configs as cfg


class ButtonClassifier(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.num_class = cfg.NUM_CLASS
        self._create_model()

    def _create_model(self):
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(in_features=128, out_features=64)
        self.fc2 = nn.Linear(in_features=64, out_features=self.num_class)
        self.softmax = nn.Softmax(dim=-1)
        # loss function
        self.loss_func = nn.CrossEntropyLoss()
        # metrics
        self.train_f1 = tm.F1Score()
        self.val_f1 = tm.F1Score()

    def forward(self, x):
        x = self.relu(self.conv3(self.relu(self.conv2(self.relu(self.conv1(x))))))
        x = torch.mean(x.view(x.size(0), x.size(1), -1), dim=2)  # mean by feature_map
        x = self.relu(self.fc1(x))
        x = self.softmax(self.fc2(x))
        return x

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        pred = self.forward(x)
        loss = self.loss_func(pred, y)
        self.train_f1.update(pred, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        pred = self.forward(x)
        loss = self.loss_func(pred, y)
        self.val_f1.update(pred, y)
        self.log('val_loss', loss)

    def training_epoch_end(self, outs):
        f1 = self.train_f1.compute()
        print("train_f1", f1)
        self.log('train_f1', f1)
        self.train_f1.reset()

    def validation_epoch_end(self, outs):
        f1 = self.val_f1.compute()
        print("val_f1", f1)
        self.log('val_f1', f1)
        self.val_f1.reset()


if __name__ == "__main__":
    cls = ButtonClassifier()
    inp = torch.rand((1, 3, cfg.IMG_SIZE, cfg.IMG_SIZE))
    out = cls(inp)
    print(out)
