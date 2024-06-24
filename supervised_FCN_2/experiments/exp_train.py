import torchmetrics
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import accuracy_score

from supervised_FCN_2.models.fcn import FCNBaseline
from supervised_FCN_2.experiments.exp_base import *
from supervised_FCN_2.utils import *


class ExpFCN(pl.LightningModule):
    def __init__(self,
                 config: dict,
                 n_train_samples: int,
                 n_classes: int,
                 ):
        super().__init__()
        self.config = config
        in_channels = config['dataset']['in_channels']

        self.fcn = FCNBaseline(in_channels, n_classes)
        self.criterion = torch.nn.CrossEntropyLoss()

    def training_step(self, batch, batch_idx):
        x, y = batch
        x = x.float()
        y = y.squeeze()

        yhat = self.fcn(x)  # (b n_classes)
        loss = self.criterion(yhat, y)

        # lr scheduler
        sch = self.lr_schedulers()
        sch.step()

        # log
        acc = accuracy_score(y.flatten().detach().cpu().numpy(),
                             yhat.argmax(dim=-1).flatten().cpu().detach().numpy())
        loss_hist = {'loss': loss, 'acc': acc}

        for k, v in loss_hist.items():
            self.log(f'train/{k}', v)

        return loss_hist

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = x.float()
        y = y.squeeze()

        yhat = self.fcn(x)  # (b n_classes)
        loss = self.criterion(yhat, y)

        # log
        acc = accuracy_score(y.flatten().detach().cpu().numpy(),
                             yhat.argmax(dim=-1).flatten().cpu().detach().numpy())
        loss_hist = {'loss': loss, 'acc': acc}

        for k, v in loss_hist.items():
            self.log(f'val/{k}', v)

        return loss_hist

    def configure_optimizers(self):
        opt = torch.optim.AdamW([{'params': self.parameters(), 'lr': self.config['exp_params']['LR']}], )
        T_max = self.config['trainer_params']['max_steps']
        return {'optimizer': opt, 'lr_scheduler': CosineAnnealingLR(opt, T_max, eta_min=0.000001)}

    def test_step(self, batch, batch_idx):
        x, y = batch
        x = x.float()
        y = y.squeeze()

        yhat = self.fcn(x)  # (b n_classes)
        loss = self.criterion(yhat, y)

        # log
        acc = accuracy_score(y.flatten().detach().cpu().numpy(),
                             yhat.argmax(dim=-1).flatten().cpu().detach().numpy())
        loss_hist = {'loss': loss, 'acc': acc}

        for k, v in loss_hist.items():
            self.log(f'val/{k}', v)

        return loss_hist
