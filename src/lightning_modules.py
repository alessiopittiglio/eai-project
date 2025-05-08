import torch
import torch.nn as nn
import lightning as L
from torchmetrics import Accuracy
from models import ResNet18, Xception3D

class DeepfakeClassifier(L.LightningModule):
    def __init__(
            self, 
            model_name: str, 
            learning_rate: float,
            optimizer_name: str,
            use_scheduler: bool,
            **kwargs
        ):
        super().__init__()
        
        # Save arguments in hparams attribute
        # This allows to access them later in the code
        # and also to log them in the experiment tracker
        self.save_hyperparameters()

        self.test_step_outputs = [] 

        if self.hparams.model_name == "resnet18_single_frame":
            self.model = ResNet18.ResNet18SingleFrame(num_classes=2)
            self.criterion = nn.CrossEntropyLoss()
            self.num_classes = 2

            self.train_acc = Accuracy(task="multiclass", num_classes=2)
            self.val_acc = Accuracy(task="multiclass", num_classes=2)
            self.test_acc = Accuracy(task="multiclass", num_classes=2)
        
        elif self.hparams.model_name == "xception3d":
            # N.B: This model requires input with shape (B, C, T, H, W)
            self.model = Xception3D.Xception3DClassifier() # Output has 1 class
            self.criterion = nn.BCEWithLogitsLoss()
            self.num_classes = 1
            self.train_acc = Accuracy(task="binary")
            self.val_acc = Accuracy(task="binary")
            self.test_acc = Accuracy(task="binary")
        
        else:
            raise ValueError(f"Model {self.hparams.model_name} not exists")
    
    def forward(self, x):
        return self.model(x) 
    
    def _common_step(self, batch, batch_idx, stage):
        x, y = batch
        logits = self(x)

        if self.num_classes == 1:
            loss = self.criterion(logits.squeeze(1), y.float())
            preds = torch.sigmoid(logits).squeeze(1) # Shape [B]
            target = y.int()
        else:
            loss = self.criterion(logits, y)
            preds = torch.softmax(logits, dim=1) 
            target = y

        acc_metric = getattr(self, f"{stage}_acc")
        acc_metric.update(preds, target)
        self.log(f'{stage}_acc', acc_metric, on_step=(stage == 'train'), on_epoch=True, prog_bar=True, logger=True)
        self.log(f'{stage}_loss', loss, on_step=(stage == 'train'), on_epoch=True, prog_bar=True, logger=True)

        return loss

    def training_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, "train")
    
    def validation_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, "val")
    
    def test_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, "test")
    
    def configure_optimizers(self):
        if self.hparams.optimizer_name.lower() == "adam":
            optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.learning_rate)
        elif self.hparams.optimizer_name.lower() == "sgd":
            optimizer = torch.optim.SGD(self.parameters(), lr=self.hparams.learning_rate, momentum=self.hparams.momentum)
        else:
            raise ValueError(f"Optimizer {self.hparams.optimizer_name} not supported")
        
        if self.hparams.use_scheduler:
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.94)

            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val_loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        else:
            return optimizer
