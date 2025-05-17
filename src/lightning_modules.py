import torch
import torch.nn as nn
import lightning as L
from torchmetrics import Accuracy, AUROC
from models import ResNet18, Xception3D, VideoTransformer
from transformers import get_linear_schedule_with_warmup

class DeepfakeClassifier(L.LightningModule):
    def __init__(
            self, 
            model_name: str, 
            num_classes: int,
            model_params: dict,
            criterion_name: str,
            criterion_params: dict,
            optimizer_name: str,
            optimizer_params: dict,
            use_scheduler: bool,
            scheduler_name: str,
            scheduler_params: dict,
            accuracy_task: str,
            accuracy_task_params: dict,
            scheduler_total_steps: int = None,
            **kwargs
        ):
        super().__init__()
        
        # Save arguments in hparams attribute
        # This allows to access them later in the code
        # and also to log them in the experiment tracker
        self.save_hyperparameters()

        self.test_step_outputs = [] 

        if self.hparams.model_name == "resnet18_single_frame":
            self.model = ResNet18.ResNet18SingleFrame(**self.hparams.model_params)
        elif self.hparams.model_name == "xception3d":
            # N.B: This model requires input with shape (B, C, T, H, W)
            self.model = Xception3D.Xception3DClassifier(**self.hparams.model_params)
        elif self.hparams.model_name == "VideoTransformer":
            # N.B: This model requires input with shape (B, C, T, H, W)
            self.model = VideoTransformer.VideoTransformer(**self.hparams.model_params)
        else:
            raise ValueError(f"Model {self.hparams.model_name} not exists")

        if self.hparams.criterion_name.lower() == "bce_logit_loss":
            self.criterion = nn.BCEWithLogitsLoss(**self.hparams.criterion_params)
        elif self.hparams.criterion_name.lower() == "cross_entropy":
            if 'weight' in self.hparams.criterion_params:
                weight = torch.tensor(self.hparams.criterion_params['weight'])
                self.hparams.criterion_params['weight'] = weight
            self.criterion = nn.CrossEntropyLoss(**self.hparams.criterion_params)
        else:
            raise ValueError(f"Criterion {self.hparams.criterion_name} not implemdented")
        
        self.train_acc = Accuracy(task=self.hparams.accuracy_task, **self.hparams.accuracy_task_params)
        self.val_acc = Accuracy(task=self.hparams.accuracy_task, **self.hparams.accuracy_task_params)
        self.test_acc = Accuracy(task=self.hparams.accuracy_task, **self.hparams.accuracy_task_params)

        self.train_auc = AUROC(task=self.hparams.accuracy_task, **self.hparams.accuracy_task_params)
        self.val_auc = AUROC(task=self.hparams.accuracy_task, **self.hparams.accuracy_task_params)
        self.test_auc = AUROC(task=self.hparams.accuracy_task, **self.hparams.accuracy_task_params)
        
    
    def forward(self, x):
        return self.model(x) 
    
    def _common_step(self, batch, batch_idx, stage):
        x, y = batch
        logits = self(x)

        if self.hparams.num_classes == 1:
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

        auc_metric = getattr(self, f"{stage}_auc")
        auc_metric.update(preds, target)
        self.log(f'{stage}_auc', auc_metric, on_step=(stage == 'train'), on_epoch=True, prog_bar=True, logger=True)

        return loss

    def training_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, "train")
    
    def validation_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, "val")
    
    def test_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, "test")
    
    def configure_optimizers(self):
        optimizers = {
            "adam": torch.optim.Adam,
            "adamw": torch.optim.AdamW,
            "sgd": torch.optim.SGD,
        }
        
        optimizer_name = self.hparams.optimizer_name.lower()
        if optimizer_name not in optimizers:
            raise ValueError(f"Optimizer {optimizer_name} not supported")
        
        optimizer_class = optimizers[optimizer_name]
        optimizer = optimizer_class(self.parameters(), **self.hparams.optimizer_params)
        
        if self.hparams.use_scheduler:
            if self.hparams.scheduler_name.lower() == "step":
                scheduler = torch.optim.lr_scheduler.StepLR(optimizer, **self.hparams.scheduler_params)
            if self.hparams.scheduler_name.lower() == "warmup":
                scheduler = get_linear_schedule_with_warmup(
                    optimizer,
                    num_training_steps=self.hparams.scheduler_total_steps,
                    **self.hparams.scheduler_params
                )
            
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
