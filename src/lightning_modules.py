import torch
import torch.nn as nn
import lightning as L
from torchmetrics import Accuracy, AUROC
from models import ResNet18, Xception3D, VideoTransformer
from transformers import AutoModelForImageClassification, get_linear_schedule_with_warmup

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
            elif self.hparams.scheduler_name.lower() == "plateau":
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **self.hparams.scheduler_params)
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

from transformers import AutoModelForImageClassification
import pytorch_lightning as pl
import torchmetrics
from torchmetrics.classification import BinaryAUROC, Accuracy
import torch.nn as nn
import torch.nn.functional as F

class DeepFakeFinetuningLightningModule(pl.LightningModule):
    def __init__(self, cfg, class_counts):
        super().__init__()
        cfg["class_counts"] = class_counts  # Store for later use
        self.save_hyperparameters(cfg)

        # 1) Load pretrained model
        self.model = AutoModelForImageClassification.from_pretrained(
            cfg["model_name"], trust_remote_code=True
        )

        # 2) Replace classification head with a 2‐class head
        # We assume the backbone's final feature size is in config.hidden_size
        hidden_size = self.model.model.head.in_features
        self.model.model.head = nn.Linear(hidden_size, cfg["num_classes"])
        # (If using another architecture, adjust accordingly.)

        # 3) Prepare class weights for weighted cross‐entropy
        # class_counts is a dict: {idx_real: count_real, idx_fake: count_fake}
        idx_real = list(class_counts.keys())[0]
        idx_fake = list(class_counts.keys())[1]
        count_real = class_counts[idx_real]
        count_fake = class_counts[idx_fake]
        total = count_real + count_fake
        # weight[c] = total/(num_classes * count[c])
        w_real = total / (cfg["num_classes"] * count_real)
        w_fake = total / (cfg["num_classes"] * count_fake)
        # But we need to map them to [weight_for_label0, weight_for_label1]
        weights = torch.zeros(cfg["num_classes"], dtype=torch.float)
        weights[idx_real] = w_real
        weights[idx_fake] = w_fake
        self.register_buffer("class_weights", weights)

        # 4) Metrics
        self.train_acc = Accuracy(task="multiclass", num_classes=cfg["num_classes"])
        self.val_acc = Accuracy(task="multiclass", num_classes=cfg["num_classes"])
        self.test_acc = Accuracy(task="multiclass", num_classes=cfg["num_classes"])

        self.train_auroc = BinaryAUROC()
        self.val_auroc = BinaryAUROC()
        self.test_auroc = BinaryAUROC()

    def forward(self, x):
        # x: [B,3,H,W]
        outputs = self.model(x)
        # HuggingFace returns a dict with "logits"
        return outputs["logits"]

    def configure_optimizers(self):
        # Differential learning rates: backbone vs new head
        backbone_params = []
        head_params = []
        for name, param in self.model.named_parameters():
            if "head" in name:
                head_params.append(param)
            else:
                backbone_params.append(param)

        optimizer = torch.optim.AdamW(
            [
                {"params": backbone_params, "lr": self.hparams["backbone_lr"]},
                {"params": head_params, "lr": self.hparams["head_lr"]},
            ],
            weight_decay=self.hparams["weight_decay"],
        )
        return optimizer

    def training_step(self, batch, batch_idx):
        imgs, labels = batch  # imgs: [B,3,H,W], labels: [B]
        logits = self.forward(imgs)
        loss = F.cross_entropy(logits, labels, weight=self.class_weights)

        preds = torch.argmax(logits, dim=1)
        # For AUROC, get probability of positive class (label==idx_fake)
        prob_pos = torch.softmax(logits, dim=1)[:, 1]

        # Update metrics
        self.train_acc.update(preds, labels)
        self.train_auroc.update(prob_pos, labels)

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log("train_acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("train_auroc", self.train_auroc, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        imgs, labels = batch
        logits = self.forward(imgs)
        loss = F.cross_entropy(logits, labels, weight=self.class_weights)

        preds = torch.argmax(logits, dim=1)
        prob_pos = torch.softmax(logits, dim=1)[:, 1]

        self.val_acc.update(preds, labels)
        self.val_auroc.update(prob_pos, labels)

        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val_acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val_auroc", self.val_auroc, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def test_step(self, batch, batch_idx):
        imgs, labels = batch
        logits = self.forward(imgs)
        loss = F.cross_entropy(logits, labels, weight=self.class_weights)

        preds = torch.argmax(logits, dim=1)
        prob_pos = torch.softmax(logits, dim=1)[:, 1]

        self.test_acc.update(preds, labels)
        self.test_auroc.update(prob_pos, labels)

        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("test_acc", self.test_acc, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("test_auroc", self.test_auroc, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss
