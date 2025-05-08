import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
import cv2

class MambaSSM(nn.Module):
    """
    Selective State Space Model (SSM) block from Mamba.
    Enables content-based propagation of sequence features.
    """
    def __init__(self, d_model, d_ssm):
        super().__init__()
        self.d_model = d_model
        self.d_ssm = d_ssm
        # parameter projections
        self.param_proj = nn.Linear(d_model, 3 * d_ssm)
        self.input_proj = nn.Linear(d_model, d_ssm)
        self.output_proj = nn.Linear(d_ssm, d_model)

        # state update
        self.register_buffer('state', torch.zeros(1, d_ssm))

    def forward(self, x):
        # x: [batch, seq, d_model]
        B, T, _ = x.shape
        params = self.param_proj(x)  # [B, T, 3*d_ssm]
        u = self.input_proj(x)      # [B, T, d_ssm]
        y = []
        s = self.state.expand(B, -1).contiguous()
        for t in range(T):
            p = params[:, t].view(B, 3, self.d_ssm)
            w, k, b = p[:,0], p[:,1], p[:,2]
            # selective update gate
            gate = torch.sigmoid(w)
            # state transition
            s = gate * (s * torch.tanh(k) + u[:,t]) + (1 - gate) * s
            y.append(s)
        y = torch.stack(y, dim=1)
        return self.output_proj(y)

class DeepfakeDataset(Dataset):
    """
    Video dataset loader for deepfake detection.
    Assumes videos stored in 'real' and 'fake' subfolders.
    """
    def __init__(self, root_dir, seq_len=16, transform=None):
        self.root = root_dir
        self.seq_len = seq_len
        self.transform = transform
        self.samples = []
        for label, cls in enumerate(['real', 'fake']):
            cls_dir = os.path.join(root_dir, cls)
            for fname in os.listdir(cls_dir):
                if fname.endswith('.mp4') or fname.endswith('.avi'):
                    self.samples.append((os.path.join(cls_dir, fname), label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        cap = cv2.VideoCapture(path)
        frames = []
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        # sample seq_len frames evenly
        indices = torch.linspace(0, total-1, steps=self.seq_len).long()
        for i in range(total):
            ret, frame = cap.read()
            if not ret:
                break
            if i in indices:
                img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                if self.transform:
                    img = self.transform(img)
                frames.append(img)
        cap.release()
        x = torch.stack(frames, dim=0)  # [seq, 3, H, W]
        return x, torch.tensor(label, dtype=torch.long)

class DeepfakeDetector(nn.Module):
    """
    Deepfake detection model with a CNN backbone + MambaSSM temporal module.
    """
    def __init__(self, seq_len=16, d_ssm=128, pretrained=True):
        super().__init__()
        # CNN feature extractor (ResNet18)
        resnet = models.resnet18(pretrained=pretrained)
        modules = list(resnet.children())[:-1]
        self.cnn = nn.Sequential(*modules)
        d_model = resnet.fc.in_features
        # temporal module
        self.mamba = MambaSSM(d_model, d_ssm)
        # classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        # x: [B, seq, 3, H, W]
        B, T, C, H, W = x.shape
        x = x.view(B*T, C, H, W)
        feats = self.cnn(x).view(B, T, -1)  # [B, T, d_model]
        seq_out = self.mamba(feats)        # [B, T, d_model]
        # aggregate temporal features (mean pooling)
        agg = seq_out.mean(dim=1)         # [B, d_model]
        logits = self.classifier(agg)
        return logits

if __name__ == '__main__':
    # example usage
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    dataset = DeepfakeDataset('/path/to/data', seq_len=16, transform=transform)
    loader = DataLoader(dataset, batch_size=4, shuffle=True)
    model = DeepfakeDetector(seq_len=16)
    for batch, labels in loader:
        logits = model(batch)
        loss = F.cross_entropy(logits, labels)
        print(loss.item())
        break
