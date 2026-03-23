import os
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, f1_score
from kan import KAN

# ==========================================\
# AYARLAR VE SABİTLEME
# ==========================================\
def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Random Seed {seed} olarak kilitlendi.")

CONFIG = {
    "csv_train": "train.csv",
    "root_dir": "multimodal_dataset/train",
    "image_size": 224,
    "batch_size": 16,
    "num_epochs": 15,
    "learning_rate": 1e-4,
    "k_folds": 5,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "save_dir": "experiments/resnet50_frozen_gaussian"
}

LABEL_MAP = {'Choroidal Hemangioma': 0, 'Choroidal Melanoma': 1, 'Choroidal Metastatic Carcinoma': 2}
os.makedirs(CONFIG['save_dir'], exist_ok=True)
seed_everything()

# ==========================================\
# DATASET
# ==========================================\
class MultimodalDataset(Dataset):
    def __init__(self, df, root_dir, transform=None):
        self.df = df.reset_index(drop=True)
        self.root_dir = root_dir
        self.transform = transform
        self.modalities = ['FA', 'ICGA', 'US']

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        patient_id = str(row['name']).zfill(6)
        label = LABEL_MAP[row['pathology']]
        
        patient_path = os.path.join(self.root_dir, patient_id)
        if not os.path.exists(patient_path):
            patient_path = os.path.join("multimodal_dataset", patient_id)

        images = {}
        for mod in self.modalities:
            mod_path = os.path.join(patient_path, mod)
            
            img_tensor = torch.randn((3, CONFIG['image_size'], CONFIG['image_size'])) 
            
            if os.path.exists(mod_path):
                files = [f for f in os.listdir(mod_path) if f.lower().endswith(('.jpg', '.png'))]
                if len(files) > 0:
                    img_path = os.path.join(mod_path, files[0])
                    try:
                        img_pil = Image.open(img_path).convert('RGB')
                        if self.transform:
                            img_tensor = self.transform(img_pil)
                    except:
                        pass
            images[mod] = img_tensor

        return images['FA'], images['ICGA'], images['US'], torch.tensor(label, dtype=torch.long)

transform = transforms.Compose([
    transforms.Resize((CONFIG['image_size'], CONFIG['image_size'])),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ==========================================\
# ResNet-50 + KAN
# ==========================================\
class HybridResNetKAN(nn.Module):
    def __init__(self, num_classes=3):
        super(HybridResNetKAN, self).__init__()
        
        # ResNet-50 Omurgası
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.flatten = nn.Flatten()
        
        # ResNet-50 Freeze
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        # 6144 boyutlu vektörü 64'e sıkıştır
        self.projection = nn.Linear(6144, 64)
        self.relu = nn.ReLU()
        
        # KAN Sınıflandırıcısı [64 -> 32 -> 3]
        self.kan = KAN([64, 32, num_classes])

    def forward(self, img_fa, img_icga, img_us):
        feat_fa = self.flatten(self.backbone(img_fa))     # [Batch, 2048]
        feat_icga = self.flatten(self.backbone(img_icga)) # [Batch, 2048]
        feat_us = self.flatten(self.backbone(img_us))     # [Batch, 2048]
        
        combined_features = torch.cat((feat_fa, feat_icga, feat_us), dim=1) # [Batch, 6144]
        
        # Boyut İndirgeme
        projected = self.relu(self.projection(combined_features)) # [Batch, 64]
        
        return self.kan(projected)

# ==========================================\
# EĞİTİM VE ÇAPRAZ DOĞRULAMA (K-FOLD)
# ==========================================\
df = pd.read_csv(CONFIG['csv_train'])
df = df[df['pathology'].isin(LABEL_MAP.keys())].reset_index(drop=True)

skf = StratifiedKFold(n_splits=CONFIG['k_folds'], shuffle=True, random_state=42)
all_true_labels, all_pred_labels = [], []

for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(df)), df['pathology'])):
    print(f"\n--- FOLD {fold+1}/{CONFIG['k_folds']} ---")
    
    train_sub = df.iloc[train_idx]
    val_sub = df.iloc[val_idx]
    
    train_dataset = MultimodalDataset(train_sub, CONFIG['root_dir'], transform=transform)
    val_dataset = MultimodalDataset(val_sub, CONFIG['root_dir'], transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'], shuffle=False)
    
    model = HybridResNetKAN(num_classes=3).to(CONFIG['device'])
    criterion = nn.CrossEntropyLoss()
    
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=CONFIG['learning_rate'], weight_decay=1e-4)
    
    # EĞİTİM
    for epoch in range(CONFIG['num_epochs']):
        model.train()
        train_loss = 0
        for fa, icga, us, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            fa, icga, us, labels = fa.to(CONFIG['device']), icga.to(CONFIG['device']), us.to(CONFIG['device']), labels.to(CONFIG['device'])
            
            optimizer.zero_grad()
            outputs = model(fa, icga, us)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
    # DOĞRULAMA
    model.eval()
    with torch.no_grad():
        for fa, icga, us, labels in val_loader:
            fa, icga, us = fa.to(CONFIG['device']), icga.to(CONFIG['device']), us.to(CONFIG['device'])
            outputs = model(fa, icga, us)
            _, preds = torch.max(outputs, 1)
            
            all_true_labels.extend(labels.cpu().numpy())
            all_pred_labels.extend(preds.cpu().numpy())

# ==========================================\
# SONUÇLAR VE RAPORLAMA
# ==========================================\
acc = np.mean(np.array(all_true_labels) == np.array(all_pred_labels)) * 100
f1 = f1_score(all_true_labels, all_pred_labels, average='macro') * 100

print(f"\nFinal Accuracy: %{acc:.2f} | Macro F1: %{f1:.2f}")

# Karmaşıklık Matrisi
cm = confusion_matrix(all_true_labels, all_pred_labels)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=LABEL_MAP.keys(), yticklabels=LABEL_MAP.keys())
plt.title('')
plt.savefig(os.path.join(CONFIG['save_dir'], 'matrix.png'))
pd.DataFrame({'True_Label': all_true_labels, 'Pred_Label': all_pred_labels}).to_csv(os.path.join(CONFIG['save_dir'], 'preds_resnet_frozen_gaussian.csv'), index=False)