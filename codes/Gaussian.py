import os
import random
import time
import datetime
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
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.manifold import TSNE
from sklearn.preprocessing import label_binarize
from kan import KAN

# ==========================================
# Sonuç Sabitleme
# ==========================================
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

# ==========================================
# AYARLAR
# ==========================================
CONFIG = {
    "csv_train": "train.csv",
    "csv_test": "test.csv",
    "root_dir": "multimodal_dataset",
    "checkpoint_path": "checkpoint_180.pth", 
    "batch_size": 8,           
    "num_epochs": 12,          
    "learning_rate": 1e-4,
    "num_classes": 3,
    "k_folds": 5,
    "input_resize": 224,
    "modality_dropout_prob": 0.25, 
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "save_dir": f"experiments/final_gaussian_no_tta_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
}

os.makedirs(CONFIG["save_dir"], exist_ok=True)
print(f"[BİLGİ] Çalışma Dizini: {CONFIG['save_dir']}")

# ==========================================
# DATASET
# ==========================================
class RobustEyeDataset(Dataset):
    def __init__(self, dataframe, root_dir, transform=None, training=False):
        self.data_frame = dataframe.copy()
        self.data_frame['name'] = self.data_frame['name'].astype(str).apply(lambda x: x.zfill(6))
        
        self.label_map = {
            'Choroidal Hemangioma': 0,
            'Choroidal Melanoma': 1,
            'Choroidal Metastatic Carcinoma': 2
        }
        
        if 'label' not in self.data_frame.columns:
            self.data_frame['label'] = self.data_frame['pathology'].map(self.label_map)
            self.data_frame = self.data_frame.dropna(subset=['label'])
            self.data_frame['label'] = self.data_frame['label'].astype(int)

        self.root_dir = root_dir
        self.transform = transform
        self.training = training 

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        row = self.data_frame.iloc[idx]
        patient_id = row['name']
        label = int(row['label'])

        possible_roots = [
            self.root_dir,
            os.path.join(self.root_dir, 'train'), 
            os.path.join(self.root_dir, 'test'),
            os.path.join(self.root_dir, 'extra_data')
        ]
        
        patient_path = None
        for r in possible_roots:
            candidate = os.path.join(r, patient_id)
            if os.path.exists(candidate):
                patient_path = candidate
                break
        
        # --- Veri yoksa gürültü ekle ---
        if patient_path is None:
            return torch.randn(7, 3, 224, 224), label, "ALL_MISSING"

        loaded_tensors = []
        missing_mods = []
        modalities = [('FA', 3), ('ICGA', 3), ('US', 1)]
        
        for mod_name, count in modalities:
            mod_path = os.path.join(patient_path, mod_name)
            files = []
            if os.path.exists(mod_path):
                files = sorted([os.path.join(mod_path, f) for f in os.listdir(mod_path) if f.lower().endswith(('.jpg', '.png'))])[:count]
            
            force_noise = False
            if self.training and len(files) > 0: 
                if np.random.rand() < CONFIG['modality_dropout_prob']:
                    force_noise = True
            
            if not files: missing_mods.append(mod_name)

            current_mod_tensors = []
            if not force_noise and files:
                for img_path in files:
                    try:
                        img = Image.open(img_path).convert('RGB')
                        if self.transform: img = self.transform(img)
                        current_mod_tensors.append(img)
                    except: 
                        current_mod_tensors.append(torch.randn(3, 224, 224))
            
            while len(current_mod_tensors) < count:
                current_mod_tensors.append(torch.randn(3, 224, 224))
                
            loaded_tensors.extend(current_mod_tensors)

        missing_str = ",".join(missing_mods) if missing_mods else "None"
        return torch.stack(loaded_tensors), label, missing_str

# ==========================================
# MODEL
# ==========================================
class HybridEfficientKAN(nn.Module):
    def __init__(self, num_classes=3, pretrained_path=None):
        super(HybridEfficientKAN, self).__init__()
        
        self.backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        self.backbone.classifier = nn.Identity()
        
        self.feature_dim = 1280 * 7 
        self.projection_dim = 64 
        self.projection = nn.Linear(self.feature_dim, self.projection_dim)
        self.relu = nn.ReLU()
        self.kan = KAN(width=[self.projection_dim, 32, num_classes], grid=5, k=3)
        self.kan.speed()

        if pretrained_path:
            if os.path.exists(pretrained_path):
                print(f"'{pretrained_path}' yükleniyor...")
                try:
                    checkpoint = torch.load(pretrained_path, map_location='cpu')
                    if 'model_state_dict' in checkpoint:
                        state_dict = checkpoint['model_state_dict']
                    else:
                        state_dict = checkpoint
                    self.load_state_dict(state_dict, strict=False)
                    print("Checkpoint yüklendi.")
                except Exception as e:
                    print(f"Yükleme hatası: {e}. Sıfırdan başlanıyor.")
            else:
                print(f"'{pretrained_path}' bulunamadı! Sıfırdan başlanıyor.")

    def forward(self, x, return_features=False):
        b, num_imgs, c, h, w = x.size()
        x = x.view(b * num_imgs, c, h, w)
        feats = self.backbone(x) 
        feats = feats.view(b, -1) 
        projected = self.relu(self.projection(feats)) 
        out = self.kan(projected)
        if return_features:
            return out, projected 
        return out

# ==========================================
# ANALİZÖR
# ==========================================
class MedicalAnalyzer:
    def __init__(self, save_dir):
        self.save_dir = save_dir
        self.classes = ['Hemangioma', 'Melanoma', 'Metastatic']

    def plot_tsne(self, features, labels):
        print("--- t-SNE Haritası oluşturuluyor ---")
        tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
        features_2d = tsne.fit_transform(features)
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], c=labels, cmap='viridis', alpha=0.7)
        plt.legend(handles=scatter.legend_elements()[0], labels=self.classes)
        plt.title('t-SNE Visualization')
        plt.savefig(os.path.join(self.save_dir, 'analysis_tsne.png'))
        plt.close()

    def plot_confidence_hist(self, correct_conf, wrong_conf):
        plt.figure(figsize=(10, 6))
        plt.hist(correct_conf, bins=20, alpha=0.7, color='green', label='Correct')
        plt.hist(wrong_conf, bins=20, alpha=0.7, color='red', label='Wrong')
        plt.xlabel('Confidence')
        plt.ylabel('Count')
        plt.title('Confidence Distribution')
        plt.legend()
        plt.savefig(os.path.join(self.save_dir, 'analysis_confidence.png'))
        plt.close()

    def plot_roc_curves(self, y_true, y_probs):
        y_true_bin = label_binarize(y_true, classes=[0, 1, 2])
        n_classes = y_true_bin.shape[1]
        plt.figure(figsize=(10, 8))
        colors = ['blue', 'red', 'green']
        for i in range(n_classes):
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_probs[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, color=colors[i], lw=2, label=f'{self.classes[i]} (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.title('ROC Curves')
        plt.legend(loc="lower right")
        plt.savefig(os.path.join(self.save_dir, 'analysis_roc.png'))
        plt.close()

# ==========================================
# PIPELINE
# ==========================================
def run_pipeline():
    # --- Veri Hazırlığı ---
    train_transforms = transforms.Compose([
        transforms.Resize((CONFIG['input_resize'], CONFIG['input_resize'])),
        transforms.RandomHorizontalFlip(), transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2), transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    val_transforms = transforms.Compose([
        transforms.Resize((224, 224)), transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    print("Veri yükleniyor...")
    df_train = pd.read_csv(CONFIG['csv_train'])
    df_test = pd.read_csv(CONFIG['csv_test'])
    df_full = pd.concat([df_train, df_test], ignore_index=True)
    
    skf = StratifiedKFold(n_splits=CONFIG['k_folds'], shuffle=True, random_state=42)
    dataset_full = RobustEyeDataset(df_full, CONFIG['root_dir'], transform=None)
    labels = dataset_full.data_frame['label'].values
    
    trained_models = []
    
    # --- Eğitim Döngüsü ---
    print(f"{CONFIG['k_folds']}-Fold Eğitim Başlıyor...")
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(labels)), labels)):
        print(f"\n--- FOLD {fold+1} ---")
        
        train_ds = RobustEyeDataset(df_full.iloc[train_idx], CONFIG['root_dir'], transform=train_transforms, training=True)
        val_ds = RobustEyeDataset(df_full.iloc[val_idx], CONFIG['root_dir'], transform=val_transforms, training=False)
        
        train_loader = DataLoader(train_ds, batch_size=CONFIG['batch_size'], shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=CONFIG['batch_size'], shuffle=False)
        
        model = HybridEfficientKAN(num_classes=CONFIG['num_classes'], pretrained_path=CONFIG['checkpoint_path']).to(CONFIG['device'])
        
        optimizer = optim.AdamW(model.parameters(), lr=CONFIG['learning_rate'])
        criterion = nn.CrossEntropyLoss()
        
        best_acc = 0
        best_state = None
        
        for epoch in range(CONFIG['num_epochs']):
            model.train()
            for imgs, lbls, _ in train_loader:
                imgs, lbls = imgs.to(CONFIG['device']), lbls.to(CONFIG['device'])
                optimizer.zero_grad()
                out = model(imgs)
                loss = criterion(out, lbls)
                loss.backward()
                optimizer.step()
            
            # Validation
            model.eval()
            correct = 0; total = 0
            with torch.no_grad():
                for imgs, lbls, _ in val_loader:
                    imgs, lbls = imgs.to(CONFIG['device']), lbls.to(CONFIG['device'])
                    out = model(imgs)
                    _, pred = torch.max(out, 1)
                    total += lbls.size(0)
                    correct += (pred == lbls).sum().item()
            
            acc = correct/total
            if acc > best_acc:
                best_acc = acc
                best_state = model.state_dict()
        
        if best_state is not None:
            model.load_state_dict(best_state)
            torch.save(best_state, os.path.join(CONFIG['save_dir'], f'model_fold_{fold+1}.pth'))
            trained_models.append(model)
            print(f"Fold {fold+1} Bitti. Best Val Acc: %{best_acc*100:.2f}")
        else:
            trained_models.append(model)
            print(f"Fold {fold+1} Bitti. (Best state bulunamadı)")

    # --- ANALİZ ---
    print("\n" + "="*40)
    print("Test Analizi")
    print("="*40)
    
    test_ds_final = RobustEyeDataset(df_test, CONFIG['root_dir'], transform=val_transforms, training=False)
    test_loader_final = DataLoader(test_ds_final, batch_size=CONFIG['batch_size'], shuffle=False)
    
    analyzer = MedicalAnalyzer(CONFIG['save_dir'])
    
    all_features = []
    all_probs = []
    all_preds = []
    all_labels = []
    correct_confs = []
    wrong_confs = []
    detailed_results = []
    
    with torch.no_grad():
        for imgs, lbls, missing_info in tqdm(test_loader_final, desc="Test Analizi"):
            imgs = imgs.to(CONFIG['device'])
            
            ensemble_logits = torch.zeros(imgs.size(0), 3).to(CONFIG['device'])
            ensemble_feats = torch.zeros(imgs.size(0), 64).to(CONFIG['device']) 
            
            for m in trained_models:
                m.eval()
                logits, feats = m(imgs, return_features=True)
                ensemble_logits += logits
                ensemble_feats += feats
            
            ensemble_logits /= len(trained_models)
            ensemble_feats /= len(trained_models)
            
            probs = torch.softmax(ensemble_logits, dim=1)
            final_confs, final_preds = torch.max(probs, 1)

            all_features.append(ensemble_feats.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
            all_preds.extend(final_preds.cpu().numpy())
            all_labels.extend(lbls.numpy())
            
            for i in range(len(lbls)):
                is_correct = (final_preds[i].item() == lbls[i].item())
                detailed_results.append({
                    "True_Label": ['Hemangioma', 'Melanoma', 'Metastatic'][lbls[i].item()],
                    "Pred_Label": ['Hemangioma', 'Melanoma', 'Metastatic'][final_preds[i].item()],
                    "Confidence": final_confs[i].item(),
                    "Correct": is_correct,
                    "Missing_Modalities": missing_info[i],
                    "TTA_Applied": False
                })
                
                if is_correct: correct_confs.append(final_confs[i].item())
                else: wrong_confs.append(final_confs[i].item())

    # Görselleştirme
    all_features = np.concatenate(all_features, axis=0)
    analyzer.plot_tsne(all_features, all_labels)
    analyzer.plot_confidence_hist(correct_confs, wrong_confs)
    all_probs = np.concatenate(all_probs, axis=0)
    analyzer.plot_roc_curves(all_labels, all_probs)
    
    class_names = ['Hemangioma', 'Melanoma', 'Metastatic']
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Tahmin Edilen')
    plt.ylabel('Gerçek Etiket')
    plt.title('Karmaşıklık Matrisi')
    plt.savefig(os.path.join(CONFIG['save_dir'], 'final_matrix.png'))
    plt.close()
    
    df_res = pd.DataFrame(detailed_results)
    df_res.to_csv(os.path.join(CONFIG['save_dir'], 'detailed_analysis_report.csv'), index=False)
    
    acc = np.mean(np.array(all_preds) == np.array(all_labels))
    print(f"\nTest Accuracy: %{acc*100:.2f}")
    print(f"Analizler '{CONFIG['save_dir']}' içine kaydedildi.")

if __name__ == '__main__':
    seed_everything(42)
    run_pipeline()