"""
TabTransformer training script that:
 - Loads and concatenates metadata from multiple congress JSONL files
 - Loads and concatenates embeddings from multiple JSONL files
 - Builds token maps and datasets
 - Trains a TabTransformer on bill metadata + text embeddings
 - Uses Focal Loss to handle class imbalance

Usage:
    python tabtransformer_train.py
"""
import os
import json
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss

class BillDataset(Dataset):
    def __init__(self, df: pd.DataFrame, embeddings_file: str, categorical_cols: List[str], numeric_cols: List[str], label_col: str):
        """
        df: metadata dataframe
        embeddings_file: path to .jsonl file containing 'embeddings' and 'bill_keys'
        """
        self.df = df.reset_index(drop=True).copy()

        data = np.load(embeddings_file, allow_pickle=True)
        self.embeddings = np.asarray(data["embeddings"])
        self.bill_keys = np.asarray(data["bill_keys"]).astype(str)
        self.key2idx = {k: i for i, k in enumerate(self.bill_keys)}

        self.categorical_cols = categorical_cols
        self.numeric_cols = numeric_cols or []
        self.label_col = label_col

        rows = []
        missing = 0
        for _, r in self.df.iterrows():
            bk = str(r["bill_key"])
            if bk not in self.key2idx:
                missing += 1
                continue
            emb_idx = self.key2idx[bk]
            meta = {c: str(r.get(c, "_NA_")) for c in categorical_cols}
            numeric = [float(r.get(c, 0.0) if (c in r and pd.notna(r.get(c))) else 0.0) for c in self.numeric_cols]
            label = int(r[self.label_col])
            rows.append({"emb_idx": int(emb_idx), "meta": meta, "numeric": numeric, "label": label})
        if missing:
            print(f"warning: {missing} metadata rows had no matching embedding and were skipped.")
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        r = self.rows[idx]
        emb = self.embeddings[r["emb_idx"]].astype(np.float32)
        meta = r["meta"]
        numeric = np.array(r["numeric"], dtype=np.float32) if len(self.numeric_cols) > 0 else None
        label = int(r["label"])
        return emb, meta, numeric, label

def parse_bill_type(bill_key: str) -> str:
    if not isinstance(bill_key, str):
        return "_NA_"
    parts = bill_key.split("-")
    if len(parts) >= 3:
        return parts[1].lower()
    return "_NA_"

def build_vocab_from_series(series: pd.Series) -> Dict[str,int]:
    uniques = sorted(list(set(series.fillna("_NA_").astype(str).tolist())))
    mapping = {v: i+1 for i, v in enumerate(uniques)} 
    mapping["_UNK"] = len(mapping) + 1
    return mapping

def build_token_maps(df: pd.DataFrame, categorical_cols: List[str]) -> Dict[str, Dict[str,int]]:
    maps = {}
    for col in categorical_cols:
        if col not in df.columns:
            maps[col] = {"_NA_": 1, "_UNK": 2}
            continue
        maps[col] = build_vocab_from_series(df[col])
    return maps

def calculate_class_weights(df: pd.DataFrame, label_col: str) -> torch.FloatTensor:
    """
    Calculate inverse frequency weights for each class.
    More common classes get lower weights, rare classes get higher weights.
    """
    class_counts = df[label_col].value_counts().sort_index()
    percentages = (class_counts / len(df) * 100).round(2)
    for idx, pct in percentages.items():
        print(f"  Stage {idx}: {pct}%")
    
    classes = np.arange(df[label_col].nunique())
    weights = compute_class_weight(
        class_weight='balanced',
        classes=classes,
        y=df[label_col].values
    )
    for idx, weight in enumerate(weights):
        print(f"  Stage {idx}: {weight:.3f}")
    
    return torch.FloatTensor(weights)

def collate_batch(batch, token_maps: Dict[str, Dict[str,int]], categorical_cols: List[str], device: str):
    text_embs, token_id_rows, numeric_rows, labels = [], [], [], []
    for emb, meta, numeric, label in batch:
        text_embs.append(torch.tensor(emb, dtype=torch.float32))
        token_ids = [token_maps[c].get(str(meta.get(c, "_NA_")), token_maps[c]["_UNK"]) for c in categorical_cols]
        token_id_rows.append(torch.tensor(token_ids, dtype=torch.long))
        labels.append(label)
        if numeric is not None:
            numeric_rows.append(torch.tensor(numeric, dtype=torch.float32))
    text_embs = torch.stack(text_embs).to(device)
    token_id_rows = torch.stack(token_id_rows).to(device)
    labels = torch.tensor(labels, dtype=torch.long).to(device)
    numeric_tensor = torch.stack(numeric_rows).to(device) if len(numeric_rows) > 0 else None
    return text_embs, token_id_rows, numeric_tensor, labels

class TabTransformer(nn.Module):
    def __init__(self,
                 token_maps: Dict[str,int],
                 categorical_embed_dim: Dict[str,int],
                 categorical_cols: List[str],
                 text_embedding_dim: int = 1536,
                 token_dim: int = 128,
                 n_heads: int = 4,
                 n_layers: int = 3,
                 mlp_hidden: List[int] = [128],
                 dropout: float = 0.2,
                 num_numeric: int = 0,
                 num_classes: int = 5,
                 use_cls: bool = False):
        super().__init__()
        self.categorical_cols = categorical_cols
        self.cat_embs = nn.ModuleDict()
        self.cat_projs = nn.ModuleDict()
        for c in categorical_cols:
            vocab_size = len(token_maps[c]) + 2
            emb_dim = categorical_embed_dim[c]
            self.cat_embs[c] = nn.Embedding(vocab_size, emb_dim)
            self.cat_projs[c] = nn.Linear(emb_dim, token_dim)

        self.text_proj = nn.Linear(text_embedding_dim, token_dim)
        self.use_cls = use_cls
        if self.use_cls:
            self.cls_token = nn.Parameter(torch.randn(1,1,token_dim))

        encoder_layer = nn.TransformerEncoderLayer(d_model=token_dim, nhead=n_heads,
                                                   dim_feedforward=token_dim*4, dropout=dropout, activation='relu')
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.layernorm = nn.LayerNorm(token_dim)

        head_in = token_dim + num_numeric
        mlp_layers = []
        in_dim = head_in
        for h in mlp_hidden:
            mlp_layers.append(nn.Linear(in_dim, h))
            mlp_layers.append(nn.ReLU())
            mlp_layers.append(nn.Dropout(dropout))
            in_dim = h
        mlp_layers.append(nn.Linear(in_dim, num_classes))
        self.mlp = nn.Sequential(*mlp_layers)

    def forward(self, text_embs: torch.Tensor, token_ids: torch.LongTensor, numeric: Optional[torch.Tensor] = None):
        B = text_embs.size(0)
        device = text_embs.device
        tok_list = [self.text_proj(text_embs).unsqueeze(1)]
        for i, c in enumerate(self.categorical_cols):
            emb = self.cat_embs[c](token_ids[:, i])
            tok_list.append(self.cat_projs[c](emb).unsqueeze(1))
        tokens = torch.cat(tok_list, dim=1)
        if self.use_cls:
            cls = self.cls_token.expand(B, -1, -1).to(device)
            tokens = torch.cat([cls, tokens], dim=1)
        tokens_t = tokens.transpose(0,1)
        out = self.transformer(tokens_t)
        out = out.transpose(0,1)
        pooled = out[:,0,:] if self.use_cls else out.mean(dim=1)
        pooled = self.layernorm(pooled)
        if numeric is not None:
            pooled = torch.cat([pooled, numeric], dim=1)
        return self.mlp(pooled)

def train_one_epoch(model, loader, optimizer, criterion, device, epoch):
    model.train()
    total_loss = 0.0
    preds, trues = [], []
    for text_embs, token_ids, numeric, labels in tqdm(loader, desc=f"Epoch {epoch} [train]"):
        text_embs, token_ids, labels = text_embs.to(device), token_ids.to(device), labels.to(device)
        if numeric is not None:
            numeric = numeric.to(device)
        optimizer.zero_grad()
        logits = model(text_embs, token_ids, numeric)
        loss = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * labels.size(0)
        preds.append(torch.argmax(logits.detach().cpu(), dim=1).numpy())
        trues.append(labels.detach().cpu().numpy())
    preds, trues = np.concatenate(preds), np.concatenate(trues)
    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(trues, preds)
    f1 = f1_score(trues, preds, average='weighted')
    return avg_loss, acc, f1

def eval_model(model, loader, criterion, device, num_classes):
    model.eval()
    total_loss = 0.0
    preds, trues = [], []
    with torch.no_grad():
        for text_embs, token_ids, numeric, labels in tqdm(loader, desc="eval"):
            text_embs, token_ids, labels = text_embs.to(device), token_ids.to(device), labels.to(device)
            if numeric is not None:
                numeric = numeric.to(device)
            logits = model(text_embs, token_ids, numeric)
            loss = criterion(logits, labels)
            total_loss += loss.item() * labels.size(0)
            preds.append(torch.argmax(logits.cpu(), dim=1).numpy())
            trues.append(labels.cpu().numpy())
    preds, trues = np.concatenate(preds), np.concatenate(trues)
    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(trues, preds)
    f1 = f1_score(trues, preds, average='weighted')
    
    target_names = [f"Stage_{i}" for i in range(num_classes)]
    report = classification_report(trues, preds, target_names=target_names, zero_division=0)
    cm = confusion_matrix(trues, preds)
    
    return avg_loss, acc, f1, trues, preds, report, cm

def load_and_concat_metadata(paths: List[str]) -> pd.DataFrame:
    """
    Load multiple metadata JSONL files and concatenate them.
    By default, later files override earlier ones for the same bill_key.
    """
    records_by_key: Dict[str, Dict] = {}
    for p in paths:
        if not p or not os.path.exists(p):
            print(f"warning: metadata file not found: {p}")
            continue
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                bk = rec.get("bill_key")
                if not bk:
                    continue
                records_by_key[bk] = rec
    records = list(records_by_key.values())
    df = pd.DataFrame(records)
    return df

def load_and_concat_embeddings(paths: List[str]) -> Tuple[np.ndarray, np.ndarray]:

    emb_dict: Dict[str, np.ndarray] = {}
    for p in paths:
        if not p or not os.path.exists(p):
            print(f"warning: embedding file not found: {p}")
            continue
        
        # Read JSONL file
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue
                k = d.get("bill_key")
                e = d.get("embedding")
                if not k or e is None:
                    continue
                if k not in emb_dict:
                    emb_dict[k] = np.asarray(e, dtype=np.float32)

    if len(emb_dict) == 0:
        raise RuntimeError("No embeddings found in provided embedding files.")

    bill_keys = list(emb_dict.keys())
    embeddings = np.stack([emb_dict[k] for k in bill_keys], axis=0).astype(np.float32)
    bill_keys = np.array(bill_keys, dtype=str)
    return embeddings, bill_keys

def main():
    METADATA_FILES = [
        "./data/116_bills_clean_metadata.jsonl",
        "./data/117_bills_clean_metadata.jsonl",
        "./data/118_bills_clean_metadata.jsonl",
    ]
    EMBEDDING_FILES = [
        "./116_bills_embeddings.jsonl",
        "./117_bills_embeddings.jsonl",
        "./118_bills_embeddings.jsonl",
    ]

    metadata_file = None 
    embeddings_file = None 
    categorical_cols = ["sponsor_party", "origin_committee", "bill_type", "policy_area", "origin_chamber"]
    numeric_cols = ["num_cosponsors"]
    label_col = "stage_label"
    out_dir = "./tabtransformer"
    os.makedirs(out_dir, exist_ok=True)

    epochs = 30
    batch_size = 64
    lr = 5e-5
    weight_decay = 1e-4
    token_dim = 128
    text_emb_dim = 1536
    n_heads = 4
    n_layers = 3
    mlp_hidden = [128]
    dropout = 0.4
    patience = 10
    use_cls = False
    focal_gamma = 1.5
    device = "cuda" if torch.cuda.is_available() else "cpu"


    df = load_and_concat_metadata(METADATA_FILES)
    if df.shape[0] == 0:
        raise RuntimeError("No metadata loaded. Check METADATA_FILES paths.")
    df = df.dropna(subset=["bill_key", label_col]).reset_index(drop=True)
    df["bill_type"] = df["bill_key"].apply(parse_bill_type)

    embeddings, bill_keys = load_and_concat_embeddings(EMBEDDING_FILES)
    emb_dim = embeddings.shape[1]
    if emb_dim != text_emb_dim:
        print(f"warning: embedding dim ({emb_dim}) != text_emb_dim ({text_emb_dim}). Adjust `text_emb_dim` if needed.")

    combined_npz = os.path.join(out_dir, "combined_embeddings.npz")
    np.savez_compressed(combined_npz, embeddings=embeddings, bill_keys=bill_keys)
    embeddings_file = combined_npz

    token_maps = build_token_maps(df, categorical_cols)
    cat_embed_dims = {c: 32 if len(token_maps[c]) < 50 else 64 for c in categorical_cols}

    if numeric_cols:
        scaler = StandardScaler()
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols].fillna(0.0))

    train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42, stratify=df[label_col])
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df[label_col])

    class_weights = calculate_class_weights(train_df, label_col)
    class_weights[0] *= 4.0
    class_weights = class_weights.to(device)

    train_ds = BillDataset(train_df, embeddings_file, categorical_cols, numeric_cols, label_col)
    val_ds = BillDataset(val_df, embeddings_file, categorical_cols, numeric_cols, label_col)
    test_ds = BillDataset(test_df, embeddings_file, categorical_cols, numeric_cols, label_col)

    collate = lambda b: collate_batch(b, token_maps, categorical_cols, device)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate)

    num_classes = int(df[label_col].nunique())
    model = TabTransformer(token_maps=token_maps,
                           categorical_embed_dim=cat_embed_dims,
                           categorical_cols=categorical_cols,
                           text_embedding_dim=text_emb_dim,
                           token_dim=token_dim,
                           n_heads=n_heads,
                           n_layers=n_layers,
                           mlp_hidden=mlp_hidden,
                           dropout=dropout,
                           num_numeric=len(numeric_cols),
                           num_classes=num_classes,
                           use_cls=use_cls)
    model.to(device)

    criterion = FocalLoss(alpha=class_weights, gamma=focal_gamma)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val_loss = float("inf")
    no_improve = 0
    for epoch in range(1, epochs+1):
        print(f"Epoch {epoch}/{epochs}")
        train_loss, train_acc, train_f1 = train_one_epoch(model, train_loader, optimizer, criterion, device, epoch)
        val_loss, val_acc, val_f1, _, _, val_report, val_cm = eval_model(model, val_loader, criterion, device, num_classes)
        
        print(f"\nTrain - Loss: {train_loss:.4f} | Acc: {train_acc:.4f} | F1: {train_f1:.4f}")
        print(f"Val   - Loss: {val_loss:.4f} | Acc: {val_acc:.4f} | F1: {val_f1:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(out_dir, "best_model.pt"))
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"\nEarly stopping triggered after {epoch} epochs (patience={patience})")
                break

    model.load_state_dict(torch.load(os.path.join(out_dir, "best_model.pt"), map_location=device))
    test_loss, test_acc, test_f1, y_true, y_pred, test_report, test_cm = eval_model(model, test_loader, criterion, device, num_classes)
    
    print(f"\nTest Set Metrics:")
    print(f"Loss: {test_loss:.4f} | Accuracy: {test_acc:.4f} | F1: {test_f1:.4f}")
    print(f"\nDetailed Classification Report:")
    print(test_report)
    print(f"\nConfusion Matrix:")
    print(test_cm)

    with open(os.path.join(out_dir, "token_maps.json"), "w", encoding="utf-8") as f:
        json.dump({k: {str(kk): vv for kk, vv in v.items()} for k, v in token_maps.items()}, f, indent=2)
    
    with open(os.path.join(out_dir, "test_results.txt"), "w") as f:
        f.write(f"Test Loss: {test_loss:.4f}\n")
        f.write(f"Test Accuracy: {test_acc:.4f}\n")
        f.write(f"Test F1: {test_f1:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(test_report)
        f.write("\n\nConfusion Matrix:\n")
        f.write(str(test_cm))

if __name__ == "__main__":
    main()