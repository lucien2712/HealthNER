

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import List, Dict, Tuple, Optional
from torchcrf import CRF
from tqdm import tqdm
from seqeval.metrics import classification_report, f1_score, precision_score, recall_score


class BiLSTMCRF(nn.Module):
    def __init__(self, 
                 vocab_size: int, 
                 tag_to_ix: Dict[str, int], 
                 embedding_dim: int = 100, 
                 hidden_dim: int = 256, 
                 num_layers: int = 2, 
                 dropout: float = 0.5,
                 pretrained_embeddings: Optional[torch.Tensor] = None):

        super(BiLSTMCRF, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.tag_to_ix = tag_to_ix
        self.tagset_size = len(tag_to_ix)
        self.num_layers = num_layers
        
        # 詞嵌入層
        if pretrained_embeddings is not None:
            self.word_embeds = nn.Embedding.from_pretrained(
                pretrained_embeddings, 
                freeze=False,
                padding_idx=0
            )
        else:
            self.word_embeds = nn.Embedding(
                vocab_size, 
                embedding_dim,
                padding_idx=0
            )
        
        # BiLSTM層
        self.lstm = nn.LSTM(
            embedding_dim, 
            hidden_dim // 2, 
            num_layers=num_layers, 
            bidirectional=True,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # 線性層
        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size)
        
        # CRF層
        self.crf = CRF(self.tagset_size)
        
        # Dropout層
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, sentence: torch.Tensor, tags: Optional[torch.Tensor] = None, mask: Optional[torch.Tensor] = None) -> torch.Tensor:

     
        if mask is None:
            mask = torch.ne(sentence, 0).to(sentence.device)
        
        # 獲取嵌入
        embeds = self.word_embeds(sentence)
        embeds = self.dropout(embeds)
        
        # 通過LSTM
        lstm_out, _ = self.lstm(embeds)
        lstm_out = self.dropout(lstm_out)
        
        # 獲取發射分數
        emissions = self.hidden2tag(lstm_out)
        
        # 如果提供了標籤，計算損失
        if tags is not None:
            return -self.crf(emissions, tags, mask=mask, reduction='mean')
        # 否則，進行解碼
        else:
            return self.crf.decode(emissions, mask=mask)
    
    def predict(self, sentence: torch.Tensor, mask: Optional[torch.Tensor] = None) -> List[List[int]]:
  
        self.eval()
        with torch.no_grad():
            return self.forward(sentence, mask=mask)


class BiLSTMCRFModel:
    def __init__(self, 
                 vocab_size: int, 
                 tag_to_ix: Dict[str, int],
                 ix_to_tag: Dict[int, str],
                 embedding_dim: int = 100, 
                 hidden_dim: int = 256, 
                 num_layers: int = 2, 
                 dropout: float = 0.5,
                 pretrained_embeddings: Optional[torch.Tensor] = None,
                 device: str = None):
 
        self.tag_to_ix = tag_to_ix
        self.ix_to_tag = ix_to_tag
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 創建模型
        self.model = BiLSTMCRF(
            vocab_size, 
            tag_to_ix, 
            embedding_dim, 
            hidden_dim, 
            num_layers, 
            dropout,
            pretrained_embeddings
        ).to(self.device)
    
    def train(self, 
              train_data_loader, 
              valid_data_loader=None, 
              epochs: int = 10, 
              learning_rate: float = 0.001, 
              weight_decay: float = 1e-5,
              patience: int = 3,
              model_path: str = None):

        optimizer = optim.Adam(
            self.model.parameters(), 
            lr=learning_rate, 
            weight_decay=weight_decay
        )
        
        best_f1 = 0
        patience_counter = 0
        history = {
            'train_loss': [],
            'valid_loss': [],
            'valid_f1': []
        }
        
        for epoch in range(epochs):
            # 訓練
            self.model.train()
            total_loss = 0
            train_bar = tqdm(train_data_loader, desc=f"Training Epoch {epoch+1}/{epochs}")
            
            for batch in train_bar:
                # 獲取數據
                inputs = batch['inputs'].to(self.device)
                tags = batch['tags'].to(self.device)
                mask = batch['mask'].to(self.device)
                
                # 清除梯度
                optimizer.zero_grad()
                
                # 前向傳播
                loss = self.model(inputs, tags, mask)
                
                # 反向傳播
                loss.backward()
                
                # 更新參數
                optimizer.step()
                
                total_loss += loss.item()
                train_bar.set_postfix(loss=f"{loss.item():.4f}")
            
            avg_train_loss = total_loss / len(train_data_loader)
            history['train_loss'].append(avg_train_loss)
            
            # 評估
            if valid_data_loader:
                valid_loss, metrics = self.evaluate(valid_data_loader)
                history['valid_loss'].append(valid_loss)
                history['valid_f1'].append(metrics['f1'])
                
                print(f"Epoch {epoch+1}/{epochs} - "
                      f"Train Loss: {avg_train_loss:.4f} - "
                      f"Valid Loss: {valid_loss:.4f} - "
                      f"Valid F1: {metrics['f1']:.4f}")
                
                # 檢查早停
                if metrics['f1'] > best_f1:
                    best_f1 = metrics['f1']
                    patience_counter = 0
                    
                    # 保存最佳模型
                    if model_path:
                        torch.save(self.model.state_dict(), model_path)
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print(f"Early stopping at epoch {epoch+1}")
                        break
            else:
                print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f}")
                
                # 保存最後一個模型
                if model_path:
                    torch.save(self.model.state_dict(), model_path)
        
        # 如果早停，加載最佳模型
        if valid_data_loader and model_path and patience_counter >= patience:
            self.model.load_state_dict(torch.load(model_path))
        
        return history
    
    def evaluate(self, data_loader) -> Tuple[float, Dict[str, float]]:
 
        self.model.eval()
        total_loss = 0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in data_loader:
                # 獲取數據
                inputs = batch['inputs'].to(self.device)
                tags = batch['tags'].to(self.device)
                mask = batch['mask'].to(self.device)
                
                # 計算損失
                loss = self.model(inputs, tags, mask)
                total_loss += loss.item()
                
                # 預測
                pred_tags = self.model.predict(inputs, mask)
                
                # 處理預測結果和真實標籤
                for i, (pred, label, m) in enumerate(zip(pred_tags, tags.tolist(), mask.tolist())):
                    mask_len = sum(m)
                    pred = pred[:mask_len]
                    label = label[:mask_len]
                    
                    pred_tags_text = [self.ix_to_tag[p] for p in pred]
                    label_tags_text = [self.ix_to_tag[l] for l in label]
                    
                    all_predictions.append(pred_tags_text)
                    all_labels.append(label_tags_text)
        
        # 計算指標
        metrics = {
            'precision': precision_score(all_labels, all_predictions),
            'recall': recall_score(all_labels, all_predictions),
            'f1': f1_score(all_labels, all_predictions)
        }
        
        return total_loss / len(data_loader), metrics
    
    def predict(self, texts: List[List[str]], word_to_ix: Dict[str, int]) -> List[List[str]]:

        self.model.eval()
        results = []
        
        # 獲取未知詞的索引
        unk_idx = word_to_ix.get('<UNK>', 0)
        
        # 轉換文本為索引
        indexed_texts = []
        for text in texts:
            indexed_text = [word_to_ix.get(word, unk_idx) for word in text]
            indexed_texts.append(indexed_text)
        
        # 將所有文本填充到相同長度
        max_len = max(len(text) for text in indexed_texts)
        padded_texts = [text + [0] * (max_len - len(text)) for text in indexed_texts]
        masks = [[1] * len(text) + [0] * (max_len - len(text)) for text in indexed_texts]
        
        # 轉換為張量
        inputs = torch.tensor(padded_texts, dtype=torch.long).to(self.device)
        masks = torch.tensor(masks, dtype=torch.bool).to(self.device)
        
        # 預測
        with torch.no_grad():
            pred_tags = self.model.predict(inputs, masks)
        
        # 轉換為標籤文本
        for i, (pred, text) in enumerate(zip(pred_tags, texts)):
            pred = pred[:len(text)]
            pred_tags_text = [self.ix_to_tag[p] for p in pred]
            results.append(pred_tags_text)
        
        return results
    
    def load_model(self, model_path: str):
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
    
    def save_model(self, model_path: str):
        torch.save(self.model.state_dict(), model_path) 