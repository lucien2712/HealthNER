import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import List, Dict, Tuple, Optional, Union
from torchcrf import CRF
from tqdm import tqdm
from transformers import BertModel, BertTokenizerFast
from seqeval.metrics import classification_report, f1_score, precision_score, recall_score


class BertBiLSTMCRF(nn.Module):
    def __init__(self, 
                 bert_model_name: str, 
                 target_size: int, 
                 hidden_dim: int = 768, 
                 lstm_layers: int = 2,
                 dropout: float = 0.1,
                 freeze_bert: bool = True):
  
        super(BertBiLSTMCRF, self).__init__()
        
        # BERT模型
        self.bert_model = BertModel.from_pretrained(bert_model_name)
        self.hidden_dim = hidden_dim
        self.target_size = target_size
        
        # 凍結BERT參數
        if freeze_bert:
            for param in self.bert_model.parameters():
                param.requires_grad_(False)
        
        # BiLSTM層
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim // 2,
            batch_first=True,
            bidirectional=True,
            num_layers=lstm_layers,
            dropout=dropout if lstm_layers > 1 else 0
        )
        
        # 分類器
        self.hidden2tag = nn.Linear(hidden_dim, self.target_size)
        
        # CRF層
        self.crf = CRF(self.target_size)
        
        # Dropout層
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, 
               sentence: torch.Tensor, 
               attention_mask: Optional[torch.Tensor] = None, 
               tags: Optional[torch.Tensor] = None, 
               crf_mask: Optional[torch.Tensor] = None) -> Union[torch.Tensor, List[List[int]]]:

        # BERT編碼
        with torch.no_grad() if self.bert_model.training is False else torch.enable_grad():
            bert_outputs = self.bert_model(sentence, attention_mask=attention_mask)
            bert_embeddings = bert_outputs.last_hidden_state  # [batch_size, seq_len, hidden_dim]
        
        # BiLSTM層
        lstm_out, _ = self.lstm(bert_embeddings)  # [batch_size, seq_len, hidden_dim]
        lstm_out = self.dropout(lstm_out)
        
        # 分類層
        emissions = self.hidden2tag(lstm_out)  # [batch_size, seq_len, target_size]
        
        # 轉換形狀以適應CRF層
        emissions_t = emissions.permute(1, 0, 2)  # [seq_len, batch_size, target_size]
        
        # 如果提供了標籤，計算訓練損失
        if tags is not None:
            if crf_mask is not None:
                crf_mask_t = crf_mask.permute(1, 0)  # [seq_len, batch_size]
                loss = -1 * self.crf.forward(
                    emissions=emissions_t,
                    tags=tags.permute(1, 0),
                    mask=crf_mask_t,
                    reduction='mean'
                )
            else:
                loss = -1 * self.crf.forward(
                    emissions=emissions_t,
                    tags=tags.permute(1, 0),
                    reduction='mean'
                )
            return loss
        # 否則，進行預測
        else:
            if crf_mask is not None:
                crf_mask_t = crf_mask.permute(1, 0)  # [seq_len, batch_size]
                predictions = self.crf.decode(emissions=emissions_t, mask=crf_mask_t)
            else:
                predictions = self.crf.decode(emissions=emissions_t)
            return predictions


class BertBiLSTMCRFModel:
    def __init__(self, 
                bert_model_name: str = "ckiplab/bert-base-chinese",
                target_size: int = 22,
                hidden_dim: int = 768,
                lstm_layers: int = 2,
                dropout: float = 0.1,
                freeze_bert: bool = True,
                device: str = None,
                max_seq_length: int = 150):

        self.bert_model_name = bert_model_name
        self.target_size = target_size
        self.hidden_dim = hidden_dim
        self.max_seq_length = max_seq_length
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 初始化分詞器和模型
        self.tokenizer = BertTokenizerFast.from_pretrained(bert_model_name)
        self.model = BertBiLSTMCRF(
            bert_model_name=bert_model_name,
            target_size=target_size,
            hidden_dim=hidden_dim,
            lstm_layers=lstm_layers,
            dropout=dropout,
            freeze_bert=freeze_bert
        ).to(self.device)
    
    def prepare_data(self, data, tag_to_idx):

        # 提取句子列表
        sentences = [s for s in data["sentence"]]
        labels = data["character_label"]
        
        # 使用BERT tokenizer處理
        inputs = self.tokenizer(
            sentences, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=self.max_seq_length
        )
        
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        
        # 處理標籤
        seq_length = input_ids.shape[1]
        for i in range(len(labels)):
            labels[i] += [21] * seq_length  # 21是填充標籤
            labels[i] = labels[i][:seq_length]
        
        # 轉換標籤為索引
        label_indices = []
        for label_list in labels:
            label_index = [tag_to_idx.get(l, 0) for l in label_list]
            label_indices.append(label_index)
        
        label_tensor = torch.LongTensor(label_indices)
        crf_mask = attention_mask.clone()
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": label_tensor,
            "crf_mask": crf_mask
        }
    
    def train(self, 
              train_data_loader, 
              valid_data_loader=None, 
              epochs: int = 15, 
              learning_rate: float = 0.01, 
              weight_decay: float = 1e-5,
              patience: int = 3,
              model_path: str = None):

        optimizer = optim.SGD(
            self.model.parameters(), 
            lr=learning_rate, 
            weight_decay=weight_decay
        )
        
        # 學習率調度器
        lr_scheduler = optim.lr_scheduler.StepLR(
            optimizer, 
            step_size=2, 
            gamma=0.9
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
            
            for step, batch in enumerate(train_bar):
                # 獲取數據
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)
                crf_mask = batch["crf_mask"].to(self.device) if "crf_mask" in batch else None
                
                # 清除梯度
                optimizer.zero_grad()
                
                # 前向傳播
                loss = self.model(
                    sentence=input_ids, 
                    attention_mask=attention_mask, 
                    tags=labels, 
                    crf_mask=crf_mask
                )
                
                # 反向傳播
                loss.backward()
                
                # 更新參數
                optimizer.step()
                
                total_loss += loss.item()
                
                if step % 100 == 0:
                    train_bar.set_postfix(loss=f"{loss.item():.4f}")
            
            # 更新學習率
            lr_scheduler.step()
            
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
        
        return history
    
    def evaluate(self, data_loader) -> Tuple[float, Dict[str, float]]:

        self.model.eval()
        total_loss = 0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(data_loader, desc="Evaluating"):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)
                crf_mask = batch["crf_mask"].to(self.device) if "crf_mask" in batch else None
                
                # 計算損失
                loss = self.model(
                    sentence=input_ids, 
                    attention_mask=attention_mask, 
                    tags=labels, 
                    crf_mask=crf_mask
                )
                total_loss += loss.item()
                
                # 獲取預測
                predictions = self.model(
                    sentence=input_ids, 
                    attention_mask=attention_mask,
                    crf_mask=crf_mask
                )
                
                # 將預測和標籤轉換為列表
                for i, pred in enumerate(predictions):
                    actual_len = attention_mask[i].sum().item()
                    pred_tags = pred[:actual_len]
                    true_tags = labels[i][:actual_len].cpu().tolist()
                    
                    all_predictions.extend(pred_tags)
                    all_labels.extend(true_tags)
        
        # 計算指標
        accuracy = np.mean([p == l for p, l in zip(all_predictions, all_labels)])
        precision = precision_score(all_labels, all_predictions, average='weighted')
        recall = recall_score(all_labels, all_predictions, average='weighted')
        f1 = f1_score(all_labels, all_predictions, average='weighted')
        
        return total_loss / len(data_loader), {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    def predict(self, sentences: List[str]) -> List[List[int]]:

        self.model.eval()
        
        # 使用BERT tokenizer處理
        inputs = self.tokenizer(
            sentences, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=self.max_seq_length
        )
        
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)
        
        with torch.no_grad():
            predictions = self.model(
                sentence=input_ids, 
                attention_mask=attention_mask, 
                crf_mask=attention_mask
            )
        
        # 處理預測結果，僅保留實際長度
        result = []
        for i, pred in enumerate(predictions):
            actual_len = attention_mask[i].sum().item()
            result.append(pred[:actual_len])
        
        return result
    
    def load_model(self, model_path: str):
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
    
    def save_model(self, model_path: str):
        torch.save(self.model.state_dict(), model_path) 