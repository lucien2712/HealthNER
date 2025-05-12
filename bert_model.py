
import os
import torch
import numpy as np
from typing import Dict, List, Tuple, Any
from transformers import (
    BertForTokenClassification,
    BertTokenizerFast,
    Trainer,
    TrainingArguments,
    EvalPrediction
)
from seqeval.metrics import classification_report, f1_score, precision_score, recall_score
from datasets import DatasetDict


class BertNerModel:
    def __init__(self, pretrained_model: str, num_labels: int, model_path: str = None):

        self.pretrained_model = pretrained_model
        self.num_labels = num_labels
        self.model_path = model_path
        
        self.tokenizer = BertTokenizerFast.from_pretrained(pretrained_model)
        
        if model_path and os.path.exists(model_path):
            self.model = BertForTokenClassification.from_pretrained(model_path)
        else:
            self.model = BertForTokenClassification.from_pretrained(
                pretrained_model, 
                num_labels=num_labels
            )
        
        self.id2label = {i: f"LABEL_{i}" for i in range(num_labels)}
        self.label2id = {v: k for k, v in self.id2label.items()}
    
    def set_label_map(self, id2label: Dict[int, str]):

        self.id2label = id2label
        self.label2id = {v: k for k, v in id2label.items()}
        self.model.config.id2label = id2label
        self.model.config.label2id = self.label2id
    
    def train(self, 
              train_dataset, 
              eval_dataset=None, 
              output_dir: str = "./results",
              num_train_epochs: int = 3,
              batch_size: int = 16,
              learning_rate: float = 5e-5,
              warmup_ratio: float = 0.1,
              weight_decay: float = 0.01,
              save_steps: int = 1000,
              save_total_limit: int = 2):
      
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            learning_rate=learning_rate,
            warmup_ratio=warmup_ratio,
            weight_decay=weight_decay,
            evaluation_strategy="epoch" if eval_dataset else "no",
            save_strategy="steps",
            save_steps=save_steps,
            save_total_limit=save_total_limit,
            load_best_model_at_end=True if eval_dataset else False,
            logging_dir="./logs",
            logging_steps=100,
        )
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=self.compute_metrics,
            tokenizer=self.tokenizer,
        )
        
        trainer.train()
        
        if self.model_path:
            self.model.save_pretrained(self.model_path)
            self.tokenizer.save_pretrained(self.model_path)
        
        return trainer
    
    def compute_metrics(self, p: EvalPrediction) -> Dict[str, float]:
   
        predictions, labels = self.align_predictions(p.predictions, p.label_ids)
        
        report = classification_report(labels, predictions, output_dict=True)
        metrics = {
            "precision": precision_score(labels, predictions),
            "recall": recall_score(labels, predictions),
            "f1": f1_score(labels, predictions),
            "accuracy": (np.array(predictions) == np.array(labels)).mean()
        }
        
        return metrics

    def align_predictions(self, predictions: np.ndarray, label_ids: np.ndarray) -> Tuple[List[List[str]], List[List[str]]]:
  
        preds = np.argmax(predictions, axis=2)
        
        batch_size, seq_len = preds.shape
        
        out_label_list = [[] for _ in range(batch_size)]
        preds_list = [[] for _ in range(batch_size)]
        
        for i in range(batch_size):
            for j in range(seq_len):
                if label_ids[i, j] != -100:
                    out_label_list[i].append(self.id2label[label_ids[i, j]])
                    preds_list[i].append(self.id2label[preds[i, j]])
        
        return preds_list, out_label_list
    
    def predict(self, text: List[str]) -> List[List[str]]:
    
        self.model.eval()
        
        inputs = self.tokenizer(
            text,
            is_split_into_words=True,
            return_tensors="pt",
            padding=True,
            truncation=True
        )
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.argmax(outputs.logits, dim=2)
        
        results = []
        for i, pred in enumerate(predictions):
            word_ids = inputs.word_ids(i)
            previous_word_idx = None
            word_preds = []
            
            for j, word_idx in enumerate(word_ids):
                if word_idx is None or word_idx == previous_word_idx:
                    continue
                word_preds.append(self.id2label[pred[j].item()])
                previous_word_idx = word_idx
            
            results.append(word_preds[:len(text)])
        
        return results 