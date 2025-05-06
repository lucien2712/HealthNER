import os
import argparse
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple

import data_utils
import config
from bert_model import BertNerModel
from bilstm_crf_model import BiLSTMCRFModel
from bert_bilstm_crf_model import BertBiLSTMCRFModel


def prepare_bert_data():
    """
    準備BERT模型的數據
    
    Returns:
        訓練集和測試集
    """
    # 加載數據
    train_data = data_utils.load_data(config.TRAIN_DATA_PATH)
    test_data = data_utils.load_data(config.TEST_DATA_PATH)
    
    # 處理數據
    train_processed = data_utils.process_data(train_data)
    test_processed = data_utils.process_data(test_data)
    
    # 獲取label列表
    labels = data_utils.get_entity_labels(train_processed)
    
    # 轉換為HuggingFace格式
    train_dataset = data_utils.convert_to_hf_dataset(train_processed)
    test_dataset = data_utils.convert_to_hf_dataset(test_processed)
    
    dataset_dict = {
        "train": train_dataset,
        "test": test_dataset
    }
    
    return dataset_dict, labels


def prepare_bilstm_data() -> Tuple[dict, dict, dict]:
    """
    準備BiLSTM-CRF模型的數據
    
    Returns:
        (數據加載器, 詞彙表, 標籤表)
    """
    # 加載數據
    train_data = data_utils.load_data(config.TRAIN_DATA_PATH)
    test_data = data_utils.load_data(config.TEST_DATA_PATH)
    
    # 處理數據
    train_processed = data_utils.process_data(train_data)
    test_processed = data_utils.process_data(test_data)
    
    # 建立詞彙表和標籤表
    word_to_ix, ix_to_word, tag_to_ix, ix_to_tag = data_utils.build_vocab(train_processed)
    
    # 轉換為DataLoader
    train_loader = data_utils.create_dataloader(
        train_processed, 
        word_to_ix, 
        tag_to_ix, 
        batch_size=config.BILSTM_CRF_CONFIG["batch_size"]
    )
    test_loader = data_utils.create_dataloader(
        test_processed, 
        word_to_ix, 
        tag_to_ix, 
        batch_size=config.BILSTM_CRF_CONFIG["batch_size"],
        shuffle=False
    )
    
    # 返回數據
    dataloader_dict = {
        "train": train_loader,
        "test": test_loader
    }
    
    vocab_dict = {
        "word_to_ix": word_to_ix,
        "ix_to_word": ix_to_word
    }
    
    tag_dict = {
        "tag_to_ix": tag_to_ix,
        "ix_to_tag": ix_to_tag
    }
    
    return dataloader_dict, vocab_dict, tag_dict


def prepare_bert_bilstm_crf_data() -> Tuple[dict, dict]:
    """
    準備BERT+BiLSTM+CRF模型的數據
    
    Returns:
        (數據加載器, 標籤字典)
    """
    # 加載數據
    train_data = data_utils.load_data(config.TRAIN_DATA_PATH)
    test_data = data_utils.load_data(config.TEST_DATA_PATH)
    
    # 處理數據
    train_processed = data_utils.process_data(train_data)
    test_processed = data_utils.process_data(test_data)
    
    # 獲取標籤列表和標籤映射
    labels = data_utils.get_entity_labels(train_processed)
    tag_to_ix = {}
    ix_to_tag = {}
    for i, tag in enumerate(labels):
        tag_to_ix[tag] = i
        ix_to_tag[i] = tag
    
    # 填充標籤
    tag_to_ix[21] = 21
    ix_to_tag[21] = "PAD"
    
    # 創建模型實例
    model = BertBiLSTMCRFModel(
        bert_model_name=config.BERT_BILSTM_CRF_CONFIG["pretrained_model"],
        target_size=len(tag_to_ix),
        hidden_dim=config.BERT_BILSTM_CRF_CONFIG["hidden_dim"],
        lstm_layers=config.BERT_BILSTM_CRF_CONFIG["lstm_layers"],
        dropout=config.BERT_BILSTM_CRF_CONFIG["dropout"],
        freeze_bert=config.BERT_BILSTM_CRF_CONFIG["freeze_bert"],
        max_seq_length=config.BERT_BILSTM_CRF_CONFIG["max_seq_length"]
    )
    
    # 準備訓練數據
    train_data_dict = model.prepare_data(train_processed, tag_to_ix)
    test_data_dict = model.prepare_data(test_processed, tag_to_ix)
    
    # 創建DataLoader
    from torch.utils.data import DataLoader, TensorDataset
    
    train_dataset = TensorDataset(
        train_data_dict["input_ids"],
        train_data_dict["attention_mask"],
        train_data_dict["labels"],
        train_data_dict["crf_mask"]
    )
    
    test_dataset = TensorDataset(
        test_data_dict["input_ids"],
        test_data_dict["attention_mask"],
        test_data_dict["labels"],
        test_data_dict["crf_mask"]
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BERT_BILSTM_CRF_CONFIG["batch_size"],
        shuffle=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BERT_BILSTM_CRF_CONFIG["batch_size"],
        shuffle=False
    )
    
    dataloader_dict = {
        "train": train_loader,
        "test": test_loader
    }
    
    tag_dict = {
        "tag_to_ix": tag_to_ix,
        "ix_to_tag": ix_to_tag
    }
    
    return dataloader_dict, tag_dict


def train_bert():
    """
    訓練BERT模型
    """
    print("準備BERT模型數據...")
    dataset_dict, labels = prepare_bert_data()
    
    # 創建模型
    print("初始化BERT模型...")
    bert_model = BertNerModel(
        pretrained_model=config.BERT_CONFIG["pretrained_model"],
        num_labels=len(labels),
        model_path=config.BERT_MODEL_PATH
    )
    
    # 設置訓練參數
    training_args = {
        "learning_rate": config.BERT_CONFIG["learning_rate"],
        "per_device_train_batch_size": config.BERT_CONFIG["batch_size"],
        "per_device_eval_batch_size": config.BERT_CONFIG["batch_size"],
        "num_train_epochs": config.BERT_CONFIG["num_train_epochs"],
        "weight_decay": config.BERT_CONFIG["weight_decay"],
        "warmup_ratio": config.BERT_CONFIG["warmup_ratio"],
        "save_steps": config.BERT_CONFIG["save_steps"],
        "save_total_limit": config.BERT_CONFIG["save_total_limit"],
        "evaluation_strategy": "epoch",
        "report_to": "none"
    }
    
    # 訓練模型
    print(f"開始訓練BERT模型，epochs={config.BERT_CONFIG['num_train_epochs']}...")
    bert_model.train(
        train_dataset=dataset_dict["train"],
        eval_dataset=dataset_dict["test"],
        labels=labels,
        training_args=training_args
    )
    
    print(f"BERT模型訓練完成，模型保存在 {config.BERT_MODEL_PATH}")


def train_bilstm_crf():
    """
    訓練BiLSTM-CRF模型
    """
    print("準備BiLSTM-CRF模型數據...")
    dataloader_dict, vocab_dict, tag_dict = prepare_bilstm_data()
    
    # 創建保存路徑
    if not os.path.exists(config.MODEL_DIR):
        os.makedirs(config.MODEL_DIR)
    
    # 創建模型
    print("初始化BiLSTM-CRF模型...")
    bilstm_model = BiLSTMCRFModel(
        vocab_size=len(vocab_dict["word_to_ix"]),
        tag_to_ix=tag_dict["tag_to_ix"],
        ix_to_tag=tag_dict["ix_to_tag"],
        embedding_dim=config.BILSTM_CRF_CONFIG["embedding_dim"],
        hidden_dim=config.BILSTM_CRF_CONFIG["hidden_dim"],
        num_layers=config.BILSTM_CRF_CONFIG["num_layers"],
        dropout=config.BILSTM_CRF_CONFIG["dropout"]
    )
    
    # 訓練模型
    print(f"開始訓練BiLSTM-CRF模型，epochs={config.BILSTM_CRF_CONFIG['num_train_epochs']}...")
    bilstm_model.train(
        train_data_loader=dataloader_dict["train"],
        valid_data_loader=dataloader_dict["test"],
        epochs=config.BILSTM_CRF_CONFIG["num_train_epochs"],
        learning_rate=config.BILSTM_CRF_CONFIG["learning_rate"],
        model_path=config.BILSTM_CRF_MODEL_PATH
    )
    
    # 保存詞彙表和標籤表
    vocab_path = os.path.join(config.MODEL_DIR, "bilstm_vocab.pt")
    torch.save({
        "word_to_ix": vocab_dict["word_to_ix"],
        "ix_to_word": vocab_dict["ix_to_word"],
        "tag_to_ix": tag_dict["tag_to_ix"],
        "ix_to_tag": tag_dict["ix_to_tag"]
    }, vocab_path)
    
    print(f"BiLSTM-CRF模型訓練完成，模型保存在 {config.BILSTM_CRF_MODEL_PATH}")
    print(f"詞彙表和標籤表保存在 {vocab_path}")


def train_bert_bilstm_crf():
    """
    訓練BERT+BiLSTM+CRF模型
    """
    print("準備BERT+BiLSTM+CRF模型數據...")
    dataloader_dict, tag_dict = prepare_bert_bilstm_crf_data()
    
    # 創建保存路徑
    if not os.path.exists(config.MODEL_DIR):
        os.makedirs(config.MODEL_DIR)
    
    # 創建模型
    print("初始化BERT+BiLSTM+CRF模型...")
    bert_bilstm_crf_model = BertBiLSTMCRFModel(
        bert_model_name=config.BERT_BILSTM_CRF_CONFIG["pretrained_model"],
        target_size=len(tag_dict["tag_to_ix"]),
        hidden_dim=config.BERT_BILSTM_CRF_CONFIG["hidden_dim"],
        lstm_layers=config.BERT_BILSTM_CRF_CONFIG["lstm_layers"],
        dropout=config.BERT_BILSTM_CRF_CONFIG["dropout"],
        freeze_bert=config.BERT_BILSTM_CRF_CONFIG["freeze_bert"],
        max_seq_length=config.BERT_BILSTM_CRF_CONFIG["max_seq_length"]
    )
    
    # 訓練模型
    print(f"開始訓練BERT+BiLSTM+CRF模型，epochs={config.BERT_BILSTM_CRF_CONFIG['num_train_epochs']}...")
    
    # 設置DataLoader格式轉換函數
    def collate_fn(batch):
        # batch是一個包含TensorDataset元素的列表
        # 每個元素是(input_ids, attention_mask, labels, crf_mask)
        input_ids = torch.stack([item[0] for item in batch])
        attention_mask = torch.stack([item[1] for item in batch])
        labels = torch.stack([item[2] for item in batch])
        crf_mask = torch.stack([item[3] for item in batch])
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "crf_mask": crf_mask
        }
    
    # 更新DataLoader的collate_fn
    train_loader = dataloader_dict["train"]
    test_loader = dataloader_dict["test"]
    
    # 訓練模型
    bert_bilstm_crf_model.train(
        train_data_loader=train_loader,
        valid_data_loader=test_loader,
        epochs=config.BERT_BILSTM_CRF_CONFIG["num_train_epochs"],
        learning_rate=config.BERT_BILSTM_CRF_CONFIG["learning_rate"],
        weight_decay=config.BERT_BILSTM_CRF_CONFIG["weight_decay"],
        model_path=config.BERT_BILSTM_CRF_MODEL_PATH
    )
    
    # 保存標籤表
    tag_path = os.path.join(config.MODEL_DIR, "bert_bilstm_crf_tags.pt")
    torch.save(tag_dict, tag_path)
    
    print(f"BERT+BiLSTM+CRF模型訓練完成，模型保存在 {config.BERT_BILSTM_CRF_MODEL_PATH}")
    print(f"標籤表保存在 {tag_path}")


def main():
    """
    主函數
    """
    parser = argparse.ArgumentParser(description="訓練中文命名實體識別模型")
    parser.add_argument("--model", type=str, choices=["bert", "bilstm", "bert_bilstm_crf", "all"], default="all", 
                        help="選擇訓練的模型類型：bert、bilstm、bert_bilstm_crf或all")
    args = parser.parse_args()
    
    # 創建模型保存目錄
    if not os.path.exists(config.MODEL_DIR):
        os.makedirs(config.MODEL_DIR)
    
    # 選擇訓練的模型
    if args.model == "bert" or args.model == "all":
        train_bert()
    
    if args.model == "bilstm" or args.model == "all":
        train_bilstm_crf()
        
    if args.model == "bert_bilstm_crf" or args.model == "all":
        train_bert_bilstm_crf()


if __name__ == "__main__":
    main() 