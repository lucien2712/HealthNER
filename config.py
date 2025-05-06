import os

# 數據文件路徑
TRAIN_DATA_PATH = "train.json"
TEST_DATA_PATH = "test.json"
REAL_TEST_DATA_PATH = "real_test.json"

# BERT 模型設定
BERT_CONFIG = {
    "pretrained_model": "ckiplab/bert-base-chinese",
    "max_seq_length": 128,
    "batch_size": 32,
    "learning_rate": 5e-5,
    "num_train_epochs": 3,
    "warmup_ratio": 0.1,
    "weight_decay": 0.01,
    "save_steps": 1000,
    "save_total_limit": 2,
}

# BiLSTM-CRF 模型設定
BILSTM_CRF_CONFIG = {
    "embedding_dim": 200,
    "hidden_dim": 256,
    "dropout": 0.5,
    "num_layers": 2,
    "batch_size": 64,
    "learning_rate": 0.001,
    "num_train_epochs": 10,
}

# BERT+BiLSTM+CRF 模型設定
BERT_BILSTM_CRF_CONFIG = {
    "pretrained_model": "ckiplab/bert-base-chinese",
    "max_seq_length": 150,
    "hidden_dim": 768,
    "lstm_layers": 2,
    "dropout": 0.1,
    "freeze_bert": True,
    "batch_size": 16,
    "learning_rate": 0.01,
    "weight_decay": 1e-5,
    "num_train_epochs": 15,
}

# 模型保存路徑
MODEL_DIR = "saved_models"
BERT_MODEL_PATH = os.path.join(MODEL_DIR, "bert_ner_model")
BILSTM_CRF_MODEL_PATH = os.path.join(MODEL_DIR, "bilstm_crf_model.pt")
BERT_BILSTM_CRF_MODEL_PATH = os.path.join(MODEL_DIR, "bert_bilstm_crf_model.pt") 