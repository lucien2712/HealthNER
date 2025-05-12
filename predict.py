import os
import json
import argparse
import torch
import pandas as pd
from typing import List, Dict

import data_utils
import config
from bert_model import BertNerModel
from bilstm_crf_model import BiLSTMCRFModel
from bert_bilstm_crf_model import BertBiLSTMCRFModel


def predict_with_bert(texts: List[str], model_path: str = None):
 
    if model_path is None:
        model_path = config.BERT_MODEL_PATH
    
    # 獲取標籤列表
    # 首先嘗試從訓練數據中獲取
    try:
        train_data = data_utils.load_data(config.TRAIN_DATA_PATH)
        train_processed = data_utils.process_data(train_data)
        labels = data_utils.get_entity_labels(train_processed)
    except Exception as e:
        print(f"無法從訓練數據獲取標籤: {e}")
        # 默認的BIOES標籤
        labels = ['O', 'B-BODY', 'I-BODY', 'E-BODY', 'S-BODY', 
                 'B-SIGNS', 'I-SIGNS', 'E-SIGNS', 'S-SIGNS',
                 'B-CHECK', 'I-CHECK', 'E-CHECK', 'S-CHECK',
                 'B-DISEASE', 'I-DISEASE', 'E-DISEASE', 'S-DISEASE',
                 'B-TREATMENT', 'I-TREATMENT', 'E-TREATMENT', 'S-TREATMENT']
    
    # 實例化模型
    model = BertNerModel(
        pretrained_model=config.BERT_CONFIG["pretrained_model"],
        num_labels=len(labels),
        model_path=model_path
    )
    
    # 從保存的權重加載模型
    try:
        model.load_model(model_path)
        print(f"成功加載BERT模型: {model_path}")
    except Exception as e:
        print(f"無法加載模型: {e}")
        return None
    
    # 進行預測
    results = model.predict(texts, labels)
    
    return results


def predict_with_bilstm_crf(texts: List[str], model_path: str = None, vocab_path: str = None):

    if model_path is None:
        model_path = config.BILSTM_CRF_MODEL_PATH
    
    if vocab_path is None:
        vocab_path = os.path.join(config.MODEL_DIR, "bilstm_vocab.pt")
    
    # 加載詞彙表和標籤表
    try:
        vocab_dict = torch.load(vocab_path)
        word_to_ix = vocab_dict["word_to_ix"]
        ix_to_word = vocab_dict["ix_to_word"]
        tag_to_ix = vocab_dict["tag_to_ix"]
        ix_to_tag = vocab_dict["ix_to_tag"]
        print(f"成功加載詞彙表: {vocab_path}")
    except Exception as e:
        print(f"無法加載詞彙表: {e}")
        return None
    
    # 實例化模型
    model = BiLSTMCRFModel(
        vocab_size=len(word_to_ix),
        tag_to_ix=tag_to_ix,
        ix_to_tag=ix_to_tag
    )
    
    # 從保存的權重加載模型
    try:
        model.load_model(model_path)
        print(f"成功加載BiLSTM-CRF模型: {model_path}")
    except Exception as e:
        print(f"無法加載模型: {e}")
        return None
    
    # 將文本處理為字符列表
    char_texts = [[char for char in text] for text in texts]
    
    # 進行預測
    results = model.predict(char_texts, word_to_ix)
    
    return results


def predict_with_bert_bilstm_crf(texts: List[str], model_path: str = None, tag_path: str = None):
  
    if model_path is None:
        model_path = config.BERT_BILSTM_CRF_MODEL_PATH
    
    if tag_path is None:
        tag_path = os.path.join(config.MODEL_DIR, "bert_bilstm_crf_tags.pt")
    
    # 加載標籤表
    try:
        tag_dict = torch.load(tag_path)
        tag_to_ix = tag_dict["tag_to_ix"]
        ix_to_tag = tag_dict["ix_to_tag"]
        print(f"成功加載標籤表: {tag_path}")
    except Exception as e:
        print(f"無法加載標籤表: {e}")
        # 嘗試從訓練數據創建標籤表
        try:
            train_data = data_utils.load_data(config.TRAIN_DATA_PATH)
            train_processed = data_utils.process_data(train_data)
            labels = data_utils.get_entity_labels(train_processed)
            tag_to_ix = {}
            ix_to_tag = {}
            for i, tag in enumerate(labels):
                tag_to_ix[tag] = i
                ix_to_tag[i] = tag
            tag_to_ix[21] = 21
            ix_to_tag[21] = "PAD"
            print("從訓練數據創建標籤表")
        except Exception as e2:
            print(f"無法創建標籤表: {e2}")
            return None
    
    # 實例化模型
    model = BertBiLSTMCRFModel(
        bert_model_name=config.BERT_BILSTM_CRF_CONFIG["pretrained_model"],
        target_size=len(tag_to_ix),
        hidden_dim=config.BERT_BILSTM_CRF_CONFIG["hidden_dim"],
        lstm_layers=config.BERT_BILSTM_CRF_CONFIG["lstm_layers"],
        dropout=config.BERT_BILSTM_CRF_CONFIG["dropout"],
        freeze_bert=config.BERT_BILSTM_CRF_CONFIG["freeze_bert"]
    )
    
    # 從保存的權重加載模型
    try:
        model.load_model(model_path)
        print(f"成功加載BERT+BiLSTM+CRF模型: {model_path}")
    except Exception as e:
        print(f"無法加載模型: {e}")
        return None
    
    # 進行預測
    predicted_indices = model.predict(texts)
    
    # 將索引轉換為標籤
    predictions = []
    for pred_seq in predicted_indices:
        tags = [ix_to_tag[idx] for idx in pred_seq]
        predictions.append(tags)
    
    return predictions


def format_result(texts: List[str], predictions: List[List[str]]):

    formatted_results = []
    
    for text, preds in zip(texts, predictions):
        result = {
            "text": text,
            "entities": []
        }
        
        # 解析BIOES標籤並提取實體
        i = 0
        while i < len(preds):
            if preds[i].startswith('B-'):
                # 開始一個實體
                entity_type = preds[i][2:]  # 提取實體類型
                start_idx = i
                
                # 尋找實體結束位置
                j = i + 1
                while j < len(preds) and preds[j].startswith('I-') and preds[j][2:] == entity_type:
                    j += 1
                
                # 檢查是否以E-結尾
                if j < len(preds) and preds[j].startswith('E-') and preds[j][2:] == entity_type:
                    end_idx = j
                    entity_text = ''.join(text[start_idx:end_idx+1])
                    result["entities"].append({
                        "text": entity_text,
                        "type": entity_type,
                        "start_idx": start_idx,
                        "end_idx": end_idx
                    })
                    i = j + 1
                else:
                    # 不是完整的實體，跳過
                    i += 1
            elif preds[i].startswith('S-'):
                # 單字實體
                entity_type = preds[i][2:]
                result["entities"].append({
                    "text": text[i],
                    "type": entity_type,
                    "start_idx": i,
                    "end_idx": i
                })
                i += 1
            else:
                # 不是實體，跳過
                i += 1
        
        formatted_results.append(result)
    
    return formatted_results


def save_predictions(predictions, output_file):

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(predictions, f, ensure_ascii=False, indent=2)
    
    print(f"預測結果已保存到 {output_file}")


def main():
   
    parser = argparse.ArgumentParser(description="使用訓練好的模型進行命名實體識別預測")
    parser.add_argument("--model", type=str, choices=["bert", "bilstm", "bert_bilstm_crf"], default="bert", 
                        help="選擇使用的模型類型：bert、bilstm或bert_bilstm_crf")
    parser.add_argument("--input", type=str, default=config.REAL_TEST_DATA_PATH, 
                        help="輸入數據文件路徑")
    parser.add_argument("--output", type=str, default="predictions.json", 
                        help="預測結果輸出文件路徑")
    parser.add_argument("--model_path", type=str, default=None, 
                        help="模型路徑，默認使用配置文件中的路徑")
    parser.add_argument("--text", type=str, default=None, 
                        help="直接輸入文本進行預測，優先於輸入文件")
    
    args = parser.parse_args()
    
    # 獲取輸入文本
    if args.text:
        texts = [args.text]
    else:
        try:
            # 嘗試加載JSON數據
            data = data_utils.load_data(args.input)
            
            if "real_test" in args.input:
                # 真實測試數據格式
                texts = data["text"].tolist()
            else:
                # 訓練數據格式
                processed_data = data_utils.process_data(data)
                texts = [''.join(tokens) for tokens in processed_data["tokens"]]
        except Exception as e:
            print(f"讀取輸入文件失敗: {e}")
            return
    
    # 使用選擇的模型進行預測
    if args.model == "bert":
        predictions = predict_with_bert(texts, model_path=args.model_path)
    elif args.model == "bilstm":
        predictions = predict_with_bilstm_crf(texts, model_path=args.model_path)
    else:
        predictions = predict_with_bert_bilstm_crf(texts, model_path=args.model_path)
    
    if predictions is None:
        print("預測失敗")
        return
    
    # 格式化並保存結果
    formatted_results = format_result(texts, predictions)
    save_predictions(formatted_results, args.output)
    
    # 打印一些示例結果
    print("\n預測示例:")
    for i, result in enumerate(formatted_results[:3]):
        print(f"文本: {result['text']}")
        print("識別出的實體:")
        for entity in result["entities"]:
            print(f"  - {entity['text']} ({entity['type']})")
        print()


if __name__ == "__main__":
    main() 