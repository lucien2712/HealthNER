# 中文醫療命名實體識別系統

本專案實現了基於深度學習的中文醫療命名實體識別系統，提供了三種不同的實現方式：
1. **基於BERT的命名實體識別**：使用預訓練的中文BERT模型進行微調
2. **基於BiLSTM-CRF的命名實體識別**：使用雙向LSTM結合CRF層進行序列標注
3. **基於BERT+BiLSTM+CRF的命名實體識別**：結合BERT的強大表示能力和BiLSTM-CRF的序列標注優勢

## 專案功能

- 識別中文醫療文本中的命名實體，包括：
  - BODY: 身體部位
  - SIGNS: 症狀和體徵
  - CHECK: 檢查和檢驗
  - DISEASE: 疾病和診斷
  - TREATMENT: 治療方式

- 支持BIOES標註方案
  - B: 實體開始位置
  - I: 實體中間位置
  - E: 實體結束位置
  - S: 單字實體
  - O: 非實體


## 專案結構

```
├── config.py             
├── data_utils.py           
├── bert_model.py           
├── bilstm_crf_model.py     
├── bert_bilstm_crf_model.py
├── train.py                
├── predict.py              
├── main.py                 
├── train.json             
├── test.json             
├── real_test.json        
└── saved_models/           
```

## 模型說明

### BERT模型
- 基於預訓練的`ckiplab/bert-base-chinese`模型
- 透過在醫療NER任務上微調，學習識別醫療實體


### BiLSTM-CRF模型
- 使用BiLSTM捕獲上下文特徵
- 結合CRF層保證輸出標籤的一致性


### BERT+BiLSTM+CRF模型
- 結合了BERT的強大表示能力和BiLSTM-CRF的序列標註優勢
- BERT負責提取深層語義特徵
- BiLSTM進一步捕獲上下文信息
- CRF層確保標籤序列的一致性

## 使用方法

### 訓練模型

```bash
# 訓練所有模型
python main.py train --model all

# 僅訓練BERT模型
python main.py train --model bert

# 僅訓練BiLSTM-CRF模型
python main.py train --model bilstm

# 僅訓練BERT+BiLSTM+CRF模型
python main.py train --model bert_bilstm_crf

# 自定義訓練輪數
python main.py train --model bert --epochs_bert 5
```

### 使用模型進行預測

```bash
# 使用BERT模型預測
python main.py predict --model bert --input real_test.json --output bert_predictions.json

# 使用BiLSTM-CRF模型預測
python main.py predict --model bilstm --input real_test.json --output bilstm_predictions.json

# 使用BERT+BiLSTM+CRF模型預測
python main.py predict --model bert_bilstm_crf --input real_test.json --output bert_bilstm_crf_predictions.json

# 直接輸入文本進行預測
python main.py predict --model bert --text "患者出現頭痛、發熱等症狀，建議服用布洛芬。"
```