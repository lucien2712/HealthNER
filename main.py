import os
import argparse
import time
from pathlib import Path

from train import train_bert, train_bilstm_crf, train_bert_bilstm_crf
from predict import predict_with_bert, predict_with_bilstm_crf, predict_with_bert_bilstm_crf, format_result, save_predictions
import config


def main():
    """
    主函數，處理命令行參數並執行相應功能
    """
    # 創建主解析器
    parser = argparse.ArgumentParser(
        description="中文醫療命名實體識別系統",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # 創建子命令解析器
    subparsers = parser.add_subparsers(dest="command", help="子命令")
    
    # 訓練子命令
    train_parser = subparsers.add_parser("train", help="訓練模型")
    train_parser.add_argument("--model", type=str, choices=["bert", "bilstm", "bert_bilstm_crf", "all"], default="all",
                            help="選擇訓練的模型類型：bert、bilstm、bert_bilstm_crf或all")
    train_parser.add_argument("--epochs_bert", type=int, default=None,
                            help="BERT模型訓練輪數，覆蓋配置文件")
    train_parser.add_argument("--epochs_bilstm", type=int, default=None,
                            help="BiLSTM-CRF模型訓練輪數，覆蓋配置文件")
    train_parser.add_argument("--epochs_bert_bilstm_crf", type=int, default=None,
                            help="BERT+BiLSTM+CRF模型訓練輪數，覆蓋配置文件")
    
    # 預測子命令
    predict_parser = subparsers.add_parser("predict", help="預測命名實體")
    predict_parser.add_argument("--model", type=str, choices=["bert", "bilstm", "bert_bilstm_crf"], default="bert",
                              help="選擇使用的模型類型：bert、bilstm或bert_bilstm_crf")
    predict_parser.add_argument("--input", type=str, default=config.REAL_TEST_DATA_PATH,
                              help="輸入數據文件路徑")
    predict_parser.add_argument("--output", type=str, default="predictions.json",
                              help="預測結果輸出文件路徑")
    predict_parser.add_argument("--model_path", type=str, default=None,
                              help="模型路徑，默認使用配置文件中的路徑")
    predict_parser.add_argument("--text", type=str, default=None,
                              help="直接輸入文本進行預測，優先於輸入文件")
    
    # 解析參數
    args = parser.parse_args()
    
    # 創建模型保存目錄
    if not os.path.exists(config.MODEL_DIR):
        os.makedirs(config.MODEL_DIR)
    
    # 執行相應的功能
    if args.command == "train":
        # 訓練模型
        if args.epochs_bert is not None:
            config.BERT_CONFIG["num_train_epochs"] = args.epochs_bert
        
        if args.epochs_bilstm is not None:
            config.BILSTM_CRF_CONFIG["num_train_epochs"] = args.epochs_bilstm
            
        if args.epochs_bert_bilstm_crf is not None:
            config.BERT_BILSTM_CRF_CONFIG["num_train_epochs"] = args.epochs_bert_bilstm_crf
        
        start_time = time.time()
        
        if args.model == "bert" or args.model == "all":
            train_bert()
        
        if args.model == "bilstm" or args.model == "all":
            train_bilstm_crf()
            
        if args.model == "bert_bilstm_crf" or args.model == "all":
            train_bert_bilstm_crf()
        
        total_time = time.time() - start_time
        print(f"訓練完成，總耗時: {total_time:.2f} 秒")
    
    elif args.command == "predict":
        # 預測
        start_time = time.time()
        
        # 獲取輸入文本
        if args.text:
            texts = [args.text]
            print(f"輸入文本: {args.text}")
        else:
            try:
                # 從data_utils中加載必要的函數
                import data_utils
                
                # 嘗試加載JSON數據
                data = data_utils.load_data(args.input)
                print(f"成功加載輸入文件: {args.input}")
                
                if "real_test" in args.input:
                    # 真實測試數據格式
                    texts = data["text"].tolist()
                else:
                    # 訓練數據格式
                    processed_data = data_utils.process_data(data)
                    texts = [''.join(tokens) for tokens in processed_data["tokens"]]
                
                print(f"文本數量: {len(texts)}")
            except Exception as e:
                print(f"讀取輸入文件失敗: {e}")
                return
        
        # 使用選擇的模型進行預測
        if args.model == "bert":
            print("使用BERT模型進行預測...")
            predictions = predict_with_bert(texts, model_path=args.model_path)
        elif args.model == "bilstm":
            print("使用BiLSTM-CRF模型進行預測...")
            predictions = predict_with_bilstm_crf(texts, model_path=args.model_path)
        else:
            print("使用BERT+BiLSTM+CRF模型進行預測...")
            predictions = predict_with_bert_bilstm_crf(texts, model_path=args.model_path)
        
        if predictions is None:
            print("預測失敗")
            return
        
        # 格式化並保存結果
        formatted_results = format_result(texts, predictions)
        save_predictions(formatted_results, args.output)
        
        total_time = time.time() - start_time
        print(f"預測完成，總耗時: {total_time:.2f} 秒")
        
        
        print("\n預測示例:")
        for i, result in enumerate(formatted_results[:3]):
            print(f"文本: {result['text']}")
            print("識別出的實體:")
            for entity in result["entities"]:
                print(f"  - {entity['text']} ({entity['type']})")
            print()
    
    else:
        # 未提供命令，顯示幫助
        parser.print_help()


if __name__ == "__main__":
    main() 