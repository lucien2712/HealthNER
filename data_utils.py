
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Any, Union
from sklearn.model_selection import train_test_split
from datasets import Dataset, ClassLabel, Sequence, Features, Value, DatasetDict


def load_data(file_path: str) -> pd.DataFrame:
 
    if "real_test" in file_path:
        data = pd.read_json(file_path)
    else:
        data = pd.read_json(file_path, lines=True)
    
    return data


def process_data(data: pd.DataFrame) -> pd.DataFrame:

    processed_data = data.loc[:, ["character", "character_label"]]
    processed_data.columns = ["tokens", "ner_tags"]
    return processed_data


def split_data(data: pd.DataFrame, test_size: float = 0.1, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:

    return train_test_split(data, test_size=test_size, random_state=random_state)


def get_tag_names(df: pd.DataFrame) -> List[str]:
 
    tag_names = []
    for tags in df.ner_tags:
        tag_names += [tag for tag in list(set(tags)) if tag not in tag_names]
    return tag_names


def df_to_dataset(df: pd.DataFrame, tag_names: List[str]) -> Dataset:

    # 創建標籤類
    tags = ClassLabel(num_classes=len(tag_names), names=tag_names)
    
    # 定義數據集結構
    features = {
        "ner_tags": Sequence(tags),
        "tokens": Sequence(feature=Value(dtype="string"))
    }
    
    # 將標籤轉換為數字
    ner_tags = [tags.str2int(tag_list) for tag_list in df["ner_tags"].values.tolist()]
    tokens = df["tokens"].values.tolist()
    
    # 創建Dataset
    dataset = Dataset.from_dict(
        {
            "ner_tags": ner_tags,
            "tokens": tokens
        },
        features=Features(features)
    )
    
    return dataset


def prepare_datasets(train_df: pd.DataFrame, test_df: pd.DataFrame, tag_names: List[str]) -> DatasetDict:

    train_dataset = df_to_dataset(train_df, tag_names)
    test_dataset = df_to_dataset(test_df, tag_names)
    
    return DatasetDict({
        "train": train_dataset,
        "test": test_dataset
    })


def tokenize_and_align_labels(examples: Dict[str, List], tokenizer, max_length: int = 128) -> Dict[str, List]:

    tokenized_inputs = tokenizer(
        examples["tokens"],
        truncation=True,
        is_split_into_words=True,
        max_length=max_length,
        padding="max_length"
    )
    
    labels = []
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        label_ids = []
        
        for word_id in word_ids:
            if word_id is None:
                label_ids.append(-100)
            else:
                if word_id < len(label):
                    label_ids.append(label[word_id])
                else:
                    label_ids.append(-100)
        
        labels.append(label_ids)
    
    tokenized_inputs["labels"] = labels
    return tokenized_inputs 