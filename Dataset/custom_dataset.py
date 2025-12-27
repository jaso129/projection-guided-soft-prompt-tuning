from datasets import load_dataset
import random
import torch
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, tokenizer, task, split="train", max_length=512, sample_size=-1):
        self.tokenizer = tokenizer
        self.task = task
        self.split = split
        self.max_length = max_length
        self.sample_size = sample_size

        # 判斷是否為 GLUE 任務
        self.glue_tasks = ['cola', 'sst2', 'mrpc', 'qqp', 'stsb', 'mnli', 'qnli', 'rte', 'wnli']
        self.is_glue = self.task in self.glue_tasks

        self.task_to_keys = {
            "cola": ("sentence", None),
            "mnli": ("premise", "hypothesis"),
            "mrpc": ("sentence1", "sentence2"),
            "qqp": ("question1", "question2"),
            "sst2": ("sentence", None),
            "rte": ("sentence1", "sentence2"),
            "stsb": ("sentence1", "sentence2"),
            "qnli": ("question", "sentence"),
            "wnli": ("sentence1", "sentence2"),
            "ag_news": ("text", None),
            "yelp_review_full": ("text", None),
            "dbpedia_14": ("title", "content")
        }

        self.task_to_labels = {
            "cola": ("not_acceptable", "acceptable"),
            "mnli": ("entailment", "neutral", "contradiction"),
            "mrpc": ("not_equivalent", "equivalent"),
            "qqp": ("not_duplicate", "duplicate"),
            "sst2": ("negative", "positive"),
            "rte": ("entailment", "not_entailment"),
            "stsb": ("low", "high"),  # placeholder
            "qnli": ("entailment", "not_entailment"),
            "wnli": ("not_entailment", "entailment"),
            "ag_news": ("world", "sports", "business", "science"),
            "yelp_review_full": ("terrible", "bad", "middle", "good", "wonderful"),
            "dbpedia_14": ("company", "educationalinstitution", "artist", "athlete", "officeholder",
                           "meanoftransportation", "building", "naturalplace", "village", "animal",
                           "plant", "album", "film", "writtenwork")
        }

        if task not in self.task_to_keys:
            raise ValueError(f"Task {task} is not supported.")

        self.data = self.load_data(task, split)

    def load_data(self, task, split):
        if self.is_glue:
            dataset = load_dataset("glue", task, split=split)
        else:
            dataset = load_dataset(task, split=split)

        if self.sample_size > 0 and self.sample_size < len(dataset):
            indices = random.sample(range(len(dataset)), self.sample_size)
            dataset = dataset.select(indices)

        return dataset

    def preprocess_function(self, example):
        keys = self.task_to_keys[self.task]
        label_key = "label"

        if keys[1] is not None:
            text = example[keys[0]] + " " + example[keys[1]]
        else:
            text = example[keys[0]]

        source = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )

        # 🛡️ 判斷是否為回歸任務（如 STS-B），直接將 float label 當作 string 預測
        if self.task == "stsb":
            label_text = str(round(float(example[label_key]), 1))  # 如 "4.3"
        else:
            label_idx = int(example[label_key])
            label_text = self.task_to_labels[self.task][label_idx]

        target = self.tokenizer(
            label_text,
            truncation=True,
            padding="max_length",
            max_length=2,
            return_tensors="pt"
        )

        return {
            "input_ids": source["input_ids"].squeeze(0),
            "attention_mask": source["attention_mask"].squeeze(0),
            "labels": target["input_ids"].squeeze(0),
            "raw_text": text
        }


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        example = self.data[idx]
        return self.preprocess_function(example)
