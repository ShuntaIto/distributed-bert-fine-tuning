import os
import torch

import pandas as pd
import tarfile
from glob import glob
import linecache
import urllib.request

download_path = "livedoor_news_corpus.tar.gz"
extract_path = "livedoor/"

def download_dataset():
    if not os.path.isfile("livedoor_news_corpus.tar.gz"):
        urllib.request.urlretrieve(
            "https://www.rondhuit.com/download/ldcc-20140209.tar.gz", download_path)
    with tarfile.open(download_path) as t:
        t.extractall(extract_path)

def preprocess():
    ## https://qiita.com/m__k/items/841950a57a0d7ff05506
    categories = [name for name in os.listdir(
        extract_path + "text") if os.path.isdir(extract_path + "text/" + name)]

    datasets = pd.DataFrame(columns=["title", "category"])
    for cat in categories:
        path = extract_path + "text/" + cat + "/*.txt"
        files = glob(path)
        for text_name in files:
            title = linecache.getline(text_name, 3)
            s = pd.Series([title, cat], index=datasets.columns)
            datasets = datasets.append(s, ignore_index=True)

    datasets = datasets.sort_values("title").reset_index(drop=True)

    categories = list(set(datasets['category']))
    id2cat = dict(zip(list(range(len(categories))), categories))
    cat2id = dict(zip(categories, list(range(len(categories)))))

    datasets['category_id'] = datasets['category'].map(cat2id)
    datasets = datasets[['title', 'category_id']]
    return datasets

class LivedoorDatasets(torch.utils.data.Dataset):
    def __init__(self, transform=None):
        self.transform = transform
        download_dataset()
        datasets = preprocess()
        self.data = list(datasets["title"])
        self.label = list(datasets["category_id"])
        if not len(self.data) == len(self.label):
            raise ValueError("Invalid dataset")
        self.datanum = len(self.data)

    def __len__(self):
        return self.datanum

    def __getitem__(self, idx):
        out_data = self.data[idx]
        out_label = self.label[idx]

        if self.transform:
            out_data = self.transform(["input_ids"][0])

        return out_data, out_label

class LivedoorDatasetPreprocesser:
    def __init__(self,train_ratio=0.8, val_ratio=0.1, generator=None):
        super().__init__()

        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = 1.0 - train_ratio - val_ratio
        self.generator = generator
        self.setup()
        
    def _prepare_data(self):
        self.dataset = LivedoorDatasets()
    
    def _load_data(self):
        self.n_samples = len(self.dataset)
        self.train_size = int(self.n_samples * self.train_ratio)
        print(f"the number of training datasets : {self.train_size}")
        self.val_size = int(self.n_samples * self.val_ratio)
        print(f"the number of validating datasets : {self.val_size}")
        self.test_size = self.n_samples - self.train_size - self.val_size
        print(f"the number of test datasets : {self.test_size}")
        self._train_dataset, self._val_dataset, self._test_dataset = torch.utils.data.random_split(
        self.dataset, [self.train_size, self.val_size, self.test_size], generator=self.generator)
 
    def setup(self):
        self._prepare_data()
        self._load_data()
    
    def train_dataset(self):
        return self._train_dataset

    def val_dataset(self):
        return self._val_dataset

    def test_dataset(self):
        return self._test_dataset