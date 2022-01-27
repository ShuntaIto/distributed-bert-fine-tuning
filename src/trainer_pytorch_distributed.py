import os
import argparse
import random

import numpy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from transformers import BertJapaneseTokenizer
from transformers import AutoTokenizer
from transformers import AdamW
import torchmetrics

from model import BERTClassificationModel
from LivedoorDataLoader import LivedoorDatasetPreprocesser

import mlflow

# seed 値の固定と決定論的アルゴリズムの使用を強制する関数
def seed_torch(seed=42):
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8" #":16:8"
    g = torch.Generator()
    g.manual_seed(seed)
    return seed, g


# トークナイズのための class
class TokenizerCollate:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def collate_fn(self, batch):
        input = [item[0] for item in batch]
        input = self.tokenizer(
            input,
            padding=True,
            max_length=512,
            truncation=True,
            return_tensors="pt")
        targets = torch.tensor([item[1] for item in batch])
        return input, targets

    def __call__(self, batch):
        return self.collate_fn(batch)

def cli_main():

    # 初期設定

    ## 再現再確保のため Seed 値の固定と決定論的アルゴリズムの使用を強制
    seed, g = seed_torch()

    ## パラメーター
    parser = argparse.ArgumentParser(description='PyTorch BERT fine-tuning on Azure ML Compute Cluster')
    parser.add_argument('--batch_size', type=int, default=256, metavar='N')
    parser.add_argument('--epochs', type=int, default=60, metavar='N')
    parser.add_argument('--lr', type=float, default=1e-4, metavar='LR')
    parser.add_argument('--num_nodes', default=4, type=int)
    parser.add_argument('--base_model', default="cl-tohoku/bert-base-japanese-v2", type=str)
    args = parser.parse_args()

    batch_size = int(args.batch_size / args.num_nodes)

    base_lr = args.lr

    model = args.base_model

    ### 環境変数 (Azure ML Compute Cluster によって初期設定済み) から分散処理に必要な値を入手
    rank = int(os.environ["NODE_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    
    ## ローカル環境に存在する GPU を特定
    device = torch.device("cuda", local_rank)

    ## クラスター間のプロセスグループを初期化
    ### gloo か nccl が使用可能
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)

    # データの用意

    ## tokenizer を入手 (使用するモデルごとに指定されている)
    tokenizer = AutoTokenizer.from_pretrained(model)

    ## Livedoor ニュースコーパスの前処理
    ldp = LivedoorDatasetPreprocesser()

    train_dataset = ldp.train_dataset()
    val_dataset = ldp.val_dataset()

    ## Distributed Data Parallel 用のサンプラーを用意
    ### 各ノード数に対して重複しないようにデータを分配する役割
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, seed=seed)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, seed=seed)

    ## DataLoader としてデータをラップ
    ### この時点で各ノードごとに分割され重複がないデータセットを保持
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        collate_fn=TokenizerCollate(tokenizer=tokenizer),
        generator=g
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        sampler=val_sampler,
        collate_fn=TokenizerCollate(tokenizer=tokenizer),
        generator=g
    )
    
    # モデルと学習の用意

    ## BERT を使用した分類モデルを初期化して GPU に転送
    model = BERTClassificationModel(model=model).to(device)

    ## DDP 用のモデルラッパーでラップ
    ddp_model = DistributedDataParallel(model, device_ids=[local_rank])

    ## optimizer のセットアップ

    bert_lr = base_lr / 2 * world_size
    output_lr = base_lr * world_size

    optimizer = AdamW([
        {
            'params': ddp_model.module.bert.encoder.layer[-1].parameters(),
            'lr': bert_lr
        },
        {
            'params': ddp_model.module.output.parameters(),
            'lr': output_lr
        }
    ])

    # 学習用の関数    
    def train(epoch):
        print(f"{epoch} epoch training phase starting...")
        model.train()
        dist.barrier()
        
        ## 損失関数として交差エントロピーを使用
        criterion = nn.CrossEntropyLoss()
        
        ## 全イテレーションの損失を保持するための tensor を用意
        ### 通常は tensor である必要はないが、今回は GPU 間で all_reduce を使用して集計するため GPU 上に置きやすい tensor を使用
        train_loss = torch.tensor([0.0], dtype=torch.double).to(device)
        #train_sampler.set_epoch(epoch)

        ## train 用の DataLoader からバッチを取得
        for batch_idx, (data, target) in enumerate(train_loader):
            ## バッチサイズ×入力長のミニバッチを取得し、GPU に転送
            (input_ids, attention_mask, token_type_ids) = (data['input_ids'].to(device), data['attention_mask'].to(device), data['token_type_ids'].to(device))
            target = target.to(device)

            ## optimizer に蓄積された勾配値を0に
            optimizer.zero_grad()

            ## モデルにデータを入力して予測を得る
            output = ddp_model(input_ids, attention_mask, token_type_ids)

            ## 予測と正解から損失を計算する
            loss = criterion(output, target)

            ## 誤差逆伝搬
            ### この時点で勾配が各ノード間で同期され、平均されて各 GPU 上のノードで同じ勾配でモデルが更新される = 各ノードで同じモデルになる
            loss.backward()

            ## モデルパラメーターの更新
            optimizer.step()

            ## 損失の値を蓄積
            train_loss += torch.tensor([loss * input_ids.size(0)], dtype=torch.double).to(device) # batch サイズ倍する

            ## ログに値を記録
            print(f"iteration number: {batch_idx} train_loss: {loss.item()}")

        ## 各ノードの train_loss を集計 (合計)
        dist.all_reduce(train_loss, op=dist.ReduceOp.SUM)

        ## rank=0 のノードでのみ Azure ML Run のメトリクスにログ書き込みを実行
        if rank == 0:
            epoch_train_loss = train_loss.item() / len(train_dataset)
            print(f"epoch: {epoch} epoch_train_loss: {epoch_train_loss}")
            mlflow.log_metric('train_loss', epoch_train_loss)

    # 評価用の関数
    def validate(epoch):
        print(f"{epoch} epoch validating phase starting...")
        model.eval()
        dist.barrier()

        ## 損失関数として交差エントロピーを使用
        criterion = nn.CrossEntropyLoss()

        ## 評価指標として正解率を算出
        accuracy = torchmetrics.Accuracy()

        ## 全イテレーションの損失および正解率を保持するための tensor を用意
        val_loss = torch.tensor([0.0], dtype=torch.double).to(device)
        val_accuracy = torch.tensor([0.0], dtype=torch.double).to(device)

        ## モデルの更新は行わないため勾配計算不要
        with torch.no_grad():

            ## validate 用の DataLoader からバッチを取得
            for batch_idx, (data, target) in enumerate(val_loader):
                ## バッチサイズ×入力長のミニバッチを取得し、GPU に転送
                (input_ids, attention_mask, token_type_ids) = (data['input_ids'].to(device), data['attention_mask'].to(device), data['token_type_ids'].to(device))
                target = target.to(device)

                ## モデルにデータを入力して予測を得る
                output = ddp_model(input_ids, attention_mask, token_type_ids)

                ## 予測と正解から損失および正解率を計算する
                loss = criterion(output, target)
                val_acc = accuracy(output.to("cpu"), target.to("cpu"))

                ## 損失と正解率の値を蓄積    
                val_loss += torch.tensor([loss * input_ids.size(0)], dtype=torch.double).to(device) 
                batch_val_acc = val_acc.item() * input_ids.size(0)
                val_accuracy += torch.tensor([batch_val_acc], dtype=torch.double).to(device)

                ## ログに値を記録
                print(f"iteration number: {batch_idx} val_loss: {loss.item()} val_acc: {val_acc.item()}")

        ## 各ノードの val_loss および val_accuracy を集計 (合計)
        dist.all_reduce(val_loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_accuracy, op=dist.ReduceOp.SUM)

        ## rank=0 のノードでのみ Azure ML Run のメトリクスにログ書き込みを実行
        if rank == 0:           
            epoch_val_loss = val_loss.item() / len(val_dataset)
            epoch_val_acc = val_accuracy.item() / len(val_dataset)
            print(f"epoch: {epoch} epoch_val_loss: {epoch_val_loss}")
            print(f"epoch: {epoch} epoch_val_acc: {epoch_val_acc}")
            mlflow.log_metric('val_loss', epoch_val_loss)
            mlflow.log_metric('val_accuracy', epoch_val_acc)

    # epoch 数だけ学習と評価を繰り返す
    for epoch in range(1, args.epochs + 1):
        train(epoch)
        validate(epoch)

    print("training process finished")

    ## 今回はスクリプト実行終了で自動的にプロセスが終了するため下記プロセスグループの明示的終了は不要
    #dist.destroy_process_group()

if __name__ == "__main__":
    cli_main()