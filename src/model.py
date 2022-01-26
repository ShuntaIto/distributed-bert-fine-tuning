import torch.nn as nn
from transformers import AutoModel
from transformers import BertModel

# BERT モデル
class BERTClassificationModel(nn.Module):
    def __init__(self, num_classes=9, fix_bert=True, model='cl-tohoku/bert-large-japanese'):
        super(BERTClassificationModel, self).__init__()

        self.num_classes = num_classes

        ## BERT の読み込み
        ### ALBERT など、他のモデルで代替することも可能
        self.bert = AutoModel.from_pretrained(model)

        ## 9クラス分類タスク用の最終層を用意
        self.output = nn.Linear(self.bert.config.hidden_size, 9)
        
        # BERT の最終層以外を更新されないように全て固定
        if fix_bert:
            # fix BERT model 
            for param in self.bert.parameters():
                param.requires_grad = False
            # unfix BERT final layer
            for param in self.bert.encoder.layer[-1].parameters():
                param.requires_grad = True
        
    def forward(self, input_ids, attention_mask, token_type_ids):
        bert_output = self.bert(input_ids, attention_mask, token_type_ids)
        ## https://huggingface.co/docs/transformers/main_classes/output#transformers.modeling_outputs.BaseModelOutput
        pred = self.output(bert_output.pooler_output)
        ## CrossEntropyLoss の PyTorch 実装は内部的に Softmax 関数による処理が含まれているため、最終出力に Softmax 関数をかける必要はない
        return pred