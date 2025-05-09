import pandas as pd
import torch
from torch.utils.data import DataLoader
import datasets
from transformers import AutoTokenizer
import torchvision.models as models
import torch
import evaluate
from transformers import AutoModelForSequenceClassification
from torchkeras import KerasModel

torch.cuda.set_device(1)
df_train = pd.read_csv('/mnt/data2/temp1/jsj/dataset/train_aug_tocp.csv')#微调macbert用race,region,gender数据集
ds_train = datasets.Dataset.from_pandas(df_train)
ds_train = ds_train.shuffle(42)


#加载模型和tokenizer
model_path="/mnt/data2/temp1/.cache/modelscope/hub/dienstag/chinese-macbert-base"
model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=2)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True) #需要和模型一致

ds_encoded = ds_train.map(lambda example:tokenizer(example["text"],
                    max_length=150,truncation=True,padding='max_length'),
                    batched=True,
                    batch_size=32,
                    num_proc=2) #支持批处理和多进程map


#转换成pytorch中的tensor
ds_encoded.set_format(type="torch",columns = ["input_ids",'attention_mask','token_type_ids','labels'])
#test.set_format(type="torch",columns = ["input_ids",'attention_mask','token_type_ids','labels'])
#分割成训练集和测试集
ds_train,ds_val = ds_encoded.train_test_split(test_size=0.2).values()
ds_val, ds_test = ds_val.train_test_split(test_size=0.5).values()

# 在collate_fn中可以做动态批处理(dynamic batching)
def collate_fn(examples):
    return tokenizer.pad(examples)  # return_tensors='pt'

dl_train = torch.utils.data.DataLoader(ds_train, batch_size=16, collate_fn=collate_fn)
dl_val = torch.utils.data.DataLoader(ds_val, batch_size=16, collate_fn=collate_fn)
dl_test = torch.utils.data.DataLoader(ds_test, batch_size=16, collate_fn=collate_fn)

# 修改StepRunner以适应transformers的数据集格式
class StepRunner:
    def __init__(self, net, loss_fn, accelerator, stage="train", metrics_dict=None,
                 optimizer=None, lr_scheduler=None
                 ):
        self.net, self.loss_fn, self.metrics_dict, self.stage = net, loss_fn, metrics_dict, stage
        self.optimizer, self.lr_scheduler = optimizer, lr_scheduler
        self.accelerator = accelerator
        if self.stage == 'train':
            self.net.train()
        else:
            self.net.eval()

    def __call__(self, batch):

        out = self.net(**batch)

        # loss
        loss = out.loss

        # preds
        preds = (out.logits).argmax(axis=1)

        # backward()
        if self.optimizer is not None and self.stage == "train":
            self.accelerator.backward(loss)
            self.optimizer.step()
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
            self.optimizer.zero_grad()

        all_loss = self.accelerator.gather(loss).sum()

        labels = batch['labels']
        acc = (preds == labels).sum() / ((labels > -1).sum())

        all_acc = self.accelerator.gather(acc).mean()

        # losses
        step_losses = {self.stage + "_loss": all_loss.item(), self.stage + '_acc': all_acc.item()}

        # metrics
        step_metrics = {}

        if self.stage == "train":
            if self.optimizer is not None:
                step_metrics['lr'] = self.optimizer.state_dict()['param_groups'][0]['lr']
            else:
                step_metrics['lr'] = 0.0
        return step_losses, step_metrics

# --------------------
# # 冻结BERT模型的部分层
#frozen_layers = ['embeddings', 'encoder.layer.0', 'encoder.layer.1','encoder.layer.2','encoder.layer.3','encoder.layer.4','encoder.layer.5','encoder.layer.6','encoder.layer.7','encoder.layer.8']
#frozen_layers = ['embeddings', 'encoder.layer.0', 'encoder.layer.1','encoder.layer.2','encoder.layer.3','encoder.layer.4','encoder.layer.5','encoder.layer.6']
#frozen_layers = ['embeddings', 'encoder.layer.0', 'encoder.layer.1','encoder.layer.2','encoder.layer.3']#86.65
#frozen_layers=['embeddings', 'encoder.layer.0', 'encoder.layer.1','encoder.layer.2','encoder.layer.3','encoder.layer.4','encoder.layer.5','encoder.layer.6','encoder.layer.7','encoder.layer.8','encoder.layer.9']
#frozen_layers=['embeddings', 'encoder.layer.0', 'encoder.layer.1','encoder.layer.2','encoder.layer.3','encoder.layer.4','encoder.layer.5','encoder.layer.6','encoder.layer.7']
# for name, param in model.named_parameters():
#     if any(frozen_layer in name for frozen_layer in frozen_layers):
#         print("1")
#         param.requires_grad = False

# --------------------训练
KerasModel.StepRunner = StepRunner
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-6)
keras_model = KerasModel(model,
                         loss_fn=None,
                         optimizer=optimizer
                         )


keras_model.fit(
    train_data = dl_train,
    val_data= dl_val,
    ckpt_path='tocp.pth',
    epochs=5,
    patience=5,
    monitor="val_acc",
    mode="max",
    plot = True,
    wandb = False,
    quiet = False
)



