
# ---------------------------------------------------------测试
# 加载评价器
import evaluate
metric = evaluate.load("/mnt/data2/temp1/jsj/Macbert/evaluate/metrics/accuracy")
#torch.save(model,'/mnt/data2/temp1/jsj/Macbert/four.pth')
# 加载模型
model.load_state_dict(torch.load('/mnt/data2/temp1/jsj/Macbert/tocp.pth'))
#model_path="/mnt/data2/temp1/.cache/modelscope/hub/dienstag/chinese-macbert-base"
# model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=2)
#model.load_state_dict(torch.load('/mnt/data2/temp1/jsj/Macbert/shanchu_four.pt'))
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True) #需要和模型一致



model.eval()

# 用于统计批次的计数器
total_samples = 0
# 用于存储每个标签的准确率
label_accuracies = {}
i=1
# 使用torch.no_grad()来关闭梯度计算
with torch.no_grad():
    for batch in dl_test:
        i = i + 1
        print(i)
        outputs = model(**batch)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        print(f"predictions: {predictions}")
        metric.add_batch(predictions=predictions, references=batch["labels"])
        # 更新总样本数
        total_samples += len(predictions)
        # 统计每个标签的准确率
        for label in range(2):  # 假设有num_labels个标签
            mask = batch["labels"] == label
            if mask.any():  # 确保标签在批次中至少出现了一次
                label_predictions = predictions[mask]
                label_references = batch["labels"][mask]
                label_accuracy = torch.sum(label_predictions == label_references).item() / len(label_predictions)
                if label not in label_accuracies:
                    label_accuracies[label] = []
                label_accuracies[label].append(label_accuracy)
# 计算评价指标
accuracy = metric.compute()
print("测试集总体精确度:", accuracy)

# 打印每个标签的准确率
for label, accuracies in label_accuracies.items():
    if accuracies:  # 确保准确率列表至少有一个值
        avg_accuracy = sum(accuracies) / len(accuracies)
        print(f"标签 {label} 的平均准确率: {avg_accuracy}")
    else:
        print(f"标签 {label} 的准确率列表为空，无法计算平均准确率。")
