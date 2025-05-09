import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertTokenizer
from transformers import BertModel
from qwen import call_agent_app
from transformers import AutoModelForSequenceClassification
#加载预训练模型
pretrained = BertModel.from_pretrained('hfl/chinese-macbert-base')
#需要移动到cuda上
device = 'cuda' if torch.cuda.is_available() else 'cpu'

#不训练,不需要计算梯度
for param in pretrained.parameters():
    param.requires_grad_(False)


import re
from pypinyin import lazy_pinyin, Style
import jieba
from Pinyin2Hanzi import DefaultHmmParams, viterbi
from pypinyin import lazy_pinyin, Style, pinyin_dict

# --------------------- 初始化 ---------------------
def convert_ue_to_ve(pinyin):
    """将 lue/nue/jue/que/xue/yue 转换为 lve/nve/jve/qve/xve/yve"""
    # 正则表达式匹配 l、n、j、q、x、y 后面的 ue，并将其替换为 ve
    return re.sub(r'([lnjqxy])ue', r'\1ve', pinyin, flags=re.IGNORECASE)

all_pinyin=['gu','qiao','qian','qve','ge','gang','ga','lian','liao','rou','zong',\
    'tu','seng','yve','ti','te','jve','ta','nong','zhang','fan','ma','gua','die','gui',\
    'guo','gun','sang','diu','zi','ze','za','chen','zu','ba','dian','diao','nei','suo',\
    'sun','zhao','sui','kuo','kun','kui','cao','zuan','kua','den','lei','neng','men',\
    'mei','tiao','geng','chang','cha','che','fen','chi','fei','chu','shui','me','tuan',\
    'mo','mi','mu','dei','cai','zhan','zhai','can','ning','wang','pie','beng','zhuang',\
    'tan','tao','tai','song','ping','hou','cuan','lan','lao','fu','fa','jiong','mai',\
    'xiang','mao','man','a','jiang','zun','bing','su','si','sa','se','ding','xuan',\
    'zei','zen','kong','pang','jie','jia','jin','lo','lai','li','peng','jiu','yi','yo',\
    'ya','cen','dan','dao','ye','dai','zhen','bang','nou','yu','weng','en','ei','kang',\
    'dia','er','ru','keng','re','ren','gou','ri','tian','qi','shua','shun','shuo','qun',\
    'yun','xun','fiao','zan','zao','rang','xi','yong','zai','guan','guai','dong','kuai',\
    'ying','kuan','xu','xia','xie','yin','rong','xin','tou','nian','niao','xiu','fo',\
    'kou','niang','hua','hun','huo','hui','shuan','quan','shuai','chong','bei','ben',\
    'kuang','dang','sai','ang','sao','san','reng','ran','rao','ming','null','lie','lia',\
    'min','pa','lin','mian','mie','liu','zou','miu','nen','kai','kao','kan','ka','ke',\
    'yang','ku','deng','dou','shou','chuang','nang','feng','meng','cheng','di','de','da',\
    'bao','gei','du','gen','qu','shu','sha','she','ban','shi','bai','nun','nuo','sen','lve',\
    'kei','fang','teng','xve','lun','luo','ken','wa','wo','ju','tui','wu','le','ji','huang',\
    'tuo','cou','la','mang','ci','tun','tong','ca','pou','ce','gong','cu','lv','dun','pu',\
    'ting','qie','yao','lu','pi','po','suan','chua','chun','chan','chui','gao','gan','zeng',\
    'gai','xiong','tang','pian','piao','cang','heng','xian','xiao','bian','biao','zhua','duan',\
    'cong','zhui','zhuo','zhun','hong','shuang','juan','zhei','pai','shai','shan','shao','pan',\
    'pao','nin','hang','nie','zhuai','zhuan','yuan','niu','na','miao','guang','ne','hai','han',\
    'hao','wei','wen','ruan','cuo','cun','cui','bin','bie','mou','nve','shen','shei','fou','xing',\
    'qiang','nuan','pen','pei','rui','run','ruo','sheng','dui','bo','bi','bu','chuan','qing',\
    'chuai','duo','o','chou','ou','zui','luan','zuo','jian','jiao','sou','wan','jing','qiong',\
    'wai','long','yan','liang','lou','huan','hen','hei','huai','shang','jun','hu','ling','ha','he',\
    'zhu','ceng','zha','zhe','zhi','qin','pin','ai','chai','qia','chao','ao','an','qiu','ni','zhong',\
    'zang','nai','nan','nao','chuo','tie','you','nu','nv','zheng','leng','zhou','lang','e',]

# all_pinyin = [convert_ue_to_ve(py) for py in all_pinyin]

from Pinyin2Hanzi import DefaultDagParams
from Pinyin2Hanzi import dag

hmm_params =  DefaultDagParams()



def convert_pinyin_to_hanzi(pinyin_list):
    """将拼音列表转换为最可能的汉字序列（带上下文优化）"""
    if not pinyin_list:
        return ""
    # path_num=1 表示仅返回最优结果
    converted_list = [convert_ue_to_ve(py) for py in pinyin_list]
    #result = viterbi(hmm_params=hmm_params, observations=converted_list, path_num=1)
    result=dag(hmm_params, converted_list, path_num=1)
    return ''.join(result[0].path) if result else ""


def filter_invalid_pinyin(pinyin_str):
    """过滤非法拼音字符（如 'c' 不属于任何拼音）"""
    valid_segments = []
    # 分割拼音和非法字符（如字母、数字）
    segments = re.findall(r'([a-zA-Z]+)|([^a-zA-Z\s])', pinyin_str)

    for pinyin_part, invalid_part in segments:
        if pinyin_part:
            # 转换为小写并检查是否为合法拼音
            pinyin_lower = pinyin_part.lower()
            if pinyin_lower in all_pinyin:
                valid_segments.append(pinyin_lower)
            else:
                # 非法拼音直接忽略或标记（此处选择忽略）
                # 替换原本的 pass 为：
                valid_segments.append(pinyin_part)  # 保留原样
        elif invalid_part:
            # 保留标点符号（如逗号、感叹号）
            valid_segments.append(invalid_part)

    return ' '.join(valid_segments)


def process_text(input_str):
    """处理混合字符串：分词优化 + 拼音转汉字 + 保留标点"""
    # Step 1: 使用 Jieba 分词优化多音字
    # words = jieba.lcut(input_str)
    # print(f"words: {words}")
    # # Step 2: 将每个词语转为拼音（自动处理多音字）
    # pinyin_list = []
    # for word in words:
    #     # 对每个词语单独转拼音（lazy_pinyin会自动处理多音字）
    #     pinyin_list.extend(lazy_pinyin(word))
    pinyin_list=[]
    pinyin_list=lazy_pinyin(input_str)

    # Step 3: 拼接拼音字符串并过滤非法字符
    pinyin_str = ' '.join(pinyin_list)
    # cleaned_str = filter_invalid_pinyin(pinyin_str)
    # print(f"cleaned_str: {cleaned_str}")
    # Step 4: 分割拼音和标点并转换
    segments = re.findall(r'([a-zA-Z]+)|([^a-zA-Z\s])', pinyin_str)
    current_pinyin = []
    final_result = []

    for pinyin_part, punct_part in segments:
        if pinyin_part:
            if pinyin_part not in all_pinyin:
                temp=punct_part
                punct_part = pinyin_part
                pinyin_part = temp

        if pinyin_part:
            current_pinyin.append(pinyin_part)
        elif punct_part:
            if current_pinyin:
                hanzi = convert_pinyin_to_hanzi(current_pinyin)
                final_result.append(hanzi)
                current_pinyin = []
            final_result.append(punct_part)

    if current_pinyin:
        hanzi = convert_pinyin_to_hanzi(current_pinyin)
        final_result.append(hanzi)

    return ''.join(final_result)


#多头注意力机制
class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super(MultiHeadAttention, self).__init__()
        # 确保隐藏层特征数能够被头数整除
        assert hidden_size % num_heads == 0
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads  # 计算每个头的维度
        # 定义线性层，用于对查询、键、值进行线性变换
        self.linear_q = nn.Linear(hidden_size, hidden_size)
        self.linear_k = nn.Linear(hidden_size, hidden_size)
        self.linear_v = nn.Linear(hidden_size, hidden_size)
        self.linear_out = nn.Linear(hidden_size, hidden_size)  # 定义输出线性层，用于整合多头注意力后的输出

    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        # 对输入进行线性变换，并将其分割为多个头
        q = self.linear_q(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.linear_k(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.linear_v(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        # 计算注意力分数
        scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float))
        attn_weights = F.softmax(scores, dim=-1)  # 计算注意力权重
        # 根据注意力权重对值进行加权求和
        context = torch.matmul(attn_weights, v).transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        out = self.linear_out(context)  # 整合多头注意力的输出
        return out
    
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(768, 512)  # 第一层全连接层
        self.fc2 = nn.Linear(512, 256)  # 第二层全连接层
        self.fc3 = nn.Linear(256, 2)    # 第三层全连接层
        self.dropout = nn.Dropout(p=0.5)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.activation = nn.ReLU()
        self.multihead_attention = MultiHeadAttention(hidden_size=768, num_heads=8)  # 多头注意力模块

    def forward(self, input_ids, attention_mask, token_type_ids):
        out = pretrained(input_ids=input_ids,
                         attention_mask=attention_mask,
                         token_type_ids=token_type_ids).last_hidden_state

        # 应用多头注意力机制
        out = self.multihead_attention(out)
        out = out[:, 0]  # 提取[CLS]标记的输出

        out = self.activation(self.bn1(self.fc1(out)))
        out = self.dropout(out)
        out = self.activation(self.bn2(self.fc2(out)))
        out = self.dropout(out)
        out = self.fc3(out)
        out = out.softmax(dim=1)
        return out

class Model_COLD4(nn.Module):
    def __init__(self):
        super(Model_COLD4, self).__init__()
        self.fc1 = nn.Linear(768, 512)  # 第一层全连接层
        self.fc2 = nn.Linear(512, 256)  # 第二层全连接层
        self.fc3 = nn.Linear(256, 4)    # 第三层全连接层
        self.dropout = nn.Dropout(p=0.5)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.activation = nn.ReLU()
        self.multihead_attention = MultiHeadAttention(hidden_size=768, num_heads=8)  # 多头注意力模块

    def forward(self, input_ids, attention_mask, token_type_ids):
        out = pretrained(input_ids=input_ids,
                         attention_mask=attention_mask,
                         token_type_ids=token_type_ids).last_hidden_state

        # 应用多头注意力机制
        out = self.multihead_attention(out)
        out = out[:, 0]  # 提取[CLS]标记的输出

        out = self.activation(self.bn1(self.fc1(out)))
        out = self.dropout(out)
        out = self.activation(self.bn2(self.fc2(out)))
        out = self.dropout(out)
        out = self.fc3(out)
        out = out.softmax(dim=1)
        return out
    
def COLD4(text, device):
    MacBERT_base_COLD4 = torch.load('/mnt/disk1/LY/作品前后端/Flask/models/MacBERT-base-COLD4.pth',weights_only=False)
    MacBERT_base_COLD4.to(device)
    tokenizer = BertTokenizer.from_pretrained('hfl/chinese-macbert-base')
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    # 将输入数据移动到相同的设备上
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        out = MacBERT_base_COLD4(**inputs) 
    out = torch.argmax(out, dim=1).item()
    return out

def load_models_and_predict(text, model, device):

    model_path = "hfl/chinese-macbert-base"
    D1 = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=2)
    D1.load_state_dict(torch.load('/mnt/disk1/LY/作品前后端/Flask/models/sex.pth'))
    D1.to(device)

    D2 = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=2)
    D2.load_state_dict(torch.load('/mnt/disk1/LY/作品前后端/Flask/models/tocp.pth'))
    D2.to(device)

    D3 = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=4)
    D3.load_state_dict(torch.load('/mnt/disk1/LY/作品前后端/Flask/models/dy_zz_xb_4.pth'))
    D3.to(device)

    # 加载字典和分词工具
    tokenizer = BertTokenizer.from_pretrained('hfl/chinese-macbert-base')

    # 对输入文本进行编码
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)

    # 将输入数据移动到相同的设备上
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # 设置模型为评估模式
    D1.eval()
    D2.eval()
    D3.eval()

    result=None
    # 分析结果
    if model == "general":
        a1=0.333
        a2=0.333
        a3=0.333
        # 进行预测
        with torch.no_grad():
            text_1=process_text(text)
            text_1 = text
            print(text_1)
            inputs_1 = tokenizer(text_1, return_tensors="pt", padding=True, truncation=True)
            inputs_1 = {k: v.to(device) for k, v in inputs_1.items()}
            out1 = D1(**inputs_1)
            logits = out1.logits
            p1 = torch.softmax(logits, dim=1)
            print(f"p1:{[p1]}")
            out1 = torch.argmax(p1, dim=-1).cpu().item()
            print(f"out1: {out1}")

        with torch.no_grad():
            out2 = D2(**inputs_1)
            logits = out2.logits
            p2 = torch.softmax(logits, dim=1)
            print(f"p2:{p2}")
            out2 = torch.argmax(p2, dim=-1).cpu().item()
            print(f"out2: {out2}")

        with torch.no_grad():
            out3 = D3(**inputs)
            logits = out3.logits
            p3 = torch.softmax(logits, dim=1)
            print(f"p3:{p3}")
            out3 = torch.argmax(p3, dim=-1).cpu().item()
            print(f"out3: {out3}"
                  )
        # out1 = torch.argmax(out1, dim=1).item() #偏见
        # out3 = torch.argmax(out3, dim=1).item() #攻击性
        if out1==out2==out3==0:
            result = "该言论无害"

        elif out1==out3==0 and out2==1:

            result = "该言论为有害言论"

        else:
            result = "该言论为有害言论"

        # if out1 == 1:
        #     result = "该言论为有害言论"
        #
        # elif out1==0:
        #     if out2 == 1:
        #         result = "该言论为有害言论"
        #     else:
        #         if out3==1:
        #             result = "该言论为有害言论"
        #         elif out3==0:
        #             result = "该言论无害"


    elif model == "sex":
        with torch.no_grad():
            text_1 = process_text(text)
            #text_1 = text
            print(text_1)
            inputs_1 = tokenizer(text_1, return_tensors="pt", padding=True, truncation=True)
            # 将输入数据移动到相同的设备上
            inputs_1 = {k: v.to(device) for k, v in inputs_1.items()}

            out1 = D1(**inputs_1)
            logits = out1.logits
            out1 = torch.argmax(logits, dim=-1).cpu().item()
            print(f"out1: {out1}")
        if out1 == 0:
            result = "该言论为无害言论"
        elif out1 == 1:
            result = "该言论涉黄"

    elif model == "abuse":
        with torch.no_grad():
            text_1 = process_text(text)
            #text_1=text
            print(text_1)
            inputs_1 = tokenizer(text_1, return_tensors="pt", padding=True, truncation=True)
            # 将输入数据移动到相同的设备上
            inputs_1 = {k: v.to(device) for k, v in inputs_1.items()}

            out2 = D2(**inputs_1)
            logits = out2.logits
            out2 = torch.argmax(logits, dim=-1).cpu().item()
            print(f"out1: {out2}")
        if out2 == 0:
            result = "该言论为无害言论"
        elif out2 == 1:
            result = "该言论还有辱骂"

    elif model == "four_offensive":
        with torch.no_grad():
            out3=D3(**inputs)
            logits = out3.logits
            out3 = torch.argmax(logits, dim=-1).cpu().item()
            print(f"out1: {out3}")

        if out3 == 0:
            result = "该言论无害"

        elif out3 == 1:
            result = "该言论具有种族偏见"
        elif out3 == 2:
            result = "该言论具有地域偏见"
        elif out3 == 3:
            result = "该言论具有性别偏见"
        # out3 = call_agent_app(text)
        # out4 = COLD4(text, device)
        # if out3 == '0':
        #     if out4 == 0:
        #         result = "这句话没有种族、地域、性别方面的攻击性"
        #     else:
        #         result = "这句话没有攻击性"
        # elif out3 == "1":
        #     if out4 == 0:
        #         result = "这句话有攻击性"
        #     elif out4 == 1:
        #         result = "这句话涉及种族方面的攻击"
        #     elif out4 == 2:
        #         result = "这句话涉及地域方面的攻击"
        #     elif out4 == 3:
        #         result = "这句话涉及性别方面的攻击"
        # else:
        #     if out4 == 0:
        #         result = "这句话没有种族、地域、性别方面的攻击性"
        #     elif out4 == 1:
        #         result = "这句话涉及种族方面的攻击"
        #     elif out4 == 2:
        #         result = "这句话涉及地域方面的攻击"
        #     elif out4 == 3:
        #         result = "这句话涉及性别方面的攻击"
    return result