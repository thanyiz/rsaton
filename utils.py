# coding: UTF-8
import os
import torch
import numpy as np
import pickle as pkl
from tqdm import tqdm
import time
from datetime import timedelta
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.font_manager as fm

MAX_VOCAB_SIZE = 10000  # 词表长度限制
UNK, PAD = "<UNK>", "<PAD>"  # 未知字，padding符号
myfont = fm.FontProperties(fname="THUCNews/font/DroidSansFallback.ttf") 

def build_vocab(file_path, tokenizer, max_size, min_freq):
    vocab_dic = {}
    with open(file_path, "r", encoding="UTF-8") as f:
        for line in tqdm(f):
            lin = line.strip()
            if not lin:
                continue
            content = lin.split("\t")[0]
            for word in tokenizer(content):
                vocab_dic[word] = vocab_dic.get(word, 0) + 1
        vocab_list = sorted(
            [_ for _ in vocab_dic.items() if _[1] >= min_freq],
            key=lambda x: x[1],
            reverse=True,
        )[:max_size]
        vocab_dic = {word_count[0]: idx for idx, word_count in enumerate(vocab_list)}
        vocab_dic.update({UNK: len(vocab_dic), PAD: len(vocab_dic) + 1})
    return vocab_dic


def build_dataset(config, ues_word):
    if ues_word:
        tokenizer = lambda x: x.split(" ")  # 以空格隔开，word-level
    else:
        tokenizer = lambda x: [y for y in x]  # char-level
    if os.path.exists(config.vocab_path):
        vocab = pkl.load(open(config.vocab_path, "rb"))
    else:
        vocab = build_vocab(
            config.train_path, tokenizer=tokenizer, max_size=MAX_VOCAB_SIZE, min_freq=1
        )
        pkl.dump(vocab, open(config.vocab_path, "wb"))
    print(f"Vocab size: {len(vocab)}")

    def load_dataset(path, pad_size=32):
        contents = []
        with open(path, "r", encoding="UTF-8") as f:
            for line in tqdm(f):
                lin = line.strip()
                if not lin:
                    continue
                content, label = lin.split("\t")
                words_line = []
                token = tokenizer(content)
                seq_len = len(token)
                if pad_size:
                    if len(token) < pad_size:
                        token.extend([PAD] * (pad_size - len(token)))
                    else:
                        token = token[:pad_size]
                        seq_len = pad_size
                # word to id
                for word in token:
                    words_line.append(vocab.get(word, vocab.get(UNK)))
                contents.append((words_line, int(label), seq_len))
        return contents  # [([...], 0), ([...], 1), ...]

    train = load_dataset(config.train_path, config.pad_size)
    dev = load_dataset(config.dev_path, config.pad_size)
    test = load_dataset(config.test_path, config.pad_size)
    return vocab, train, dev, test


class DatasetIterater(object):
    def __init__(self, batches, batch_size, device):
        self.batch_size = batch_size
        self.batches = batches
        self.n_batches = len(batches) // batch_size
        self.residue = False  # 记录batch数量是否为整数
        if len(batches) % self.n_batches != 0:
            self.residue = True
        self.index = 0
        self.device = device

    def _to_tensor(self, datas):
        x = torch.LongTensor([_[0] for _ in datas]).to(self.device)
        y = torch.LongTensor([_[1] for _ in datas]).to(self.device)

        # pad前的长度(超过pad_size的设为pad_size)
        seq_len = torch.LongTensor([_[2] for _ in datas]).to(self.device)
        return (x, seq_len), y

    def __next__(self):
        if self.residue and self.index == self.n_batches:
            batches = self.batches[self.index * self.batch_size : len(self.batches)]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

        elif self.index >= self.n_batches:
            self.index = 0
            raise StopIteration
        else:
            batches = self.batches[
                self.index * self.batch_size : (self.index + 1) * self.batch_size
            ]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

    def __iter__(self):
        return self

    def __len__(self):
        if self.residue:
            return self.n_batches + 1
        else:
            return self.n_batches


def build_iterator(dataset, config):
    iter = DatasetIterater(dataset, config.batch_size, config.device)
    return iter


def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))


def plot_confusion_matrix(config, confusion_matrix):
    # 拼接保存路径
    save_path = os.path.join("result", config.model_name + "confusion_matrix")
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.heatmap(
        confusion_matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=config.class_list,
        yticklabels=config.class_list,
    )
    plt.xticks(fontproperties=myfont)  # 设置x轴标签字体
    plt.yticks(fontproperties=myfont)  # 设置y轴标签字体
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.savefig(save_path)


def plot_test_report(config, test_report):
    save_path = os.path.join("result", config.model_name + "test_report")
    categories = []  # 存储类别名
    precisions = []  # 存储精确率
    recalls = []  # 存储召回率

    # 解析 test_report，提取类别名、精确率和召回率
    for line in test_report.split("\n")[2:-5]:
        category, precision, recall, _, _ = line.split()
        categories.append(category)
        precisions.append(float(precision))
        recalls.append(float(recall))

    # 绘制柱状图
    bar_width = 0.35
    index = np.arange(len(categories))
    plt.figure(figsize=(10, 10))
    plt.bar(index, precisions, bar_width, label="Precision")
    plt.bar(index + bar_width, recalls, bar_width, label="Recall")
    plt.xlabel("Categories")
    plt.ylabel("Scores")
    plt.title("Precision and Recall for Each Category")
    plt.xticks(index + bar_width / 2, categories,fontproperties=myfont)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)


def plot_accuracy_loss(
    config, train_accuracies, dev_accuracies, train_losses, dev_losses
):
    save_path = os.path.join("result", config.model_name + "accuracy_loss")
    epochs = range(1, len(train_accuracies) + 1)

    # 绘制准确率曲线
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_accuracies, "bo-", label="Training Accuracy")
    plt.plot(epochs, dev_accuracies, "r*-", label="Validation Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()

    # 绘制损失曲线
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_losses, "bo-", label="Training Loss")
    plt.plot(epochs, dev_losses, "r*-", label="Validation Loss")
    plt.title("Training and Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_path)


if __name__ == "__main__":
    """提取预训练词向量"""
    # 下面的目录、文件名按需更改。
    train_dir = "./THUCNews/data/train.txt"
    vocab_dir = "./THUCNews/data/vocab.pkl"
    pretrain_dir = "./THUCNews/data/sgns.sogou.char"
    emb_dim = 300
    filename_trimmed_dir = "./THUCNews/data/embedding_SougouNews"
    if os.path.exists(vocab_dir):
        word_to_id = pkl.load(open(vocab_dir, "rb"))
    else:
        # tokenizer = lambda x: x.split(' ')  # 以词为单位构建词表(数据集中词之间以空格隔开)
        tokenizer = lambda x: [y for y in x]  # 以字为单位构建词表
        word_to_id = build_vocab(
            train_dir, tokenizer=tokenizer, max_size=MAX_VOCAB_SIZE, min_freq=1
        )
        pkl.dump(word_to_id, open(vocab_dir, "wb"))

    embeddings = np.random.rand(len(word_to_id), emb_dim)
    f = open(pretrain_dir, "r", encoding="UTF-8")
    for i, line in enumerate(f.readlines()):
        # if i == 0:  # 若第一行是标题，则跳过
        #     continue
        lin = line.strip().split(" ")
        if lin[0] in word_to_id:
            idx = word_to_id[lin[0]]
            emb = [float(x) for x in lin[1:301]]
            embeddings[idx] = np.asarray(emb, dtype="float32")
    f.close()
    np.savez_compressed(filename_trimmed_dir, embeddings=embeddings)
