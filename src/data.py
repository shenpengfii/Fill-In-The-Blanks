import torch
import pickle as pkl

from torch import Tensor
from torch.utils.data import Dataset


class BrownDataset(Dataset):
    """
    基于Brown Corpus英文词性标注数据集的Pytorch封装

    变量：
        `sentences`：一个二维列表，每个子列都代表一个数值化后的句子

        `tags`：一个二维列表，每个子列都代表一个数值化后的标注

        `length`：表示数据集的长度
    """

    def __init__(self, rawSentences: list[list[str]], rawTags: list[list[str]], word2id: dict, tag2id: dict) -> "BrownDataset":
        """
        参数：
            `rawSentences`：一组句子，以单词作为元素

            `rawTags`：一组句子的对应标注，以单词对应的标注作为元素

            `word2id`：一个单词到数值的映射字典

            `tag2id`：一个标注词性到数值的映射字典
        """
        super(BrownDataset, self).__init__()
        self.sentences, self.tags = [], []
        self.length = len(rawTags)

        # 将句子和标注分别进行数值化
        for words in rawSentences:
            self.sentences.append([word2id[word] for word in words])
        for labels in rawTags:
            self.tags.append([tag2id[label] for label in labels])


    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index) -> tuple[Tensor, Tensor]:
        return torch.tensor(self.sentences[index]), torch.tensor(self.tags[index])


#########################################
#            Testing Block              #
#########################################
def test():
    # 已处理过的数据文件地址
    src = './data/data.pkl'

    # 读取文件数据
    with open(src, 'rb') as f:
        train_data, test_data, train_label, test_label, bag_of_word, bag_of_tag = pkl.load(f)
    
    # 创建词和标注的数值化映射字典
    word2id = dict(zip(bag_of_word, range(len(bag_of_word))))
    tag2id = dict(zip(bag_of_tag, range(len(bag_of_tag))))

    # 创建对应的数据集封装实例
    trainset = BrownDataset(train_data, train_label, word2id, tag2id)
    testset = BrownDataset(test_data, test_label, word2id, tag2id)

    # 测试
    print(len(trainset))
    print(len(testset))
    

if '__main__' == __name__:
    test()
