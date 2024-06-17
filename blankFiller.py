import os
import torch
import numpy as np
import pickle as pkl

from data import BrownDataset
from bilstm_crf import BiLSTM_CRF


class BlankFiller():
    """
    英文完形填空做题机

    变量：
        `word2id`：一个单词到数值的映射字典

        `tag2id`：一个标注词性到数值的映射字典

        `bilstm_crf`：对BiLSTM_CRF模型的封装对象，提供了用于训练、推理和评估的方法

    方法：
        `predict(self, sentence: str, choices: list[str]) -> str`：完形填空的做题方法
    """

    def __init__(self, embeddingDim: int, hiddenDim: int, lrBilstm: float, lrCrf: float, 
        wdBilstm: float, wdCrf: float, epochs: int, evalIntervals: int, data_url: str, model_url: tuple[str, str] | None = None,
        startTag: str = "<START>", stopTag: str = "<STOP>") -> "BlankFiller":
        """
        参数：
            `embeddingDim`：词嵌入向量的维度
            
            `hiddenDim`：BiLSTM网络隐状态的维度
            
            `lrBilstm`：BiLSTM层的学习率
            
            `lrCrf`：CRF层的学习率
            
            `wdBilstm`：BiLSTM层的权重衰减
            
            `wdCrf`：CRF层的权重衰减
            
            `epochs`：训练周期数
            
            `evalIntervals`：隔多个轮训练评估一次

            `data_url`: 数据文件的地址

            `model_url`： 模型文件的地址

            `startTag`： 句子的起始标签
            
            `stopTag`： 句子的终止标签
        """

        # 导入数据
        with open(data_url, 'rb') as f:
            train_data, test_data, train_label, test_label, bag_of_word, bag_of_tag = pkl.load(f)
        self.word2id = dict(zip(bag_of_word, range(len(bag_of_word))))
        self.tag2id = dict(zip(bag_of_tag, range(len(bag_of_tag))))

        # 加入起始和终止标记
        len_t = len(self.tag2id)
        self.tag2id[startTag] = len_t
        self.tag2id[stopTag] = len_t + 1

        # 创建数据集
        trainset = BrownDataset(train_data, train_label, self.word2id, self.tag2id)
        testset = BrownDataset(test_data, test_label, self.word2id, self.tag2id)

        # 创建模型
        self.bilstm_crf = BiLSTM_CRF(len(self.word2id), self.tag2id, embeddingDim, hiddenDim, lrBilstm, lrCrf, wdBilstm, wdCrf, startTag, stopTag)
        if os.path.exists(model_url[0]) and os.path.exists(model_url[1]):
            self.bilstm_crf.load(model_url)
        else:
            print(f'trainset size: {len(trainset)}')
            print(f'testsize size: {len(testset)}')
            print(f'bag-of-words size: {len(self.word2id)}')
            print(f'bag-of-tags size: {len(self.tag2id)}')
            self.bilstm_crf.train(trainset, testset, epochs, evalIntervals, showLog=True)
            self.bilstm_crf.save(model_url)
        

    def predict(self, sentence: str, choices: list[str]) -> str:
        """
        完形填空预测函数，返回预测后的完整句子

        参数：
            `sentence`: 一个用`()`表示空缺词的英文句子

            `choices`: 一个选项列表
        """
        scores = []
        sentence = sentence.replace('()', '{}')
        with torch.no_grad():
            for choice in choices:
                tokens = [self.word2id[word] for word in sentence.format(choice).split()]
                # 测试对应选项
                score, _ = self.bilstm_crf.forward(torch.tensor(tokens), showScore=True)
                scores.append(score)
        # 返回最优的填空句
        return sentence.format(choices[np.array(scores).argmax()])


#########################################
#            Testing Block              #
#########################################
def test():
    # 模型超参数
    EMBEDDING_DIM = 128
    HIDDEN_DIM = 32
    LR_BILSTM = 0.01
    LR_CRF = 0.1
    WD_BILSTM = 1e-4
    WD_CRF = 1e-4
    EPOCHS = 10
    EVAL_INTERVALS = 1
    
    # 数据和模型文件地址
    data_url = './data/data.pkl'
    model_url = (
        './model/bilstm.pth',
        './model/crf.pth'
    )

    # 创建模型实例
    # torch.manual_seed(1) # 训练的时候固定随机数种子，方便检查问题
    blankFiller = BlankFiller(EMBEDDING_DIM, HIDDEN_DIM, LR_BILSTM, LR_CRF, WD_BILSTM, WD_CRF, EPOCHS, EVAL_INTERVALS, data_url, model_url)
    
    # 完型填空测试1
    problem = 'Dear Sirs : let me begin by clearing up any possible misconception in your minds , () you are .'
    choices = ['wherever', 'whatever', 'at', 'where']
    answer = 'Dear Sirs : let me begin by clearing up any possible misconception in your minds , wherever you are .'

    result = blankFiller.predict(problem, choices)

    print(f'# Prediction: {result}')
    print(f'# Result: {result == answer}')
    print('')

    # 完型填空测试2
    problem = 'Several defendants in () Summerdale police burglary trial made statements indicating their guilt at the time of their arrest , Judge James was told yesterday .'
    choices = ['the', 'a', 'at', 'where']
    answer = 'Several defendants in the Summerdale police burglary trial made statements indicating their guilt at the time of their arrest , Judge James was told yesterday .'

    result = blankFiller.predict(problem, choices)

    print(f'# Prediction: {result}')
    print(f'# Result: {result == answer}')


if '__main__' == __name__:
    test()
