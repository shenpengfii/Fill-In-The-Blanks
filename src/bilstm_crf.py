import torch
import torch.optim as optim

from torch import Tensor
from torch.utils.data import Dataset
from bilstm import BILSTM
from crf import CRF


class BiLSTM_CRF():
    """
    用于序列标注应用的双向长短时条件随机场网络

    变量：
        `bilstm`：BiLSTM层

        `crf`：CRF层

        `optimBilstm`：BiLSTM的优化器

        `optimCrf`：CRF的优化器
    """

    def __init__(self, vocabLength: int, labelEncodeDict: dict, embeddingDim: int, 
        hiddenDim: int, lrBilstm: float, lrCrf: float, wdBilstm: float, 
        wdCrf: float, startTag: str, stopTag: str) -> "BiLSTM_CRF":
        """
        参数：
            `vocabLength`：词袋长度

            `labelEncodeDict`：标签到编号的映射字典

            `embeddingDim`：词嵌入向量的维度

            `hiddenDim`：BiLSTM的隐藏层激活值维度
            
            `lrBilstm`：BiLSTM层的学习率
            
            `lrCrf`：CRF层的学习率
            
            `wdBilstm`：BiLSTM层的权重衰减
            
            `wdCrf`：CRF层的权重衰减

            `startTag`： 句子的起始标签
            
            `stopTag`： 句子的终止标签
        """

        super(BiLSTM_CRF, self).__init__()

        # 创建网络层
        labelLength = len(labelEncodeDict)
        self.bilstm = BILSTM(vocabLength, labelLength, embeddingDim, hiddenDim)
        self.crf = CRF(labelEncodeDict, labelLength, startTag, stopTag)

        # 创建对应的优化器
        self.optimBilstm = optim.SGD(self.bilstm.parameters(), lr=lrBilstm, weight_decay=wdBilstm)
        self.optimCrf = optim.SGD(self.crf.parameters(), lr=lrCrf, weight_decay=wdCrf)


    def train(self, trainset: Dataset, testset: Dataset, epochs: int, evalInterval: int, 
        showLog: bool = False) -> list[tuple[int, float]]:
        """
        模型训练方法，返回一个以训练轮次序号和对应的评估精度为元素的列表

        参数：
            `trainset`：训练集

            `testset`：测试集

            `epochs`：训练轮数

            `evalInterval`：隔多少轮评估一次

            `showLog`：是否输出训练过程中的评估信息
        """
        
        cur = -1
        precisions = []

        # 训练模型
        for epoch in range(1, epochs + 1):
            for sentence, labelIndexs in trainset:
                self.bilstm.zero_grad()
                self.crf.zero_grad()
                
                loss = self._calcLoss(sentence, labelIndexs)
                loss.backward()

                self.optimCrf.step()
                self.optimBilstm.step()

            # 评估模型
            if epoch // evalInterval != cur:
                cur = epoch // evalInterval
                precision = self._eval(testset)
                precisions.append((epoch, precision))

                if showLog:
                    print(f'Epoch {epoch}: precision {precision * 100: .2f}%')

        return precisions


    def _calcLoss(self, sentence: Tensor, labelIndexs: Tensor) -> Tensor:
        """
        损失计算函数，返回损失

        参数：
            `sentence`：一个数值化后的句子张量

            `labelIndexs`：该句子对应的标注张量
        """
        return self.crf.calcLoss(self.bilstm.forward(sentence), labelIndexs)
    

    def _eval(self, testset: Dataset) -> float:
        """
        模型评估方法，返回在测试集上的精度

        参数：
            `testset`：测试集
        """
        total, correct = 0, 0
        with torch.no_grad():
            for sentence, labelIndexs in testset:
                res = torch.tensor(self.forward(sentence)) == labelIndexs
                total += len(sentence)
                correct += res.sum()
        return correct / total
    

    def forward(self, sentence: Tensor, showScore: bool = False) -> list[int] | tuple[float, list[int]]:
        """
        推理函数，返回句子对应的标注编码序列和打分

        参数：
            `sentence`：一个数值化后的句子张量

            `showScore`：是否返回打分
        """
        return self.crf.forward(self.bilstm.forward(sentence), showScore)
    

    def save(self, modelUrl: tuple[str, str]) -> None:
        """
        模型保存方法

        参数：
            `modelUrl`： 模型文件的保存地址
        """
        torch.save(self.bilstm.state_dict(), modelUrl[0])
        torch.save(self.crf.state_dict(), modelUrl[1])


    def load(self, modelUrl: tuple[str, str]) -> None:
        """
        模型导入方法

        参数：
            `modelUrl`： 模型文件的地址
        """
        self.bilstm.load_state_dict(torch.load(modelUrl[0]))
        self.crf.load_state_dict(torch.load(modelUrl[1]))
