import torch
import torch.nn as nn

from torch import Tensor


class CRF(nn.Module):
    """
    条件随机场，用于搭配BiLISTM的序列标注应用

    变量：
    - `labelEncodeDict`
        一个标签到编号的映射字典
    - `labelLength`
        标签数
    - `startTag`
        句子的起始标签
    - `stopTag`
        句子的终止标签
    - `transMatrix`
        CRF转移矩阵
    
    方法：
    - `calcLoss(self, emitScores: Tensor, labelIndexs: Tensor) -> Tensor`
        损失计算函数
    - `forward(self, emitScores: Tensor, showScore: bool) -> list[int] | tuple[float, list[int]]`
        维特比算法推理一个句子对应的标注编码序列
    """

    def __init__(self, labelEncodeDict: dict, labelLength: int, startTag: str, stopTag: str) -> "CRF":
        """
        参数：
            `labelEncodeDict`：标签到编号的映射字典

            `labelLength`：标签数

            `startTag`： 句子的起始标签
            
            `stopTag`： 句子的终止标签
        """
        super(CRF, self).__init__()
        self.labelEncodeDict = labelEncodeDict
        self.labelLength = labelLength
        self.StartTag = startTag
        self.StopTag = stopTag

        # CRF转移矩阵，以正态分布变量初始化该矩阵
        self.transMatrix = nn.Parameter(torch.randn(labelLength, labelLength))


    def calcLoss(self, emitScores: Tensor, labelIndexs: Tensor) -> Tensor:
        """
        根据CRF算法公式计算某个句子的损失

        参数：
            `emitScores`：某句子计算后得到的一组发射向量

            `labelIndexs`：该句子对应的编码后的标签向量
        """
        return self._calcAllPathScores(emitScores) - self._calcSentenceScore(emitScores, labelIndexs)


    def _calcAllPathScores(self, emitScores: Tensor) -> Tensor:
        """
        计算CRF算法中所有路径的分数指数的对数和

        参数：
            `emitScores`：某句子计算后得到的一组发射向量
        """

        # 以StartTag作为序列起始点，初始化状态向量alphas，除StartTag外的其他位置分数设为无穷小
        alphas = torch.full((1, self.labelLength), -10000.)
        alphas[0][self.labelEncodeDict[self.StartTag]] = 0.

        # 动态规划递推
        for emitScore in emitScores:
            # 利用广播机制计算矩阵
            alphas = self._logSumExp(alphas.T + emitScore.unsqueeze(0) + self.transMatrix)

        # 以StopTag作为结尾，利用广播机制求出所有路径分数和
        return self._logSumExp(alphas.T + self.transMatrix[:, [self.labelEncodeDict[self.StopTag]]])
    

    def _calcSentenceScore(self, emitScores: Tensor, labelIndexs: Tensor) -> Tensor:
        """
        计算给定句子和标签的CRF算法路径分数

        参数：
            `emitScores`：某句子计算后得到的一组发射向量

            `labels`：该句子对应的一组编码后的标注
        """
        
        pathScore = torch.zeros(1)
        labelIndexs = torch.cat((torch.tensor(self.labelEncodeDict[self.StartTag]).unsqueeze(0), labelIndexs), dim=0)
        for i, emitScore in enumerate(emitScores):
            pathScore += emitScore[labelIndexs[i + 1]] + self.transMatrix[labelIndexs[i], labelIndexs[i + 1]]
        return pathScore + self.transMatrix[labelIndexs[-1], self.labelEncodeDict[self.StopTag]]
    

    def _logSumExp(self, values: Tensor) -> Tensor:
        """
        CRF算法中用到的logSumExp运算
        
        参数：
            `values`: 一个具有特殊意义的二维矩阵
        """

        # 按列求矩阵的最大值，并与原矩阵维度保持一致
        maxs = values.max(0, keepdim=True).values
        # 对矩阵中的每列减去该列的最大值，以避免数据溢出，再按照特定规则逐步进行logSumExp还原计算公式
        return (values - maxs).exp().sum(0, keepdim=True).log() + maxs

    
    def forward(self, emitScores: Tensor, showScore: bool) -> list[int] | tuple[float, list[int]]:
        """
        维特比解码算法推理，输出句子对应的预测标注编码和打分

        参数：
            `emitScores`：某句子计算后得到的一组发射向量

            `showScore`：是否返回打分
        """
        backTraces = []

        # 以StartTag作为序列起始点，初始化状态向量alphas，除StartTag外的其他位置分数设为无穷小
        alphas = torch.full((1, self.labelLength), -10000.)
        alphas[0][self.labelEncodeDict[self.StopTag]] = 0

        # 递推，同时记录每一步的回溯路径
        for emitScore in emitScores:
            alphas = alphas.T + emitScore.unsqueeze(0) + self.transMatrix
            backTraces.append(alphas.argmax(0).tolist())
            # 这里求logSumExp是为了防止数据溢出，并同步计算出对应的路径分数
            alphas = self._logSumExp(alphas)

        # 以StopTag作为结尾，利用广播机制求末端的最优标签
        alphas = alphas.T + self.transMatrix[:, [self.labelEncodeDict[self.StopTag]]]
        # 回溯求最优标签序列
        bestPath = [alphas.flatten().argmax().item()]
        for i, trace in enumerate(reversed(backTraces[1:])):
            bestPath.append(trace[bestPath[i]])
        bestPath.reverse()

        if showScore:
            return self._logSumExp(alphas).exp(), bestPath
        else:
            return bestPath