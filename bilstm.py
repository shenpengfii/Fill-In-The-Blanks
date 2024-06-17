import torch
import torch.nn as nn

from torch import Tensor


class BILSTM(nn.Module):
    """
    双向长短期记忆网络，适用于序列标注应用  

    变量：
        `hiddenDim`：隐藏层的输出维度

        `embeddingLayer`：词嵌入层

        `bilstmLayer`：BiLSTM隐藏层

        `emissionLayer`BiLSTM线性输出层，也即发射层
    
    方法：
        `forward(self, sentence: Tensor) -> Tensor`
            推理函数，接收一个编码后的词序列张量
    """
    
    def __init__(self, vocabLength: int, labelLength: int, embeddingDim: int, hiddenDim: int) -> "BILSTM":
        """
        参数：
            `vocabLength`：词袋长度

            `labelLength`：标签数

            `embeddingDim`：词嵌入向量的维度

            `hiddenDim`：BiLSTM的隐藏层的输出维度
        """
        super(BILSTM, self).__init__()
        self.hiddenDim = hiddenDim

        # 词嵌入层
        self.embeddingLayer = nn.Embedding(vocabLength, embeddingDim)
        # BiLSTM隐藏层，由官方文档可知，bidirectional为True时，输出维度是实际参数的2倍
        self.bilstmLayer = nn.LSTM(embeddingDim, hiddenDim // 2, bidirectional=True)
        # BiLSTM线性输出层，也即发射层
        self.emissionLayer = nn.Linear(hiddenDim, labelLength)


    def forward(self, sentence: Tensor) -> Tensor:
        """
        返回一组关于该句子的发射分数张量
        
        参数：
            sentence：一个数值化后的句子
        """

        # 将数值化的词编码到词向量，并返回BiLSTM网络需要的（seq_length, batch_size, 词向量维度）格式
        embeds = self.embeddingLayer(sentence).view(len(sentence), 1, -1)
        # 将词向量序列送入BiLISTM隐藏层，得到激活值
        lstm_out, _ = self.bilstmLayer(embeds, self._init_hidden())
        # 对激活值维度进行处理，并送入线性层运算得到发射向量
        return self.emissionLayer(lstm_out.view(len(sentence), -1))
    

    def _init_hidden(self):
        """
        创建随机初始化的BiLSTM网络隐藏层中的hidden state和cell state张量
        """

        # 按照官方文档中提供的格式进行书写
        return (
            torch.randn(2, 1, self.hiddenDim // 2),  # h_0: hidden state
            torch.randn(2, 1, self.hiddenDim // 2))  # c_0: cell state