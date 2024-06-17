import sys
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QLineEdit, QTextEdit, QPushButton
from blankFiller import BlankFiller


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


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()

        # 创建模型实例
        # torch.manual_seed(1) # 训练的时候固定随机数种子，方便检查问题
        self.blankFiller = BlankFiller(EMBEDDING_DIM, HIDDEN_DIM, LR_BILSTM, LR_CRF, WD_BILSTM, WD_CRF, EPOCHS, EVAL_INTERVALS, data_url, model_url)

        self.initUI()


    def initUI(self):
        self.setWindowTitle('BlankFiller')

        layout = QVBoxLayout()

        self.input1 = QLineEdit()
        self.input1.setPlaceholderText("请输入问题")
        layout.addWidget(self.input1)

        self.input2 = QLineEdit()
        self.input2.setPlaceholderText("请输入选项")
        layout.addWidget(self.input2)

        self.output = QTextEdit()
        layout.addWidget(self.output)

        self.submit_btn = QPushButton('提交')
        self.submit_btn.clicked.connect(self.submit)
        layout.addWidget(self.submit_btn)

        self.clear_btn = QPushButton('清空')
        self.clear_btn.clicked.connect(self.clear)
        layout.addWidget(self.clear_btn)

        self.setLayout(layout)


    def submit(self):
        problem = self.input1.text()
        choices = self.input2.text().split()

        # 根据输入的问题和选项调用 predict 方法获取答案
        result = self.blankFiller.predict(problem, choices)

        # 将结果显示在输出框中
        if result:
            self.output.setText(result)


    def clear(self):
        self.input1.clear()
        self.input2.clear()
        self.output.clear()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())