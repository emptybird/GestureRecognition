import sys

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import *
from PyQt5.QtGui import QIcon, QPixmap
from ViT import Model
from PIL import Image
from torchvision import transforms
import torch

class Window(QWidget):
    def __init__(self):
        super().__init__()
        self.init_vit()
        self.init_ui()

    def init_vit(self):
        self.model = Model()

    def init_ui(self):
        # 设置窗口的大小
        self.resize(800, 500)
        self.center()
        # 设置窗口的标题
        self.setWindowTitle('Gesture Recognition 👍🏻')
        # 设置窗口的图标，引用当前目录下的web.png图片
        self.setWindowIcon(QIcon('web.png'))

        hbox = QHBoxLayout()

        # 左侧展示选择的图片
        self.picture = QLabel()
        hbox.addWidget(self.picture)

        # 右侧展示按钮及识别结果
        vbox = QVBoxLayout()

        btn_layout = QHBoxLayout()
        # 选择图片按钮
        btn_select = QPushButton('选择图片', self)
        btn_select.clicked.connect(self.select_image)
        # 识别按钮
        btn_recognize = QPushButton('识别', self)
        btn_recognize.clicked.connect(self.recognize)
        btn_layout.addWidget(btn_select)
        btn_layout.addWidget(btn_recognize)

        # 识别结果
        self.num = QLCDNumber(self)
        self.num.display('1')

        vbox.addLayout(btn_layout)
        vbox.addWidget(self.num)

        hbox.addLayout(vbox)

        self.setLayout(hbox)
        # 显示窗口
        self.show()

    def recognize(self, img):
        transform = transforms.Compose([
            transforms.PILToTensor(), transforms.ConvertImageDtype(dtype=torch.float32)
        ])
        img_tensor = transform(img)
        img_tensor = img_tensor[None, :]
        print(img_tensor.shape)
        print(self.model.recognize_tensor(img_tensor))

    def select_image(self):
        imgName, imgType = QFileDialog.getOpenFileName(self, "选择图片", "", "Image Files(*.jpg;*.jpeg;*.png)")
        jpg = QPixmap(imgName).scaled(self.picture.width(), self.picture.height(), aspectRatioMode = Qt.KeepAspectRatio)
        self.picture.setPixmap(jpg)
        image = Image.open(imgName)
        self.recognize(image)


    # 控制窗口显示在屏幕中心
    def center(self):
        # 获得窗口
        qr = self.frameGeometry()
        # 获得屏幕中心点
        cp = QDesktopWidget().availableGeometry().center()
        # 显示到屏幕中心
        qr.moveCenter(cp)
        self.move(qr.topLeft())

if __name__ == '__main__':
    #创建应用程序和对象
    app = QApplication(sys.argv)
    ex = Window()
    sys.exit(app.exec_())