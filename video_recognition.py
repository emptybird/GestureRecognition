import sys
import cv2
import torch

from PyQt5 import QtCore
from PyQt5.QtCore import QRunnable, QThreadPool
from PyQt5.QtWidgets import *
from PyQt5.QtGui import QIcon, QPixmap, QImage
from handutil import HandDetector
from ViT import *
from playsound import playsound

class Window(QWidget):
    def __init__(self):
        super().__init__()
        self.init_detector()
        self.init_vit()
        # self.tts = pyttsx3.init()
        self.init_ui()

    def init_detector(self):
        # 初始化定时器
        self.timer_camera = QtCore.QTimer()
        self.timer_camera.timeout.connect(self.show_camera)
        # 初始化手部检测器
        self.detector = HandDetector()
        # 初始化摄像头
        self.cap = cv2.VideoCapture()
        self.count = -1

    def init_vit(self):
        # self.model = Model()
        pass

    def init_ui(self):
        # 设置窗口的大小
        self.resize(1000, 500)
        self.center()
        # 设置窗口的标题
        self.setWindowTitle('Gesture Recognition 👍🏻')
        # 设置窗口的图标，引用当前目录下的web.png图片
        self.setWindowIcon(QIcon('web.png'))

        hbox = QHBoxLayout()

        # 左侧展示选择的图片
        self.picture = QLabel()
        hbox.addWidget(self.picture, 4)

        # 右侧展示按钮及识别结果
        vbox = QVBoxLayout()

        btn_layout = QHBoxLayout()
        # 打开摄像头按钮
        self.button_open_camera = QPushButton('打开摄像头', self)
        self.button_open_camera.clicked.connect(self.open_camera)
        # 识别按钮
        btn_recognize = QPushButton('识别', self)
        btn_recognize.clicked.connect(self.recognize)
        btn_layout.addWidget(self.button_open_camera)
        btn_layout.addWidget(btn_recognize)

        # 识别结果
        self.num = QLCDNumber(self)
        self.num.display('0')

        vbox.addLayout(btn_layout)
        vbox.addWidget(self.num)

        hbox.addLayout(vbox,1)

        self.setLayout(hbox)
        # 显示窗口
        self.show()

    def recognize(self):
        count = self.detector.finger_count(self.gesture_img)
        if count == -1:
            return
        tts = TTS('sound/' + str(count) + '.mp3')
        QThreadPool.globalInstance().start(tts)

    def open_camera(self):
        if self.timer_camera.isActive() == False:
            result = self.cap.open(0)
            if result == False:
                msg = QMessageBox.Warning(self, u'Warning', u'请检测相机与电脑是否连接正确', buttons=QMessageBox.Ok,
                                          defaultButton=QMessageBox.Ok)
            else:
                self.timer_camera.start(30)
                self.button_open_camera.setText(u'关闭摄像头')
        else:
            self.timer_camera.stop()
            self.cap.release()
            self.picture.clear()
            self.button_open_camera.setText(u'打开摄像头')

    def show_camera(self):
        success, img = self.cap.read()
        if success:
            # 检测手势
            img = self.detector.find_hands(img, draw=True)
            # 缓存当前手势图片
            self.gesture_img = img
            current_count = self.detector.finger_count(img)
            if current_count > -1 and current_count != self.count:
                self.count = current_count
                tts = TTS('sound/' + str(current_count) + '.mp3')
                QThreadPool.globalInstance().start(tts)
                self.num.display(str(current_count))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # vit = Model()
            # print(vit.recognize(img))
            show_img = QImage(img.data, img.shape[1], img.shape[0], QImage.Format_RGB888)
            self.picture.setPixmap(QPixmap.fromImage(show_img).scaled(self.picture.width(), self.picture.height()))
            self.picture.setScaledContents(True)

    # 控制窗口显示在屏幕中心
    def center(self):
        # 获得窗口
        qr = self.frameGeometry()
        # 获得屏幕中心点
        cp = QDesktopWidget().availableGeometry().center()
        # 显示到屏幕中心
        qr.moveCenter(cp)
        self.move(qr.topLeft())

class TTS(QRunnable):
    def __init__(self, filename):
        super(QRunnable,self).__init__()
        self.filename = filename
    def run(self):
        playsound(self.filename)

if __name__ == '__main__':
    #创建应用程序和对象
    app = QApplication(sys.argv)
    ex = Window()
    ex.show()
    sys.exit(app.exec_())