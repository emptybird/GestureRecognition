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
    # ViT识别
    MODEL_VIT  = 0
    # MediaPipe识别
    MODEL_MEDIAPIPE = 1

    def __init__(self):
        super().__init__()
        self.init_detector()
        self.init_model()
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
        self.frameCount = 0
        self.last_num = -1

    def init_model(self):
        self.model = Model()
        self.model_type = Window.MODEL_VIT

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
        # 切换按钮
        self.btn_switch_model = QPushButton('ViT', self)
        self.btn_switch_model.clicked.connect(self.switch_model_type)
        btn_layout.addWidget(self.button_open_camera)
        btn_layout.addWidget(self.btn_switch_model)

        # 识别结果
        self.num = QLCDNumber(self)
        self.num.display('0')

        vbox.addLayout(btn_layout)
        vbox.addWidget(self.num)

        hbox.addLayout(vbox,1)

        self.setLayout(hbox)
        # 显示窗口
        self.show()

    def switch_model_type(self):
        self.model_type = 1 - self.model_type
        if self.model_type == Window.MODEL_VIT:
            self.btn_switch_model.setText("ViT")
        else:
            self.btn_switch_model.setText("MediaPipe")

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
            self.num.display("0")

    def show_camera(self):
        success, img = self.cap.read()
        if success:
            # 检测手势
            img = self.detector.find_hands(img, draw=self.model_type == Window.MODEL_MEDIAPIPE)
            # 手掌目标检测
            palm_rect = self.detector.palm_detection(img, cv2)

            if self.model_type == Window.MODEL_MEDIAPIPE:
                # 缓存当前手势图片
                self.gesture_img = img
                current_count = self.detector.finger_count(img)
                if current_count == self.count:
                    self.frameCount += 1
                else:
                    self.count = current_count
                    self.frameCount = 0

                if current_count > -1 and current_count != self.last_num and self.frameCount >= 4:
                    self.last_num = current_count
                    tts = TTS('sound/' + str(current_count) + '.mp3')
                    QThreadPool.globalInstance().start(tts)
                    self.num.display(str(current_count))
            elif palm_rect:
                vit = ViT(self.model, img, palm_rect, self.num)
                QThreadPool.globalInstance().start(vit)

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
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
# 语音播放
class TTS(QRunnable):
    def __init__(self, filename):
        super(QRunnable,self).__init__()
        self.filename = filename
    def run(self):
        playsound(self.filename)

# ViT分类 语音播放
class ViT(QRunnable):
    def __init__(self, model, img, palm_rect, view):
        super(QRunnable,self).__init__()
        self.model = model
        self.img = img
        self.palm_rect = palm_rect
        self.view = view

    def run(self):
        if self.palm_rect:
            try:
                resized_img = self.resize_img(self.img, self.palm_rect)
                result = self.model.recognize(resized_img)
                finger_count = torch.max(result, 1)[1].data.numpy()[0]
                print(finger_count)
                # playsound('sound/' + str(finger_count) + '.mp3')
                self.view.display(str(finger_count))
            except Exception as e:
                print(str(e))

    def resize_img(self, img, palm_rect):
        if palm_rect:
            border_w = min(palm_rect[2], img.shape[1])
            border_h = min(palm_rect[3], img.shape[0])
            region = img[palm_rect[1]:border_h, palm_rect[0]:border_w]
            w, h = border_w - palm_rect[0], border_h - palm_rect[1]
            max_length = max(w, h)
            ratio = 64.0 / max_length
            new_w, new_h = int(ratio * w), int(ratio * h)
            resized = cv2.resize(region, (new_w, new_h))

            W, H = 64, 64
            top = bottom = (H - new_h) // 2
            if top + bottom + new_h < H:
                bottom += (H - top - bottom - new_h)
            left = right =  (W - new_w) // 2
            if left + right + new_w < W:
                right += (W - left - right - new_w)
            pad_image = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0,0,0))
            # cv2.imshow("resize image",pad_image)
            return pad_image

if __name__ == '__main__':
    #创建应用程序和对象
    app = QApplication(sys.argv)
    ex = Window()
    ex.show()
    sys.exit(app.exec_())