import numpy as np
import cv2
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QLineEdit, QPushButton, QHBoxLayout, QVBoxLayout, \
    QWidget, QFileDialog
from PyQt5.QtCore import Qt


class ImageViewer(QMainWindow):
    def __init__(self):
        super().__init__()

        self.images = []  # 存储加载的图像
        self.current_image_index = 0  # 当前图像索引

        self.init_ui()

    def init_ui(self):
        # 创建控件
        self.address_edit = QLineEdit()
        self.select_button = QPushButton("选择文件")
        self.prev_button = QPushButton("<")
        self.next_button = QPushButton(">")
        self.index_label = QLabel()
        self.image_label = QLabel()

        # 设置控件属性
        self.index_label.setAlignment(Qt.AlignCenter)  # 居中显示
        self.image_label.setAlignment(Qt.AlignCenter)  # 居中显示
        self.image_label.setScaledContents(True)  # 自适应大小

        # 创建布局
        address_layout = QHBoxLayout()
        address_layout.addWidget(self.address_edit)
        address_layout.addWidget(self.select_button)

        button_layout = QHBoxLayout()
        button_layout.addWidget(self.prev_button)
        button_layout.addWidget(self.index_label)
        button_layout.addWidget(self.next_button)

        main_layout = QVBoxLayout()
        main_layout.addLayout(address_layout)
        main_layout.addWidget(self.image_label)
        main_layout.addLayout(button_layout)

        # 创建主窗口部件
        main_widget = QWidget()
        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)

        # 绑定按钮点击事件
        self.select_button.clicked.connect(self.select_file)
        self.prev_button.clicked.connect(self.prev_image)
        self.next_button.clicked.connect(self.next_image)

    def select_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "选择文件", "", "Numpy文件 (*.npy)")
        if file_path:
            self.load_images(file_path)

    def load_images(self, file_path):
        try:
            data = np.load(file_path)  # 从.npy文件中加载np.array数据
            image_shape = data.shape
            num_images = image_shape[0]
            self.images = []

            for i in range(num_images):
                image_data = data[i]
                # 将BGR形式的NumPy数组转换为RGB形式
                rgb_image_data = cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB)
                image = QImage(rgb_image_data.data, image_shape[2], image_shape[1], QImage.Format_RGB888)
                self.images.append(image)

            if self.images:
                self.current_image_index = 0
                self.update_image_label()

        except Exception as e:
            print("发生错误:", str(e))

    def prev_image(self):
        if self.current_image_index > 0:
            self.current_image_index -= 1
            self.update_image_label()

    def next_image(self):
        if self.current_image_index < len(self.images) - 1:
            self.current_image_index += 1
            self.update_image_label()

    def update_image_label(self):
        image = self.images[self.current_image_index]
        self.image_label.setPixmap(QPixmap.fromImage(image))

        self.index_label.setText(f"第 {self.current_image_index+1} 张")


if __name__ == "__main__":
    app = QApplication([])
    viewer = ImageViewer()
    viewer.show()
    app.exec()
