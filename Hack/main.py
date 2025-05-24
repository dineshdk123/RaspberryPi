import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import QMainWindow, QApplication, QFileDialog
from PyQt5.QtGui import QImage, QPixmap
from gui import Ui_MainWindow
import logic

class MainApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.image = None
        self.bind_events()

    def bind_events(self):
        self.ui.uploadButton1.clicked.connect(self.upload_image)
        self.ui.Grayscaled.clicked.connect(lambda: self.apply_filter(logic.to_grayscale))
        self.ui.Thresholdingd.clicked.connect(lambda: self.apply_filter(logic.threshold))
        self.ui.GaussianBlurd.clicked.connect(lambda: self.apply_filter(logic.gaussian_blur))
        self.ui.Dilationd.clicked.connect(lambda: self.apply_filter(logic.dilation))
        self.ui.Erosiond.clicked.connect(lambda: self.apply_filter(logic.erosion))
        self.ui.MedianBlurd.clicked.connect(lambda: self.apply_filter(logic.median_blur))
        self.ui.CannyEdgeDetectiond.clicked.connect(lambda: self.apply_filter(logic.canny_edge))
        self.ui.sobelEdgeDetectiond.clicked.connect(lambda: self.apply_filter(logic.sobel_edge))
        self.ui.ImageRotationd.clicked.connect(lambda: self.apply_filter(logic.rotate_image))
        self.ui.ImageResizingd.clicked.connect(lambda: self.apply_filter(logic.resize_image))
        self.ui.save1.clicked.connect(self.save_image)
        self.ui.Reset1.clicked.connect(self.reset_image)
        self.ui.brightnessSlider.valueChanged.connect(self.adjust_brightness_contrast)
        self.ui.contrastSlider.valueChanged.connect(self.adjust_brightness_contrast)


    def upload_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Image Files (*.png *.jpg *.jpeg)")
        if path:
            self.image = cv2.imread(path)
            self.display_image(self.image)

    def display_image(self, img):
        if len(img.shape) == 2:  # Grayscale
            qimg = QImage(img, img.shape[1], img.shape[0], img.strides[0], QImage.Format_Grayscale8)
        else:  # Color image
            rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            qimg = QImage(rgb_image.data, rgb_image.shape[1], rgb_image.shape[0], QImage.Format_RGB888)

        self.ui.outputimage.setPixmap(QPixmap.fromImage(qimg).scaled(
            self.ui.outputimage.width(), self.ui.outputimage.height()))


    def apply_filter(self, func):
        if self.image is not None:
            filtered = func(self.image)
            self.display_image(filtered)
            self.image = filtered

    def save_image(self):
        if self.image is not None:
            path, _ = QFileDialog.getSaveFileName(self, "Save Image", "", "PNG Files (*.png)")
            if path:
                cv2.imwrite(path, self.image)

    def reset_image(self):
        self.upload_image()
    
    def adjust_brightness_contrast(self):
        if self.image is not None:
            brightness = self.ui.brightnessSlider.value() - 100  # -100 to +100
            contrast = self.ui.contrastSlider.value() - 100      # -100 to +100

            img = self.image.astype(np.int16)
            img = img * (1 + contrast / 100.0) + brightness
            img = np.clip(img, 0, 255).astype(np.uint8)

            self.display_image(img)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainApp()
    window.show()
    sys.exit(app.exec_())
