import csv
import os
import shutil
import sys
import numpy as np
from PyQt5 import QtWidgets, QtCore, QtGui, sip
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from xlsxwriter.workbook import Workbook
import SimpleITK as sitk 
from time import sleep
import qtawesome
import cv2
import shutil
from copy import deepcopy
import subprocess
import xlwt
import logging
from PIL import Image, ImageQt
from utils import *

class WindowSlider(QWidget):
    def __init__(self,
                 parent,
                 title="Window Center",
                 minimum=-1000,
                 maximum=1000,
                 single_step=1,
                 value=0,
                 interval=50,
                 add_slider=True,
                 value_top_margin=10,
                 slider_top_margin=5):
        super().__init__(parent)

        Vlayer = QVBoxLayout()

        # Qlayer = QHBoxLayout()
        # self.Ltitle = QLabel(title)
        # self.Ltitle.move(0, 0)
        # Qlayer.addWidget(self.Ltitle)
        # self.Evalue = QLineEdit()
        # self.Evalue.setText(str(value))
        # self.Evalue.move(value_left_margin, 0)
        # self.Evalue.returnPressed.connect(self._textValueChangedFunc)
        # Qlayer.addWidget(self.Evalue)
        # tmp = QWidget()
        # tmp.setLayout(Qlayer)
        # Vlayer.addWidget(tmp)

        self.Ltitle = QLabel(title)
        self.Ltitle.move(0, 0)
        Vlayer.addWidget(self.Ltitle)

        if add_slider:
            self.Slider = QSlider(Qt.Horizontal)
            self.Slider.move(0, slider_top_margin)
            self.Slider.setMinimum(minimum)
            self.Slider.setMaximum(maximum)
            self.Slider.setSingleStep(single_step)
            self.Slider.setValue(value)
            self.Slider.setTickPosition(QSlider.TicksBelow)
            self.Slider.setTickInterval(interval)
            self.Slider.valueChanged.connect(self._sliderValueChangedFunc)
            Vlayer.addWidget(self.Slider)

        self.Evalue = QLineEdit()
        self.Evalue.setText(str(value))
        self.Evalue.move(0, 5)
        self.Evalue.returnPressed.connect(self._textValueChangedFunc)
        Vlayer.addWidget(self.Evalue)

        self.setLayout(Vlayer)

    def _sliderValueChangedFunc(self):
        cur_value = int(self.Evalue.text())
        cur_slider_value = int(self.Slider.value())
        if cur_value == cur_slider_value:
            return 
        self.Evalue.setText(str(cur_slider_value))

    def _textValueChangedFunc(self):
        cur_value = int(self.Evalue.text())
        cur_slider_value = int(self.Slider.value())
        if cur_value == cur_slider_value:
            return 
        self.Slider.setValue(cur_value)


class LabelSelector(QWidget):
    def __init__(self, 
                 parent,
                 title="Label Selector",
                 labels=["MPSI", "MPSO", "MVEN", "SAT", "VAT"],
                 button_top_margin=30,
                 button_left_margin=50,
                 button_margin=30,
                 color_box_size=(30, 20),
                 left_margin=10):
        super().__init__(parent)

        self.Ltitle = QLabel(title, self)
        self.Ltitle.move(left_margin, 0)

        self.Buttons = []
        for i, label in enumerate(labels):
            label_id = i+1
            mode = f"{label} (Label#{label_id})"
            Button = QCheckBox(mode, self)
            Button.setChecked(True)
            Button.mode = mode
            Button.toggled.connect(self._modeChangedFunc)
            Button.move(button_left_margin, button_top_margin+button_margin*i)
            Lcolor = QLabel(self)
            Lcolor.resize(*color_box_size)
            Lcolor.move(left_margin, button_top_margin*1.2+button_margin*i)
            color = cmap[label_to_id[label]][::-1]
            Lcolor.setStyleSheet(f"background-color:rgb({color[0]},{color[1]},{color[2]})")

            self.Buttons.append(Button)

        # self.BshowImage = QRadioButton("Show image", self)
        # self.BshowImage.setChecked(True)
        # self.BshowImage.move(button_left_margin, button_top_margin+button_margin*len(labels))
        
    def _modeChangedFunc(self):
        self.mode = self.sender().mode 


class SuperQLabel(QLabel):
    def __init__(self,
                 parent,
                 background_file="Icons/black.jpg"):
        super().__init__(parent)

        self.imgPixmap = QPixmap(background_file)                                   # 载入图片
        self.scaledImg = self.imgPixmap.scaled(self.size())                    # 初始化缩放图

        self.singleOffset = QPoint(0, 0)                                       # 初始化偏移值
        
        self.isLeftPressed = bool(False)                                       # 图片被点住(鼠标左键)标志位
        self.isImgLabelArea = bool(True) 

    def setPixmap(self, image_file):
        self.imgPixmap = QPixmap(image_file)                                   # 载入图片
        self.scaledImg = self.imgPixmap.scaled(self.size())                    # 初始化缩放图
        super().setPixmap(self.scaledImg)  

    '''重载绘图: 动态绘图'''
    def paintEvent(self,event):
        self.imgPainter = QPainter()                                           # 用于动态绘制图片
        self.imgFramePainter = QPainter()                                      # 用于动态绘制图片外线框
        self.imgPainter.begin(self)                                            # 无begin和end,则将一直循环更新
        self.imgPainter.drawPixmap(self.singleOffset, self.scaledImg)          # 从图像文件提取Pixmap并显示在指定位置
        self.imgFramePainter.setPen(QColor(168, 34, 3))  # 不设置则为默认黑色   # 设置绘图颜色/大小/样式
        self.imgFramePainter.drawRect(10, 10, 480, 480)                        # 为图片绘外线狂(向外延展1)
        self.imgPainter.end()                                                  # 无begin和end,则将一直循环更新

# =============================================================================
# 图片移动: 首先,确定图片被点选(鼠标左键按下)且未左键释放;
#          其次,确定鼠标移动;
#          最后,更新偏移值,移动图片.
# =============================================================================
#     '''重载一下鼠标按下事件(单击)'''
#     def mousePressEvent(self, event):
#         if event.buttons() == QtCore.Qt.LeftButton:                            # 左键按下
#             print("鼠标左键单击")  # 响应测试语句
#             self.isLeftPressed = True;                                         # 左键按下(图片被点住),置Ture
#             self.preMousePosition = event.pos()                                # 获取鼠标当前位置
#         elif event.buttons () == QtCore.Qt.RightButton:                        # 右键按下
#             print("鼠标右键单击")  # 响应测试语句
#         elif event.buttons() == QtCore.Qt.MidButton:                           # 中键按下
#             print("鼠标中键单击")  # 响应测试语句
#         elif event.buttons() == QtCore.Qt.LeftButton | QtCore.Qt.RightButton:  # 左右键同时按下
#             print("鼠标左右键同时单击")  # 响应测试语句
#         elif event.buttons() == QtCore.Qt.LeftButton | QtCore.Qt.MidButton:    # 左中键同时按下
#             print("鼠标左中键同时单击")  # 响应测试语句
#         elif event.buttons() == QtCore.Qt.MidButton | QtCore.Qt.RightButton:   # 右中键同时按下
#             print("鼠标右中键同时单击")  # 响应测试语句
#         elif event.buttons() == QtCore.Qt.LeftButton | QtCore.Qt.MidButton \
#              | QtCore.Qt.RightButton:                                          # 左中右键同时按下
#             print("鼠标左中右键同时单击")  # 响应测试语句
                        
#     '''重载一下滚轮滚动事件'''
#     def wheelEvent(self, event):
# #        if event.delta() > 0:                                                 # 滚轮上滚,PyQt4
#         # This function has been deprecated, use pixelDelta() or angleDelta() instead.
#         angle=event.angleDelta() / 8                                           # 返回QPoint对象，为滚轮转过的数值，单位为1/8度
#         angleX=angle.x()                                                       # 水平滚过的距离(此处用不上)
#         angleY=angle.y()                                                       # 竖直滚过的距离
#         if angleY > 0:                                                         # 滚轮上滚
#             print("鼠标中键上滚")  # 响应测试语句
#             self.scaledImg = self.imgPixmap.scaled(self.scaledImg.width()+5,
#                                                    self.scaledImg.height()+5)
#             newWidth = event.x() - (self.scaledImg.width() * (event.x()-self.singleOffset.x())) \
#                         / (self.scaledImg.width()-5)
#             newHeight = event.y() - (self.scaledImg.height() * (event.y()-self.singleOffset.y())) \
#                         / (self.scaledImg.height()-5)
#             self.singleOffset = QPoint(newWidth, newHeight)                    # 更新偏移量
#             self.repaint()                                                     # 重绘
#         else:                                                                  # 滚轮下滚
#             print("鼠标中键下滚")  # 响应测试语句
#             self.scaledImg = self.imgPixmap.scaled(self.scaledImg.width()-5,
#                                                    self.scaledImg.height()-5)
#             newWidth = event.x() - (self.scaledImg.width() * (event.x()-self.singleOffset.x())) \
#                         / (self.scaledImg.width()+5)
#             newHeight = event.y() - (self.scaledImg.height() * (event.y()-self.singleOffset.y())) \
#                         / (self.scaledImg.height()+5)
#             self.singleOffset = QPoint(newWidth, newHeight)                    # 更新偏移量
#             self.repaint()                                                     # 重绘
            
#     '''重载一下鼠标键公开事件'''
#     def mouseReleaseEvent(self, event):
#         if event.buttons() == QtCore.Qt.LeftButton:                            # 左键释放
#             self.isLeftPressed = False;  # 左键释放(图片被点住),置False
#             print("鼠标左键松开")  # 响应测试语句
#         elif event.button() == Qt.RightButton:                                 # 右键释放
#             self.singleOffset = QPoint(0, 0)                                   # 置为初值
#             self.scaledImg = self.imgPixmap.scaled(self.size())                # 置为初值
#             self.repaint()                                                     # 重绘
#             print("鼠标右键松开")  # 响应测试语句
 
#     '''重载一下鼠标移动事件'''
#     def mouseMoveEvent(self,event):
#         if self.isLeftPressed:                                                 # 左键按下
#             print("鼠标左键按下，移动鼠标")  # 响应测试语句
#             self.endMousePosition = event.pos() - self.preMousePosition        # 鼠标当前位置-先前位置=单次偏移量
#             self.singleOffset = self.singleOffset + self.endMousePosition      # 更新偏移量
#             self.preMousePosition = event.pos()                                # 更新当前鼠标在窗口上的位置，下次移动用
#             self.repaint()                                                     # 重绘

def replace_color(img, src_clr, dst_clr):
    img_arr = np.asarray(img, dtype=np.double)
        
    r_img = img_arr[:,:,0].copy()
    g_img = img_arr[:,:,1].copy()
    b_img = img_arr[:,:,2].copy()

    img = r_img * 256 * 256 + g_img * 256 + b_img
    src_color = src_clr[0] * 256 * 256 + src_clr[1] * 256 + src_clr[2] #编码

    r_img[img == src_color] = dst_clr[0]
    g_img[img == src_color] = dst_clr[1]
    b_img[img == src_color] = dst_clr[2]
        
    dst_img = np.array([r_img, g_img, b_img], dtype=np.uint8)
    dst_img = dst_img.transpose(1,2,0)
        
    return dst_img

def extract_color(img, dst_clr):
    img_arr = np.asarray(img, dtype=np.double)
        
    r_img = img_arr[:,:,0].copy()
    g_img = img_arr[:,:,1].copy()
    b_img = img_arr[:,:,2].copy()

    img = r_img * 256 * 256 + g_img * 256 + b_img
    dst_color = dst_clr[0] * 256 * 256 + dst_clr[1] * 256 + dst_clr[2] #编码
 
    r_img[img != dst_color] = 0
    g_img[img != dst_color] = 0
    b_img[img != dst_color] = 0
        
    dst_img = np.array([r_img, g_img, b_img], dtype=np.uint8)
    dst_img = dst_img.transpose(1,2,0)
        
    return dst_img

def merge_image(img1, img2):
    mask = img1.max(axis=-1)[..., np.newaxis]
    img = img1 * mask + img2 * (1-mask)
    return img

class SuperDrawQLabel(QLabel):
    def __init__(self, parent, cache_dir, background_file="Icons/black.jpg"):
        super().__init__(parent)

        #self.imgPixmap = QPixmap('black.jpg')                                   # 载入图片
        self.imgPixmap = QPixmap(background_file)
        self.imgPixmap.fill(Qt.black)
        self.scaledImg = self.imgPixmap.scaled(self.size())                    # 初始化缩放图
        
        self.emptyPixmap = QPixmap(background_file)
        self.emptyPixmap.fill(Qt.black)
        self.emptyImg = self.emptyPixmap.scaled(self.size())

        self.singleOffset = QPoint(0, 0)                                       # 初始化偏移值
        self.offset_width = 0
        self.offset_height = 0
        
        self.isLeftPressed = bool(False)                                       # 图片被点住(鼠标左键)标志位
        self.isImgLabelArea = bool(True) 

        self.lastPoint = QPoint()
        self.endPoint = QPoint()
        self.offset = QPoint(0, 0)

        self.openPaint = False
        self.curLabel = 'BACKGROUND'
        self.penColor = Qt.red
        self.imageFile = None

        self.base_mask_file = os.path.join(cache_dir, "Outputs", "show_seg.jpg")
        self.base_mask_with_img_file = os.path.join(cache_dir, "Outputs", "show_seg_with_img.jpg")
        self.cache_dir = os.path.join(cache_dir, 'Drawing')
        os.makedirs(self.cache_dir, exist_ok=True)
        self.num_back = 10
        self.cur_idx = 0
        self.num_back_step = 0
        self.ni_to_np_seg = {ni: None for ni in range(self.num_back)}

        self.wheel_width = 0
        self.wheel_height = 0
        self.penWidth = 4

    def setPixmap(self, image_file):
        self.imgPixmap = QPixmap(image_file)                                   # 载入图片
        self.scaledImg = self.imgPixmap.scaled(self.size())                    # 初始化缩放图
        self.emptyImg = self.emptyPixmap.scaled(self.size())
        super().setPixmap(self.scaledImg)  

        self.scaledImg.save(os.path.join(self.cache_dir, f'{self.cur_idx}.jpg'))

    def setBaseSeg(self, np_seg):
        self.np_seg = np_seg # (H, W)
        self.ni_to_np_seg[self.cur_idx] = self.np_seg

    def _updateBaseSeg(self, pre_idx, cur_idx, draw_file):
        diff_img = np.asarray(Image.open(draw_file).convert("L")).astype(np.uint8)
        H, W = diff_img.shape
        T = min(H, W)
        padding = (W-T, H-T)
        diff_img = diff_img[padding[1]//2:padding[1]//2+T, padding[0]//2:padding[0]//2+T]
        diff_img = cv2.resize(diff_img, self.np_seg.shape, interpolation=cv2.INTER_NEAREST)
        np_seg = deepcopy(self.ni_to_np_seg[pre_idx])
        np_seg[diff_img > 200] = label_to_id[self.curLabel]
        self.ni_to_np_seg[cur_idx] = np_seg
        #cv2.imwrite("1.jpg", (self.np_seg == label_to_id[self.curLabel]).astype(np.uint8) * 255)

        seg = deepcopy(np_seg)
        seg[seg == 100] = 0
        seg[seg != label_to_id[self.curLabel]] = 0
        seg = cmap[seg]
        seg = cv2.resize(seg, (T, T), interpolation=cv2.INTER_NEAREST)
        new_seg = np.zeros((H, W, seg.shape[2]))
        padding = [W-T, H-T]
        new_seg[padding[1]//2:padding[1]//2+T,
                padding[0]//2:padding[0]//2+T, :] = seg
        cv2.imwrite(self.base_mask_file, new_seg)

    def paintEvent(self, event):
        if not self.openPaint:
            self.imgPainter = QPainter()                                           # 用于动态绘制图片
            self.imgFramePainter = QPainter()                                      # 用于动态绘制图片外线框
            self.imgPainter.begin(self)                                            # 无begin和end,则将一直循环更新
            self.imgPainter.drawPixmap(self.singleOffset, self.scaledImg)          # 从图像文件提取Pixmap并显示在指定位置
            self.imgFramePainter.setPen(QColor(168, 34, 3))  # 不设置则为默认黑色   # 设置绘图颜色/大小/样式
            self.imgFramePainter.drawRect(10, 10, 480, 480)                        # 为图片绘外线狂(向外延展1)
            self.imgPainter.end() 
        else:
            pp = QPainter(self.scaledImg)
            # pen = QPen(self.penColor, 2, Qt.SolidLine)
            pen = QPen(QColor(*cmap[label_to_id[self.curLabel]][::-1]), self.penWidth, Qt.SolidLine)
            pp.setPen(pen)
            # 根据鼠标指针前后两个位置绘制直线
            # print(self.lastPoint, self.endPoint, self.offset)
            pp.drawLine(self.lastPoint, self.endPoint)
            # 让前一个坐标值等于后一个坐标值，
            # 这样就能实现画出连续的线
            #self.lastPoint = self.endPoint
            painter = QPainter(self)
            #绘制画布到窗口指定位置处
            painter.drawPixmap(self.singleOffset, self.scaledImg)

            pp = QPainter(self.emptyImg)
            # pen = QPen(self.penColor, 2, Qt.SolidLine)
            pen = QPen(QColor(255, 255, 255), self.penWidth, Qt.SolidLine)
            pp.setPen(pen)
            # 根据鼠标指针前后两个位置绘制直线
            # print(self.lastPoint, self.endPoint, self.offset)
            pp.drawLine(self.lastPoint, self.endPoint)
            # 让前一个坐标值等于后一个坐标值，
            # 这样就能实现画出连续的线
            self.lastPoint = self.endPoint

# =============================================================================
# 图片移动: 首先,确定图片被点选(鼠标左键按下)且未左键释放;
#          其次,确定鼠标移动;
#          最后,更新偏移值,移动图片.
# =============================================================================
    '''重载一下鼠标按下事件(单击)'''
    def mousePressEvent(self, event):
        if not self.openPaint:
            pass
            if event.buttons() == QtCore.Qt.LeftButton:                            # 左键按下
                #print("鼠标左键单击")  # 响应测试语句
                self.isLeftPressed = True;                                         # 左键按下(图片被点住),置Ture
                self.preMousePosition = event.pos()                                # 获取鼠标当前位置
            elif event.buttons () == QtCore.Qt.RightButton:                        # 右键按下
                #print("鼠标右键单击")  # 响应测试语句
                pass
            elif event.buttons() == QtCore.Qt.MidButton:                           # 中键按下
                #print("鼠标中键单击")  # 响应测试语句
                pass
            elif event.buttons() == QtCore.Qt.LeftButton | QtCore.Qt.RightButton:  # 左右键同时按下
                #print("鼠标左右键同时单击")  # 响应测试语句
                pass
            elif event.buttons() == QtCore.Qt.LeftButton | QtCore.Qt.MidButton:    # 左中键同时按下
                #print("鼠标左中键同时单击")  # 响应测试语句
                pass
            elif event.buttons() == QtCore.Qt.MidButton | QtCore.Qt.RightButton:   # 右中键同时按下
                #print("鼠标右中键同时单击")  # 响应测试语句
                pass
            elif event.buttons() == QtCore.Qt.LeftButton | QtCore.Qt.MidButton \
                | QtCore.Qt.RightButton:                                          # 左中右键同时按下
                #print("鼠标左中右键同时单击")  # 响应测试语句
                pass
        else:
            # 鼠标左键按下
            if event.button() == Qt.LeftButton:
                self.lastPoint = event.pos() - self.singleOffset #self.offset
                # 上面这里减去一个偏移量，否则鼠标点的位置和线的位置不对齐
                self.endPoint = self.lastPoint
                #print(self.endPoint)
                        
#     '''重载一下滚轮滚动事件'''
#     def wheelEvent(self, event, stepwidth=15):
# #        if event.delta() > 0:                                                 # 滚轮上滚,PyQt4
#         # This function has been deprecated, use pixelDelta() or angleDelta() instead.
#         angle=event.angleDelta() / 8                                           # 返回QPoint对象，为滚轮转过的数值，单位为1/8度
#         angleX=angle.x()                                                       # 水平滚过的距离(此处用不上)
#         angleY=angle.y()                                                       # 竖直滚过的距离
#         if angleY > 0:                                                         # 滚轮上滚
#             #print("鼠标中键上滚")  # 响应测试语句
#             self.scaledImg = self.imgPixmap.scaled(self.scaledImg.width()+stepwidth,
#                                                 self.scaledImg.height()+stepwidth)
#             self.wheel_width += stepwidth
#             self.wheel_height += stepwidth
#             newWidth = event.x() - (self.scaledImg.width() * (event.x()-self.singleOffset.x())) \
#                         / (self.scaledImg.width()-stepwidth)
#             newHeight = event.y() - (self.scaledImg.height() * (event.y()-self.singleOffset.y())) \
#                         / (self.scaledImg.height()-stepwidth)
#             self.singleOffset = QPoint(newWidth, newHeight)                    # 更新偏移量
#             self.offset_width += newWidth
#             self.offset_height += newHeight
#             self.repaint()                                                     # 重绘
#         else:                                                                  # 滚轮下滚
#             #print("鼠标中键下滚")  # 响应测试语句
#             self.scaledImg = self.imgPixmap.scaled(self.scaledImg.width()-stepwidth,
#                                                 self.scaledImg.height()-stepwidth)
#             self.wheel_width -= stepwidth
#             self.wheel_height -= stepwidth
#             newWidth = event.x() - (self.scaledImg.width() * (event.x()-self.singleOffset.x())) \
#                         / (self.scaledImg.width()+stepwidth)
#             newHeight = event.y() - (self.scaledImg.height() * (event.y()-self.singleOffset.y())) \
#                         / (self.scaledImg.height()+stepwidth)
#             self.singleOffset = QPoint(newWidth, newHeight)                    # 更新偏移量
#             self.offset_width += newWidth
#             self.offset_height += newHeight
#             self.repaint()                                                     # 重绘

    def backFunc(self):
        if self.num_back_step > self.num_back:
            return 
        if not os.path.isfile(os.path.join(self.cache_dir, f"{(self.cur_idx + self.num_back - 1) % self.num_back}.jpg")):
            return
        self.cur_idx = (self.cur_idx + self.num_back - 1) % self.num_back
        cur_file = os.path.join(self.cache_dir, f'{self.cur_idx}.jpg')
        self.setPixmap(cur_file)
        shutil.copy(cur_file, self.base_mask_with_img_file)
        self.scaledImg = self.imgPixmap.scaled(self.scaledImg.width()+self.wheel_width,
                                            self.scaledImg.height()+self.wheel_height)
        self.repaint()  
        self.num_back_step += 1
        
    '''重载一下鼠标键公开事件'''
    def mouseReleaseEvent(self, event):
        if not self.openPaint:
            pass
            if event.buttons() == QtCore.Qt.LeftButton:                            # 左键释放
                self.isLeftPressed = False;  # 左键释放(图片被点住),置False
                #print("鼠标左键松开")  # 响应测试语句
            elif event.button() == Qt.RightButton:                                 # 右键释放
                self.singleOffset = QPoint(0, 0)                                   # 置为初值
                self.scaledImg = self.imgPixmap.scaled(self.size())                # 置为初值
                self.repaint()                                                     # 重绘
                #print("鼠标右键松开")  # 响应测试语句
        else:
            # 鼠标左键释放
            if event.button() == Qt.LeftButton:
                self.endPoint = event.pos() - self.singleOffset #self.offset
                # 进行重新绘制
                # self.update()
                self.pre_idx = deepcopy(self.cur_idx)
                self.cur_idx = (self.cur_idx + 1) % self.num_back
                cur_file = os.path.join(self.cache_dir, f'{self.cur_idx}.jpg')
                self.scaledImg.save(cur_file)
                draw_file = os.path.join(self.cache_dir, "draw.jpg")
                self.emptyImg.save(draw_file)
                self._updateBaseSeg(self.pre_idx, self.cur_idx, draw_file)
                shutil.copy(cur_file, self.base_mask_with_img_file)
                self.setPixmap(cur_file)

                self.scaledImg = self.imgPixmap.scaled(self.scaledImg.width()+self.wheel_width,
                                                    self.scaledImg.height()+self.wheel_height)
                self.imgPainter = QPainter()                                           # 用于动态绘制图片
                self.imgFramePainter = QPainter()                                      # 用于动态绘制图片外线框
                self.imgPainter.begin(self)                                            # 无begin和end,则将一直循环更新
                self.imgPainter.drawPixmap(QPoint(self.offset_width, self.offset_height), self.scaledImg)          # 从图像文件提取Pixmap并显示在指定位置
                self.imgFramePainter.setPen(QColor(168, 34, 3))  # 不设置则为默认黑色   # 设置绘图颜色/大小/样式
                self.imgFramePainter.drawRect(10, 10, 480, 480)                        # 为图片绘外线狂(向外延展1)
                self.imgPainter.end()  

                # self._updateBaseSeg(pre_file, cur_file)

                self.num_back_step = 0
 
    '''重载一下鼠标移动事件'''
    def mouseMoveEvent(self,event):
        if not self.openPaint:
            pass
            if self.isLeftPressed:                                                 # 左键按下
                #print("鼠标左键按下，移动鼠标")  # 响应测试语句
                self.endMousePosition = event.pos() - self.preMousePosition        # 鼠标当前位置-先前位置=单次偏移量
                self.singleOffset = self.singleOffset + self.endMousePosition      # 更新偏移量
                self.preMousePosition = event.pos()                                # 更新当前鼠标在窗口上的位置，下次移动用
                self.repaint()                                                     # 重绘
        else:
            # 鼠标左键按下的同时移动鼠标
            if event.buttons() and Qt.LeftButton:
                self.endPoint = event.pos() - self.singleOffset #self.offset
                # 进行重新绘制
                self.update()