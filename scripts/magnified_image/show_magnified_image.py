import cv2
import numpy as np
import matplotlib.pyplot as plt

# 定义全局变量
selected_roi = None
selecting = False

# 鼠标事件回调函数
def mouse_callback(event, x, y):
    global selected_roi, selecting

    H, W, _ = image.shape

    if event == cv2.EVENT_LBUTTONDOWN:
        # 鼠标左键按下，开始选择区域
        selected_roi = (x, y)
        selecting = True

    elif event == cv2.EVENT_LBUTTONUP:
        # 鼠标左键释放，结束选择区域
        selecting = False
        roi_width = x - selected_roi[0]
        roi_height = y - selected_roi[1]

        # 获取选择的区域
        selected_roi = (selected_roi[0], selected_roi[1], roi_width, roi_height)

        # 进行局部放大
        enlarged_roi = cv2.resize(image[selected_roi[1]:selected_roi[1] + selected_roi[3],
                                       selected_roi[0]:selected_roi[0] + selected_roi[2]], 
                                   (selected_roi[2] * 2, selected_roi[3] * 2), interpolation=cv2.INTER_LINEAR)
        
        roi_h, roi_w, _, = enlarged_roi.shape

        # 在原图上绘制矩形框
        cv2.rectangle(image, (selected_roi[0], selected_roi[1]), 
                      (selected_roi[0] + selected_roi[2], selected_roi[1] + selected_roi[3]), 
                      (0, 255, 0), 2)

        # 合并原始图像和局部放大的图像
        image[H - roi_h: H, W - roi_w: W] = enlarged_roi

        # 显示结果
        cv2.imshow('Result', image)
        cv2.imwrite("roi_enlarge.png", image)

# 读取图片
image = cv2.imread('datasets/GoPro/test/input/GOPR0384_11_00-000001.png')

# 创建窗口并设置鼠标回调函数
cv2.namedWindow('Result')
cv2.setMouseCallback('Result', mouse_callback)

while True:
    cv2.imshow('Result', image)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # 按下ESC键退出
        break

cv2.destroyAllWindows()
