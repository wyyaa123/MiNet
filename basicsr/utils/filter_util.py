import numpy as np
import cv2 as cv
from torch.nn import functional as F

def distance(shape):		# 计算每个像素到中心原点的距离
    m, n = shape
    u = np.arange(n)
    v = np.arange(m)
    u, v = np.meshgrid(u, v)
    return np.sqrt((u - n//2)**2 + (v - m//2)**2) + 0.0001
    
def ideal_filter(shape, d0):
    d = distance(shape)
    mask = d > d0
    return mask.astype(int)

def butterworth_filter(shape, d0, order=1):
    d = distance(shape)
    mask = 1 / (1 + (d0 / d)**(2 * order))
    return mask

def exponential_filter(shape, d0, order=1):
    d = distance(shape)
    mask = np.exp(-(d0 / d)**order)
    return mask

if __name__ == "__main__":
    img = cv.imread('images/1.png', cv.IMREAD_GRAYSCALE)
    fft_shift = np.fft.fftshift(np.fft.fft2(img))	# 变换后将零频分量移到频谱中心
    fft_img = np.log(np.abs(fft_shift))				# 可视化

    mask = butterworth_filter(img.shape, 20)

    ifft_shift = np.fft.ifftshift(fft_shift * mask)
    ifft_img = np.abs(np.fft.ifft2(ifft_shift))

    cv.imshow("ft_img", fft_img)
    cv.imshow("filter_img", fft_img)
    cv.waitKey(0)
