# 所有前置内容
import cv2
import numpy as np
import matplotlib.pyplot as plt
import threading

xScale = 1.0
yScale = 1.0


def setScale(xScale_, yScale_):
    global xScale, yScale
    xScale = xScale_
    yScale = yScale_


def drawAllPic(titles,
               imgs,
               fsize=4,
               sizey=4,
               outputName="./result.jpg",
               showHist=True,
               imLim_L=[]):
    global xScale, yScale
    # 绘制
    tlen = len(titles)
    sizex = int(np.ceil((tlen / sizey)))
    show_v = 1
    if showHist:
        sizex *= 2
        show_v = 2
    if len(imLim_L) == 0:
        imLim_L = [(0, 255)] * tlen
    fig = plt.figure(figsize=(fsize * sizey * xScale, fsize * sizex * yScale))
    cmap_use = "gray"
    print(tlen, sizex, sizey, 2 * np.ceil((tlen / sizey)))
    for i in range(tlen):
        # print(i)
        plt.subplot(sizex, sizey,
                    int(i / sizey) * show_v * sizey + i % sizey + 1)
        print(
            i,
            int(i / sizey) * 2 * sizey + i % sizey + 1,
            int(i / sizey) * 2 * sizey + i % sizey + sizey + 1,
        )
        plt.imshow(imgs[i], cmap=cmap_use, clim=imLim_L[i])
        plt.axis('off')
        plt.title(titles[i])

        if showHist:
            h, w = imgs[i].shape[:2]
            im = np.uint8(imgs[i]).reshape([
                h * w,
            ])
            barw = int(256 / 3)
            plt.subplot(sizex, sizey,
                        int(i / sizey) * 2 * sizey + i % sizey + sizey + 1)
            histogram, _, _ = plt.hist(im, barw, histtype="bar")
            plt.yticks([])  # 去掉y轴
            plt.title(titles[i] + "_hist")
            plt.xlim(imLim_L[i][0], imLim_L[i][1])
        # plt.ylim(0, 300)
    plt.tight_layout()
    plt.savefig(outputName)
    plt.show()


def nonlinearFilter(pic_, func, filterSize_=(3, 3)):
    hfilterSize = (int(filterSize_[0] >> 1), int(filterSize_[1] >> 1))
    textp = cv2.copyMakeBorder(
        pic_,
        hfilterSize[1],
        hfilterSize[1],
        hfilterSize[0],
        hfilterSize[0],
        cv2.BORDER_DEFAULT,
    )
    x, y = pic_.shape[:2]
    x, y = x - 1, y - 1
    result = [[[
        func(textp[i:i + filterSize_[0], j:j + filterSize_[1]].reshape(-1, ))
    ] for j in range(y)] for i in range(x)]
    return np.array(result)


def g_blur(src, ksize=(3, 3)):
    out = np.zeros(src.shape)
    pin = np.array(src, dtype=float)
    hk0 = np.int(ksize[0] / 2)
    m, n = src.shape[:2]
    hm = int(m / 2)
    k0, k1 = ksize
    iK = 1 / (k0 * k1)
    x = np.array([[i] * k1 for i in range(-hk0, k0 - hk0)])
    y = np.transpose(x).reshape(-1)
    x = x.reshape(-1)
    for i in range(m):
        xx = x + i
        xx[xx < 0] = 0
        xx[xx >= m] = m - 1
        for j in range(n):
            yy = y + j
            yy[yy < 0] = 0
            yy[yy >= n] = n - 1
            out[i, j] = np.prod(pin[xx, yy])**iK
    out[out > 255] = 255
    out[out < 0] = 0
    # cv2.imshow("asdf", np.uint8(noise))
    # cv2.waitKey(0)
    return np.uint8(out)


def ih_blur(src, ksize=(3, 3), Q=0.5):
    out = np.zeros(src.shape)
    pin = np.array(src, dtype=float)
    hk0 = np.int(ksize[0] / 2)
    m, n = src.shape[:2]
    hm = int(m / 2)
    k0, k1 = ksize
    iK = k0 * k1
    x = np.array([[i] * k1 for i in range(-hk0, k0 - hk0)])
    y = np.transpose(x).reshape(-1)
    x = x.reshape(-1)
    for i in range(m):
        xx = x + i
        xx[xx < 0] = 0
        xx[xx >= m] = m - 1
        for j in range(n):
            yy = y + j
            yy[yy < 0] = 0
            yy[yy >= n] = n - 1
            a1 = np.sum(np.power(pin[xx, yy], Q + 1))
            a2 = np.sum(np.power(pin[xx, yy], Q))
            if a2 == 0:
                a2 = 1e-32
            out[i, j] = a1 / a2
    out[out > 255] = 255
    out[out < 0] = 0
    # cv2.imshow("asdf", np.uint8(noise))
    # cv2.waitKey(0)
    return np.uint8(out)


# 我想用std表示方差，但不是所有函数都需要它，所以有些没用到。
# mean是均值。
def gauss_n(shape, mean, std):
    return np.random.normal(mean, std**0.5, shape)


def gamma_n(shape, mean, std):
    if np.abs(mean) < 1e-7:
        mean = 1e-7
    if np.abs(std) < 1e-7:
        std = 1e-7
    return np.random.gamma(mean**2 / std, std / mean, shape)


def rayleigh_n(shape, mean, std):
    return np.random.rayleigh(mean / np.sqrt(np.pi / 2), shape)


def logistic_n(shape, mean, std=0):
    if np.abs(mean) < 1e-7:
        mean = 1e-7
    return np.random.logistic(std, mean, shape)


def exponential_n(shape, mean, std):
    if np.abs(mean) < 1e-7:
        mean = 1e-7
    return np.random.exponential(mean, shape)


def uniform_n(shape, mean, std):
    return np.random.uniform(-mean, mean, shape)


# 椒盐噪声比较特殊，mean我表示概率，输入范围是[0,1]
def choice_n(shape, mean, std):
    if mean > 1:
        mean = 1
    noise = np.zeros(shape)
    for i in range(shape[0]):
        for j in range(shape[1]):
            r = np.random.random()
            if r < mean:
                noise[i, j] = -255
                continue
            if r > 1 - std:
                noise[i, j] = 255
    return noise


def add_noise(image, funName, mean, std=-1):
    funmap = {
        "gauss": gauss_n,
        "gamma": gamma_n,
        "rayleigh": rayleigh_n,
        "logistic": logistic_n,
        "exponential": exponential_n,
        "uniform": uniform_n,
        "choice": choice_n,
    }
    func = funmap.get(funName)
    if func == None:
        print("No function name call: " + funName)
        return image, np.zeros(image.shape)
    noise = func(image.shape, mean, std)
    out = image + noise
    out[out > 255] = 255
    out[out < 0] = 0
    return np.uint8(out), noise
