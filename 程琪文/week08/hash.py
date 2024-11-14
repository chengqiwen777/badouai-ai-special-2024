import cv2
import numpy as np


def aHash(img):
    gray = cv2.resize(img, (8, 8), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)

    s_pixel = 0
    for i in range(8):
        for j in range(8):
            s_pixel += gray[i][j]
    average_pixel = s_pixel / 64
    print('type(average_pixel)', type(average_pixel))
    print('average_pixel: ', average_pixel)
    s_hash = ''
    for i in range(8):
        for j in range(8):
            if gray[i][j] > average_pixel:
                s_hash += '1'
            else:
                s_hash += '0'
    return s_hash


def dHash(img):
    gray = cv2.resize(img, (9, 8), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    s_hash = ''
    for i in range(8):
        for j in range(8):
            if gray[i][j] > gray[i][j + 1]:
                s_hash += '1'
            else:
                s_hash += '0'
    return s_hash


def cmpHash(hash_str1, hash_str2):
    n = 0
    if len(hash_str1) != len(hash_str2):
        return -1

    for i in range(len(hash_str1)):
        if hash_str1[i] != hash_str2[i]:
            n += 1
    return 1 - n / len(hash_str1)


if __name__ == '__main__':
    img1 = cv2.imread('iphone1.png')
    img2 = cv2.imread('iphone2.png')

    img1_aHash = aHash(img1)
    img2_aHash = aHash(img2)
    print('img1_aHash: ', img1_aHash)
    print('img2_aHash: ', img2_aHash)
    cmp_aHash = cmpHash(img1_aHash, img2_aHash)
    print('均值哈希比较：', cmp_aHash)

    img1_dHash = dHash(img1)
    img2_dHash = dHash(img2)
    print('img1_dHash: ', img1_dHash)
    print('img2_dHash: ', img2_dHash)
    cmp_dHash = cmpHash(img1_dHash, img2_dHash)
    print('差值哈希比较：', cmp_dHash)

