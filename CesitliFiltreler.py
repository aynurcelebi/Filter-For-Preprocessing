# -*- coding: utf-8 -*-
"""
Created on Sun Jan  2 17:55:07 2022

@author: MYPC
"""


import numpy as np
import cv2
from matplotlib import pyplot as plt
from PIL import Image, ImageFilter



#-------------------------  Mean Filtresi  ----------------------------------------
#image = cv2.imread('C:/Users/MYPC/Desktop/x/img.png') # reads the image
#image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV) # convert to HSV
#figure_size = 9 # the dimension of the x and y axis of the kernal.
#new_image = cv2.blur(image,(figure_size, figure_size))
#plt.figure(figsize=(11,6))
#plt.subplot(121), plt.imshow(cv2.cvtColor(image, cv2.COLOR_HSV2RGB)),plt.title('Orijinal')
#plt.xticks([]), plt.yticks([])
#plt.subplot(122), plt.imshow(cv2.cvtColor(new_image, cv2.COLOR_HSV2RGB)),plt.title('Mean filtresi')
#plt.xticks([]), plt.yticks([])
#plt.show()

#-------------------------  Mean Filtresi + Gri Tonlama  ----------------------------------------

#image2 = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
#image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
#figure_size = 9
#new_image = cv2.blur(image2,(figure_size, figure_size))
#plt.figure(figsize=(11,6))
#plt.subplot(121), plt.imshow(image2, cmap='gray'),plt.title('Original')
#plt.xticks([]), plt.yticks([])
#plt.subplot(122), plt.imshow(new_image, cmap='gray'),plt.title('Mean filter')
#plt.xticks([]), plt.yticks([])
#plt.show()

#-------------------------  Gaussian Filtresi  ----------------------------------------
#
#new_image = cv2.GaussianBlur(image, (figure_size, figure_size),0)
#plt.figure(figsize=(11,6))
#plt.subplot(121), plt.imshow(cv2.cvtColor(image, cv2.COLOR_HSV2RGB)),plt.title('Orijinal')
#plt.xticks([]), plt.yticks([])
#plt.subplot(122), plt.imshow(cv2.cvtColor(new_image, cv2.COLOR_HSV2RGB)),plt.title('Gaussian Filtresi')
#plt.xticks([]), plt.yticks([])
#plt.show()


#-------------------------  Gaussian Filtresi + Gri Tonlama ----------------------------------------

#new_image_gauss = cv2.GaussianBlur(image2, (figure_size, figure_size),0)
#plt.figure(figsize=(11,6))
#plt.subplot(121), plt.imshow(image2, cmap='gray'),plt.title('Original')
#plt.xticks([]), plt.yticks([])
#plt.subplot(122), plt.imshow(new_image_gauss, cmap='gray'),plt.title('Gaussian Filter')
#plt.xticks([]), plt.yticks([])
#plt.show()

#-------------------------  Median  Filtresi  ----------------------------------------
#
#new_image = cv2.medianBlur(image, figure_size)
#plt.figure(figsize=(11,6))
#plt.subplot(121), plt.imshow(cv2.cvtColor(image, cv2.COLOR_HSV2RGB)),plt.title('Orijinal')
#plt.xticks([]), plt.yticks([])
#plt.subplot(122), plt.imshow(cv2.cvtColor(new_image, cv2.COLOR_HSV2RGB)),plt.title('Median Filtresi')
#plt.xticks([]), plt.yticks([])
#plt.show()

##-------------------------  Median Filtresi + Gri Tonlama ----------------------------------------
#
#new_image = cv2.medianBlur(image2, figure_size)
#plt.figure(figsize=(11,6))
#plt.subplot(121), plt.imshow(image2, cmap='gray'),plt.title('Original')
#plt.xticks([]), plt.yticks([])
#plt.subplot(122), plt.imshow(new_image, cmap='gray'),plt.title('Median Filter')
#plt.xticks([]), plt.yticks([])
#plt.show()


#-------------------------  Laplacian Filtresi  ----------------------------------------

#
#new_image = cv2.Laplacian(image,cv2.CV_64F)
#plt.figure(figsize=(11,6))
#plt.subplot(131), plt.imshow(image, cmap='gray'),plt.title('Orijinal')
#plt.xticks([]), plt.yticks([])
#plt.subplot(132), plt.imshow(new_image, cmap='gray'),plt.title('Laplacian Filtresi')
#plt.xticks([]), plt.yticks([])
#plt.subplot(133), plt.imshow(image + new_image, cmap='gray'),plt.title('Sonuç')
#plt.xticks([]), plt.yticks([])
#plt.show()

#-------------------------  Keskinliği Azaltma Filtresi  ----------------------------------------
#
#image = Image.fromarray(image.astype('uint8'))
#new_image = image.filter(ImageFilter.UnsharpMask(radius=2, percent=150))
#plt.subplot(121),plt.imshow(image, cmap = 'gray')
#plt.title('Input Image'), plt.xticks([]), plt.yticks([])
#plt.subplot(122),plt.imshow(new_image, cmap = 'gray')
#plt.title('Keskinlik azaltan filtre'), plt.xticks([]), plt.yticks([])
#plt.show()


#-------------------------  Robinson Filtresi  ----------------------------------------
#

#
#kernel_x = np.array(
#    [
#        [-1, 0, 1],
#        [-1, 0, 1],
#        [-1, 0, 1]
#    ]
#)
#
#im = cv2.imread('C:/Users/MYPC/Desktop/x/img.png',cv2.IMREAD_GRAYSCALE)
#convolved_x = cv2.filter2D(im, -1, kernel_x)
#convolved_y = cv2.filter2D(im, -1, kernel_x.T)
#
#plt.subplot(131)
#plt.title("Gorsel")
#plt.imshow(im, cmap="gray")
#plt.subplot(132)
#plt.title("X Ekseninde Gradyan")
#plt.imshow(convolved_x, cmap="gray")
#plt.subplot(133)
#plt.title("Y Ekseninde Gradyan")
#plt.imshow(convolved_y, cmap="gray")
#
#plt.show()
#
##-------------------------  Sobel Filtresi  ----------------------------------------
#
#kernel_x = np.array(
#    [
#        [-1, 0, 1],
#        [-2, 0, 2],
#        [-1, 0, 1]
#    ]
#)
#
#
#im = cv2.imread('C:/Users/MYPC/Desktop/x/img.png',cv2.IMREAD_GRAYSCALE)
#convolved_x = cv2.filter2D(im, -1, kernel_x)
#convolved_y = cv2.filter2D(im, -1, kernel_x.T)
#
#plt.subplot(131)
#plt.title("Gorsel")
#plt.imshow(im, cmap="gray")
#plt.subplot(132)
#plt.title("X Ekseninde Gradyan")
#plt.imshow(convolved_x, cmap="gray")
#plt.subplot(133)
#plt.title("Y Ekseninde Gradyan")
#plt.imshow(convolved_y, cmap="gray")
#
#plt.show()


#-------------------------  Kirsch Filtresi  ----------------------------------------
#
#kernel_x = np.array(
#    [
#        [-3, 0, 3],
#        [-5, 0, 5],
#        [-3, 0, 3]
#    ]
#)
#
#image = cv2.imread('C:/Users/MYPC/Desktop/x/img.png') # reads the image
#convolved_x = cv2.filter2D(im, -1, kernel_x)
#convolved_y = cv2.filter2D(im, -1, kernel_x.T)
#
#plt.subplot(131)
#plt.title("Gorsel")
#plt.imshow(im, cmap="gray")
#plt.subplot(132)
#plt.title("X Ekseninde Gradyan")
#plt.imshow(convolved_x, cmap="gray")
#plt.subplot(133)
#plt.title("Y Ekseninde Gradyan")
#plt.imshow(convolved_y, cmap="gray")
#
#plt.show()


#-------------------------  Canny Filtresi  ----------------------------------------

#edges = cv2.Canny(image,50,150)
#
#plt.subplot(121),plt.imshow(image,cmap = 'gray')
#plt.title('Orijinal'), plt.xticks([]), plt.yticks([])
#plt.subplot(122),plt.imshow(edges,cmap = 'gray')
#plt.title('Canny'), plt.xticks([]), plt.yticks([])
#
#plt.show()

#-------------------------  Prewitt Filtresi  ----------------------------------------


#im = cv2.imread('C:/Users/MYPC/Desktop/x/img.png',cv2.IMREAD_GRAYSCALE)
#kernelx = np.array([[1,1,1],[0,0,0],[-1,-1,-1]])
#kernely = np.array([[-1,0,1],[-1,0,1],[-1,0,1]])
#img_prewittx = cv2.filter2D(images, -1, kernelx)
#img_prewitty = cv2.filter2D(images, -1, kernely)

#cv2.imshow("Prewitt X", img_prewittx)
#cv2.imshow("Prewitt Y", img_prewitty)
#cv2.imshow("Prewitt", img_prewittx + img_prewitty)



