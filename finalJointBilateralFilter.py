
import numpy as np
import cv2
import math
from numpy import array

def distance(x, y, i, j):
    return math.sqrt((x-i)**2 + (y-j)**2)

def GaussianFunction(x, sigma):
	return (1 / (2 * math.pi * (sigma ** 2))**(1/2)) * np.exp(- (x ** 2) / (2 * (sigma ** 2)))

def applyBilateralFilter(source, processedImage, x, y, diameter, sigma_range, sigma_space):
    half = diameter//2
    newPixel = 0
    Wp = 0
    i = 0
    while i < diameter:
        j = 0
        while j < diameter:
            neighbourX = x - (half - i)
            neighbourY = y - (half - j)
            if neighbourX >= len(source):
                neighbourX -= len(source)
            if neighbourY >= len(source[0]):
                neighbourY -= len(source[0])
#            print("neighbour[",neighbourX,",",neighbourY, "]" ," to originial source[",x,",",y , "]")
            p=source[neighbourX][neighbourY]-source[x][y]
            gi=[]
#           split into 3 channels and apply Gaussian on all three
            for z in range (3):
                gi.append(GaussianFunction(p[z],sigma_range))
            gi=array(gi)
            gs = GaussianFunction(distance(neighbourX, neighbourY, x, y), sigma_space)
            w = gi * gs
            newPixel += source[neighbourX][neighbourY] * w
            Wp += w
            j += 1
        i += 1
    
    newPixel = newPixel / Wp
    temp=[]
    for k in range(3):
        temp.append(int(newPixel[k]))
    processedImage[x][y]=temp
    return processedImage


def bilateralFilter(source, filterDiameter, sigma_range, sigma_space):
    processedImage = np.zeros(source.shape,int)
    i = 0
    while i < len(source):
        j = 0
        while j < len(source[0]):
            applyBilateralFilter(source, processedImage, i, j, filterDiameter, sigma_range, sigma_space)
            j += 1
        i += 1
        print(100*i/len(source), "% completed")
    return processedImage
        
def applyJointBilateralFilter(source1, source2, processedImage, x, y, diameter, sigma_range, sigma_space):
    half = diameter//2
    newPixel = 0
    Wp = 0
    i = 0
    while i < diameter:
        j = 0
        while j < diameter:
            neighbourX = x - (half - i)
            neighbourY = y - (half - j)
            if neighbourX >= len(source2):
                neighbourX -= len(source2)
            if neighbourY >= len(source2[0]):
                neighbourY -= len(source2[0])
            p=source2[neighbourX][neighbourY]-source2[x][y]
            gi=[]
            for z in range (3):
                gi.append(GaussianFunction(p[z],sigma_range))
            gi=array(gi)
            gs = GaussianFunction(distance(neighbourX, neighbourY, x, y), sigma_space)
            w = gi * gs
            newPixel += source1[neighbourX][neighbourY] * w
            Wp += w
            j += 1
        i += 1
    newPixel = newPixel / Wp
    temp=[]
    for k in range(3):
        temp.append(int(newPixel[k]))
    processedImage[x][y]=temp
    return processedImage   

def jointBilateralFilter(source1,source2,filterDiameter,sigma_range,sigma_space):
    processedImage = np.zeros(source2.shape,int)
    i = 0
    while i < len(source2):
        j = 0
        while j < len(source2[0]):
            applyJointBilateralFilter(source1, source2, processedImage, i, j, filterDiameter, sigma_range, sigma_space)
            j += 1
        i += 1
        print(100*i/len(source2), "% completed")
    return processedImage



#implementing joint bilateral filter
nonflash_img = cv2.imread('./test3a.jpg', cv2.IMREAD_COLOR);
flash_img=cv2.imread('./test3b.jpg', cv2.IMREAD_COLOR);
if not nonflash_img is None and not nonflash_img is None:
    source1=nonflash_img
    source2=flash_img
    jointBilateralImage =jointBilateralFilter(source1,source2,7,5,60)
    cv2.imwrite("jointly_processedImage_own_temp(7,5,60).png", jointBilateralImage)
#    processedImage_OpenCV = cv2.bilateralFilter(source, 3, 60, 60)
#    cv2.imwrite("processedImage_OpenCV.png", processedImage_OpenCV)
else:
    print("No image file successfully loaded.");

"""
#implementing bilateral filter
img=cv2.imread('./test2.png', cv2.IMREAD_COLOR)
if not img is None:
    source=img
    bilateralImage=bilateralFilter(source,5,80,150)
    cv2.imwrite("girl_processed(5,80,150).png",bilateralImage)
else:
    print("No image file successfully loaded.");
"""