from math import floor
import math 
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
desayuno = Image.open('yes\Y8.jpg','r')

def convolucion(img, kernel):
    kernelint = floor(kernel.shape[0]/2)
    R, G, B = 0, 1, 2
    matrizExtendidaR = np.pad(img[:, :, R], kernelint, mode='constant')
    matrizExtendidaG = np.pad(img[:, :, G], kernelint, mode='constant')
    matrizExtendidaB = np.pad(img[:, :, B], kernelint, mode='constant')
    matrixRes = np.zeros((img.shape[0], img.shape[1], img.shape[2]))
    for i in range(kernelint, matrizExtendidaR.shape[0] - kernelint):
        for j in range(kernelint, matrizExtendidaR.shape[1] - kernelint):
            sumaR, sumaG, sumaB = 0, 0, 0
            for m in range(kernel.shape[0]):
                for n in range(kernel.shape[1]):
                    sumaR = sumaR + (kernel[m][n] * matrizExtendidaR[i-kernelint+m][j-kernelint+n])
                    sumaG = sumaG + (kernel[m][n] * matrizExtendidaG[i-kernelint+m][j-kernelint+n])
                    sumaB = sumaB + (kernel[m][n] * matrizExtendidaB[i-kernelint+m][j-kernelint+n])
            matrixRes[i-kernelint][j-kernelint][R] = round(sumaR)
            matrixRes[i-kernelint][j-kernelint][G] = round(sumaG)
            matrixRes[i-kernelint][j-kernelint][B] = round(sumaB)
    return matrixRes

def laplace(img, umbral, k):
    matrixRes=(convolucion(img, k))
    matrixRes = normalize(matrixRes)
    matrixRes = thresholding(matrixRes, umbral)
    return matrixRes.astype(np.uint8)
    
def normalize (img):
    for i in range(0,2):
        for row in range(img.shape[0]):
            for column in range(img.shape[1]):
                if(img[row][column][i] > 255):
                    img[row][column][i] = 255
                if (img[row][column][i]) < 0:
                    img[row][column][i] = 0
    return img


def thresholding(D, threshold): # 0 or 255 since the threshold
    R, G, B = 0, 1, 2
    width_D, height_D, RGBD = D.shape
    thresholdImg = np.copy(D)
    for row in range(width_D):
        for column in range(height_D):
            if(D[row][column][R]>threshold):
                thresholdImg[row][column][R] = 255
            else:
                thresholdImg[row][column][R] = 0
            if(D[row][column][G]>threshold):
                thresholdImg[row][column][G] = 255
            else:
                thresholdImg[row][column][G] = 0
            if(D[row][column][B]>threshold):
                thresholdImg[row][column][B] = 255
            else:
                thresholdImg[row][column][B] = 0
    
    return thresholdImg

def factorial(x):
    return 1 if x == 0 else x * factorial(x - 1)

def triangle(n):
    return [[factorial(i) / (factorial(j) * factorial(i - j)) for j in range(i + 1)] for i in range(n)]


def kernelGauss(n):
    h = triangle(n)[n-1]
    v = np.vstack(h)
    return h*v

def kerLaplace():
    return np.array([[-1, -1, -1],[-1, 8, -1],[-1, -1, -1]])

def kernelGaussX(n):
    kernelGaussX = kernelGauss(n)
    kernelGaussX[:,0] = kernelGaussX[:,0] * -1
    kernelGaussX[:,1] = kernelGaussX[:,1] * 0
    return kernelGaussX

def kernelGaussY(n):
    kernelGaussY = kernelGauss(n)
    kernelGaussY[0, :] = kernelGaussY[0, :] * -1
    kernelGaussY[1, :] = kernelGaussY[1, :] * 0
    return kernelGaussY

def sobel(img, n, umbral):
    matrixRes = np.zeros((img.shape[0], img.shape[1], img.shape[2]))
    matrixResX = (convolucion(img, kernelGaussX(n)))
    matrixResY = (convolucion(img, kernelGaussY(n)))
    
    for i in range(0,2):
        for x in range(img.shape[0]):
            for y in range(img.shape[1]):
                matrixRes[x][y][i] = math.sqrt(matrixResX[x][y][i]**2 + matrixResY[x][y][i] **2)
    #muestra de la capa X y la capa Y
    '''         
    matrixResY = thresholding(matrixResY, umbral-200)
    matrixResX = thresholding(matrixResX, umbral-200)
    fig, axs = plt.subplots(1,3,figsize=(20, 5))
    axs[0].imshow(matrixResX.astype(np.uint8))
    axs[0].set_title("Gauss X")
    axs[1].imshow(matrixResY.astype(np.uint8))
    axs[1].set_title("Gauss Y")
    axs[2].imshow(matrixRes.astype(np.uint8))
    axs[2].set_title("Sobel bordering")
    '''
    matrixRes = normalize(matrixRes)
    matrixRes = thresholding(matrixRes, umbral)
    return matrixRes.astype(np.uint8)

def perfilado(img,n,a, umbral):
    kernelPerf =np.array([[0, -a, 0], [-a, (n*a)+1, -a], [0, -a, 0]])  
    matrixRes = convolucion(img, kernelPerf)
    matrixRes = normalize(matrixRes)
    matrixRes = thresholding(matrixRes, umbral)
    return matrixRes.astype(np.uint8)

laplaceImg = laplace(np.array(desayuno), 70, kerLaplace())
sobelImg = sobel(np.array(desayuno),3, 250)
perfiladoImg = perfilado(np.array(desayuno),4, 1, 125)

fig, axs = plt.subplots(1,4,figsize=(20, 5))
axs[0].imshow(np.array(desayuno))
axs[0].set_title("Original Image")
axs[1].imshow(sobelImg)
axs[1].set_title("Sobel bordering")
axs[2].imshow(laplaceImg)
axs[2].set_title("Laplace bordering")
axs[3].imshow(perfiladoImg)
axs[3].set_title("Perfilado")
plt.show()