import numpy as np
import cv2

IMG_IN = 'GT2.BMP'
#IMG_IN = 'Wind Waker GC.bmp'

THRESHOULD = 0.52
SIGMA = 1.1
F_ALPHA = 0.7
F_BETA = 0.3

img = cv2.imread(IMG_IN)
img = img.astype (np.float32) / 255

gray = cv2.cvtColor (img, cv2.COLOR_BGR2GRAY)
gray= gray.reshape ((img.shape [0], img.shape [1], 1))

light_mask = np.where ((gray <= THRESHOULD), 0, img)
#cv2.imshow ('mascara', light_mask)

'''
# Versão 1: usando Filtro Gaussiano
sig = SIGMA
layer = light_mask.copy ()
sum = light_mask.copy() #perguntar aqui (a soma é outra imagem?)

while sig <= (SIGMA * 64):
    cv2.GaussianBlur (light_mask, (0,0), sigmaX=sig, dst=layer) #perguntar aqui -> estamos borrando a mascara, fazendo uma nova layer cada vez
    sum = sum + layer #aqui soma as layers
    sum = np.where ((sum > 1.0), 1.0, sum) 
    sig *= 2.0

result = img * F_ALPHA + sum * F_BETA

cv2.imshow ('teste', sum)
cv2.imshow ('final', result)
'''

# Versão 2: usando Box-Blur
box_sig = 1
layer = light_mask.copy ()
sum = light_mask.copy()

while box_sig <= 64:
    x = 0.0
    for x in range (box_sig * 3):
        cv2.boxFilter(layer, -1, (15,15), layer)
    sum = sum + layer #aqui soma as layers
    sum = np.where ((sum > 1.0), 1.0, sum) 
    box_sig *= 2

result = img * F_ALPHA + sum * F_BETA

cv2.imshow ('teste', sum)
cv2.imshow ('final', result)

print (sum.max())
print (sum.min())

print (result.max())
print (result.min())

cv2.waitKey ()
cv2.destroyAllWindows ()