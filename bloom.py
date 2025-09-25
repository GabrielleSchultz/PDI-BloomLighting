#===============================================================================
# Alunas: Gabrielle Schultz e Laura Pelisson
# Universidade Tecnológica Federal do Paraná
#===============================================================================

import numpy as np
import cv2

IMG_IN = 'GT2.BMP'
#IMG_IN = 'Wind Waker GC.bmp'

THRESHOULD = 0.52
SIGMA = 1.1
F_ALPHA = 0.9
F_BETA = 0.1

img = cv2.imread(IMG_IN)
img = img.astype (np.float32) / 255

gray = cv2.cvtColor (img, cv2.COLOR_BGR2GRAY)
gray= gray.reshape ((img.shape [0], img.shape [1], 1))

light_mask = np.where ((gray <= THRESHOULD), 0, img)
#cv2.imshow ('mascara', light_mask)

# Versão 1: usando Filtro Gaussiano
sig = SIGMA
layer = light_mask.copy ()
sum = np.empty ((layer.shape), np.float32) 

while sig <= (SIGMA * 64):
    cv2.GaussianBlur (light_mask, (0,0), sigmaX=sig, dst=layer) 
    sum = sum + layer 
    sig *= 2.0

sum = np.clip (sum, 0,1) 
result_Gaussian = (img * F_ALPHA) + (sum * F_BETA)

#cv2.imshow ('teste - Gaussian', sum)
cv2.imshow ('Bloom - Gaussian', result_Gaussian)

# Versão 2: usando Box-Blur
window = 4
layer = light_mask.copy ()
sum = np.empty ((layer.shape), np.float32) 

while window <= 128:
    x = 0
    for x in range (5):
        cv2.boxFilter(layer, -1, (window + 1, window + 1), layer)
    sum = sum + layer
    window *= 2

sum = np.clip (sum, 0,1) 
result_box = (img * F_ALPHA) + (sum * F_BETA)

#cv2.imshow ('teste - box', sum)
cv2.imshow ('Bloom - Box-Filter', result_box)

cv2.waitKey ()
cv2.destroyAllWindows ()