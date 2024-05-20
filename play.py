import sys
import numpy as np
import cv2

loaded = np.load(sys.argv[1])

images = loaded["images"]  # Camera FOV: 2.22 radians
gpsvs = loaded["gps"]
compassvs = loaded["compass"]

# Cria um canvas (uma imagem preta) de 800x800 pixels, com 3 canais de cor (RGB), para representar a posição e orientação do veículo.
canvas = np.zeros((800,800,3))

for i in range(images.shape[0]):
    # clear canvas
    canvas[:,:,:] = 0
    # read next 'frame' from file
    image, gps, compass = images[i], gpsvs[i], compassvs[i]

    # show image
    cv2.imshow('image', image)
    
    # draw position and orientation of the robot
    # Desenha um círculo vermelho no canvas representando a posição do veículo
    x_in_map =                    int(gps[0]*150)+canvas.shape[1]//2
    y_in_map = canvas.shape[0]//2-int(gps[1]*150)-canvas.shape[0]//4
    cv2.circle(canvas, (x_in_map, y_in_map), 12, (0,0,255), 2)

    # Calcula o ângulo de orientação do veículo a partir dos dados da bússola
    angle = np.arctan2(compass[1], compass[0])-np.pi/2

    # Desenha uma linha verde no canvas representando a orientação do veículo
    nx_in_map = x_in_map + int(18*np.cos(angle))
    ny_in_map = y_in_map + int(18*np.sin(angle))
    cv2.line(canvas, (x_in_map, y_in_map), (nx_in_map, ny_in_map), (0,255,0), 1)
    
    cv2.imshow('map', canvas)

    # exit when 'esc' or 'q' is pressed
    k = cv2.waitKey(10)
    if k == 113 or k == 27:
        break


