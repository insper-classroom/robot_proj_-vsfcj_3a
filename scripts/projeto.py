#! /usr/bin/env python3
# -*- coding:utf-8 -*-

from __future__ import print_function, division


import rospy 
import statsmodels.api as sm
import numpy as np
import cv2
import time
import math
from geometry_msgs.msg import Twist, Vector3
from sensor_msgs.msg import LaserScan
from sensor_msgs.msg import Image, CompressedImage, LaserScan
from cv_bridge import CvBridge, CvBridgeError
import cv2.aruco as aruco
import sys

# roslaunch my_simulation forca.launch


# ARUCO
aruco_dict  = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
parameters  = aruco.DetectorParameters_create()
parameters.minDistanceToBorder = 0
parameters.adaptiveThreshWinSizeMax = 1000





#Variaveis

font = cv2.FONT_HERSHEY_PLAIN

id = None
avarage_x = 0
DISTANCEOK = False 
FIND150 = False 
FIND50 = False 
contador = 0

blueArea = 0
greenArea=0
redArea=0
centro_blue = 0
centro_green = 0
centro_red = 0

ranges = None
minv = 0
maxv = 10
image_center = [0,0]

bridge = CvBridge()



#funções do projeto

#corta a imagem
def image_cut(mask):
    avarage_x = int(mask.shape[1]/2)
    
    img_left = mask[0:mask.shape[0], 0: avarage_x + 50]
    img_right = mask[0:mask.shape[0], avarage_x+100: mask.shape[1]]
    
    return img_left, img_right

#linear adjustment

def ajuste_linear_x_fy(mask):
    """
        y = coef_angular*x + coef_linear
    """ 
    pontos = np.where(mask==255)
    if len(pontos[0]) != 0:
        ximg = pontos[1]
        yimg = pontos[0]
        yimg_c = sm.add_constant(yimg)
        model = sm.OLS(ximg,yimg_c)
        results = model.fit()
        coef_angular = results.params[1] 
        coef_linear =  results.params[0] 
        return coef_angular, coef_linear
    return None, None


def ajuste_linear(mask):
    """Recebe uma imagem já limiarizada e faz um ajuste linear
        retorna coeficientes linear e angular da reta
        e equação é da forma
        y = coef_angular*x + coef_linear
    """ 
    pontos = np.where(mask==255)
    ximg = pontos[1]
    yimg = pontos[0]
    yimg_c = sm.add_constant(yimg)
    ximg_c = sm.add_constant(ximg)
    model = sm.OLS(yimg,ximg_c)
    results = model.fit()
    coef_angular = results.params[1] # Pegamos o beta 1
    coef_linear =  results.params[0] # Pegamso o beta 0
    return coef_angular, coef_linear


def ajuste_linear_grafico_x_fy(mask):

    coef_angular, coef_linear = ajuste_linear_x_fy(mask)
    
    if coef_angular != None:
        pontos = np.where(mask==255) 
        ximg = pontos[1]
        yimg = pontos[0]
        y_bounds = np.array([min(yimg), max(yimg)])
        x_bounds = coef_angular*y_bounds + coef_linear
        x_int = x_bounds.astype(dtype=np.int64)
        y_int = y_bounds.astype(dtype=np.int64)
        mask_rgb =  cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
        x_f = x_int[0]
        y_f = y_int[0]
        cv2.line(mask_rgb, (x_int[0], y_int[0]), (x_int[1], y_int[1]), color=(0,0,255), thickness=11);    
        return mask_rgb, x_f, y_f 
    return None, None, None
      

#Tres próximas para segmentar a cor do creeper 

def segmenta_red(bgr):
    
    
    bgr2 = bgr.copy()
    hsv = cv2.cvtColor(bgr2, cv2.COLOR_BGR2HSV)
    
   
    cor_hsv1  = np.array([ 0, 200, 190], dtype=np.uint8)
    cor_hsv2 = np.array([ 15, 255, 255], dtype=np.uint8)
    
    yellow_mask = cv2.inRange(hsv, cor_hsv1, cor_hsv2)
    
    yellow_blur = cv2.blur(yellow_mask, (1,1))

    mask_morpho = morpho_limpa(yellow_blur)
    
    return mask_morpho

def segmenta_green(bgr):

    bgr2 = bgr.copy()
    hsv = cv2.cvtColor(bgr2, cv2.COLOR_BGR2HSV)
    
    
    cor_hsv1  = np.array([ 40, 50, 50], dtype=np.uint8)
    cor_hsv2 = np.array([ 70, 255, 255], dtype=np.uint8)
  
    yellow_mask = cv2.inRange(hsv, cor_hsv1, cor_hsv2)
    
    yellow_blur = cv2.blur(yellow_mask, (1,1))

    mask_morpho = morpho_limpa(yellow_blur)
    
    return mask_morpho

def segmenta_blue(bgr):

    
    bgr2 = bgr.copy()
    hsv = cv2.cvtColor(bgr2, cv2.COLOR_BGR2HSV)
    
    
    cor_hsv1  = np.array([ 0, 50, 50], dtype=np.uint8)
    cor_hsv2 = np.array([ 15, 170, 255], dtype=np.uint8)

    yellow_mask = cv2.inRange(hsv, cor_hsv1, cor_hsv2)
    
    yellow_blur = cv2.blur(yellow_mask, (1,1))

    mask_morpho = morpho_limpa(yellow_blur)
    
    return mask_morpho


#Segmenta linha amarelo para seguir pista

def segmenta_yellow(bgr):

    bgr2 = bgr.copy()
    hsv = cv2.cvtColor(bgr2, cv2.COLOR_BGR2HSV)
    
    
    cor_hsv1  = np.array([ 27, 200, 249], dtype=np.uint8)
    cor_hsv2 = np.array([ 32, 230, 255], dtype=np.uint8)
    
    mascara_amarela = cv2.inRange(hsv, cor_hsv1, cor_hsv2)
    
    mascara_amarela_blur = cv2.blur(mascara_amarela, (1,1))

    mask_morpho = morpho_limpa(mascara_amarela_blur)
    
    
    return mask_morpho

#Trabalho com contornos.
def find_countours(mask):
    
    contornos, arvore = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
   
    return contornos

def big_countour(contornos):
    bigger_count = None

    area = 0
    for i in contornos:
        areaC = cv2.contourArea(i)
        if areaC > area:
            bigger_count = i
            area = areaC
    
    return bigger_count

def area_countours(contornos):
    area = 0

    for i in contornos:
        areaC = cv2.contourArea(i)
    
        area += areaC

    return area

#Centro do contorno por moments

def countours_center(img,contornos):
    
    lista_x = []
    lista_y = []
    
    for i in contornos:
        M = cv2.moments(i)
        if M["m00"] == 0: 
            M["m00"] = 1
        
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        crosshair(img, (cX,cY), 10, (255,255,0))
        
        if (cX and cY) != 0:
        
            lista_x.append(cX)
            lista_y.append(cY)
        
    return img, lista_x, lista_y

#MORPHO igual da aula

def morpho_limpa(mask):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    mask = cv2.morphologyEx( mask, cv2.MORPH_OPEN, kernel )
    mask = cv2.morphologyEx( mask, cv2.MORPH_CLOSE, kernel )    
    return mask



def center_of_contour(contorno):
    """ Retorna uma tupla (cx, cy) que desenha o centro do contorno"""
    M = cv2.moments(contorno)
    if M["m00"] == 0:
        M["m00"] = 1
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    return (int(cX), int(cY))

#Crosshair

def crosshair(img, point, size, color):
    
    x,y = point
    cv2.line(img,(x - size,y),(x + size,y),color,2)
    cv2.line(img,(x,y - size),(x, y + size),color,2)


def center_regression(img, X,Y):
 
    
    x = np.array(X)
    y = np.array(Y)
    
    
    y_c = sm.add_constant(y)
    
    model = sm.OLS(x,y_c)
    
    results = model.fit()
    
    coef_angular = results.params[1]
    coef_linear = results.params[0]
    
    y_min = 0
    y_max = img.shape[0]
    
    x_min = int(coef_angular*y_min + coef_linear)
    x_max = int(coef_angular*y_max + coef_linear)

    cv2.line(img, (x_min, y_min), (x_max, y_max), (255,255,0), thickness=3);       
    
    return img, [(x_min, y_min),(x_max,y_max)]

### fim das funções de calculo.###

### Definição da maquina estados que ira variar entre as cores de creeper escolhido ou estado de andar pela pista
### Caso a variavel abaixo estiver vazia ele somente ira vagar pela pista caso contrario somente setar a missão a ser feita

MISSION = 'RED'

def scaneou(dado):
    global ranges
    global minv
    global maxv
    global DISTANCEOK 
    global leituras
    #dados.range para distancia...
    leituras = np.array(dado.ranges).round(decimals=2)
    ranges = np.array(dado.ranges).round(decimals=2)
    minv = dado.range_min 
    maxv = dado.range_max
    

    if leituras[0] < 0.75:
        DISTANCEOK = True
        
    
    else:
        DISTANCEOK = False
 


def roda_todo_frame(imagem):

    try:
        global blueArea 
        global greenArea 
        global redArea

        global blue_center
        global green_center
        global red_center
        global image_center
        global avarage_x
        global FIND150
        global FIND50
        
        cv_image = bridge.compressed_imgmsg_to_cv2(imagem, "bgr8")
        hsv = cv2.cvtColor(cv_image.copy(), cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

    
        if ids is not None:
            for i_d in ids:
                if i_d == [150]:
                    FIND150 = True
                else:
                    FIND150 = False

                if i_d == [50]:
                    FIND50 = True
                else:
                    FIND50 = False

        
        #Aplicando todas as funções feitas para utilização

        mask_blue = segmenta_blue(hsv.copy())
        countours_blue = find_countours(mask_blue)
        bigger_countour_blue = big_countour(countours_blue)
        blue_center = center_of_contour(bigger_countour_blue)
        
        
        mask_red = segmenta_red(cv_image.copy())
        countours_red = find_countours(mask_red)
        bigger_countour_red = big_countour(countours_red)
        red_center = center_of_contour(bigger_countour_red)

        mask_green = segmenta_green(cv_image.copy())
        countours_green = find_countours(mask_green)
        bigger_countour_green = big_countour(countours_green)
        green_center = center_of_contour(bigger_countour_green)


                #Area do creeper
        if bigger_countour_blue is not None:
            blueArea = cv2.contourArea(bigger_countour_blue)
        else:
            blueArea = 0
        
        if bigger_countour_green is not None:
            greenArea = cv2.contourArea(bigger_countour_green)
        else:
            greenArea = 0
        if bigger_countour_red is not None:
            redArea = cv2.contourArea(bigger_countour_red)
        else:
            redArea = 0

        

        #Agora puxando as funções necessarias.

        image_center = (int(cv_image.shape[1]/2), int(cv_image.shape[0]/2))
        img_cont = cv_image.copy()

        mask = segmenta_yellow(hsv)

        img_left, img_right = image_cut(mask)

        mask_regression, x_f, y_f = ajuste_linear_grafico_x_fy(img_left)
        avarage_x = x_f
        contornos = find_countours(mask)
        areaTotal = area_countours(contornos)
        contornos_filtrados = area_countours(contornos)

        mask_regression, x_f, y_f = ajuste_linear_grafico_x_fy(img_left)
        x_medio = x_f

        #Desenha os contornos
        cv2.drawContours(img_cont, contornos, -1, [255, 255, 255], 3)
        cv2.imshow("Contornos", img_cont)
        cv2.imshow("Mask",mask)
        
        if mask_regression is not None:

            cv2.circle(mask_regression, (x_f, y_f), 10, (255,0, 0), -1)
            cv2.circle(img_cont, (x_f,y_f), 10, (0,255, 0), -1)
            

        cv2.waitKey(1)
    except CvBridgeError as e:
        print('ex', e)

if __name__=="__main__":

    rospy.init_node("q3")

    topico_imagem = "/camera/image/compressed"
    velocidade_saida = rospy.Publisher("/cmd_vel", Twist, queue_size = 3 )
    recebe_scan = rospy.Subscriber("/scan", LaserScan, scaneou)
    recebedor = rospy.Subscriber(topico_imagem, CompressedImage, roda_todo_frame, queue_size=4, buff_size = 2**24)

    while not rospy.is_shutdown():
        #Vel de inicio para ambientação
        vel = Twist(Vector3(0,0,0), Vector3(0,0,-0.1))
        
        if MISSION == 'GREEN' and (greenArea >= 1000):
            if (green_center != 0) and (len(image_center) != 0):
                print("Estagio: Centralizando creeper verde")   
                if (image_center[0] <= green_center[0] + 15) and (image_center[0] >= green_center[0] -15):

                    print("Centralizei")
                    print(leituras[0])
                    if leituras[0]<0.4:
                        print("To close heading back")
                        MISSION=''

                    vel = Twist(Vector3(0.4,0,0), Vector3(0,0,0))
            
                else:
                   
                    if (image_center[0] > green_center[0]):
                        vel = Twist(Vector3(0.2,0,0), Vector3(0,0,0.2))
                        
                    if (image_center[0] < green_center[0]):
                       vel = Twist(Vector3(0.2,0,0), Vector3(0,0,-0.2))
                       
       
        elif MISSION == 'RED' and (redArea>= 1000):
            if (red_center[0] != 0) and (len(image_center) != 0):
                print("Estagio: Centralizando creeper vermelho")
                if (image_center[0] <= red_center[0] + 15) and (image_center[0] >= red_center[0] -15):

                    print("Centralized")
                    print("Distância do creeper",leituras[0])
                    if leituras[0]<0.4:
                        print("To close heading back")
                        MISSION=''
                    vel = Twist(Vector3(0.4,0,0), Vector3(0,0,0))
                  
                else:
                    
                    if (image_center[0] > red_center[0]):
                        vel = Twist(Vector3(0.2,0,0), Vector3(0,0,0.2))
                        
                    if (image_center[0] < red_center[0]):
                        vel = Twist(Vector3(0.2,0,0), Vector3(0,0,-0.2))
                        
        
        elif MISSION == 'BLUE' and (blueArea >= 1000):
            if (blue_center != 0) and (len(image_center) != 0):
                print("Estagio: Centalizando creeper azul")

                if (image_center[0] <= blue_center[0] + 15) and (image_center[0] >= blue_center[0] -15):
                    print("Centralized")
                    print(leituras[0])
                    if leituras[0]<0.4:
                        print("To close heading back")
                        MISSION=''
                    
                    vel = Twist(Vector3(0.4,0,0), Vector3(0,0,0))
                else:
                    if (image_center[0] > blue_center[0]):
                        vel = Twist(Vector3(0.2,0,0), Vector3(0,0,0.2))
                        
                    if (image_center[0] < blue_center[0]):
                       vel = Twist(Vector3(0.2,0,0), Vector3(0,0,-0.2))
                       
        
        
        else:
            print('Estagio: De role tranquilao')  
            if (DISTANCEOK and FIND150) or (DISTANCEOK and FIND50):
                    #Girando até "se achar"...
                    vel = Twist(Vector3(0,0,0), Vector3(0,0,-0.4))
                    velocidade_saida.publish(vel)
                    rospy.sleep(0.5)
            else:
                if (len(image_center) != 0) and (avarage_x != 0 and avarage_x != None):

                    if (image_center[0] <= avarage_x + 15) and (image_center[0] >= avarage_x -15):
                        
                        vel = Twist(Vector3(0.25,0,0), Vector3(0,0,0))
                        

                    else:
                        if (image_center[0] > avarage_x):
                            vel = Twist(Vector3(0.3,0,0), Vector3(0,0,0.2))
                            
                        if (image_center[0] < avarage_x):
                            vel = Twist(Vector3(0.3,0,0), Vector3(0,0,-0.2))
                            

        velocidade_saida.publish(vel)