''''
    ==> yüzler datasetin içine yerleştirilmeli ve 1,2,3 gibi rakamlar verilmeli                   
    ==> trainer dosyasının içine trainer.yml eklenmeli 
'''
import time
import cv2 #OpenCv (cv2) kütüphanesi import edilir 
import numpy as np #Numpy görsel işleme kütüphanesi
import os
import pygame

pygame.mixer.init()
pygame.mixer.set_num_channels(6)
voice=pygame.mixer.Channel(2)
fartSound = pygame.mixer.Sound('/home/pi/beep2.wav') # ses dosyasını 
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer/trainer.yml')
cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath); # cascade algoritmasını kullanıyoruz. 

font = cv2.FONT_HERSHEY_SIMPLEX

#id sayaci
id = 0

# id lerine göre isimler mesela id=1 => batuhan
names = ['None', 'batuhan', 'taylor', 'non', 'Z', 'W'] 

# Gerçek zamanlı video başlatması 
cam = cv2.VideoCapture(-1) # burasi -1 olmak zorundadır . windowsta olsa 0 yapmalı.
cam.set(3, 640) # videonun genişliği
cam.set(4, 480) # videonun yüksekliği 

# yüzün algılayacagı pencere boyutu 
minW = 0.1*cam.get(3)
minH = 0.1*cam.get(4)

while True:

    ret, img =cam.read()
    img = cv2.flip(img, -1) # dikey olarak çevirme 
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale( 
        gray,
        scaleFactor = 1.2,
        minNeighbors = 5,
        minSize = (int(minW), int(minH)),
       )

    for(x,y,w,h) in faces:
        if (id == 'batuhan') :
         cv2.rectangle(img, (x,y), (x+w,y+h), (0,0,255), 2)
         voice.play(fartSound)
         
        else :
         cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2) 

        id, confidence = recognizer.predict(gray[y:y+h,x:x+w])

        # Yüzün ne kadar ona ait olduğunu gösteren doğrulayacı 
        if (confidence < 100):
            id = names[id]
            confidence = "  {0}%".format(round(100 - confidence))
        else:
            id = "unknown"
            confidence = "  {0}%".format(round(100 - confidence))
        
        cv2.putText(img, str(id), (x+5,y-5), font, 1, (255,255,255), 2)
        cv2.putText(img, str(confidence), (x+5,y+h-5), font, 1, (255,255,0), 1)  
    
    cv2.imshow('camera',img) 

    k = cv2.waitKey(10) & 0xff # ESC ' basıp videodan çıkma 
    if k == 27:
        break

# Program temizliği .
print("\n [INFO] Program temizlendi .")
cam.release()
cv2.destroyAllWindows()
