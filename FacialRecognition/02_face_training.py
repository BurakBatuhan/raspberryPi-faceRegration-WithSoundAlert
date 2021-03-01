''''

    ==> yüzlere 1,2,3 gibi numaralandırılmalı                   
    ==> trainer dosyası yaratın.
    ==>  "pip install pillow"
'''

import cv2 
import numpy as np
from PIL import Image
import os

# resimler için yol 
path = 'dataset'

recognizer = cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml");

# görüntülerin verilerini alma fonksiyonu 
def getImagesAndLabels(path):

    imagePaths = [os.path.join(path,f) for f in os.listdir(path)]     
    faceSamples=[]
    ids = []

    for imagePath in imagePaths:

        PIL_img = Image.open(imagePath).convert('L') # resmi griye çevirme 
        img_numpy = np.array(PIL_img,'uint8')

        id = int(os.path.split(imagePath)[-1].split(".")[1])
        faces = detector.detectMultiScale(img_numpy)

        for (x,y,w,h) in faces:
            faceSamples.append(img_numpy[y:y+h,x:x+w])
            ids.append(id)

    return faceSamples,ids

print ("\n [Bilgilendirme] Yuzler ogreniliyor lutfen bekleyiniz.")
faces,ids = getImagesAndLabels(path)
recognizer.train(faces, np.array(ids))

#  trainer/trainer.yml içine kaydetmek 
recognizer.write('trainer/trainer.yml') # write yerine save yapılırsa mac te çalışır . ama pi de çalışmaz 

# Yüzlerin öğrenilip programndan çıkılması 
print("\n [Bilgilendirme] {0} Yuzler ogrenildi programdan cikabilirsiniz.".format(len(np.unique(ids))))
