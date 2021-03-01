''''
    ==> dataset adında bir dosya olusturun.Burada tanıtacagınız yuzler oraya kaydedilecektir 
    ==> ,yuzler için id olarak 1,2,3 gibi isimler verin                   
  
'''

import cv2
import os

cam = cv2.VideoCapture(-1) # video baslaması 
cam.set(3, 640) # videonun genişliği
cam.set(4, 480) #videonun yüksekliği 

face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Tanıtcagınız yuzu id lendirin 
face_id = input('\n id giriniz <return> ==>  ')

print("\n [bilgi] Yuzunuz cekilecektir lutfen kameraya bakınız .")
# Bireysel örnekleme yüz sayısını başlat
count = 0

while(True):

    ret, img = cam.read()
    img = cv2.flip(img, -1) # video görüntüsünü dikey olarak çevirilir
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #Başarılı sonuçlar için görsel siyah-beyaz forma çevrilir
    faces = face_detector.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in faces:

        cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)     
        count += 1

        # Resmileri datasete kaydedilmesi .
        cv2.imwrite("dataset/User." + str(face_id) + '.' + str(count) + ".jpg", gray[y:y+h,x:x+w])

        cv2.imshow('resim', img)

    k = cv2.waitKey(100) & 0xff # Esc ' ye basıp videodan çıkabilirsiniz
    if k == 27:
        break
    elif count >= 30: # 30 yuz ogreni alınıp program durduruluyor 
         break

# Temizlik .
print("\n [Bilgilendirme] Program temizlendi .")
cam.release()
cv2.destroyAllWindows()


