# C:\Users\user\PycharmProjects\pythonProject1\detect\eyeshadow
import os

product = '립스틱' #제품명
folders = os.listdir('C:/Users/user/PycharmProjects/pythonProject1/detect/eyeshadow') #폴더경로
total = 0
for folder in folders:
      print(folder)
      files = os.listdir('C:/Users/user/PycharmProjects/pythonProject1/detect/eyeshadow' + '/' + folder)
      files = [file for file in files if file.endswith(".jpg" or "png")]
      #print(len(files))
      total += len(files)
print(total)#작업 완료 이미지 개수