from detect_face import detect
import cv2
img=cv2.imread("./data/test01.jpg")
print(detect(img))
print(detect(img))
