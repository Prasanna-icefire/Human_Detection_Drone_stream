import cv2
import numpy as np

net = cv2.dnn.readNet("/home/icefire/ML/Yolo_object/weights/yolov3.weights","cfg/yolov3.cfg")
classes = []
with open("data/coco.names","r") as f:
    classes = [line.strip() for line in f.readlines()]
#print(classes)

#getting layer names
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0]-1]for i in net.getUnconnectedOutLayers()]#In this line we use only the output layers
#print(output_layers)  #u get to see the output layer numbers XD


img = cv2.imread("data/test.jpg")
cv2.resize(img,None,fx=0.4,fy=0.4)
height, width, channels = img.shape
#For getting features from image  1st blob it
blob = cv2.dnn.blobFromImage(img,0.00392,(416,416),(0,0,0),True,crop=False)
net.setInput(blob)
outs = net.forward(output_layers) # we have suceesfully obtained the detections, we now have to display it properly

for out in outs:
    for detection in out:
        # we first obtain the confidence of the detection,for this we 1st obtain the scores
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.5:
            center_x = int(detection[0]*width)
            center_y = int(detection[1]*height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)

            x = int(center_x - w/2)
            y = int(center_y - h/2)
            cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)

#display


cv2.imshow("Image",img)


cv2.waitKey(0)
cv2.destroyAllWindows()