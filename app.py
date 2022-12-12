import cv2
import numpy as np


#Video capture 
cap = cv2.VideoCapture('video.mp4')

#Initialize the algorithm
algo = cv2.bgsegm.createBackgroundSubtractorMOG()

count_line_position = 550


#Minimum Height of Rectangle
min_height_rect = 80 

#Minimum width of Rectangle
min_width_rect = 80


#Bounding box center
def centre_cal(x,y,w,h):
    x1 = int(w/2)
    y1 = int(h/2)
    cx = x+x1
    cy = y+y1
    return (cx,cy)


detect =[]
offset = 6 # Allowable Error in pixel
counter = 0


while True:
    ret, frame = cap.read()
    grey = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    blurr = cv2.GaussianBlur(grey,(3,3),5)
    # print(frame.shape)

    img_apply = algo.apply(blurr)
    dilat = cv2.dilate(img_apply,np.ones((5,5)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    dilatada = cv2.morphologyEx(dilat,cv2.MORPH_CLOSE,kernel)
    dilatada = cv2.morphologyEx(dilatada,cv2.MORPH_CLOSE,kernel)
    countershape ,h  = cv2.findContours(dilatada,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    cv2.line(frame,(25,count_line_position),(1200,count_line_position),(0,0,255),5)
    cv2.putText(frame,'Vehicle counter : '+str(counter),(400,50),5,cv2.FONT_HERSHEY_DUPLEX,(0,255,0),3)

    for (i,c) in enumerate(countershape):
        (x,y,w,h) = cv2.boundingRect(c)
        if  (w>=min_width_rect) and (h>=min_height_rect):
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),3)
            cv2.putText(frame,'Vehicle'+str(counter),(x,y-20),5,cv2.FONT_HERSHEY_DUPLEX,(255,255,0),1)

            #Defining the centre of the bounding box for each objects detected
            centre = centre_cal(x,y,w,h)
            detect.append(centre)
            cv2.circle(frame,centre,4,(255,0,255),-1)
            for (x,y) in detect:
                if y<(count_line_position+offset) and y>(count_line_position-offset):
                    counter+= 1
                    cv2.line(frame,(25,count_line_position),(1200,count_line_position),(0,125,255),3)
                    detect.remove((x,y))
                    # print('Vehicle counter:'+str(counter) )



    cv2.imshow('video',frame)

    key = cv2.waitKey(30)
    if key ==13:
        break


cap.release()
cv2.destroyAllWindows()