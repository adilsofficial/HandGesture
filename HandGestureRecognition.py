import cv2
from subprocess import check_output
import os
import numpy as np

# Load all the images
print(check_output(["ls", "temp_images//"]).decode("utf8"))
folder = "temp_images//"
onlyfiles = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]

# Capturing video from Camera
cap = cv2.VideoCapture(0)

#--------------------- Frame and Matching for Victory Gesture ---------------------#

# Reading image for Victory-template
print(onlyfiles[0])
vic_temp1 = cv2.imread("temp_images//" + onlyfiles[0], 0)
vic_temp2 = cv2.imread("temp_images//" + onlyfiles[0], 1)
w, h = vic_temp1.shape[::-1]
# Change color of image to YCR
ycc1 = cv2.cvtColor(vic_temp2,cv2.COLOR_BGR2YCR_CB)
vic_bw_temp, cr1, cb1 = cv2.split(ycc1)
# Change color of image to black and white (bw)
cv2.threshold(cr1, 0, 255,cv2.THRESH_BINARY + cv2.THRESH_OTSU, vic_bw_temp)
kernel1 = np.ones((9,9), np.uint8)
# Binary Image
vic_bw_temp = cv2.morphologyEx(vic_bw_temp,cv2.MORPH_CLOSE,kernel1)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    # Our operations on the frame come here
    cv2.flip(frame,1,frame)
    # Change color of image to gray scale
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    # Change color of image to YCR
    ycc = cv2.cvtColor(frame,cv2.COLOR_BGR2YCR_CB)
    # Change color of image to bw
    vic_bw_frame, cr, cb = cv2.split(ycc)
    # Change color of image to binary
    cv2.threshold(cr, 0, 255,cv2.THRESH_BINARY + cv2.THRESH_OTSU, vic_bw_frame)
    kernel = np.ones((9,9), np.uint8)
    vic_bw_frame = cv2.morphologyEx(vic_bw_frame,cv2.MORPH_CLOSE,kernel)
    # Apply template Matching
    res = cv2.matchTemplate(vic_bw_frame, vic_bw_temp, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    # Print Precision value for Victory
    print(max_val)
    # if max_val is great than 0.65 then show a rec
    if max_val > 0.65:
        top_left = max_loc
        bottom_right = (top_left[0] + w, top_left[1] + h)
        # Rectangle around the detected gesture
        cv2.rectangle(frame, top_left, bottom_right, (0, 0, 255), 2)
        # Text for detected gesture
        cv2.putText(frame,'victory',(30,30),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
        cv2.rectangle(vic_bw_frame, top_left, bottom_right, (0, 0, 255), 4)
        cv2.putText(vic_bw_frame, 'victory', (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    # Display the resulting frame
    cv2.imshow('Victory-Gesture',frame)
    # Show the Binary version of the template image and camera video
    #cv2.imshow('VICTORY-FRAME_BW',vic_bw_frame)
    #cv2.imshow('VICTORY-TEMPLATE', vic_bw_temp)
    # Run the frame until user press 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# --------------------- Frame and Matching for Victory Gesture ---------------------#

#--------------------- Frame and Matching for Like Gesture ---------------------#

# Reading image for Like-template
print(onlyfiles[2])
like_temp1 = cv2.imread("temp_images//" + onlyfiles[2], 0)
like_temp2 = cv2.imread("temp_images//" + onlyfiles[2], 1)
w, h = like_temp1.shape[::-1]
# Change color of image to gray scale
ycc1 = cv2.cvtColor(like_temp2,cv2.COLOR_BGR2YCR_CB)
like_bw_temp, cr1, cb1 = cv2.split(ycc1)
cv2.threshold(cr1, 0, 255,cv2.THRESH_BINARY + cv2.THRESH_OTSU, like_bw_temp)
kernel1 = np.ones((9,9), np.uint8)
like_bw_temp = cv2.morphologyEx(like_bw_temp,cv2.MORPH_CLOSE,kernel1)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    # Our operations on the frame come here
    cv2.flip(frame,1,frame)
    # Change color of image to gray scale
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    ycc = cv2.cvtColor(frame,cv2.COLOR_BGR2YCR_CB)
    like_bw_frame, cr, cb = cv2.split(ycc)
    cv2.threshold(cr, 0, 255,cv2.THRESH_BINARY + cv2.THRESH_OTSU, like_bw_frame)
    kernel = np.ones((9,9), np.uint8)
    like_bw_frame = cv2.morphologyEx(like_bw_frame,cv2.MORPH_CLOSE,kernel)
    # Apply template Matching
    res = cv2.matchTemplate(like_bw_frame, like_bw_temp, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    # Print Precision value for Like
    print(max_val)
    if max_val > 0.80:
        top_left = max_loc
        bottom_right = (top_left[0] + w, top_left[1] + h)
        cv2.rectangle(frame, top_left, bottom_right, (0, 0, 255), 4)
        cv2.putText(frame,'Like',(30,30),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
        cv2.rectangle(like_bw_frame, top_left, bottom_right, (0, 0, 255), 4)
        cv2.putText(like_bw_frame, 'Like', (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    # Display the resulting frame
    cv2.imshow('Like-Gesture',frame)
    #cv2.imshow('Like-FRAME_BW',like_bw_frame)
    #cv2.imshow('Like-TEMPLATE', like_bw_temp)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# --------------------- Frame and Matching for Like Gesture ---------------------#


#--------------------- Frame and Matching for Fist Gesture ---------------------#

# Reading image for Fist-template
print(onlyfiles[1])
fist_temp1 = cv2.imread("temp_images//" + onlyfiles[1], 0)
fist_temp2 = cv2.imread("temp_images//" + onlyfiles[1], 1)
w, h = like_temp1.shape[::-1]
# Change color of image to gray scale
ycc1 = cv2.cvtColor(fist_temp2,cv2.COLOR_BGR2YCR_CB)
fist_bw_temp, cr1, cb1 = cv2.split(ycc1)
cv2.threshold(cr1, 0, 255,cv2.THRESH_BINARY + cv2.THRESH_OTSU, fist_bw_temp)
kernel1 = np.ones((9,9), np.uint8)
fist_bw_temp = cv2.morphologyEx(fist_bw_temp,cv2.MORPH_CLOSE,kernel1)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    # Our operations on the frame come here
    cv2.flip(frame,1,frame)
    # Change color of image to gray scale
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    ycc = cv2.cvtColor(frame,cv2.COLOR_BGR2YCR_CB)
    fist_bw_frame, cr, cb = cv2.split(ycc)
    cv2.threshold(cr, 0, 255,cv2.THRESH_BINARY + cv2.THRESH_OTSU, fist_bw_frame)
    kernel = np.ones((9,9), np.uint8)
    fist_bw_frame = cv2.morphologyEx(fist_bw_frame,cv2.MORPH_CLOSE,kernel)
    # Apply template Matching
    res = cv2.matchTemplate(fist_bw_frame, fist_bw_temp, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    # Print Precision value for Fist
    print(max_val)
    # 0.67 far
    if max_val > 0.67:
        top_left = max_loc
        bottom_right = (top_left[0] + w, top_left[1] + h)
        cv2.rectangle(frame, top_left, bottom_right, (0, 0, 255), 2)
        cv2.putText(frame,'Fist',(30,30),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
        cv2.rectangle(like_bw_frame, top_left, bottom_right, (0, 0, 255), 4)
        cv2.putText(like_bw_frame, 'Fist', (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    # Display the resulting frame
    cv2.imshow('Fist-Gesture',frame)
    #cv2.imshow('Fist-FRAME_BW',fist_bw_frame)
    #cv2.imshow('Fist-TE, fist_bw_temp)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# --------------------- Frame and Matching for fist Gesture ---------------------#

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
