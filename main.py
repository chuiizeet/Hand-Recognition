import cv2
import numpy as np
from sklearn.metrics import pairwise
from time import sleep

# Global
backGround = None
accumulated_weight = 0.5

roi_top = 20
roi_bottom = 300
roi_right = 300
roi_left = 600

def calc_accum_avg(frame,accumulated_weight):

    global backGround

    if backGround is None:
        backGround = frame.copy().astype('float')
        return None

        cv2.accumulateWeighted(frame,backGround,accumulated_weight)


def segment(frame,threshold_min=65):

    diff = cv2.absdiff(backGround.astype('uint8'),frame)
    ret, thresh = cv2.threshold(diff,threshold_min,255,
                               cv2.THRESH_BINARY)
    image, cts, hierarchy = cv2.findContours(thresh.copy(),
                                            cv2.RETR_EXTERNAL,
                                            cv2.CHAIN_APPROX_SIMPLE)

    if len(cts) == 0:
        return None

    else:
        # The largest external contour in roi, is the hand.
        hand_segment = max(cts,key=cv2.contourArea)

        return (thresh,hand_segment)


def count_fingers(thresh,hand_segment):

    conv_hull = cv2.convexHull(hand_segment)

    top = tuple(conv_hull[conv_hull[:,:,1].argmin()][0])
    bottom = tuple(conv_hull[conv_hull[:,:,1].argmax()][0])
    left = tuple(conv_hull[conv_hull[:,:,0].argmin()][0])
    right = tuple(conv_hull[conv_hull[:,:,0].argmax()][0])

    cX = (left[0] + right[0]) // 2
    cY = (top[1] + bottom[1]) // 2

    distance = pairwise.euclidean_distances([[cX,cY]], Y=[left,right,top,bottom])[0]

    max_distance = distance.max()

    #Depends on the size of the hand
    radius = int(0.9 * max_distance)
    circumference = (2 * np.pi * radius)

    print(thresh[:2])
    npThresh = np.array(thresh)
    print(npThresh[:2])

    circular_roi = np.zeros_like(thresh)

    cv2.circle(circular_roi,(cX,cY),radius,255,10)

    circular_roi = cv2.bitwise_and(thresh,thresh,
                                  mask=circular_roi)

    image, cts, hierarchy = cv2.findContours(circular_roi.copy(),
                                            cv2.RETR_EXTERNAL,
                                            cv2.CHAIN_APPROX_SIMPLE)


    count = 0

    for cnt in cts:

        (x,y,w,h) = cv2.boundingRect(cnt)

        # Limits
        out_of_wrist = (cY + (cY*0.25)) > (y+h)
        limit_points = ((circumference*0.25) > cnt.shape[0])

        if out_of_wrist and limit_points:
            count += 1

    return count

cam = cv2.VideoCapture(0)
n_frames = 0

while True:

    ret, frame = cam.read()
    frame_copy = frame.copy()
    roi = frame[roi_top:roi_bottom,roi_right:roi_left]
    gray = cv2.cvtColor(roi,cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray,(7,7),0)

    if n_frames < 60:

        calc_accum_avg(gray,accumulated_weight)

        if n_frames <= 59:
            cv2.putText(frame_copy,'WAIT, GETTING BACKGROUND',
                       (200,300),
                       cv2.FONT_HERSHEY_SIMPLEX,
                       1,
                       (0,0,255),
                       2)

            cv2.imshow('Finger count', frame_copy)

    else:

        hand = segment(gray)
        if hand is not None:

            thresh,hand_segment = hand

            # Draw contours
            cv2.drawContours(frame_copy,[hand_segment+(roi_right,roi_top)],-1,
                             (0,255,0),
                            5)

            fingers = count_fingers(thresh,hand_segment)
            cv2.putText(frame_copy,
                       str(fingers),
                       (70,50),
                       cv2.FONT_HERSHEY_SIMPLEX,
                       1,
                       (0,0,255),
                       2)

            # Thresholded image
            cv2.imshow('Thresholded',thresh)

    cv2.rectangle(frame_copy,(roi_left,roi_top),(roi_right,roi_bottom),
                 (0,0,255),
                 5)

    n_frames += 1
    print(n_frames)

    cv2.imshow('Finger Count',frame_copy)

    k = cv2.waitKey(1) & 0xFF

    if k == 27:
        break

cam.release()
cv2.destroyAllWindows()