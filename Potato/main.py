import cv2 
import numpy as np 

# video source 
# change source file
source = cv2.VideoCapture("")

captured_frames = list()
_, mean = source.read()
mean = cv2.cvtColor(mean,cv2.COLOR_BGR2GRAY)

def mean_of_frames(frames :list) -> list:
    """
    Function to compute the median of all frames

    Params:
    frames -> List of frames

    Returns:
    mean_frames -> The median of all frames
    """

    mean_frames = np.mean(frames, axis=0).astype(dtype=np.uint8)
    return mean_frames

while True:
    
    """  MEAN METHOD """
    _, frame = source.read()

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    captured_frames.append(frame)

    if len(captured_frames) == 10:
        mean = mean_of_frames(captured_frames)
        captured_frames = list()        

    
    difference = cv2.absdiff(frame,mean)
    _,threshold = cv2.threshold(difference,100,255,cv2.THRESH_BINARY)

    cnts,_ = cv2.findContours(threshold, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    final = frame.copy()
    for c in cnts:
        if cv2.contourArea(c) < 8000 or cv2.contourArea(c) > 15000:
            continue
        
        cv2.drawContours(final,c,-1,(0,0,0),4)
        # (x, y, w, h) = cv2.boundingRect(c)

        # cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow("Area",final[315:1000,500:800])
    cv2.imshow("Overall",final)


    key = cv2.waitKey(1)

    if key == ord("q"):
        break

cv2.destroyAllWindows()