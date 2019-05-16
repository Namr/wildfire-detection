import cv2
import numpy as np
import skvideo.io

writer = skvideo.io.FFmpegWriter("outputvideo.mp4")
cap = cv2.VideoCapture('Gatlinburg Wildfire 300% Speed Time Lapse.mp4')

frameCount = 0
while(cap.isOpened()):
    ret, frame = cap.read()
    if ret:
        isFire = False
        
        frameCount = cap.get(cv2.CAP_PROP_POS_FRAMES)
        # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # change color spaces
        hsvFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                
        # set bounds for what is "green" and threshold based on that values
        lowerHSVBound = np.array([0, 100, 100])
        upperHSVBound = np.array([30, 255, 255])
        maskFrame = cv2.inRange(hsvFrame, lowerHSVBound, upperHSVBound)

        # perform morphological transformation to remove noise from the image
        # read more here:
        # https://docs.opencv.org/3.4/d9/d61/tutorial_py_morphological_ops.html
        kernel = np.ones((5, 5), np.uint8)
        maskFrame = cv2.morphologyEx(maskFrame, cv2.MORPH_OPEN, kernel)

        # get contours of the detected tape
        edgeFrame = cv2.Canny(maskFrame, 100, 200)
        contours, heirarchy = cv2.findContours(edgeFrame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            if cv2.contourArea(contour) > 30:
                isFire = True

        if isFire:
            # write the flipped frame
            writer.writeFrame(frame)
            cv2.putText(frame, "CURRENTLY ON FIRE", (100, 100), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 0, 255), 1, cv2.LINE_AA)
            # cv2.imshow("mask", maskFrame)
            
        cv2.imshow("frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

print(frameCount)
cap.release()
cv2.destroyAllWindows()
