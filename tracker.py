import cv2
import numpy as np
import imutils
import hickle as hkl

map = None

def getCamFeed():
    return cv2.VideoCapture('data/04.webm')
    #return cv2.VideoCapture(0)

def updateMap(params):
    return map

if __name__ == '__main__':

    # get camera feed
    cam_feed = getCamFeed()
    # sample image
    (_,sample) = cam_feed.read()
    sample = imutils.resize(sample, width=500)
    # init map
    map = np.zeros(sample.shape)

    # previous frame
    prev_frame = None

    frame_num = 0
    avg_x = []
    avg_y = []

    # setup params
    H = 300
    W = 100

    while True:
        # get a frame
        (active,frame) = cam_feed.read()

        if active:
            frame_num += 1
            # resize the frame, convert it to grayscale, and blur it
            frame = imutils.resize(frame, width=500)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (21, 21), 0)

            W0 = frame.shape[0]
 
                # if the first frame is None, initialize it
            if prev_frame is None:
                prev_frame = gray
                continue

            # compute the absolute difference between the current frame and
            # first frame
            frameDelta = cv2.absdiff(prev_frame, gray)
            thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]

            # dilate the thresholded image to fill in holes, then find contours
            # on thresholded image
            thresh = cv2.dilate(thresh, None, iterations=15)
            (cnts, _) = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)


            MAX_AREA = -1
            maxC = None
            for c in cnts:
                # if the contour is too small, ignore it
                cA = cv2.contourArea(c)
                if cA > 300 and cA > MAX_AREA:
                    MAX_AREA = cA
                    maxC = c

            # compute the bounding box for the contour, draw it on the frame,
            # and update the text
            if maxC is not None:
                (x, y, w, h) = cv2.boundingRect(maxC)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                #_x = 1 - ( (MAX_AREA/(frame.shape[0]*frame.shape[1]))**2 )

                _y = 997.15*32/w
                _x = (x + w/2)*100/W0
                avg_x.append(_x)
                avg_y.append(_y)

                if len(avg_x) > 5:
                    if _y > H:
                        H = min(700,int(_y) + 1)
                    if _x > W:
                        W = min(400,int(_x) + 1)
                    map = np.zeros([H,W])
                    avx = reduce(lambda a, b: a + b, avg_x[:-1]) / len(avg_x[:-1])
                    avy = reduce(lambda a, b: a + b, avg_y[:-1]) / len(avg_y[:-1])
                    uavx = avx*0.60 + avg_x[-1]*0.40
                    uavy = avy*0.70 + avg_y[-1]*0.30
                    cv2.circle(map,( int(uavx),int(uavy)) ,5,(255,0,0),-1)
                    #print int((x+w/2)*100/W0),int(_x)
                    avg_x = [uavx,uavx]
                    avg_y = [uavy,uavy]

            cv2.imshow("Webcam Feed", frame)
            cv2.imshow("Map",map)
            key = cv2.waitKey(23) & 0xFF

            # if the `q` key is pressed, break from the lop
            if key == ord("q"):
                break


    cam_feed.release()
    cv2.destroyAllWindows()
