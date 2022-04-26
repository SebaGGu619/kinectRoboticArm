import cv2
import time
import numpy as np
import freenect
import math
import pygame
from pygame import K_r
import serial
import struct
import pyfirmata
from pyfirmata import SERVO

ctx = freenect.init()
dev = freenect.open_device(ctx, 0)
freenect.set_led(dev, freenect.LED_BLINK_GREEN)
freenect.set_tilt_degs(dev, 25)
freenect.close_device(dev)

board = pyfirmata.Arduino('/dev/tty.usbserial-1420')

time.sleep(1)

board.digital[9].mode = SERVO
board.digital[11].mode = SERVO
board.digital[3].mode = SERVO

time.sleep(1)

iter8 = pyfirmata.util.Iterator(board)
iter8.start()

time.sleep(1)

board.digital[3].write(90)
board.digital[11].write(90)
board.digital[9].write(90)

time.sleep(1)


def calculateDistance(x1, y1, x2, y2):
    dist = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return dist


def get_video():
    array, _ = freenect.sync_get_video()
    array = cv2.cvtColor(array, cv2.COLOR_RGB2BGR)
    return array


def get_depth():
    array, _ = freenect.sync_get_depth()
    # array = array.astype(np.uint8)
    return array


tracker = cv2.TrackerCSRT_create()

protoFile = "hand/pose_deploy.prototxt"
weightsFile = "hand/pose_iter_102000.caffemodel"
nPoints = 22
POSE_PAIRS = [[0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8], [0, 9], [9, 10], [10, 11], [11, 12],
              [0, 13], [13, 14], [14, 15], [15, 16], [0, 17], [17, 18], [18, 19], [19, 20]]

threshold = 0.2

pygame.init()
pygame.display.init()

frame = get_video()
frame = cv2.flip(frame, 1)
frameCopy = np.copy(frame)

frameWidth = frame.shape[1]
frameHeight = frame.shape[0]

aspect_ratio = frameWidth / frameHeight

inHeight = 368
inWidth = int(((aspect_ratio * inHeight) * 8) // 8)

net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)
k = 0

cv2.imshow('Output Tracking Point', frameCopy)

manaX1 = 0
manaY1 = 0
manaX2 = 0
manaY2 = 0
while 1:
    frameDepth = get_depth()
    frameDepth = cv2.flip(frameDepth, 1)
    frame = get_video()
    frame = cv2.flip(frame, 1)
    frameCopy = np.copy(frame)
    if manaY1 == 0 or manaY2 == 0 or manaX1 == 0 or manaX2 == 0:
        manaX1 = 0
        manaY1 = 0
        manaX2 = 0
        manaY2 = 0

        k += 1

        inpBlob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (inWidth, inHeight),
                                        (0, 0, 0), swapRB=False, crop=False)

        net.setInput(inpBlob)

        output = net.forward()

        points = []

        probMap = output[0, 9, :, :]
        probMap = cv2.resize(probMap, (frameWidth, frameHeight))

        minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)

        if prob > threshold:
            cv2.circle(frameCopy, (int(point[0]), int(point[1])), 6, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
            cv2.putText(frameCopy, "{}".format(9), (int(point[0]), int(point[1])), cv2.FONT_HERSHEY_SIMPLEX, .8,
                        (0, 0, 255), 2, lineType=cv2.LINE_AA)

            points.append((int(point[0]), int(point[1])))
            manaX1 = int(point[0])
            manaY1 = int(point[1])
        else:
            points.append(None)

        points = []

        probMap = output[0, 12, :, :]
        probMap = cv2.resize(probMap, (frameWidth, frameHeight))

        minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)

        if prob > threshold:
            cv2.circle(frameCopy, (int(point[0]), int(point[1])), 6, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
            cv2.putText(frameCopy, "{}".format(12), (int(point[0]), int(point[1])), cv2.FONT_HERSHEY_SIMPLEX, .8,
                        (0, 0, 255), 2, lineType=cv2.LINE_AA)

            points.append((int(point[0]), int(point[1])))
            manaX2 = int(point[0])
            manaY2 = int(point[1])
        else:
            points.append(None)

        dist = calculateDistance(manaX1, manaY1, manaX2, manaY2)
        dist = dist * 3 / 2
        bbox = (manaX1 - dist / 3, manaY1 - dist / 3, dist / 1.2, dist / 1.2)
        p1 = (int(bbox[0]), int(bbox[1]))
        p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
        cv2.rectangle(frameCopy, p1, p2, (0, 0, 255), 2, 1)
        tracker = cv2.TrackerCSRT_create()
        ok = tracker.init(frameCopy, bbox)
    else:
        keys = pygame.key.get_pressed()
        if keys[K_r]:
            manaX1 = 0

        bboxInainte1 = int(bbox[0])
        bboxInainte2 = int(bbox[1])

        ok, bbox = tracker.update(frameCopy)

        bbox1 = int(bbox[0])
        bbox2 = int(bbox[1])

        if bbox1 > 479:
            bbox1 = 479
        if bbox2 > 479:
            bbox2 = 479
        if bbox1 < 0:
            bbox1 = 0
        if bbox2 < 0:
            bbox2 = 0

        bbox3 = int(frameDepth[bbox1][bbox2])

        cv2.putText(frameCopy, "X:", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
        cv2.putText(frameCopy, str(bbox1), (28, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
        cv2.putText(frameCopy, "Y:", (5, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
        cv2.putText(frameCopy, str(bbox2), (28, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
        cv2.putText(frameCopy, "Z:", (5, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
        cv2.putText(frameCopy, str(bbox3), (28, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
        cv2.circle(frameDepth, (bbox1, bbox2), 1, (255, 255, 255))

        time.sleep(0.1)

        # if bboxInainte1 > bbox1 + 1 or bboxInainte1 < bbox1 - 1:
        bbox1 = bbox1 - 100
        bbox1 = bbox1 / 3
        bbox1 = int(bbox1)
        if bbox1 < 0:
            bbox1 = 0
        if bbox1 > 360:
            bbox1 = 360
        board.digital[11].write(bbox1)

        # if bboxInainte2 > bbox2 + 1 or bboxInainte2 < bbox2 - 1:
        bbox2 = bbox2 - 50
        if bbox2 < 150:
            bbox2 = 150
        if bbox2 > 360:
            bbox2 = 360
        bbox2 = bbox2 / 3
        bbox2 = int(bbox2)
        board.digital[9].write(bbox2)
        board.digital[3].write(bbox2)
        # print(bbox1)
        # print(bbox2)
        # print("-------")
        if ok:
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            cv2.rectangle(frameCopy, p1, p2, (255, 255, 255), 2, 1)
        else:
            manaX1 = 0
            tracker = 0
            print("nu e ok")

    # cv2.imshow('Depth Map', frameDepth)

    cv2.imshow('Output Tracking Point', frameCopy)

    key = cv2.waitKey(1)
    if key == 27:
        break
cv2.destroyAllWindows()
exit(0)
