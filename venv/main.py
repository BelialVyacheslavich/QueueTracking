import numpy as np
import cv2
import multiprocessing as mp
import datetime
from centroidtracker import CentroidTracker

now1 = datetime.datetime.today()
# Variables
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))
color_box = (50, 50, 230)  # Blue Green Red
color_text = (200, 200, 200)  # Blue Green Red
color_area = (50, 50, 50)

nettype = 0
if nettype is 0:
    net = cv2.dnn.readNetFromCaffe('nets\\base\\net.prototxt.txt', 'nets\\base\\net.caffemodel')
    min_confidence = 0.2
else:
    net = cv2.dnn.readNetFromCaffe('nets\\face\\net.prototxt', 'nets\\face\\net.caffemodel')
    min_confidence = 0.5

video = cv2.VideoCapture('videos\\test3.mp4')

frames_delay = 40

maxDisappeared = 10  # test 1 = 5;
ct = CentroidTracker(maxDisappeared)


def draw_frame(frame, ROI_box, obj_list):
    # frame = cv2.resize(frame, (800, 600))
    cv2.rectangle(frame, ROI_box, color_area, 2)
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
                                 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()

    # tracker = cv2.MultiTracker_create()

    rects = []
    for i in np.arange(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with
        # the prediction
        confidence = detections[0, 0, i, 2]

        # filter out weak detections by ensuring the `confidence` is
        # greater than the minimum confidence
        idx = int(detections[0, 0, i, 1])

        cur_class = "{}".format(CLASSES[idx])
        condition_1 = (confidence > min_confidence and cur_class is "person") and nettype is 0
        condition_2 = nettype is 1 and confidence > min_confidence
        if condition_1 or condition_2:
            # idx_mas.append(idx)
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            if startX >= ROI_box[0] and startY >= ROI_box[1] and endX <= ROI_box[2] and endY <= ROI_box[3]:
                rects.append(box.astype("int"))
                cv2.rectangle(frame, (startX, startY), (endX, endY), color_box, 2)

                # y = startY - 15 if startY - 15 > 15 else startY + 15
                # cv2.putText(frame, label, (startX, y),
                # cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_text, 2)
    objects = ct.update(rects)

    # loop over the tracked objects
    for (objectID, centroid) in objects.items():
        # draw both the ID of the object and the centroid of the
        # object on the output frame
        text = "ID {}".format(objectID)
        cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)

    # Output info on screen
    text = "Queue count: " + ct.objects.__len__().__str__()
    cv2.putText(frame, text, (2, 32), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
    cv2.putText(frame, text, (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (230, 230, 230), 2)
    return obj_list


def main():
    fps = 0
    ROI_box = None
    person_num = 0;
    obj_list = list(())
    while (fps < frames_delay + 1):
        ret, frame = video.read()
        if ret:
            if ROI_box is None:
                ROI_box = cv2.selectROI("Select active zone and press 'spacebar'", frame, fromCenter=False,
                                        showCrosshair=True)
                cv2.destroyAllWindows()
            if fps is frames_delay:
                obj_list = draw_frame(frame, ROI_box, obj_list)
                cv2.imshow("Frame", frame)
                fps = 0
                if cv2.waitKey(frames_delay) & 0xFF == ord('q'):
                    break
            else:
                fps += 1
        else:
            print("End of file")
            cv2.destroyAllWindows()
            break;
    ct.endCentroind()
    obj_count = ct.nextObjectID
    # print("Было зарегестрированно " + obj_count.__str__() + " объектов")
    # print("Было зарегестрированно " + (ct.detected_objects.__len__() + ct.objects.__len__()).__str__() + " объектов")
    det_objs = ct.detected_objects
    print("Было зарегестрированно " + det_objs.__len__().__str__() + " объектов")
    strform = "Hours:%H, Minutes:%M, Seconds:%S, Microseconds:%f"
    for obj in det_objs:
        obj_id = obj.id + 1
        obj_time = obj.getDetectionTime()
        str = "Объект №" + obj_id.__str__() + " был зарегестрирован " + obj_time.__str__() + " секунд"
        print(str)
    now2 = datetime.datetime.today()
    delta = now2 - now1
    print("Времени прошло с начала работы программы " + delta.seconds.__str__() + " секунд")


main()
