import cv2
import argparse
import orien_lines
import datetime
from imutils.video import VideoStream
from utils import detector_utils as detector_utils
import pandas as pd
from datetime import date
import xlrd
from xlwt import Workbook
from xlutils.copy import copy
import numpy as np
from object_detection.utils import label_map_util

# cap = cv2.VideoCapture('170609_A_Delhi_058_2.mp4')
# start_time = datetime.datetime.now()
# num_frames = 0
# while(cap.isOpened()):
#     ret, frame = cap.read()
#     print(frame)
#     num_frames += 1
#     elapsed_time = (datetime.datetime.now() -
#                     start_time).total_seconds()
#     fps = num_frames / elapsed_time
#
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB )
#     show = cv2.resize(frame, (960, 540))
#     detector_utils.draw_text_on_image("FPS : " + str("{0:.2f}".format(fps)), show )
#
#     cv2.imshow('frame',show)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#
# cap.release()
# cv2.destroyAllWindows()

cap = cv2.VideoCapture('170609_A_Delhi_058_2.mp4')

while (cap.isOpened()):
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = QImage(frame, frame.shape[1], frame.shape[0], QImage.Format_RGB888)
    pix = QPixmap.fromImage(img)
    pix = pix.scaled(600, 400, Qt.KeepAspectRatio, Qt.SmoothTransformation)
    self.ui.label_7.setPixmap(pix)

    self.ui.frame = pix  # or img depending what `ui.frame` needs

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()