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

from PyQt5 import QtCore, QtGui, QtWidgets
# from vehicle.hand_detection import detection
from PyQt5.QtCore import pyqtSlot
from PyQt5.Qt import QImage,QPixmap,QFileDialog
from PyQt5 import Qt
import cv2


from object_detection.utils import visualization_utils as vis_util

lst1=[]
lst2=[]
ap = argparse.ArgumentParser()
ap.add_argument('-d', '--display', dest='display', type=int,
                        default=1, help='Display the detected images using OpenCV. This reduces FPS')
args = vars(ap.parse_args())

detection_graph, sess = detector_utils.load_inference_graph()

from PyQt5 import QtCore, QtGui, QtWidgets
# from vehicle.hand_detection import detection
from PyQt5.QtCore import pyqtSlot
from PyQt5.Qt import QImage,QPixmap,QFileDialog
from PyQt5 import Qt
import cv2



class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(800, 600)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.widget = QtWidgets.QWidget(self.centralwidget)
        self.widget.setGeometry(QtCore.QRect(10, 10, 781, 541))
        self.widget.setObjectName("widget")
        self.gridLayout = QtWidgets.QGridLayout(self.widget)
        self.gridLayout.setContentsMargins(0, 0, 0, 0)
        self.gridLayout.setObjectName("gridLayout")
        self.label = QtWidgets.QLabel(self.widget)
        self.label.setText("")
        self.label.setScaledContents(True)
        self.label.setObjectName("label")
        self.gridLayout.addWidget(self.label, 0, 0, 1, 4)
        self.pushButton = QtWidgets.QPushButton(self.widget)
        self.pushButton.setObjectName("pushButton")

        self.gridLayout.addWidget(self.pushButton, 1, 0, 1, 2)
        self.label_3 = QtWidgets.QLabel(self.widget)
        self.label_3.setAlignment(QtCore.Qt.AlignCenter)
        self.label_3.setObjectName("label_3")
        self.gridLayout.addWidget(self.label_3, 1, 2, 1, 1)
        self.lineEdit_2 = QtWidgets.QLineEdit(self.widget)
        self.lineEdit_2.setObjectName("lineEdit_2")
        self.gridLayout.addWidget(self.lineEdit_2, 1, 3, 1, 1)
        self.label_2 = QtWidgets.QLabel(self.widget)
        self.label_2.setAlignment(QtCore.Qt.AlignCenter)
        self.label_2.setObjectName("label_2")
        self.gridLayout.addWidget(self.label_2, 2, 0, 1, 1)
        self.lineEdit = QtWidgets.QLineEdit(self.widget)
        self.lineEdit.setObjectName("lineEdit")
        self.gridLayout.addWidget(self.lineEdit, 2, 1, 1, 1)
        self.label_4 = QtWidgets.QLabel(self.widget)
        self.label_4.setAlignment(QtCore.Qt.AlignCenter)
        self.label_4.setObjectName("label_4")
        self.gridLayout.addWidget(self.label_4, 2, 2, 1, 1)
        self.lineEdit_3 = QtWidgets.QLineEdit(self.widget)
        self.lineEdit_3.setText("")
        self.lineEdit_3.setObjectName("lineEdit_3")
        self.gridLayout.addWidget(self.lineEdit_3, 2, 3, 1, 1)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 800, 21))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        # self.new_window = QtWidgets.QDialog(self)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)
        self.pushButton.clicked.connect(self.open)

        # self.pushButton.clicked(self.detect_video())


    # def detect_video(self):
    #     cap = cv2.VideoCapture('170609_A_Delhi_058_2.mp4')
    #
    #     while (cap.isOpened()):
    #         ret, frame = cap.read()
    #         if not ret:
    #             break
    #
    #         frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #         img = QImage(frame, frame.shape[1], frame.shape[0], QImage.Format_RGB888)
    #         pix = QPixmap.fromImage(img)
    #         print(pix)
    #         pix = pix.scaled(600, 400)
    #         self.label.setPixmap(pix)
    #
    #         self.label.frame = pix  # or img depending what `ui.frame` needs
    #
    #         if cv2.waitKey(25) & 0xFF == ord('q'):
    #             break
    #
    #     cap.release()

    def open(self):
        filename = QFileDialog.getOpenFileName()
        print(filename)
        self.detection(filename[0])

        # return filename[0]

    # def activate(self):
    #
    #     print("dir")
    #     self.open()
    #     # return filename


    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Vehicle detection"))
        self.pushButton.setText(_translate("MainWindow", "Load"))
        self.label_3.setText(_translate("MainWindow", "cars"))
        self.label_2.setText(_translate("MainWindow", "FPS"))
        self.label_4.setText(_translate("MainWindow", "bus"))


    def save_data(self,no_of_time_hand_detected, no_of_time_hand_crossed):

        try:
            today = date.today()
            today=str(today)
            #loc = (r'C:\Users\rahul.tripathi\Desktop\result.xls')

            rb = xlrd.open_workbook('result.xls')
            sheet = rb.sheet_by_index(0)
            sheet.cell_value(0, 0)


            #print(sheet.nrows)
            q=sheet.cell_value(sheet.nrows-1,1)

            rb = xlrd.open_workbook('result.xls')
            #rb = xlrd.open_workbook(loc)
            wb=copy(rb)
            w_sheet=wb.get_sheet(0)

            if q==today:
                w=sheet.cell_value(sheet.nrows-1,2)
                e=sheet.cell_value(sheet.nrows-1,3)
                w_sheet.write(sheet.nrows-1,2,w+no_of_time_hand_detected)
                w_sheet.write(sheet.nrows-1,3,e+no_of_time_hand_crossed)
                wb.save('result.xls')
            else:
                w_sheet.write(sheet.nrows,0,sheet.nrows)
                w_sheet.write(sheet.nrows,1,today)
                w_sheet.write(sheet.nrows,2,no_of_time_hand_detected)
                w_sheet.write(sheet.nrows,3,no_of_time_hand_crossed)
                wb.save('result.xls')
        except FileNotFoundError:
            today = date.today()
            today=str(today)


            # Workbook is created
            wb = Workbook()

            # add_sheet is used to create sheet.
            sheet = wb.add_sheet('Sheet 1')

            sheet.write(0, 0, 'Sl.No')
            sheet.write(0, 1, 'Date')
            sheet.write(0, 2, 'Number of times hand detected')
            sheet.write(0, 3, 'Number of times hand crossed')
            m=1
            sheet.write(1, 0, m)
            sheet.write(1, 1, today)
            sheet.write(1, 2, no_of_time_hand_detected)
            sheet.write(1, 3, no_of_time_hand_crossed)

            wb.save('result.xls')

    # if __name__ == '__main__':
    def detection(self,file):
        # Detection confidence threshold to draw bounding box
        score_thresh = 0.5

        #vs = cv2.VideoCapture('rtsp://192.168.1.64')
        # vs = VideoStream(0).start()
        print(file)
        vs = cv2.VideoCapture(file)
        category_index = label_map_util.create_category_index_from_labelmap(r"C:\Darshan\ml\vehicle detection\vehicle\frozen_graphs\mscoco_label_map01.pbtxt", use_display_name=True)

        #Oriendtation of machine
        Orientation= 'bt'
        #input("Enter the orientation of hand progression ~ lr,rl,bt,tb :")

        #For Machine
        #Line_Perc1=float(input("Enter the percent of screen the line of machine :"))
        Line_Perc1=float(15)

        #For Safety
        #Line_Perc2=float(input("Enter the percent of screen for the line of safety :"))
        Line_Perc2=float(50)

        # max number of hands we want to detect/track
        num_hands_detect = 100

        # Used to calculate fps
        start_time = datetime.datetime.now()
        num_frames = 0

        im_height, im_width = (None, None)
        # cv2.namedWindow('Detection', cv2.WINDOW_NORMAL)
        def count_no_of_times(lst):
            x=y=cnt=0
            for i in lst:
                x=y
                y=i
                if x==0 and y==1:
                    cnt=cnt+1
            return cnt



        try:
            while(vs.isOpened()):
                rect, scene = vs.read()
                frame_org = vs.read()


                frame = np.array(frame_org[1:])
                # print(frame,"gddgdgg")
                # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                if im_height == None:

                    im_height, im_width = frame.shape[1:3]
                    print(im_height,im_width)

                # Convert image to rgb since opencv loads images in bgr, if not accuracy will decrease
                # try:
                #     frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # except Exception as e:
                #     print("Error converting to RGB",e)
                #cv2.line(img=frame, pt1=(0, Line_Position1), pt2=(frame.shape[1], Line_Position1), color=(255, 0, 0), thickness=2, lineType=8, shift=0)

                #cv2.line(img=frame, pt1=(0, Line_Position2), pt2=(frame.shape[1], Line_Position2), color=(255, 0, 0), thickness=2, lineType=8, shift=0)
                # print(frame.shape,"hi")
                # Run image through tensoqrflow graph
                boxes, scores, classes = detector_utils.detect_objects(
                    frame[0,:,:,:], detection_graph, sess)
                # print(boxes, scores, classes, "oooo")

                Line_Position2=orien_lines.drawsafelines(frame[0,:,:,:],Line_Perc1,Line_Perc2)

                a,b=detector_utils.draw_box_on_image(
                    num_hands_detect, score_thresh, scores, boxes, classes, im_width, im_height, scene,Line_Position2,Orientation)
                # lst1.append(a)
                # lst2.append(b)

                no_of_time_hand_detected=no_of_time_hand_crossed=0
                # Calculate Frames per second (FPS)
                num_frames += 1
                elapsed_time = (datetime.datetime.now() -
                                start_time).total_seconds()
                fps = num_frames / elapsed_time

                if args['display']:

                    # Display FPS on frame
                    # detector_utils.draw_text_on_image("FPS : " + str("{0:.2f}".format(fps)), scene)
                    # cap = cv2.VideoCapture('170609_A_Delhi_058_2.mp4')
                    #
                    # while (cap.isOpened()):
                    #     ret, frame = cap.read()
                    #     if not ret:
                    #         break

                    frame = cv2.cvtColor(scene, cv2.COLOR_BGR2RGB)
                    img = QImage(frame, frame.shape[1], frame.shape[0], QImage.Format_RGB888)
                    pix = QPixmap.fromImage(img)
                    print(pix)
                    pix = pix.scaled(600, 400)
                    self.label.setPixmap(pix)

                    self.label.frame = pix  # or img depending what `ui.frame` needs
                    self.lineEdit.setText(str(round(fps,2)))
                    self.lineEdit_2.setText(str(a))
                    self.lineEdit_3.setText(str(b))
                    if cv2.waitKey(25) & 0xFF == ord('q'):
                        break
                    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    # show = cv2.resize(scene, (960, 540))
                    # print(frame[0,:,:,:],"ppppp")q
                    # cv2.set(cv2.CAP_PROP_BUFFERSIZE, 2)
                    # cv2.waitKey(0)
                    # cv2.imshow('Detection', show)
                    # cv2.imshow('Detection', show)
                    # if cv2.waitKey(25) & 0xFF == ord('q'):
                    #     cv2.destroyAllWindows()
                    #     vs.stop()
                    #     break

            no_of_time_hand_detected=count_no_of_times(lst2)
            #no_of_time_hand_detected=b
            no_of_time_hand_crossed=count_no_of_times(lst1)
            #print(no_of_time_hand_detected)
            #print(no_of_time_hand_crossed)
            self.save_data(no_of_time_hand_detected, no_of_time_hand_crossed)
            print("Average FPS: ", str("{0:.2f}".format(fps)))
        except Exception as e:
            print(e)
            pass

        # #
        # except KeyboardInterrupt:
        #     no_of_time_hand_detected=count_no_of_times(lst2)
        #     no_of_time_hand_crossed=count_no_of_times(lst1)
        #     today = date.today()
        #     save_data(no_of_time_hand_detected, no_of_time_hand_crossed)

          #     print("Average FPS: ", str("{0:.2f}".format(fps)))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    MainWindow.show()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)

    sys.exit(app.exec_())
