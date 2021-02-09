from PyQt5 import QtWidgets, QtGui
from PyQt5.QtCore import Qt, QFileInfo, QTimer, QDateTime, QSize, QEventLoop
from PyQt5.QtGui import QImage, QPixmap, QPalette, QPainter
from PyQt5.QtPrintSupport import QPrintDialog, QPrinter
from PyQt5.QtWidgets import (QLabel, QSizePolicy, QScrollArea, QMessageBox, QMainWindow, QMenu, QTabWidget, QAction, QWidget, QTableWidget, 
    qApp, QFileDialog, QToolBar, QProgressBar, QProgressDialog, QGroupBox, QRadioButton, QCheckBox, QVBoxLayout, QHBoxLayout, QPushButton, QProgressDialog,
    QTextEdit, QLineEdit, QSpinBox, QDateTimeEdit, QSlider, QComboBox, QScrollBar, QDial, QGridLayout, QListView, QListWidget, QListWidgetItem)



class doc_main(QMainWindow):
    def __init__(self):
        super().__init__()

        self.printer = QPrinter()
        self.scaleFactor = 0.0
        self.selected_page = 0
        self.selected_document = ""

        self.createToolbar()
        self.createActions()
        self.createMenus()
        #self.createProgressBar()
        
        # statub Bar
        self.statusBar().showMessage("Ready")
        # self.setStatusBar(QStatusBar(self))      

        self.createLeftGroupBox()
        self.createRightGroupBox()
        
        #self.createProgressBar()

        self.mainLayout = QGridLayout()
        self.mainLayout.addWidget(self.topLeftGroupBox, 1, 0)
        self.mainLayout.addWidget(self.RightGroupBox, 1, 1)
        
        #mainLayout.addWidget(self.progressBar, 3, 0, 1, 2)
        
        self.mainLayout.setColumnStretch(0, 1)
        self.mainLayout.setColumnStretch(1, 1)

        wid = QtWidgets.QWidget(self)
        self.setCentralWidget(wid)
        wid.setLayout(self.mainLayout)

        self.setWindowTitle("Jabil DHR Document Inspector 1.0")
        self.showMaximized()
        #self.resize(1280, 680)

        self.load_video_feed()
   

#########  END OF MAINDHR CLASS AND DEFINITION ##########




########################################################################
##############            UI LAYOUT DEFINITIONS  #######################
########################################################################
            

    def createLeftGroupBox(self):
        self.topLeftGroupBox = QGroupBox("Place DHR Document")
        
        #doc_list_old = ["TJ-COC-2019-0142", "TJ-VF-2005-0189", "TJ-VF-2020-3636", "TJ-VF-2019-3568"]
        #doc_list = self.load_template_names()        
        self.numDocLabel = QLabel()
        self.numDocLabel.setText("# Documento Detectado: ")

        self.videoLabel = QLabel()
        self.videoLabel.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        #self.videoLabel.resize(640, 480)
        #self.videoLabel.move(280,120)                
               
        # define INSPECT BUTTON
        self.inspectButton = QPushButton("Click button to Inspect Document", self)
        self.inspectButton.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        #self.inspectButton.resize(300,200)
        #self.inspectButton.setDefault(True)     
        self.inspectButton.clicked.connect(self.inspect_document)

        # define layout and assignet to LeftGroupBox
        layout = QVBoxLayout()        
        #layout.addStretch(1)             
        
        layout.addWidget(self.videoLabel)
        layout.addWidget(self.numDocLabel)        
        layout.addWidget(self.inspectButton)

        self.topLeftGroupBox.setLayout(layout)    



    def createRightGroupBox(self):
        self.RightGroupBox = QGroupBox("DHR Document Display")
        

        self.imageLabel = QLabel()
        self.imageLabel.setBackgroundRole(QPalette.Base)
        self.imageLabel.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.imageLabel.setScaledContents(True)

        self.scrollArea = QScrollArea()
        self.scrollArea.setBackgroundRole(QPalette.Dark)
        self.scrollArea.setWidget(self.imageLabel)
        self.scrollArea.setVisible(False)
        
        layout = QVBoxLayout()
        layout.addWidget(self.scrollArea)
        #layout.addStretch(1)
        self.RightGroupBox.setLayout(layout)
       


########################################################################
#########   START OF Definition of WIDGETS Components   ################
########################################################################

    def displayMessageBox(self, msgtext):
        QMessageBox.information(self, "[INFO] ", msgtext)

    def createProgressBar(self):
        self.progressBar = QProgressBar()
        self.progressBar.setRange(0, 10000)
        self.progressBar.setValue(0)

        timer = QTimer(self)
        timer.timeout.connect(self.advanceProgressBar)
        timer.start(1000)

    def advanceProgressBar(self):
        curVal = self.progressBar.value()
        maxVal = self.progressBar.maximum()
        self.progressBar.setValue(curVal + (maxVal - curVal) / 100)
             


    def print_(self):
        dialog = QPrintDialog(self.printer, self)
        if dialog.exec_():
            painter = QPainter(self.printer)
            rect = painter.viewport()
            size = self.imageLabel.pixmap().size()
            size.scale(rect.size(), Qt.KeepAspectRatio)
            painter.setViewport(rect.x(), rect.y(), size.width(), size.height())
            painter.setWindow(self.imageLabel.pixmap().rect())
            painter.drawPixmap(0, 0, self.imageLabel.pixmap())

    def zoomIn(self):
        self.scaleImage(1.25)

    def zoomOut(self):
        self.scaleImage(0.8)

    def normalSize(self):
        self.imageLabel.adjustSize()
        self.scaleFactor = 1.0

    def fitToWindow(self):
        fitToWindow = self.fitToWindowAct.isChecked()
        self.scrollArea.setWidgetResizable(fitToWindow)
        if not fitToWindow:
            self.normalSize()

        self.updateActions()

    def about(self):
        QMessageBox.about(self, "About Jabil DHR ML",
                          "<p>The <b>Jabil DHR Document ML Viewer</b> is a Machine Learning application "
                          "designed and developed using Artificial Intlligence technology to visually inspect "
                          "any DHR document generated in the Jabil manufacturing process in order to reduce poor documentation practices "
                          "due to typos, missing or incorrent information on DHR documents."                          
                          "<p>(Jabil DHR ML), can be customized to inspect any type of document "
                          "to comply with Jabil Good Documentation Practices requirements.</p>"
                          "<p>New feaures are permanently under development by Jabil QA Department (JHB3)."
                          "</p>")

    def createActions(self):
        self.openAct = QAction("&Open ...", self, shortcut="Ctrl+O", triggered=self.open)
        self.printAct = QAction("&Print...", self, shortcut="Ctrl+P", enabled=False, triggered=self.print_)
        self.exitAct = QAction("E&xit", self, shortcut="Ctrl+Q", triggered=self.close)
        self.zoomInAct = QAction("Zoom &In (25%)", self, shortcut="Ctrl++", enabled=False, triggered=self.zoomIn)
        self.zoomOutAct = QAction("Zoom &Out (25%)", self, shortcut="Ctrl+-", enabled=False, triggered=self.zoomOut)
        self.normalSizeAct = QAction("&Normal Size", self, shortcut="Ctrl+S", enabled=False, triggered=self.normalSize)
        self.fitToWindowAct = QAction("&Fit to Window", self, enabled=False, checkable=True, shortcut="Ctrl+F",
                                      triggered=self.fitToWindow)
        self.aboutAct = QAction("&About Jabil DHR ML", self, triggered=self.about)


    def createToolbar(self):
         # main Toolbar definition
        self.toolbar = QToolBar("DHR ML Toolbar")
        #toolbar.setIconSize(QSize(16,16))
        self.addToolBar(self.toolbar)        

        icono = QtGui.QIcon("blue-document-arrow.png")
        button_action = QAction(icono, "Import DHR", self)
        button_action.setStatusTip("Import DHR Document")
        button_action.triggered.connect(self.open)
        button_action.setCheckable(False)
        self.toolbar.addAction(button_action)        
                
        icono_zoomIn = QtGui.QIcon("binocular-plus.png")
        button_action_ZoomIn = QAction(icono_zoomIn, "Zoom In", self, triggered=self.zoomIn)        
        button_action_ZoomIn.setStatusTip("Zoom In document")
        button_action_ZoomIn.setCheckable(False)
        self.toolbar.addAction(button_action_ZoomIn)

        icono_zoomOut = QtGui.QIcon("binocular-minus.png")
        button_action_ZoomOut = QAction(icono_zoomOut, "Zoom Out", self, triggered=self.zoomOut)        
        button_action_ZoomOut.setStatusTip("Zoom Out document")
        button_action_ZoomOut.setCheckable(False)
        self.toolbar.addAction(button_action_ZoomOut)

        


    def createMenus(self):
        self.fileMenu = QMenu("&File", self)
        self.fileMenu.addAction(self.openAct)
        self.fileMenu.addAction(self.printAct)
        self.fileMenu.addSeparator()
        self.fileMenu.addAction(self.exitAct)

        self.viewMenu = QMenu("&View", self)
        self.viewMenu.addAction(self.zoomInAct)
        self.viewMenu.addAction(self.zoomOutAct)
        self.viewMenu.addAction(self.normalSizeAct)
        self.viewMenu.addSeparator()
        self.viewMenu.addAction(self.fitToWindowAct)

        self.helpMenu = QMenu("&Help", self)
        self.helpMenu.addAction(self.aboutAct)        

        self.menuBar().addMenu(self.fileMenu)
        self.menuBar().addMenu(self.viewMenu)
        self.menuBar().addMenu(self.helpMenu)

    def updateActions(self):
        self.zoomInAct.setEnabled(not self.fitToWindowAct.isChecked())
        self.zoomOutAct.setEnabled(not self.fitToWindowAct.isChecked())
        self.normalSizeAct.setEnabled(not self.fitToWindowAct.isChecked())

    def scaleImage(self, factor):
        self.scaleFactor *= factor
        self.imageLabel.resize(self.scaleFactor * self.imageLabel.pixmap().size())

        self.adjustScrollBar(self.scrollArea.horizontalScrollBar(), factor)
        self.adjustScrollBar(self.scrollArea.verticalScrollBar(), factor)

        self.zoomInAct.setEnabled(self.scaleFactor < 3.0)
        self.zoomOutAct.setEnabled(self.scaleFactor > 0.333)

    def adjustScrollBar(self, scrollBar, factor):
        scrollBar.setValue(int(factor * scrollBar.value()
                               + ((factor - 1) * scrollBar.pageStep() / 2)))


########################################################################
#############   METHODS (HELPERS) LOGIC FUNCTIONALITY   ################
########################################################################

      # open PDF documents and import it into the ImageListView
    def open(self):
        options = QFileDialog.Options()
        # fileName = QFileDialog.getOpenFileName(self, "Open File", QDir.currentPath())
        fileName, _ = QFileDialog.getOpenFileName(self, 'QFileDialog.getOpenFileName()', '',
                                                  'PDF (*.pdf)', options=options)
        #fileName, _ = QFileDialog.getOpenFileName(self, 'QFileDialog.getOpenFileName()', '',
        #                                          'Images (*.png *.jpeg *.jpg *.bmp *.gif)', options=options)
        
        if fileName:           
            
            f = QFileInfo(fileName).fileName()
            print("FILENAME: " + f)                     

                                            
            # 2) LOAD  IMAGE(s) from OS Directory
            import os
            #os.startfile() #code to launch an application                                
            files = []        
            for file in os.listdir("."):
                if file.startswith(f[:-5]) and file.endswith(".jpeg"):
                    files.append(os.path.join(os.getcwd(), file))

            total_files = len(files)
            files.sort()       


    def inspect_document(self):
        import cv2
        import imutils
        from imutils.perspective import four_point_transform
        from skimage import data
        from skimage.filters import threshold_local
        # custom tools
        from doc_scanner import scan_doc
        from doc_header import get_header_data


        # Version 2.0 CODE        
        
        # STEP 3: Perspective transform
        ratio = 1
        warped = four_point_transform(self.frame, self.screenCnt.reshape(4,2) * ratio)
        # convert the warped image to grayscale, then threshold it to give it 'black and white' paper effect
        warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        warped = cv2.resize(warped, (768, 1024), interpolation=cv2.INTER_CUBIC)  #extra
        #T = threshold_local(warped, 11, offset = 4, method = "gaussian")
        #warped = (warped > T).astype("uint16") * 255

        # save and display scanned image
        cv2.imwrite("IMG_SCANEADA.jpeg", warped, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
        image = QImage("IMG_SCANEADA.jpeg")

        self.imageLabel.setPixmap(QPixmap.fromImage(image))
        self.scaleFactor = 1.0
        self.scrollArea.setVisible(True)
        self.printAct.setEnabled(True)
        self.fitToWindowAct.setEnabled(True)
        self.updateActions()


        #  capture and saves IMAGEN from VIDEO FEED
        #cv2.imwrite("imagen_capturada.jpeg", self.frame)
        
        # Clean image, prepare it for OCR (Scanned)
        #scan_doc("imagen_capturada.jpeg")

        # Get DOC HEADER INFORMATION
        '''
        doc_header_data = get_header_data("imagen_scaneada.jpeg")
        print("NOMBRE DOCUMENTO: " + doc_header_data[0].docname.strip())

        if len(doc_header_data[0].docname.strip()) < 15: 
            self.numDocLabel.setText("# Documento Detectado: NO DISPONIBLE")
            self.displayMessageBox("[INFO] No. de Documento no legible, intente tomar nueva imagen.")
            return False

        else:
            # Si el numero de documento es LEGIBLE, ejecuta la deteccion de espacios en blanco.
            # Detect Blanks from SCANNED image
            clean_docname = doc_header_data[0].docname.strip()
            self.numDocLabel.setText("# Documento Detectado: " + clean_docname)
            self.numDocLabel.repaint()
            temp_filename = self.detect_blanks("imagen_scaneada.jpeg", clean_docname, "1")
            #temp_filename = self.detect_blanks("imagen_scaneada.jpeg", "TJ-VF-2005-0198", "1") 
            
            print("temp_filename: " + temp_filename)

            if temp_filename != "NADA":
                # Desplegar la imagen con su resultado
                image = QImage(temp_filename) 
                if image.isNull():
                    QMessageBox.information(self, "Image Viewer", "Cannot load %s." % temp_filename)
                    return

                self.imageLabel.setPixmap(QPixmap.fromImage(image))
                self.scaleFactor = 1.0
                self.scrollArea.setVisible(True)
                self.printAct.setEnabled(True)
                self.fitToWindowAct.setEnabled(True)
                self.updateActions()

                if not self.fitToWindowAct.isChecked():
                    self.imageLabel.adjustSize()

                return True
            else:
                # No se despliega nada.
                return False
                '''
        return True
    
    # Loads Template Names from COOORDS_TEMPLATE DB
    def load_template_names(self):
        from sqlalchemy import create_engine

        import psycopg2 
        import json
        import pandas as pd

        from collections import namedtuple

        # LOAD DATA FROM SQL
        engine = create_engine('postgresql://postgres:Halend2009@localhost/DHRML_DB')
        conn = psycopg2.connect("host=localhost dbname=DHRML_DB user=postgres password=Halend2009")        
        templates_sql_table = 'select doc_name from public.coords_template group by doc_name order by doc_name'  

        with engine.connect() as conn, conn.begin():
            templates_data = pd.read_sql_query(templates_sql_table, conn)           
        
        
        templates_list = templates_data['doc_name'].tolist()
        print ("templates list:")
        print (templates_list)

        return templates_list
    

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_F3:
            print("Space key pressed")
            self.displayMessageBox("Tecla Space pressed !")


    #################################################    
    # loads VIDEO FEED from Web Camera
    def load_video_feed(self):
        import cv2
        import sys
        import imutils
        from imutils.perspective import four_point_transform
        from skimage import data
        from skimage.filters import threshold_local
               

        #cv2.namedWindow("video_preview")
        vc = cv2.VideoCapture(1)  # 0-Camara web integrada /  1-Camara web externa BRIO 8MP
       
        # max settings for Logitech BRIO (8.1 MB) - VERTICAL
        vc.set(cv2.CAP_PROP_FRAME_WIDTH, 2160)
        vc.set(cv2.CAP_PROP_FRAME_HEIGHT, 3840)

        # Asjust CAMERA settings
        # Change the camera setting using the set() function
        # cap.set(cv2.cv.CV_CAP_PROP_EXPOSURE, -6.0)
        # cap.set(cv2.cv.CV_CAP_PROP_GAIN, 4.0)
        # cap.set(cv2.cv.CV_CAP_PROP_BRIGHTNESS, 144.0)
        # cap.set(cv2.cv.CV_CAP_PROP_CONTRAST, 27.0)
        # cap.set(cv2.cv.CV_CAP_PROP_HUE, 13.0) # 13.0
        # cap.set(cv2.cv.CV_CAP_PROP_SATURATION, 28.0)

        # Read the current setting from the camera        
        test = vc.get(cv2.CAP_PROP_POS_MSEC)
        ratio = vc.get(cv2.CAP_PROP_POS_AVI_RATIO)
        frame_rate = vc.get(cv2.CAP_PROP_FPS)
        width = vc.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = vc.get(cv2.CAP_PROP_FRAME_HEIGHT)
        brightness = vc.get(cv2.CAP_PROP_BRIGHTNESS)
        contrast = vc.get(cv2.CAP_PROP_CONTRAST)
        saturation = vc.get(cv2.CAP_PROP_SATURATION)
        hue = vc.get(cv2.CAP_PROP_HUE)
        gain = vc.get(cv2.CAP_PROP_GAIN)
        exposure = vc.get(cv2.CAP_PROP_EXPOSURE)
        print("Test: ", test)
        print("Ratio: ", ratio)
        print("Frame Rate: ", frame_rate)
        print("Height: ", height)
        print("Width: ", width)
        print("Brightness: ", brightness)
        print("Contrast: ", contrast)
        print("Saturation: ", saturation)
        print("Hue: ", hue)
        print("Gain: ", gain)
        print("Exposure: ", exposure)
        


        if vc.isOpened(): # try to get the first frame
            rval, self.frame = vc.read()
        else:
            rval = False
            print("No camera available!")
                        

        img_id = 0
        while rval:                
            
            rval, self.frame = vc.read()            

            # Countour actions
            # STEP 1: EDGE Detection - Detecting Edges in Document    
            gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (5,5), 0)          
            edged = cv2.Canny(gray, 75, 200)

            # STEP 2: Document Contour - Detecting Document Contours
            cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            cnts = imutils.grab_contours(cnts)
            cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:5]            

            # Check is the countours are closed
            for c in cnts:
                peri = cv2.arcLength(c, True)
                approx = cv2.approxPolyDP(c, 0.02 * peri, True)

                if len(approx) == 4:
                    self.screenCnt = approx
                    cv2.drawContours(self.frame, [self.screenCnt], -1, (0, 255, 0), 2)
                    break    



             # Convert FRAME to a QTImage object
            rgbImage = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgbImage.shape
            bytesPerLine = ch * w
            convertToQtFormat = QImage(rgbImage.data, w, h, bytesPerLine, QImage.Format_RGB888)
            self.p = convertToQtFormat.scaled(1080, 1920, Qt.KeepAspectRatio)
            self.videoLabel.setPixmap(QPixmap.fromImage(self.p))
            
           #input actions
            key = cv2.waitKey(20)
            if key == 27: #exit on ESC
                break
            elif key == 99: # C letter
                img_id = img_id + 1
        
                # STEP 3: Perspective transform
                ratio = 1
                warped = four_point_transform(self.frame, screenCnt.reshape(4,2) * ratio)
                # convert the warped image to grayscale, then threshold it to give it 'black and white' paper effect
                warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
                T = threshold_local(warped, 11, offset = 10, method = "gaussian")
                warped = (warped > T).astype("uint8") * 255


                cv2.imwrite('IMG_' + str(img_id)  + '.jpeg', warped)
                cv2.imshow("img1", warped)

        vc.release()
        #cv2.destroyWindow("video_preview")
 

    # Load COORDS TEMPLATES  from DATA Server
    def import_template_locations(self, docname, docpage):
        from sqlalchemy import create_engine

        import psycopg2 
        import json
        import pandas as pd

        from collections import namedtuple

        # LOAD DATA FROM SQL
        engine = create_engine('postgresql://postgres:Halend2009@localhost/DHRML_DB')
        conn = psycopg2.connect("host=localhost dbname=DHRML_DB user=postgres password=Halend2009")
        #dhrml_sql_table = 'coords_template'
        dhrml_sql_table = 'select * from public.coords_template where doc_name = %(num_doc)s'  

        with engine.connect() as conn, conn.begin():
            dhrml_data = pd.read_sql_query(dhrml_sql_table, conn, params={'num_doc': str(docname)})
            #dhrml_data = pd.read_sql_table(dhrml_sql_table, conn)

        print("dhrml_data:")
        print(dhrml_data) 

        if dhrml_data.empty:
            self.displayMessageBox("[INFO] No hay coordenadas definidas para documento detectado.")
            self.OCR_Locations = []
            return False
        else:
            # PRE PROCESS DATA
            # Filter DHRML data LOCATIONS by Document Template Name and PAGE
                   
            dhr_template_locations = dhrml_data[dhrml_data["doc_page"].astype('int32') == int(docpage)]
            #dhr_template_locations = dhrml_data[(dhrml_data["doc_name"] == self.combo_docname.currentText())]
            locations_list = dhr_template_locations.values.tolist()
            print ("localions_list:")
            print (locations_list)
            #  Define OCR Location TUPLE
            OCRLocation = namedtuple("OCRLocation", ["id", "bbox", "filter_keywords"])
            # Asign DHR Template LOCATIONS to the OCR_Locations list
            self.OCR_Locations = []
            count = 0
            while count < len(locations_list):
                self.OCR_Locations.append(OCRLocation(locations_list[count][5], (locations_list[count][0], locations_list[count][1], locations_list[count][2], locations_list[count][3]), ["solo", ]))
                count = count + 1

            print ("Locations Results \n")
            print(self.OCR_Locations)
            return True



    # This function will detect EMPTY FIELDS on the provided image DHR Document, it will be compared to its TEMPLATE
    def detect_blanks(self, docimagen, docname, docpage):
        from collections import namedtuple
        import pytesseract
        import argparse
        import imutils
        import cv2
        #import align_images as am

        pytesseract.pytesseract.tesseract_cmd = r'C:\Users\2638986\AppData\Local\Tesseract-OCR\tesseract.exe'
        OCRLocation = namedtuple("OCRLocation", ["id", "bbox", "filter_keywords"])

    # define the locations of each area oh the document we with to OCR
           
        self.import_template_locations(docname, docpage)
        # verifica si hay coordenadas en el documento
        #self.import_template_locations("TJ-VF-2005-0198","1")

        # load the input image and template from disk
        print ("[INFO] loading images...")        
        self.statusBar().showMessage("Loading document...")   
        image = cv2.imread(docimagen)             
        self.statusBar().showMessage("Aligning document...")          
        aligned = image # skip align 

        # Initilize a results list to store the document OCR parsing results
        print ("[INFO] OCR'ing document...")        
        self.statusBar().showMessage("Inspecting document...")   
        parsingResults = []
        emptyField = "blank field"

        # loop over the locations of the document we are going to OCR
        for loc in self.OCR_Locations:
            # extract the OCR ROI from the ailgned image
            (x, y, w, h) = loc.bbox
            roi = aligned[y:y + h, x:x + w]

            # OCR the ROI using Tesseract
            config = ('-l eng --oem 1 --psm 11')  # Config OCR Parameters    
            rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
            #text = pytesseract.image_to_string(rgb, config=config)
            text = pytesseract.image_to_boxes(rgb, config=config)

            parsingResults.append((loc, text))

        results = {}

        # Loop over the results of parsing the documents
        for (loc, line) in parsingResults:
            # grab any existing OCR result for the current ID of the document
            r = results.get(loc.id, None)

            # if the results is None, initialize it using the text and locations named Tuple (converting it to a dictionary as namedtuples are not hastable)
            if r is None:
                results[loc.id] = (line, loc._asdict())

            # otherwise , there exists a OCR result for the current area of the document , so we append out existing line
            else:
                # unpack the existing OCR result and append line to existing text
                (existingText, loc) = r
                text = "{}\n{}".format(existingText, line)

                # update our result dictionary
                results[loc["id"]] = (text, loc)


        # Visualize OCR results
        for (locID, result) in results.items():
            # unpack the results tupple
            (text, loc) = result

            # display OCR results to terminal
            print(loc["id"])
            print("=" * len(loc["id"]))
            print("{}\n\n".format(text))

            # extract the bounding box coordinates of the OCR location
            (x, y, w, h) = loc["bbox"]

            if len(text) <= 0:
                # draw a boundind box around the text
                cv2.rectangle(aligned, (x, y), (x + w, y + h), (0, 255, 0), 2)    
                # draw a empty field label on each blank field
                cv2.putText(aligned, emptyField, (x, y + 12), cv2.FONT_HERSHEY_SIMPLEX, .9, (0, 0, 255), 2)


        # generate tenporal image file
        temporal_filename = "temporal.jpeg"
        cv2.imwrite(temporal_filename, aligned)              
        return temporal_filename


# This code will call the MAIN CLASS TO EXECUTE START THE APPLICATION
# It creates the QApplication Widget to call and Execute the MainDHR Class
# 
if __name__ == '__main__':
    import sys
    from PyQt5.QtWidgets import QApplication

    app = QApplication(sys.argv)
    docViewer = doc_main()
    docViewer.show()
    sys.exit(app.exec_())
    # TODO QScrollArea support mouse
    # base on https://github.com/baoboa/pyqt5/blob/master/examples/widgets/imageviewer.py
    #
    # if you need Two Image Synchronous Scrolling in the window by PyQt5 and Python 3
    # please visit https://gist.github.com/acbetter/e7d0c600fdc0865f4b0ee05a17b858f2