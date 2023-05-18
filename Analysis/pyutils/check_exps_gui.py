# %% 

#!/usr/bin/python3
"""
GUI too loop over pngs in expfolders 
Up/Down -- swap between experiments 
Left/Right -- swap between pngs
in main you can use queryExp-like inputs to call the desired experiments

"""
import sys,glob
from pathlib import Path
pinkRig_path= glob.glob(r'C:\Users\*\Documents\Github\PinkRigs')
pinkRig_path = Path(pinkRig_path[0])
sys.path.insert(0, (pinkRig_path.__str__()))

# Qt imports 
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QMainWindow, QDesktopWidget, QApplication,QLabel,QWidget,QVBoxLayout
from PyQt5.QtGui import QPixmap
# PinkRig imports 
from Admin.csv_queryExp import queryCSV

def check_file(my_path,filenamestring):
    """
    function to check if file exists.
    """
    imagedat = list(Path(my_path).glob('*%s.png' % filenamestring))
    if len(imagedat)==1: 
        out_path = imagedat[0]
    else:
        out_path = pinkRig_path / r'Analysis/pyutils/nodat.png'
    
    return out_path
    
class Viewer(QMainWindow):

    def __init__(self,**kwargs):
        super().__init__()

        self.initUI(**kwargs)

    def initUI(self,**kwargs):

        self.resize(180, 380)
        self.center()
        self.setWindowTitle('Exp')
        self.load_dat(**kwargs)

        self.central_widget = QWidget()               
        self.setCentralWidget(self.central_widget)    
        self.lay = QVBoxLayout(self.central_widget)
        self.label = QLabel()
        self.lay.addWidget(self.label)
        self.show()
        self.show_image(self.images[self.curr_dataset][self.curr_image])

    def center(self):
        """centers the window on the screen"""
        screen = QDesktopWidget().screenGeometry()
        size = self.geometry()
        self.move(int((screen.width() - size.width()) / 2),
                  int((screen.height() - size.height()) / 2))

    def load_dat(self,**kwargs):
        recordings = queryCSV(**kwargs)
        self.expFolders = recordings.expFolder.values
        # search for jpegs & pngs within the folder
        self.datatypes = [
            'frontCam_alignment', 
            'eyeCam_alignment',
            'sideCam_alignment',
            'frontCam_roi_crop_and_mask',
            'eyeCam_roi_crop_and_mask',
            'sideCam_roi_crop_and_mask',
            'frontCam_audTriggeredMovement',
            'eyeCam_audTriggeredMovement',
            'sideCam_audTriggeredMovement'            
        ]

        self.images = [[check_file(exp,d) for d in self.datatypes] for exp in self.expFolders]
        self.curr_dataset = 0
        self.curr_image = 0

    def show_image(self,image_address):
        #self.hide()
        self.im = QPixmap(image_address.__str__())
        self.im = self.im.scaled(1000, 1700,Qt.KeepAspectRatio)
        self.label.setPixmap(self.im)
        self.setGeometry(50,50,320,200)
        #self.resize(180, 380)
        self.setWindowTitle('%s\%s' %(self.expFolders[self.curr_dataset],self.datatypes[self.curr_image]))
        self.label.update()
        self.update()

    def step_dataset(self,stepsize):
        n_datasets = len(self.images)
        self.curr_dataset = self.curr_dataset+stepsize
        if (self.curr_dataset==-1):
            self.curr_dataset = n_datasets-1
        elif (self.curr_dataset==n_datasets):
            self.curr_dataset = 0
        #self.curr_image=0

    def step_image(self,stepsize):
        n_images = len(self.images[self.curr_dataset])
        self.curr_image = self.curr_image+stepsize
        print(self.curr_image)
        if (self.curr_image==-1):
            self.curr_image = n_images-1
        elif (self.curr_image==n_images):
            self.curr_image = 0

    def keyPressEvent(self, event):
        """processes key press events"""
        key = event.key()
        if key == Qt.Key_Left:
            self.step_image(1)
            self.show_image(self.images[self.curr_dataset][self.curr_image])
        elif key == Qt.Key_Right:
            self.step_image(-1)
            self.show_image(self.images[self.curr_dataset][self.curr_image])
        elif key == Qt.Key_Down:
            self.step_dataset(1)
            self.show_image(self.images[self.curr_dataset][self.curr_image])
        elif key == Qt.Key_Up:
            self.step_dataset(-1)
            self.show_image(self.images[self.curr_dataset][self.curr_image])
       
def main(**kwargs):

    app = QApplication([])
    viewer = Viewer(**kwargs)
    sys.exit(app.exec_())

main(subject='AV030')

