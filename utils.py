from __init__ import *

label_to_id = {
    'BACKGROUND': 0,
    'MPSI': 1,
    'MPSO': 2,
    'MVEN': 3,
    'SAT': 4,
    'VAT': 5,
}

cmap = np.array(
    [
        (0, 0, 0),
        (255, 255, 0),
        (0, 205, 0),
        (72, 118, 255),
        (0, 0, 139),
        (255, 0, 0),
    ],
    dtype=np.uint8,
)

def read_dcm(dcm_dir):
    reader = sitk.ImageSeriesReader()
    img_name = reader.GetGDCMSeriesFileNames(dcm_dir)
    reader.SetFileNames(img_name)
    image = reader.Execute()
    return image

class MyThread(QThread):
    signalForText = pyqtSignal(str)

    def __init__(self, data=None, parent=None):
        super(MyThread, self).__init__(parent)
        self.data = data

    def write(self, text):
        self.signalForText.emit(str(text))  # 发射信号

    def run(self):
        log = os.popen(self.data)
        print(log.read())