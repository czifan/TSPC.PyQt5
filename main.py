from __init__ import *
from modules import *
from utils import *
from nnunet_utils import *

class SetupWindow(QtWidgets.QMainWindow):
    def __init__(self,
                 title="TSPC",
                 logo_file="Icons/logo.png",
                 window_scale_ratio=0.9,
                 base_width=1750,
                 base_height=975,
                 folder="DATASET/nnUNet_trained_models/nnUNet/2d/Task100_fat/nnUNetTrainerV2__nnUNetPlansv2.1",
                 folds=[4,],
                 checkpoint_name="model_last"):
        super().__init__()

        self.title = title
        self.logo_file = logo_file
        self.base_width = base_width
        self.base_height = base_height
        self.window_scale_ratio = window_scale_ratio
        self.folder = folder
        self.folds = folds
        self.checkpoint_name = checkpoint_name
        self.single_view = False

        self._initParams()
        self._initUI()
        self._initSegor()

        self.thread = MyThread()
        self.thread.signalForText.connect(self._onUpdateText)
        sys.stdout = self.thread

    def _onUpdateText(self,text):
        cursor = self.Console.textCursor()
        cursor.movePosition(QTextCursor.End)
        cursor.insertText(text)
        self.Console.setTextCursor(cursor)
        self.Console.ensureCursorVisible()

    def _initSegor(self):
        self.segmentor = Segmentor(self.folder, self.folds, self.checkpoint_name)

    def _initParams(self):
        desktop = QApplication.desktop()
        self.desktop_width = desktop.width()
        self.desktop_height = desktop.height()

        self.width = int(self.desktop_width * self.window_scale_ratio)
        self.height = int(self.desktop_height * self.window_scale_ratio)
        self.scale_ratio = min(1.0*self.width/self.base_width,
                               1.0*self.height/self.base_height)
        
        self.cacheDir = "Caches"
        if os.path.isdir(self.cacheDir):
            shutil.rmtree(self.cacheDir)
        self.cacheInputDir = os.path.join(self.cacheDir, "Inputs")
        self.cacheOutputDir = os.path.join(self.cacheDir, "Outputs")
        os.makedirs(self.cacheInputDir, exist_ok=True)
        os.makedirs(self.cacheOutputDir, exist_ok=True)

        self.window_level_value = 60
        self.window_width_value = 180

    # Views
    def centerOnScreen(self):
        resolution = QDesktopWidget().screenGeometry()
        self.move(int((resolution.width() / 2) - (self.width / 2)),
                int((resolution.height() / 2) - (self.height / 2)) - 40)

    def _initUI(self, style_file="styles.qss"):
        self.setWindowTitle(self.title)
        self.setWindowIcon(QIcon(self.logo_file))
        self.setGeometry(0, 0, self.width, self.height)
        self.centerOnScreen()

        self._initUI_MenuBar()
        self._initUI_Selector()
        self._initUI_CTView()
        self._initUI_Console()
        self._initUI_ParamView()

        try:
            with open(style_file, 'r') as fh:
                self.setStyleSheet(fh.read())
        except:
            print("Can't load custom stylesheet.")

    def _initUI_MenuBar(self):
        self.MenubarWidth = self.width
        self.MenubarHeight = int(0.026*self.height)

        self.Menubar = QtWidgets.QMenuBar(self)
        self.Menubar.setGeometry(QtCore.QRect(0, 0, self.MenubarWidth, self.MenubarHeight))
        self.setMenuBar(self.Menubar)

        # File
        self.MenubarFile = QtWidgets.QMenu("File", self.Menubar)
        
        self.AopenFile = QtWidgets.QAction("Open `.nii.gz` file ...", self)
        self.AopenFile.triggered.connect(self._openNiiFileFunc)
        self.MenubarFile.addAction(self.AopenFile)
        
        self.AopenDir = QtWidgets.QAction("Open `.dicom` file ...", self)
        self.AopenDir.triggered.connect(self._openDicomDirFunc)
        self.MenubarFile.addAction(self.AopenDir)
        
        self.Menubar.addAction(self.MenubarFile.menuAction())

        # Save
        self.MenubarSave = QtWidgets.QMenu("Save", self.Menubar)

        self.AsaveSeg = QtWidgets.QAction("Save segmentation as ...", self)
        self.AsaveSeg.triggered.connect(self._saveSegFunc)
        self.MenubarSave.addAction(self.AsaveSeg)

        self.AsaveExcel = QtWidgets.QAction("Save excel as ...", self)
        self.AsaveExcel.triggered.connect(self._saveExcelFunc)
        self.MenubarSave.addAction(self.AsaveExcel)

        self.AsaveLog = QtWidgets.QAction("Save log as ...", self)
        self.AsaveLog.triggered.connect(self._saveLogFunc)
        self.MenubarSave.addAction(self.AsaveLog)

        self.Menubar.addAction(self.MenubarSave.menuAction())

        # Help
        self.MenubarHelp = QtWidgets.QMenu("Help", self.Menubar)

        self.AhelpAbout = QtWidgets.QAction(f"About {self.title}", self)
        self.AhelpAbout.triggered.connect(self._helpAboutFunc)
        self.MenubarHelp.addAction(self.AhelpAbout)

        self.AhelpDoc = QtWidgets.QAction(f"Documentation", self)
        self.AhelpDoc.triggered.connect(self._helpDocFunc)
        self.MenubarHelp.addAction(self.AhelpDoc)

        self.Menubar.addAction(self.MenubarHelp.menuAction())

    def _initUI_Selector(self):
        self.WSMargin = int(0.006*self.width)
        self.WSWidth = int(0.109*self.width)
        self.WSHeight = int(0.103*self.height)

        qp = QPainter()
        qp.begin(self)
        pen = QPen(Qt.black, 10, Qt.SolidLine)

        self.WSlevel = WindowSlider(self,
                                    title="Win.(Level)",
                                    minimum=-1000,
                                    maximum=1000,
                                    single_step=1,
                                    value=self.window_level_value,
                                    interval=50,
                                    add_slider=True)
        self.WSlevel.Slider.valueChanged.connect(self._windowLevelChangedFunc)
        self.WSlevel.resize(self.WSWidth, self.WSHeight)
        self.WSlevel.setGeometry(self.WSMargin, 2*self.WSMargin, self.WSWidth, self.WSHeight)

        self.WSwidth = WindowSlider(self,
                                    title="Win.(Width)",
                                    minimum=0,
                                    maximum=2000,
                                    single_step=1,
                                    value=self.window_width_value,
                                    interval=50,
                                    add_slider=True)
        self.WSwidth.Slider.valueChanged.connect(self._windowWidthChangedFunc)
        self.WSwidth.resize(self.WSWidth, self.WSHeight)
        self.WSwidth.setGeometry(self.WSMargin, 2*self.WSMargin+(self.WSMargin+self.WSHeight), self.WSWidth, self.WSHeight)

        self.WStraSlice = WindowSlider(self,
                                    title="Tra. Sli.",
                                    minimum=0,
                                    maximum=100,
                                    single_step=1,
                                    value=1,
                                    interval=4,
                                    add_slider=True)
        self.WStraSlice.Slider.valueChanged.connect(self._windowTraSliChangedFunc)
        self.WStraSlice.resize(self.WSWidth, self.WSHeight)
        self.WStraSlice.setGeometry(self.WSMargin, 2*self.WSMargin+2*(self.WSMargin+self.WSHeight), self.WSWidth, self.WSHeight)

        self.WSsagSlice = WindowSlider(self,
                                    title="Sag. Sli.",
                                    minimum=0,
                                    maximum=500,
                                    single_step=1,
                                    value=1,
                                    interval=20,
                                    add_slider=True)
        self.WSsagSlice.Slider.valueChanged.connect(self._windowSagSliChangedFunc)
        self.WSsagSlice.resize(self.WSWidth, self.WSHeight)
        self.WSsagSlice.setGeometry(self.WSMargin, 2*self.WSMargin+3*(self.WSMargin+self.WSHeight), self.WSWidth, self.WSHeight)

        self.WScorSlice = WindowSlider(self,
                                    title="Cor. Sli.",
                                    minimum=0,
                                    maximum=500,
                                    single_step=1,
                                    value=1,
                                    interval=20,
                                    add_slider=True)
        self.WScorSlice.Slider.valueChanged.connect(self._windowCorSliChangedFunc)
        self.WScorSlice.resize(self.WSWidth, self.WSHeight)
        self.WScorSlice.setGeometry(self.WSMargin, 2*self.WSMargin+4*(self.WSMargin+self.WSHeight), self.WSWidth, self.WSHeight)

        self.WSheight = WindowSlider(self,
                                    title="Height(cm)",
                                    minimum=0,
                                    maximum=250,
                                    single_step=1,
                                    value=170,
                                    interval=1,
                                    add_slider=True)
        self.WSheight.resize(self.WSWidth, self.WSHeight)
        self.WSheight.setGeometry(self.WSMargin, 2*self.WSMargin+5*(self.WSMargin+self.WSHeight), self.WSWidth, self.WSHeight)

        self.LSselector = LabelSelector(self, 
                                        title="Label Selector",
                                        labels=["MPSI", "MPSO", "MVEN", "SAT", "VAT"])
        self.LSselector.resize(self.WSWidth, self.WSHeight*2)
        self.LSselector.setGeometry(self.WSMargin, 2*self.WSMargin+6*(self.WSMargin+self.WSHeight), self.WSWidth, self.WSHeight*2)
        for Button in self.LSselector.Buttons:
            Button.toggled.connect(self._showLabelChangedFunc)
        #self.LSselector.BshowImage.toggled.connect(self._showImageChangedFunc)

        self.Bgenerate = QPushButton("Generate", self)
        self.Bgenerate.setGeometry(self.WSMargin, 3*self.WSMargin+7*(self.WSMargin+self.WSHeight)+self.WSHeight, self.WSWidth, self.WSHeight//3)
        self.Bgenerate.clicked.connect(self._generateSegFunc)

    def _initUI_CTView(self):
        self.CTVLeftMargin = 2*self.WSMargin+self.WSWidth+int(0.006*self.width)
        self.CTVTopMargin = int(0.026*self.height)
        self.CTVWidth = int(0.360*self.width)
        self.CTVHeight = int(0.380*self.height)
        self.CTVBWidth = int(0.026*self.height)
        self.CTVBHeight = int(0.026*self.height)

        self.SLtra = SuperQLabel(self)
        self.SLtra.setGeometry(self.CTVLeftMargin, self.CTVTopMargin, self.CTVWidth, self.CTVHeight)
        self.SLtra.setStyleSheet('background-color: black')
        self.SLtra.setObjectName('SLtra')

        self.SLsag = SuperQLabel(self)
        self.SLsag.setGeometry(self.CTVLeftMargin+self.CTVWidth, self.CTVTopMargin, self.CTVWidth, self.CTVHeight)
        self.SLsag.setStyleSheet('background-color: black')
        self.SLsag.setObjectName('SLsag')

        self.SLcor = SuperQLabel(self)
        self.SLcor.setGeometry(self.CTVLeftMargin, self.CTVTopMargin+self.CTVHeight, self.CTVWidth, self.CTVHeight)
        self.SLcor.setStyleSheet('background-color: black')
        self.SLcor.setObjectName('SLcor')

        self.SLseg = SuperDrawQLabel(self, self.cacheDir)
        self.SLseg.setGeometry(self.CTVLeftMargin+self.CTVWidth, self.CTVTopMargin+self.CTVHeight, self.CTVWidth, self.CTVHeight)
        self.SLseg.setStyleSheet('background-color: black')
        self.SLseg.setObjectName('SLseg')


        self.Badd = QPushButton(self)
        self.Badd.setIcon(QIcon("Icons/add.png"))
        self.Badd.resize(self.CTVBWidth, self.CTVBHeight)
        self.Badd.setGeometry(self.CTVLeftMargin+self.CTVWidth*2-5*self.CTVBWidth,
                         self.CTVTopMargin+self.CTVHeight*2-self.CTVBHeight,
                         self.CTVBWidth, self.CTVBHeight)
        self.Badd.setEnabled(False)
        self.Badd.setToolTip("Add")
        self.Badd.mode = "Add"
        self.Badd.clicked.connect(self._drawChangedFunc)

        self.Berase = QPushButton(self)
        self.Berase.setIcon(QIcon("Icons/erase.png"))
        self.Berase.resize(self.CTVBWidth, self.CTVBHeight)
        self.Berase.setGeometry(self.CTVLeftMargin+self.CTVWidth*2-4*self.CTVBWidth,
                         self.CTVTopMargin+self.CTVHeight*2-self.CTVBHeight,
                         self.CTVBWidth, self.CTVBHeight)
        self.Berase.setEnabled(False)
        self.Berase.setToolTip("Erase")
        self.Berase.mode = "Erase"
        self.Berase.clicked.connect(self._drawChangedFunc)

        self.Bback = QPushButton(self)
        self.Bback.setIcon(QIcon("Icons/back.png"))
        self.Bback.resize(self.CTVBWidth, self.CTVBHeight)
        self.Bback.setGeometry(self.CTVLeftMargin+self.CTVWidth*2-3*self.CTVBWidth,
                         self.CTVTopMargin+self.CTVHeight*2-self.CTVBHeight,
                         self.CTVBWidth, self.CTVBHeight)
        self.Bback.setEnabled(False)
        self.Bback.setToolTip("Back")
        self.Bback.mode = "Back"
        self.Bback.clicked.connect(self._drawChangedFunc)

        self.Bexit = QPushButton(self)
        self.Bexit.setIcon(QIcon("Icons/exit.png"))
        self.Bexit.resize(self.CTVBWidth, self.CTVBHeight)
        self.Bexit.setGeometry(self.CTVLeftMargin+self.CTVWidth*2-2*self.CTVBWidth,
                         self.CTVTopMargin+self.CTVHeight*2-self.CTVBHeight,
                         self.CTVBWidth, self.CTVBHeight)
        self.Bexit.setEnabled(False)
        self.Bexit.setToolTip("Exit")
        self.Bexit.mode = "Exit"
        self.Bexit.clicked.connect(self._drawChangedFunc)

        self.Bview = QPushButton(self)
        self.Bview.setIcon(QIcon("Icons/zoom_in.png"))
        self.Bview.resize(self.CTVBWidth, self.CTVBHeight)
        self.Bview.setGeometry(self.CTVLeftMargin+self.CTVWidth*2-1*self.CTVBWidth,
                         self.CTVTopMargin+self.CTVHeight*2-self.CTVBHeight,
                         self.CTVBWidth, self.CTVBHeight)
        self.Bview.setEnabled(False)
        self.Bview.setToolTip("View")
        self.Bview.mode = "View"
        self.Bview.clicked.connect(self._viewChangedFunc)

        self.SpenWidth = QSlider(Qt.Horizontal, self)
        self.SpenWidth.setGeometry(self.CTVLeftMargin+self.CTVWidth*2-10*self.CTVBWidth,
                                self.CTVTopMargin+self.CTVHeight*2-self.CTVBHeight,
                                self.CTVBWidth*5*0.95, self.CTVBHeight)
        self.SpenWidth.setEnabled(False)
        self.SpenWidth.setMinimum(1)
        self.SpenWidth.setMaximum(10)
        self.SpenWidth.setSingleStep(1)
        self.SpenWidth.setValue(4)
        self.SpenWidth.setTickInterval(1)
        self.SpenWidth.setTickPosition(QSlider.TicksBelow)
        self.SpenWidth.valueChanged.connect(self._penWidthChangedFunc)

        self.Salpha = QSlider(Qt.Horizontal, self)
        self.Salpha.setGeometry(self.CTVLeftMargin+self.CTVWidth*2-10*self.CTVBWidth, 
                               self.CTVTopMargin*1.5+self.CTVHeight, 
                               self.CTVBWidth*10*0.95, self.CTVBHeight)
        self.Salpha.setEnabled(True)
        self.Salpha.setMinimum(0)
        self.Salpha.setMaximum(100)
        self.Salpha.setSingleStep(5)
        self.Salpha.setValue(30)
        self.Salpha.setTickInterval(5)
        self.Salpha.setTickPosition(QSlider.TicksBelow)
        self.Salpha.valueChanged.connect(self._alphaChangedFunc)

        self.DrawButtons = [
            self.Badd,
            self.Berase,
            self.Bback,
            self.Bexit,
            self.SpenWidth,
        ]

    def _initUI_Console(self):
        self.ConsoleTopMargin = 2*self.CTVHeight+self.CTVTopMargin
        self.ConsoleLeftMargin = self.CTVLeftMargin
        self.ConsoleWidth = 2*self.CTVWidth
        self.ConsoleHeight = self.height-self.ConsoleTopMargin

        self.Console = QTextEdit(self, readOnly=True)
        self.Console.ensureCursorVisible()
        self.Console.setLineWrapColumnOrWidth(self.ConsoleWidth)
        self.Console.setLineWrapMode(QTextEdit.FixedPixelWidth)
        self.Console.setFixedWidth(self.ConsoleWidth)
        self.Console.setFixedHeight(self.ConsoleHeight)
        self.Console.setGeometry(self.ConsoleLeftMargin, self.ConsoleTopMargin, 
                                self.ConsoleWidth, self.ConsoleHeight)

    def _initUI_ParamView(self):
        self.ParamViewLeftMargin = self.ConsoleLeftMargin+self.ConsoleWidth
        self.ParamViewTopMargin = self.CTVTopMargin
        self.ParamViewWidth = self.width - self.ParamViewLeftMargin
        self.ParamViewHeight = self.height - self.ParamViewTopMargin

        self.ParamView = QTableWidget(self)
        self.ParamView.setGeometry(self.ParamViewLeftMargin, self.ParamViewTopMargin, 
                                self.ParamViewWidth, self.ParamViewHeight)
        self.ParamView.setRowCount(100)
        self.ParamView.setColumnCount(2)
        self.ParamView.horizontalHeader().setSortIndicatorShown(False)

    # Controls
    def _showTable(self):
        for row_id in range(len(self.table_data)):
            self.ParamView.setRowHeight(row_id, 30)
            self.ParamView.setItem(row_id, 0, QTableWidgetItem(self.table_data[row_id][0]))
            self.ParamView.setItem(row_id, 1, QTableWidgetItem(self.table_data[row_id][1]))

    def _convertImage(self, img, mode, eps=1e-6):
        minv = int(self.window_level_value - self.window_width_value / 2.0)
        maxv = int(self.window_level_value + self.window_width_value / 2.0)
        img = img.clip(minv, maxv)
        img = (img - minv) / (maxv - minv + eps)
        img = (img * 255).astype(np.uint8)

        H, W = img.shape 
        T = min(self.CTVWidth, self.CTVHeight)
        img = cv2.resize(img, (T, T))
        new_img = np.zeros((self.CTVHeight, self.CTVWidth))
        padding = [self.CTVWidth-T, self.CTVHeight-T]
        new_img[padding[1]//2:padding[1]//2+T,
                padding[0]//2:padding[0]//2+T] = img 
        new_img = np.stack([new_img, new_img, new_img], axis=-1)
        cv2.imwrite(os.path.join(self.cacheOutputDir, f"{mode}_ori.jpg"), new_img)

        if mode == 'transverse':
            tmp_sag_slice_idx = int(1.0*self.sag_slice_idx/self.shape[2]*new_img.shape[1])
            for k in range(0, new_img.shape[0], 5):
                cv2.circle(new_img, (tmp_sag_slice_idx, k), 1, (0, 0, 255), -1) 
            tmp_cor_slice_idx = int(1.0*self.cor_slice_idx/self.shape[1]*new_img.shape[0])
            for k in range(0, new_img.shape[1], 5):
                cv2.circle(new_img, (k, tmp_cor_slice_idx), 1, (0, 0, 255), -1)
        elif mode == 'sagittal':
            tmp_tar_slice_idx = int(1.0*self.tra_slice_idx/self.shape[0]*new_img.shape[0])
            for k in range(0, new_img.shape[1], 5):
                cv2.circle(new_img, (k, tmp_tar_slice_idx), 1, (0, 0, 255), -1)
            tmp_cor_slice_idx = int(1.0*self.cor_slice_idx/self.shape[1]*new_img.shape[1])
            for k in range(0, new_img.shape[0], 5):
                cv2.circle(new_img, (tmp_cor_slice_idx, k), 1, (0, 0, 255), -1)
        elif mode == 'coronal':
            tmp_tar_slice_idx = int(1.0*self.tra_slice_idx/self.shape[0]*new_img.shape[0])
            for k in range(0, new_img.shape[1], 5):
                cv2.circle(new_img, (k, tmp_tar_slice_idx), 1, (0, 0, 255), -1)
            tmp_sag_slice_idx = int(1.0*self.sag_slice_idx/self.shape[2]*new_img.shape[1])
            for k in range(0, new_img.shape[0], 5):
                cv2.circle(new_img, (tmp_sag_slice_idx, k), 1, (0, 0, 255), -1)

        cache_file = os.path.join(self.cacheOutputDir, f"{mode}.jpg")
        cv2.imwrite(cache_file, new_img)
        return cache_file

    def _showImage(self):
        self.tra_cache_file = self._convertImage(self.np_image[self.tra_slice_idx, ...], "transverse")
        self.SLtra.setPixmap(self.tra_cache_file)

        self.sag_cache_file = self._convertImage(self.np_image[..., self.sag_slice_idx], "sagittal")
        self.SLsag.setPixmap(self.sag_cache_file)

        self.cor_cache_file = self._convertImage(self.np_image[:, self.cor_slice_idx, :], "coronal")
        self.SLcor.setPixmap(self.cor_cache_file)

    def _showSeg(self):
        seg = sitk.ReadImage(self.seg_file)
        seg = sitk.GetArrayFromImage(seg)
        self.np_seg = seg[0]
        self._computeParams(self.np_seg)
        self._showTable()

        self._showLabelChangedFunc()

    def _computeParams(self, np_seg):
        for d in self.table_data:
            if "SMI" == d[0]:
                return 
        
        slice_image = self.np_image[self.tra_slice_idx]

        patient_height = int(self.WSheight.Slider.value()) / 100.0

        mpsi_area = 1.0 * np.sum(np_seg == 1) * self.spacing[0] * self.spacing[1] / 100.0
        self.table_data.insert(0, ['SMI (MPSI)', f'{1.0 * mpsi_area / patient_height / patient_height:.6f}'])

        mpso_area = 1.0 * np.sum(np_seg == 2) * self.spacing[0] * self.spacing[1] / 100.0
        self.table_data.insert(0, ['SMI (MPSO)', f'{1.0 * mpso_area / patient_height / patient_height:.6f}'])

        mven_area = 1.0 * np.sum(np_seg == 3) * self.spacing[0] * self.spacing[1] / 100.0
        self.table_data.insert(0, ['SMI (MVEN)', f'{1.0 * mven_area / patient_height / patient_height:.6f}'])

        self.table_data.insert(0, ['SMI', f'{1.0 * (mpsi_area + mpso_area + mven_area) / patient_height / patient_height:.6f}'])

        self.table_data.insert(0, ['SMD (MPSI)', f'{np.mean(slice_image[np_seg == 1]):.6f}'])
        self.table_data.insert(0, ['SMD (MPSO)', f'{np.mean(slice_image[np_seg == 2]):.6f}'])
        self.table_data.insert(0, ['SMD (MVEN)', f'{np.mean(slice_image[np_seg == 3]):.6f}'])
        np_seg[np_seg == 0] = 100
        self.table_data.insert(0, ['SMD', f'{np.mean(slice_image[np_seg <= 3]):.6f}'])

        sat_area = 1.0 * np.sum(np_seg == 4) * self.spacing[0] * self.spacing[1] / 100.0
        self.table_data.insert(0, ['SFA', f'{sat_area:.6f}'])

        vat_area = 1.0 * np.sum(np_seg == 5) * self.spacing[0] * self.spacing[1] / 100.0
        self.table_data.insert(0, ['VFA', f'{vat_area:.6f}'])

        self.table_data.insert(0, ['VSR', f'{vat_area/sat_area:.6f}'])

    def _initImage(self):
        self.table_data = []
        for row_id, k in enumerate(self.image.GetMetaDataKeys()):
            v = self.image.GetMetaData(k)
            self.table_data.append([k, v])
        self._showTable()

        self.np_image = sitk.GetArrayFromImage(self.image)[::-1]
        self.shape = self.np_image.shape 
        self.spacing = self.image.GetSpacing()

        self.tra_slice_idx = self.shape[0]//2
        self.cor_slice_idx = self.shape[1]//2
        self.sag_slice_idx = self.shape[2]//2
        self.WStraSlice.Slider.setValue(self.tra_slice_idx)
        self.WScorSlice.Slider.setValue(self.cor_slice_idx)
        self.WSsagSlice.Slider.setValue(self.sag_slice_idx)

        self.WStraSlice.Slider.setMaximum(self.shape[0]-1)
        self.WScorSlice.Slider.setMaximum(self.shape[1]-1)
        self.WSsagSlice.Slider.setMaximum(self.shape[2]-1)

        self._showImage()

    def _openNiiFileFunc(self):
        filename, filetype = QFileDialog.getOpenFileName(self,
                                                        "select files",
                                                        os.getcwd(),
                                                        'All File(*);;Nii Files(*.nii.gz)')
        self.filename = filename
        self.setWindowTitle(f'{self.title} ({os.path.basename(self.filename).split(".")[0]})')
        self.filetype = filetype
        self.image = sitk.ReadImage(self.filename)
        self._initImage()
        
    def _openDicomDirFunc(self):
        self.dicom_dir = QtWidgets.QFileDialog.getExistingDirectory(None, "select dicom path", os.getcwd())
        self.image = read_dcm(self.dicom_dir)
        self._initImage()

    def _saveSegFunc(self):
        return 

    def _saveExcelFunc(self):
        return 

    def _saveLogFunc(self):
        return 

    def _helpAboutFunc(self):
        return 

    def _helpDocFunc(self):
        return

    def _windowLevelChangedFunc(self):
        self.window_level_value = int(self.WSlevel.Slider.value())
        self._showImage() 

    def _windowWidthChangedFunc(self):
        self.window_width_value = int(self.WSwidth.Slider.value())
        self._showImage() 

    def _windowTraSliChangedFunc(self):
        self.tra_slice_idx = int(self.WStraSlice.Slider.value())
        self._showImage() 

    def _windowSagSliChangedFunc(self):
        self.sag_slice_idx = int(self.WSsagSlice.Slider.value())
        self._showImage() 

    def _windowCorSliChangedFunc(self):
        self.cor_slice_idx = int(self.WScorSlice.Slider.value())
        self._showImage() 

    def _convertSeg(self, seg, save_file):
        H, W = seg.shape[:2]
        T = min(self.CTVWidth, self.CTVHeight)
        seg = cv2.resize(seg, (T, T), interpolation=cv2.INTER_NEAREST)
        new_seg = np.zeros((self.CTVHeight, self.CTVWidth, seg.shape[2]))
        padding = [self.CTVWidth-T, self.CTVHeight-T]
        new_seg[padding[1]//2:padding[1]//2+T,
                padding[0]//2:padding[0]//2+T, :] = seg
        cv2.imwrite(save_file, new_seg)
        return save_file

    # def _showImageChangedFunc(self):
    #     if self.LSselector.BshowImage.isChecked():
    #         self.show_seg_file = os.path.join(self.cacheOutputDir, "show_seg_with_img.jpg")
    #         image = cv2.imread(os.path.join(self.cacheOutputDir, "transverse_ori.jpg"))
    #         seg = cv2.imread(os.path.join(self.cacheOutputDir, "show_seg.jpg"))

    #         image = cv2.addWeighted(image, float(self.Salpha.value())/100.0, 
    #                                 seg, 1.0-float(self.Salpha.value())/100.0, 0)
    #         cv2.imwrite(self.show_seg_file, image)
    #     else:
    #         self.show_seg_file = os.path.join(self.cacheOutputDir, "show_seg.jpg")    
    #     self.SLseg.setPixmap(self.show_seg_file)

    def _showLabelChangedFunc(self):
        keep_label_ids = []
        for Button in self.LSselector.Buttons:
            if Button.isChecked():
                keep_label_ids.append(label_to_id[Button.mode.split(" ")[0]])
                self.curLabel = Button.mode.split(" ")[0]

        if len(keep_label_ids) == 1:
            self._openDraw()
        else:
            self._closeDraw()

        try:
            tmp_np_seg = deepcopy(self.np_seg)
            for i in np.unique(tmp_np_seg):
                if i not in keep_label_ids:
                    tmp_np_seg[tmp_np_seg == i] = 0
            self._convertSeg(cmap[tmp_np_seg], os.path.join(self.cacheOutputDir, "show_seg.jpg"))

            self.show_seg_file = os.path.join(self.cacheOutputDir, "show_seg_with_img.jpg")
            image = cv2.imread(os.path.join(self.cacheOutputDir, "transverse_ori.jpg"))
            seg = cv2.imread(os.path.join(self.cacheOutputDir, "show_seg.jpg"))
            image = cv2.addWeighted(image, float(self.Salpha.value())/100.0, 
                                    seg, 1.0-float(self.Salpha.value())/100.0, 0)
            cv2.imwrite(self.show_seg_file, image)
            self.SLseg.setPixmap(self.show_seg_file)

            self.Bview.setEnabled(True)
        except:
            return

    def _generateSegFunc(self):
        self.patient_height = int(self.WSheight.Slider.value())

        slice_image = self.np_image[self.tra_slice_idx:self.tra_slice_idx+1]
        slice_image = sitk.GetImageFromArray(slice_image)
        slice_image.SetSpacing(self.spacing)
        
        input_file = os.path.join(self.cacheInputDir, "example_0000.nii.gz")
        sitk.WriteImage(slice_image, input_file)
        output_file = os.path.join(self.cacheOutputDir, os.path.basename(input_file).replace("_0000.nii.gz", ".nii.gz"))
        self.segmentor.predict_case(input_file, output_file)
        self.seg_file = output_file

        self._showSeg()

    def _viewChangedFunc(self):
        if self.single_view == False:
            self.Bview.setIcon(QIcon("Icons/zoom_out.png"))
            self.SLseg.setGeometry(self.CTVLeftMargin, self.CTVTopMargin, self.CTVWidth*2, self.CTVHeight*2)
            self.Salpha.setGeometry(self.CTVLeftMargin+self.CTVWidth*2-10*self.CTVBWidth, 
                                self.CTVTopMargin*1.5, 
                                self.CTVBWidth*10*0.95, self.CTVBHeight)
            self.single_view = True
        else:
            self.Bview.setIcon(QIcon("Icons/zoom_in.png"))
            self.SLseg.setGeometry(self.CTVLeftMargin+self.CTVWidth, self.CTVTopMargin+self.CTVHeight, self.CTVWidth, self.CTVHeight)
            self.Salpha.setGeometry(self.CTVLeftMargin+self.CTVWidth*2-10*self.CTVBWidth, 
                                self.CTVTopMargin*1.5+self.CTVHeight, 
                                self.CTVBWidth*10*0.95, self.CTVBHeight)
            self.single_view = False
        self.SLseg.setPixmap(self.show_seg_file)

    def _drawChangedFunc(self):
        if self.sender().mode == "Add":
            self.SLseg.openPaint = True 
            self.SLseg.curLabel = self.curLabel
            self._closeSelector()
        elif self.sender().mode == "Erase":
            self.SLseg.openPaint = True 
            self.SLseg.curLabel = "BACKGROUND"
            self._closeSelector()
        elif self.sender().mode == "Back":
            self._closeSelector()
            self.SLseg.backFunc()
        elif self.sender().mode == "Exit":
            self.SLseg.openPaint = False
            self._openSelector()

    def _penWidthChangedFunc(self):
        self.SLseg.penWidth = int(self.SpenWidth.value()) 

    def _alphaChangedFunc(self):
        #if self.LSselector.BshowImage.isChecked():
        self.show_seg_file = os.path.join(self.cacheOutputDir, "show_seg_with_img.jpg")
        image = cv2.imread(os.path.join(self.cacheOutputDir, "transverse_ori.jpg"))
        seg = cv2.imread(os.path.join(self.cacheOutputDir, "show_seg.jpg"))

        image = cv2.addWeighted(image, float(self.Salpha.value())/100.0, 
                                seg, 1.0-float(self.Salpha.value())/100.0, 0)
        cv2.imwrite(self.show_seg_file, image)
        self.SLseg.setPixmap(self.show_seg_file)

    def _closeDraw(self):
        for obj in self.DrawButtons:
            obj.setEnabled(False)

    def _openDraw(self):
        for obj in self.DrawButtons:
            obj.setEnabled(True)

    def _closeSelector(self):
        for obj in self.LSselector.Buttons:
            obj.setEnabled(False)
        #self.LSselector.BshowImage.setEnabled(False)

    def _openSelector(self):
        for obj in self.LSselector.Buttons:
            obj.setEnabled(True)
        #self.LSselector.BshowImage.setEnabled(True)

    def closeEvent(self, event):
        sys.stdout = sys.__stdout__
        super().closeEvent(event)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.aboutToQuit.connect(app.deleteLater)
    ex = SetupWindow()
    ex.show()
    sys.exit(app.exec_())