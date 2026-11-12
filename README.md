深度学习目标检测算法使用YOLOv8训练油茶果实数据集模型，识别果园种出现的油茶进行检测
@[toc]

以下文字及代码仅供参考，
油茶果实数据集，

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/1052123a7d8a4545b855a2e87c5827de.png)
txt标签

数据集分为3个子数据集，包括1012个"训练"样本、337个"验证"样本和328个"测试"样本![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/140a9f10970d443980ab4187e1215d23.png)1
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/02a236e3aec548ebb8b33ffd47bb3d49.png)
从数据预处理、模型训练、评估以及构建一个用于实时检测的GUI应用程序。我们将使用YOLOv8作为目标检测模型，并利用PyQt5来创建用户界面。

### 项目结构

```
Camellia_Fruit_Detection/
├── data/                       # 数据集
│   ├── images/                 # 图片文件夹
│   ├── labels_txt/             # 标签文件(txt格式)
│   ├── train.txt               # 训练集列表
│   ├── val.txt                 # 验证集列表
│   ├── test.txt                # 测试集列表
├── models/                     # 模型权重
│   ├── yolov8.pt               # YOLOv8预训练权重
│   ├── best.pt                 # 训练后的最佳权重
├── utils/                      # 工具函数
│   ├── dataset_utils.py        # 数据集处理工具
│   ├── eval_utils.py           # 评估工具
├── gui/                        # GUI 文件
│   ├── main_gui.py             # 主界面逻辑
│   ├── ui_main.ui              # PyQt5 UI 设计文件
├── requirements.txt            # Python 依赖包列表
├── train.py                    # 训练脚本
├── evaluate.py                 # 评估脚本
├── inference.py                # 推理脚本
└── README.md                   # 项目说明文档
```

### 1. 安装依赖

在 `requirements.txt` 中列出所需的依赖库：

```txt
torch==2.0.1
torchvision==0.15.2
opencv-python==4.7.0.72
PyQt5==5.15.9
ultralytics==8.0.43          # YOLOv8 库
numpy==1.23.5
matplotlib==3.7.1
```

安装依赖：

```bash
pip install -r requirements.txt
```

### 2. 数据集准备

根据描述，数据集已经按照训练、验证和测试进行了划分。我们需要将这些信息写入对应的 `.txt` 文件中。

#### 创建训练、验证和测试集列表 (`dataset_utils.py`)

```python
import os

def create_dataset_files(image_dir, label_dir, output_dir):
    sets = ['train', 'val', 'test']
    counts = {'train': 1012, 'val': 337, 'test': 328}

    for set_name in sets:
        with open(os.path.join(output_dir, f"{set_name}.txt"), "w") as f:
            for i in range(counts[set_name]):
                image_file = os.path.join(image_dir, f"{set_name}_{i+1}.jpg")
                if os.path.exists(image_file):
                    f.write(f"{image_file}\n")

# 调用示例
create_dataset_files("data/images", "data/labels_txt", "data")
```

### 3. 配置 YOLOv8

创建 `yolov8.yaml` 文件，定义类别和数据集路径。

```yaml
train: data/train.txt
val: data/val.txt
test: data/test.txt

nc: 1  # 类别数（假设只有油茶果实一个类别）
names: ['camellia_fruit']
```

### 4. 训练与评估

#### 训练脚本 (`train.py`)

```python
from ultralytics import YOLO

# 加载预训练模型
model = YOLO("models/yolov8.pt")

# 开始训练
model.train(data="data/yolov8.yaml", epochs=100, imgsz=640, batch=16, device=0)
```

#### 评估脚本 (`evaluate.py`)

```python
from ultralytics import YOLO

# 加载训练好的模型
model = YOLO("runs/train/weights/best.pt")

# 在验证集上评估
metrics = model.val()
print(metrics)
```

### 5. 构建 GUI 应用程序

#### `main_gui.py`

```python
from PyQt5.QtWidgets import QMainWindow, QFileDialog, QLabel
from PyQt5.QtGui import QImage, QPixmap
from ui_main import Ui_MainWindow
from inference import run_inference
import cv2

class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.setupUi(self)

        # 连接按钮事件
        self.btn_image.clicked.connect(self.select_image)
        self.btn_video.clicked.connect(self.select_video)
        self.btn_camera.clicked.connect(self.start_camera)

    def select_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "选择图片", "", "Image Files (*.jpg *.jpeg *.png)")
        if file_path:
            result_image = run_inference(file_path)
            self.display_image(result_image)

    def select_video(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "选择视频", "", "Video Files (*.mp4 *.avi)")
        if file_path:
            cap = cv2.VideoCapture(file_path)
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                result_frame = run_inference(frame)
                self.display_image(result_frame)
                if cv2.waitKey(30) & 0xFF == ord('q'):
                    break
            cap.release()
            cv2.destroyAllWindows()

    def start_camera(self):
        cap = cv2.VideoCapture(0)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            result_frame = run_inference(frame)
            self.display_image(result_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()

    def display_image(self, image):
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        q_img = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.label_image.setPixmap(QPixmap.fromImage(q_img))
```

### 6. 推理脚本 (`inference.py`)

```python
from ultralytics import YOLO
import cv2

# 加载训练好的模型
model = YOLO("runs/train/weights/best.pt")

def run_inference(input_data):
    if isinstance(input_data, str):  # 如果是图片路径
        results = model(input_data)
        img = cv2.imread(input_data)
    else:  # 如果是视频帧
        results = model(input_data)
        img = input_data

    # 绘制检测框
    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        label = model.names[int(box.cls)]
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    return img
```

### 7. 运行程序

```bash
python main_gui.py
```


