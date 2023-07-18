!git clone https://github.com/ultralytics/yolov5  
%cd yolov5
%pip install -qr requirements.txt 
%pip install -q roboflow

import torch
import os
from IPython.display import Image, clear_output 

print(f"Setup complete. Using torch {torch.__version__} ({torch.cuda.get_device_properties(0).name if torch.cuda.is_available() else 'CPU'})")

from roboflow import Roboflow
rf = Roboflow(model_format="yolov5", notebook="ultralytics")

#環境構築
os.environ["DATASET_DIRECTORY"] = "/content/datasets"

!pip install roboflow

from roboflow import Roboflow
rf = Roboflow(api_key="3wAYldO7HvCevuQXEwZD")
project = rf.workspace("cist-cplyz").project("puromen_moto")
dataset = project.version(1).download("yolov5")

#学習
!python train.py  --img 416 --batch 50 --epochs 200 --data {dataset.location}/data.yaml --cfg yolov5s.yaml --weights ''

!pip install roboflow

from roboflow import Roboflow
rf = Roboflow(api_key="")
project = rf.workspace("cist-cplyz").project("project_3-vj8ns")
dataset = project.version(1).download("yolov5")

!python train.py  --img 416 --batch 50 --epochs 200 --data {dataset.location}/data.yaml --weights runs/train/exp/weights/best.pt --cache --freeze 10

%load_ext tensorboard
%tensorboard --logdir runs

#推論
!python detect.py --weights runs/train/exp2/weights/best.pt --img 416 --conf 0.5 --source {dataset.location}/test/images

#グラフの表示
import glob
from IPython.display import Image, display

for imageName in glob.glob('/content/yolov5/runs/detect/exp/*.jpg'): #assuming JPG
    display(Image(filename=imageName))
    print("\n")