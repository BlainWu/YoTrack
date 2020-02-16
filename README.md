# YoTrack
Realise a TDL system which realizes long term object tracking .The detection module is implemented by yolov3 .

# Demo
![Only-aeroplane](https://img-blog.csdnimg.cn/20200216163940371.gif)
## YoloV3: Tracking by detection
Realize tracking only by detection :  **Yolo-Tracking.py**
You can run it by:  (defaults detect all 80 kinds of objects)
```
python Yolo-Tracking.py
```
or if you just want to searching for some kinds of objects, you can pass parameters to a program: 
```
python Yolo-Tracking.py --targets "targets lists,split by ','  "
```
take an example,if you want search for "aeroplane"and "person" :
```
python Yolo-Tracking.py --targets aeroplane,person
```
for more kinds of objects ,you can see **models/coco.names** file
# YoloV3
To better adapt to video object detection, I made a few adjustments. The most of YoloV3 files stored in YoloV3 .Weights files and objects lists stored in *models* .  
You can find the original codes here  ---> [YoloV3](https://github.com/ayooshkathuria/YOLO_v3_tutorial_from_scratch)  
For better understanding ,I collected related papers stored in papers.  
