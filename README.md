# Utilities for Metropolis Microservcies on Jetson
The mmj_utils package provides some convenient classes to help integrate AI applications into Metropolis Microservices. 

## Setup
The mmj_utils package requires jetson_utils to be installed. You can follow the build instructions [here](https://github.com/dusty-nv/jetson-utils?tab=readme-ov-file#building-from-source) to install jetson_utils on your system or build a container with jetson_utils using the [jetson_containers project](https://github.com/dusty-nv/jetson-containers). 


Once you have jetson_utils installed, you can install mmj_utils in two ways: 

```
pip install git+https://github.com/NVIDIA-AI-IOT/mmj_utils
```

Or you can clone the repository and install it
```
git clone https://github.com/NVIDIA-AI-IOT/mmj_utils
pip install ./mmj_utils/
```

## Overview
Currently, mmj_utils provides functions to accomplish three tasks 

1) Interfacing with the Video Storage Toolkit (VST)
2) Creating Overlays for machine learning vision tasks 
3) Generating meta data in Metropolis Minimal Schema

To accomplish this, mmj_utils is split in three submodules: 

- vst.py
    - VST
- overlay_gen.py
    - DetectionOverlayCUDA
- schema_gen.py
    - SchemaGenerator

To use these classes, import them into your Python scripts once mmj_utils has been installed with pip. 

```
from moj_utils.schema_gen import SchemaGenerator
from moj_utils.overlay_gen import DetectionOverlayCUDA
from moj_utils.vst import VST 
```

### VST
The VST object should be instantiated with the VST host address.
```
url = http://0.0.0.0:81
VST(url)
```

Once instantiated its methods can be called to get a list of available streams and add new ones. 

```
stream_list = vst.get_rtsp_streams()
vst.add_rtsp_stream(self, "rtsp://0.0.0.0:8554/stream", "camera1")
```
For more details read the docstrings in vst.py


### Overlay Generation
The DetectionOverlayCUDA class can be used to generate an overlay of bounding boxes and text labels for object detection tasks. You can instantiate the object with several configurations to change the colors, text size, bounding box size, etc. through keyword arguments. For a full list view, the docstring in overlay_gen.py. 

```
overlay_gen = DetectionOverlayCUDA(draw_bbox=True, draw_text=True, text_size=45)
```

The object can then be called and passed an image as an numpy array, torch tensor or a CUDAImage. You must also pass a list of text labels and bounding boxes. It will modify the passed in image and add the bounding boxes and text labels. 

```
objects = ["a person", "a box", "a box", "a vest", "a person"]
bboxes = [(0,0, 15,15), (20,20,50,50), (10,15, 25,40), (10,21, 12,36), (50, 50, 55, 55)]
overlay_gen = DetectionOverlayCUDA(draw_bbox=True, draw_text=True, text_size=45)
img = np.ones((60,60,3)) #make a test image
overlay_img = gen(img, objects, bboxes) #draw the bboxes and text
```

For more details read the docstrings in overlay_gen.py

### Schema Generation

The schema generator can be instanited and optionally supplied with camera settings like the ID, location and type. You can also supply a redis host, port and stream name to automatically connect and output metadata to redis. 

```
schema_gen = SchemaGenerator(redis_host="0.0.0.0", redis_port=6379, redis_stream="owl")
```


You can then call the schema_gen object and pass it a list of labels and bounding boxes. This will convert the information into metropolis minimal schema and return a serialized json string. If a redis configuration was supplied, then it will also write out the json string to the redis stream. This is important if you want the metadata to be consumed by an analytic application for downstream tasks like object counting, tracking and heatmapping. 

```
objects = ["a person", "a box", "a box", "a vest", "a person"]
bboxes = [(0,0, 15,15), (20,20,50,50), (10,15, 25,40), (10,21, 12,36), (50, 50, 55, 55)]
out = schema_gen(objects, bboxes)
print(out)
```

For more details read the docstrings in schema_gen.py