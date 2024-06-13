# Utilities for AI services that integrate with Jetson Platform Services 
The mmj_utils package provides convenient classes to build AI services to integrate with Jetson Platform Service. This package is presented as a standard python module that implements common functionality in the AI server found in the [jetson-platform-services repository](https://github.com/NVIDIA-AI-IOT/jetson-platform-services). This includes modular components for RTSP streaming, overlay generation, REST API integration and more. 

Reference examples for AI services that use mmj_utils can be found on the [jetson-platform-services repository](https://github.com/NVIDIA-AI-IOT/jetson-platform-services). 

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
Currently, mmj_utils provides classes to do the following 

- Interface with the Video Storage Toolkit (VST)
- Create Overlays for detection and VLMs 
- Handle RTSP streaming input and output 
- Interface with a VLM chat server with custom callbacks and live stream inputs 
- REST API Server base class with stream control endpoints 

mmj_utils is split into several submodules

- vst.py
    - VST
- overlay_gen.py
    - DetectionOverlayCUDA
    - VLMOverlay
- schema_gen.py
    - SchemaGenerator
- streaming.py
    - VideoSource
    - VideoOutput
- vlm.py
    - VLM
- api_server.py
    - APIServer


To use these classes, import them into your Python scripts once mmj_utils has been installed with pip or using a container with mmj_utils installed. 

```
from mmj_utils.schema_gen import SchemaGenerator
from mmj_utils.overlay_gen import DetectionOverlayCUDA, VLMOverlay
from mmj_utils.vst import VST 
from mmj_utils.vlm import VLM
from mmj_utils.streaming import VideoSource, VideoOutput
from mmj_utils.api_server import APIServer 
```

### VST
The VST object should be instantiated with the VST host address.
```
url = http://0.0.0.0:81
VST(url)
```

Once instantiated its methods can be called to get a list of available streams and add new ones. For example: 

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

For more details read the docstrings in overlay_gen.py and view how it is used in the Zero Shot Detection AI service in the [jetson-platform-services repository](https://github.com/NVIDIA-AI-IOT/jetson-platform-services).


The VLMOverlay class can be used to generate an overlay for VLM input and output strings. This is useful to visualize the VLM responses on a live streaming video. Similar to the DetectionOverlayCUDA class, there are several options to configure the formatting of the output. 

View the docstrings for the VLMOverlay class in overlay_gen.py and to see example usage, view the VLM AI Service source code in the [jetson-platform-services repository](https://github.com/NVIDIA-AI-IOT/jetson-platform-services) 

### Schema Generation

The schema generator can be instantiated and optionally supplied with camera settings like the ID, location and type. You can also supply a redis host, port and stream name to automatically connect and output metadata to redis. 

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

### Streaming

To handle RTSP input and output, the VideoSource and VideoOutput classes are implemented. 

The VideoSource object allows you to easily connect to an RTSP stream and then retrieve individual frames to use as input to an AI model. 

```
v_input = VideoSource()
v_input.connect_stream(rtsp_link, camera_id=message.data["stream_id"])
...

while True: 
    ...
    frame:=v_input()
    ...
```

This frame can then be modified with one of the overlay objects and then output as a new RTSP stream. 

```
v_output = VideoOutput("rtsp://0.0.0.0:5011/out")

...
while True: 
    ...
    v_output(output_frame)
    ...
```

The output frames can then be viewed in a media player like VLC by connecting to the defined output RTSP link. 

For more details view the source code in the streaming.py file and view how it is used in the Zero Shot Detection and VLM AI services in the [jetson-platform-services repository](https://github.com/NVIDIA-AI-IOT/jetson-platform-services). 

### VLM

An abstraction for interacting with the VLM chat server implemented in the VLM AI service found [jetson-platform-services repository](https://github.com/NVIDIA-AI-IOT/jetson-platform-services) is provided as a VLM class. This VLM class allows the developer to create custom "egos" that have an associated system prompt and callback function to enable each VLM call to take some specific action like sending a notification. 

```
#Define the callback function
def vlm_alert_handler(response, **kwargs):
    print(response)
```

```
#Create VLM object and add an ego 
vlm = VLM(config.chat_server)
vlm.add_ego("alert", system_prompt="Answer the user's alert as yes or no.", callback=alert_callback_fn)
```

```
#Call VLM with prompt and frame input. 
frame:=v_input()
vlm("alert", "is there a fire?", frame)
```

After calling the VLM object, it will not return a response, instead it will wait for the VLM to finish processing in the background then pass the completed response to the specified callback function. Optional kwargs are also supported to pass through to the callback function. 

For further example usage, view the VLM AI service implemented in the [jetson-platform-services repository](https://github.com/NVIDIA-AI-IOT/jetson-platform-services)

### API Server

A base API server built with FastAPI and implementing the live-stream endpoints used by the Zero Shot Detection and VLM services found in the [jetson-platform-services repository](https://github.com/NVIDIA-AI-IOT/jetson-platform-services) is implemented in the api_server.py file. 

This base class can be extended to add new endpoints and also supports inter thread communication to send commands and receive responses from other threads. 

To see how this is used, view the Zero Shot Detection and VLM services implemented in the [jetson-platform-services repository](https://github.com/NVIDIA-AI-IOT/jetson-platform-services). 

