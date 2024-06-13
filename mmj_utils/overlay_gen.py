# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from jetson_utils import cudaFont, cudaDrawRect #cuda accelerated functions 
from jetson_utils import cudaFromNumpy
import matplotlib.pyplot as plt
import numpy as np
import torch 


__all__ = ["DetectionOverlayCUDA", "VLMOverlay"]

def _cudaFromTorch(tensor):
    """
    https://github.com/dusty-nv/jetson-utils/blob/master/python/examples/cuda-from-pytorch.py
    Determine the cudaImage format string (eg 'rgb32f', 'rgba32f', ect) from a PyTorch tensor.
    Only float and uint8 tensors are supported because those datatypes are supported by cudaImage.
    """
    if tensor.dtype != torch.float32 and tensor.dtype != torch.uint8:
        raise ValueError(f"PyTorch tensor datatype should be torch.float32 or torch.uint8 (was {tensor.dtype})")
        
    if len(tensor.shape)>= 4:     # NCHW layout
        channels = tensor.shape[1]
    elif len(tensor.shape) == 3:   # CHW layout
        channels = tensor.shape[0]
    elif len(tensor.shape) == 2:   # HW layout
        channels = 1
    else:
        raise ValueError(f"PyTorch tensor should have at least 2 image dimensions (has {tensor.shape.length})")
        
    if channels == 1:
        im_format = 'gray32f' if tensor.dtype == torch.float32 else 'gray8'
    elif channels == 3:
        im_format = 'rgb32f'  if tensor.dtype == torch.float32 else 'rgb8'
    elif channels == 4: 
        im_format = 'rgba32f' if tensor.dtype == torch.float32 else 'rgba8'
    
    raise ValueError(f"PyTorch tensor should have 1, 3, or 4 image channels (has {channels})")
    cuda_img = cudaImage(ptr=tensor.data_ptr(), width=tensor.shape[-1], height=tensor.shape[-2], format=tensor_image_format(tensor))
    return cuda_img




class VLMOverlay:

    def __init__(self, **kwargs):
        """
        Creates an object that can be called to generate VLM overlays given VLM input and output strings. 
        Optional kwargs can be passed to configure the drawing options of the text. All kwargs have defaults that should work in most cases. 

        Args:
            text_size (int): Font size for the text overlay
            x (int): Flag to enable drawing text for each bounding box
            y (int): Size in pixels of the bbox line width
            input_prefix (str): Prefix to append to the beginning of input text such as "VLM Input:" 
            output_prefix (str): Prefix to append to the beginning of the output such as "VLM Output:"
            input_text_color (Tuple): Color of input text 
            input_background_color (Tuple): Color of input text background
            output_text_color (Tuple): Color of output text
            output_background_color (Tuple): Color of output text background
            decay_rate (int): Decay rate of text overlay. This value is subtracted from the alpha channel of the text overlay each frame. 
            line_spacing (int): Spacing in pixels between lines 
            line_length (int): Maximum line length before wrapping text
            
        Returns:
            None
        """

        self.text_size = kwargs.get("text_size", 32) #font size

        #Starting location for text overlay 
        self.x = kwargs.get("x", 5)
        self.y = kwargs.get("y", 5)

        #Input and output prefix
        self.input_prefix = kwargs.get("input_prefix", "VLM Input:")
        self.output_prefix = kwargs.get("output_prefix", "VLM Output:")
        
        #Color options 
        self.input_text_color = kwargs.get("input_text_color", (120,215,21))
        self.input_background_color = kwargs.get("input_background_color", (40,40,40))
        self.output_text_color = kwargs.get("output_text_color", (255,255,255))
        self.output_background_color = kwargs.get("output_background_color", (40,40,40))
        self.input_background_color = kwargs.get("output_background_color", (40,40,40))
        self.decay_rate = kwargs.get("decay_rate", 1)

        #Line options
        self.line_spacing = kwargs.get("line_spacing", 42)
        self.line_length = kwargs.get("line_length", None)

        self.font = cudaFont(size=self.text_size)
        self.alpha = 255

        #text output 
        self._input_text = None 
        self._output_text = None 
   
    @property
    def input_text(self):
        return self._input_text
    
    @input_text.setter
    def input_text(self, text):
        self._input_text = text 

    @property
    def output_text(self):
        return self._output_text
    
    @output_text.setter
    def output_text(self, text):
        self._output_text = text

    def _filter_string(self, text):
        """cuda font only supports ascii codes 32-254 """
        return ''.join([ch if 32 <= ord(ch) <= 254 else ' ' for ch in text])


    def _draw_lines(self, image, text, x, y, **kwargs):
        """"Draw text on a image with word wrapping"""

        if text is None:
            return y
        
        text_color=kwargs.get("text_color", self.font.White) 
        background_color=kwargs.get("background_color", self.font.Gray40)
        line_spacing = kwargs.get("line_spacing", 42)
        line_length = (image.width-x) // 16 if kwargs.get("line_length") is None else kwargs.get("line_length")
        alpha = kwargs.get("alpha", 255)

        #Add transparency 
        text_color = text_color[:3] + (alpha,)
        background_color = background_color[:3] + (alpha,)

        #Split text across lines at word boundaries 
        text = text.split()
        current_line = ""
        line_count = 0
        for word in text:
            if len(current_line) + len(word) <= line_length:
                current_line = current_line + word + " "
            else:
                current_line = current_line.strip()
                self.font.OverlayText(image, text=current_line, x=x, y=y, color=text_color, background=background_color)
                current_line = word + " "
                line_count+=1
                y=y+line_spacing
        
        
        current_line = current_line.strip() 
        current_line = self._filter_string(current_line)
        self.font.OverlayText(image, text=current_line, x=x, y=y, color=text_color, background=background_color)

        return y+line_spacing #return Y offset for next line start 

    def reset_decay(self):
        """Reset the text alpha decay."""
        self.alpha=255

    def set_input_text(self, text):
        """Set input text for the overlay"""
        self.input_text = text 
    
    def set_output_text(self, text):
        """Set output text for the overlay"""
        self.output_text = text 
        self.reset_decay()

    def __call__(self, image, input_required=True, output_required=True, decay=True):
        """Drawing input and output text on the input frame. Call set_input_text and set_output_text prior to calling."""

        if input_required and self._input_text is None:
            return image 
        
        if output_required and self._output_text is None:
            return image 

        y_offset = self._draw_lines(image, f"{self.input_prefix} {self._input_text}", self.x, self.y, text_color=self.input_text_color, background_color=self.input_background_color, line_spacing=self.line_spacing, line_length=self.line_length, alpha=self.alpha)
        self._draw_lines(image, f"{self.output_prefix} {self._output_text}", self.x, y_offset, text_color=self.output_text_color, background_color=self.output_background_color, line_spacing=self.line_spacing, line_length=self.line_length, alpha=self.alpha)

        if decay:
            self.alpha = max(0, self.alpha - self.decay_rate)

        return image 

class DetectionOverlayCUDA:

    def __init__(self, **kwargs):
        """
        Creates an object that can be called to generate detection overlays given bounding boxes and text. 
        Optional kwargs can be passed to configure the drawing options of the bboxes and text. All kwargs have defaults that should work in most cases. 

        Args:
            draw_bbox (bool): Flag to enable drawing bounding boxes
            draw_text (bool): Flag to enable drawing text for each bounding box
            bbox_width (int): Size in pixels of the bbox line width
            text_size (int): Font size of the text
            text_offset_y (int): Text offset in the y direction from the top left of the associated bbox. Can be + or -
            text_offset_x (int): Text offset in the x direction from the top left of the associated bbox. Can be + or -
            max_objects (int): Expected unique objects. If the number of unique objects exceeds this value, then bounding box colors will repeat. If the value is too high then the color difference between bboxes will be very small. 

        Returns:
            None
        """

        #Optional configurations through keyword args 
        self.draw_bbox = kwargs.get("draw_bbox", True)
        self.draw_text = kwargs.get("draw_text", True)
        self.bbox_width = kwargs.get("bbox_width", 3) #bbox line width 
        self.text_size = kwargs.get("text_size", 45) #font size
        self.text_offset_y = kwargs.get("text_offset_y", 12) #offset text in Y direction from top left bbox corner
        self.text_offset_x = kwargs.get("text_offset_x", 0) #offset text in X direction from top left bbox corner
        self.max_objects = kwargs.get("max_objects", 10) #Used for color mapping. If value is lower than the number object classes, then colors will repeat. Set to higher value if you don't want duplicate colors. 
        
        cmap = plt.get_cmap("rainbow", self.max_objects)  #color map for bounding boxes 
        self.cmap = []
        for i in range(self.max_objects):
            color = cmap(i)
            color = [int(255 * value) for value in color]
            self.cmap.append(tuple(color))

        self.font = cudaFont(size=self.text_size)
        self.object_classes = {}

    #TODO handle image input errors/wrong types etc 
    def __call__(self, image, objects, bboxes):
        """
        Draws bounding boxes and text on an image using cuda accelerated draw functions.
        objects and bboxes should be lists of the same length. Each bbox will be matched with the object based on list order. 

        Args:
            image (jetson_utils.cudaImage, numpy.array, torch.Tensor ): The image to draw the overlay on.
            objects (list[str]): Text to draw for each bounding box.
            bboxes (list[(int,int,int,int), ...]): List of bounding boxes. Each bounding box should be in this format (x1, y1, x2, y2) Where x1,y1 is the top left and x2,y2 is the bottom right coordinate of the bbox. 

        Returns:
            image (jetson_utils.cudaImage, numpy.array, torch.Tensor): Same image reference with text and bounding boxes drawn. Will return the same type as the input. 
        """
        if len(objects) != len(bboxes):
            raise Exception("objects and bboxes not the same length.")
        
        #Convert tensor or np array to cudaImage
        if type(image) == torch.Tensor:
            image = _cudaFromTorch(image)
        elif type(image) == np.ndarray:
            image = cudaFromNumpy(image)
        
        #Loop through objects and draw bboxes & labels 
        for i in range(len(objects)):
            box = bboxes[i]
            obj = objects[i]

            #Add to obj to dict to track color mapping if not there already
            if obj not in self.object_classes:
                self.object_classes[obj] = len(self.object_classes)

            box = [int(x) for x in box]
            box = tuple(box) 
            color = self.cmap[self.object_classes[obj]%len(self.cmap)]

            if self.draw_bbox:
                cudaDrawRect(image, box, line_color=color, line_width=self.bbox_width)

            if self.draw_text:
                self.font.OverlayText(image, text=obj, 
                            x=box[0]+self.text_offset_x, y=box[1]+self.text_offset_y,
                            color=color, background=(0,0,0,0))

        return image 


if __name__ == "__main__":
    """
    Example usage
    """
    objects = ["a person", "a box", "a box", "a vest", "a person"]
    bboxes = [(0,0, 15,15), (20,20,50,50), (10,15, 25,40), (10,21, 12,36), (50, 50, 55, 55)]
    gen = DetectionOverlayCUDA()
    img = np.ones((60,60,3))
    img = gen(img, objects, bboxes)
    img = np.array(img)
    for x in img:
        for y in x:
            if y[0] != 1 and y[1] != 1 and y[2] != 1:
                print(y)


