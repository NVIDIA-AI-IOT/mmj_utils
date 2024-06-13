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

from jetson_utils import videoSource, videoOutput, cudaAllocMapped, cudaDeviceSynchronize
from threading import Thread 
import logging 
from time import sleep 
import queue 

class VideoSource:

    def __init__(self, url=None):
        """Initialize video source object. If url supplied, it will attempt to connect. Otherwise call connect_stream to connect after initialization."""

        self.v_input = None 
        self.url = url 
        self.connected = False 
        self.camera_name = None 
        self.camera_id = None 

        if self.url is not None:
            self.connect_stream(self.url)
        
    def close_stream(self):
        """Close stream if connected"""
        if self.v_input is not None:
            try:
                self.v_input.Close() 
            except Exception as e:
                logging.error("Failed to close stream")
            self.v_input = None 
            self.url = None 
            self.connected = False 
            self.camera_name = None 
            self.camera_id = None 

    def connect_stream(self, url, retries=5, camera_name="", camera_id=""):
        """Closes current stream if available and tries to connect to a new one """
        count = 0
        while count <= retries:
            try:
                if self.v_input is not None:
                    self.v_input.Close()
                    self.v_input = None 
                    self.connected = False
                    self.camera_name = None 
                    self.camera_id = None 
                    self.url = None 
            except Exception as e:
                logging.info(f"exception occured closing video source {self.url}")
            
            try:
                v_input = videoSource(url)
                self.v_input = v_input 
                self.connected = True 
                self.url = url 
                self.camera_name = camera_name 
                self.camera_id = camera_id 
                logging.info("Successfully connected to stream")
                return 
            
            except Exception as e:
                logging.info(f"Failed to create video source")
                sleep(0.5)
            
            count+=1
        logging.error("Failed to connect to stream")
        return 


    def __call__(self,retries=8):
        """Returns the most recent frame from the connected stream"""

        if self.v_input is None:
            #logging.info("Called VideoSource but not connected to stream.")
            return None 
        
        count = 0
        while count < retries:
            try:
                frame = self.v_input.Capture()
                return frame 
            except Exception as e:
                frame = None 
            count+=1
        #Failed to get frame 
        logging.error("Failed to get frame from input stream. Reconnecting.")
        self.connect_stream(self.url, camera_name=self.camera_name, camera_id=self.camera_id)
        return None 

class VideoOutput:

    def __init__(self, url):
        """Initialize an output RTSP stream."""
        self.url = url 

        self.frame_queue = queue.Queue(maxsize = 3) 

        self.v_output = videoOutput(self.url, options={'save': '/tmp/null.mp4'})
        self.timeout = 1/5 
        self.backup_frame = cudaAllocMapped(width=1920, height=1080, format="rgb8")
        cudaDeviceSynchronize()

        self.thread = Thread(target=self._stream_out, daemon=True)
        self.thread.start() 

    def _stream_out(self):
        """Runs as a background thread to keep a persistent RTSP output. Will output a black frame if nothing is in the output queue."""
        while True:
            try:
                frame = self.frame_queue.get(timeout=self.timeout)
                self.v_output.Render(frame)
            except queue.Empty: #todo only catch queue timeout handle others differently 
                self.v_output.Render(self.backup_frame)

            except Exception as e:
                logging.error(e)
                sleep(self.timeout)

    def __call__(self, frame):
        """Call to place frame in the output queue to render on RTSP stream."""
        self.frame_queue.put(frame)