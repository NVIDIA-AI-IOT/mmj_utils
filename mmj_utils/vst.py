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

import requests 
import argparse 
import json 

__all__ = ["VST"]

class VST:
    """
    Wrapper for the VST REST API
    Provides convenient functions to get streams from VST
    """

    def __init__(self, url):
        """
        Provide the URL to a VST server. 
        Ex)     url = http://0.0.0.0:81
                VST(url)
        Args:
            url (string): Address of VST server. Ex) "http://0.0.0.0:81/"

        Returns:
            None
        """
        url = url + "/" if url[-1] != "/" else url #make sure address ends in a /
        self.url = url 
        self.rtsp_streams = []
        self.streams = []
        self.sensors = []


    def _update_streams(self):
        streams = requests.get(f"{self.url}api/v1/live/streams")
        self.streams = streams.json()
    def _update_sensors(self):
        sensors = requests.get(f"{self.url}api/v1/sensor/list")
        self.sensors = sensors.json()
        
    def add_rtsp_stream(self, url, name, location=""):
        """
        Add an RTSP stream to VST 

        Args:
            url (str): A valid rtsp URL. ex: "rtsp://0.0.0.0:8554/stream"
            name (str): A name to be associated with the stream in vst. ex: "cam1"
            location (str): A location to be associated with the stream in vst. ex: "office"

        Returns:
            none
        """
        stream_info = {"sensorUrl":url, "name":name, "username":"", "password":"", "location":location}
        stream_info = json.dumps(stream_info)
        resp = requests.post(f"{self.url}api/v1/sensor/add", data=stream_info)
        return resp 

    def remove_rtsp_stream(self, sensorId):
        """
        Deletes a sensor and its associated RTSP stream from VST. 

        Args:
            sensorId (str): The sensor ID. Need to get this from VST REST api. 

        Returns:
            none
        """
        requests.delete(f"{self.url}api/v1/sensor/{sensorId}")

    def get_sensor_id(self, name):
        """
        Get the sensor ID by name of sensor.  

        Args:
            name (str): Name of the sensor. 

        Returns:
            sensorId (str): Returns the sensorId if found, else None. 
        """
        self._update_sensors()
        for sensor in self.sensors:
            if sensor["name"] == name:
                return sensor["sensorId"]
        return None
        

    def readd_rtsp_stream(self,url,name,location=""):
        """
        Add an RTSP stream. If the stream already exists, then delete it and add it. 
        """
        resp = self.add_rtsp_stream(url, name,location)
        if resp.status_code == 400:
            if resp.json()["error_message"] == "User given name is invalid or already exists":
                sid = self.get_sensor_id(name)
                if sid is not None:
                    self.remove_rtsp_stream(sid)
                    self.add_rtsp_stream(url, name,location)

    def get_rtsp_streams(self):
        """
        Returns a list of available RTSP streams from VST. 

        Args:
            none

        Returns:
            rtsp_streams (list[dict(streamID:, name:, url:,)]): List containing stream information
        """

        self._update_streams()
        self.rtsp_streams = []
        for stream in self.streams:
            for streamID,v in stream.items():
                stream_info = {"streamID":streamID}
                for substream in v:
                    if substream["isMain"]:
                        url = substream["url"]
                        if url != "":
                            stream_info["name"] = substream["name"]
                            stream_info["url"] = url
                            self.rtsp_streams.append(stream_info)
    
        return self.rtsp_streams

if __name__=="__main__":
    """
    Example Usage of VST
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("vst_addr", type=str, default="", nargs='?', help="VST address (http://0.0.0.0:81/)")
    args = parser.parse_args()

    vst_addr = args.vst_addr 
    vst = VST(vst_addr)

    rtsp_streams = vst.get_rtsp_streams()
    for rs in rtsp_streams:
        print(rs)
