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

import redis 
import datetime 
import json

__all__ = ["SchemaGenerator"]

class SchemaGenerator:

    def __init__(self, **kwargs):
        """
        Creates an object that can be called to generate object detection metadata in metropolis minimal schema 

        Args:
            sensor_id (int): ID of the sensor
            sensor_loc ([float,float,float]): 
            sensor_type (str): Type of sensor 
            redis_stream (str): Redis Stream name to output metadata
            redis_host (str): Redis server host to connect to 
            redis_port (int): Port of redis server

        Returns:
            None
        """
        self.sensor_id = kwargs.get("sensor_id", 1)
        self.sensor_loc = kwargs.get("sensor_loc", [-1,-1,-1])
        self.sensor_type = kwargs.get("sensor_type", "camera")
        self.redis_stream = kwargs.get("redis_stream", None)
        self.redis_port = kwargs.get("redis_port", None)
        self.redis_host = kwargs.get("redis_host", None)
        self.redis_connected = False
        self.frame_counter = 0
        self.id_counter = 0

        #If user supplied redis info then connect
        if self.redis_stream != None and self.redis_port != None and self.redis_stream!= None:
            self.connect_redis(self.redis_host, self.redis_port, self.redis_stream)

        self.frame_id = 0

    def connect_redis(self, host, port, stream):
        """
        Connects object to a redis server. 

        Args:
            host (str): Redis server host ex: "0.0.0.0"
            port (int): Redis server port ex: 6387
            stream (str): Name of redis stream to output ex: "owl"

        Returns:
            none
        """
        self.redis_stream = stream 
        self.redis_server = redis.Redis(host=host, port=port, decode_responses=True)
        self.redis_connected = True

    def _gen_schema(self, objects, bboxes, object_ids = None, frame_id=None):
        if frame_id is None:
            frame_id = self.frame_counter
            self.frame_counter+=1

        timestamp = datetime.datetime.utcnow()
        timestamp = timestamp.isoformat("T")[0:-3] + "Z"
        #sensor_schema = {"sensor":{"id":self.sensor_id, "type":self.sensor_type, "location":{"lat":self.sensor_loc[0], "lon":self.sensor_loc[1], "alt":self.sensor_loc[2]}}}
        
        object_list = []
        if len(objects) != len(bboxes):
            raise Exception("objects and bboxes not the same length.")

        for i in range(len(objects)):
            box = bboxes[i]
            obj = objects[i]
            if object_ids:
                obj_id = object_ids[i]
            else:
                obj_id = self.id_counter
                self.id_counter += 1

            object_schema = f"{obj_id}|{box[0]}|{box[1]}|{box[2]}|{box[3]}|{obj}"
            #object_schema = {"bbox": {"leftX":box[0], "topY":box[1],"rightX":box[2], "bottomY":box[3]}, "type":obj}
            object_list.append(object_schema)
    
        frame_schema = {"version": "4.0", "id":frame_id, "@timestamp":timestamp, "sensorId": self.sensor_id, "objects":object_list}
        frame_schema = json.dumps(frame_schema)
        return frame_schema

    def _redis_out(self, metadata):
        self.redis_server.xadd(self.redis_stream, {"metadata":metadata})
    
    def __call__(self, objects, bboxes):
        """
        Converts list of objects and associated bounding boxes into metropolis minimal schema and outputs on a redis stream if available. 

        Args:
            objects (list[str]): Text to draw for each bounding box.
            bboxes (list[(int,int,int,int), ...]): List of bounding boxes. Each bounding box should be in this format (x1, y1, x2, y2) Where x1,y1 is the top left and x2,y2 is the bottom right coordinate of the bbox. 

        Returns:
            out (str): A serialized json string with the metadata in Metropolis minimal schema. Will also write this to a redis stream if connected.
        """
        out = self._gen_schema(objects, bboxes)
      
        if self.redis_connected:
            self._redis_out(out)

        return out 

if __name__ == "__main__":
    from time import sleep  
    schema_gen = SchemaGenerator()
    schema_gen.connect_redis("0.0.0.0", "6379", "owl")
    for i in range(100):
        objects = ["a person", "a box", "a box", "a vest", "a person"]
        bboxes = [(0,0, 15,15), (20,20,50,50), (10,15, 25,40), (10,21, 12,36), (50, 50, 55, 55)]
        out = schema_gen(objects, bboxes)
        sleep(0.2)
        print(out)

