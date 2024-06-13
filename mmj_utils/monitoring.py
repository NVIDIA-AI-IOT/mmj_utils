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

from dataclasses import dataclass 
from prometheus_client import start_http_server, Gauge
import logging 
from time import time 


@dataclass 
class Alert:
    """Schema for websocket alerts"""
    string: str 
    state: int #1 or 0
    cooldown: bool 
    trigger_time: int # time.time() stamp 

class AlertMonitor:
    """Class to handle true/false alert value output to prometheus"""
    def __init__(self, num_rules, port, cooldown_time=60):
        self.num_rules = num_rules
        self.prometheus_metric = Gauge("alert_status", "test", ["alert_number", "alert_string"])
        self.prometheus_metric.labels("", "") 
        start_http_server(port)
        self.alerts = {} #{"r0":Alert, ...}
        self.cooldown_time = cooldown_time 

    def set_alert_states(self, new_alert_states):
        """Updates alert states. new_alert_states {"r0": 0, "r1": 1}"""

        for alert_key, new_alert_state in new_alert_states.items():
            
            if alert_key in self.alerts:
                alert = self.alerts[alert_key]

                if self.cooldown_time > 0:
                    #Reset Cooldown
                    if alert.cooldown:
                        if time() - alert.trigger_time > self.cooldown_time:
                            alert.cooldown = False 
                            alert.trigger_time = None 
                    else:
                        if new_alert_state == 1:
                            alert.trigger_time = time() 
                            alert.cooldown = True 

                alert.state = new_alert_state 

                self.alerts[alert_key].state = new_alert_state  #update alert state value 
      
        self._update_prometheus(self.alerts)

    def set_alerts(self, alert_strings):
        """Set alert strings. input alert_strings {"r0": "is there fire?", "r1": "is there smoke?" ...}"""
        self.clear_alerts()

        for key in alert_strings:
            self.alerts[key] = Alert(string=alert_strings[key], state=0, cooldown=False, trigger_time=None)

    def _update_prometheus(self, alerts):
        """Update prometheus metrics with alert states. input alerts dict:  {"r0": Alert, "r1":Alert ...}"""
        for k, v in alerts.items():
            self.prometheus_metric.labels(k, v.string).set(v.state)

    def clear_alerts(self):
        "Set all alerts to false"
        old_alerts = self.alerts 
        self.alerts = {}

        for key in old_alerts:
            old_alerts[key].state = 0
        
        self._update_prometheus(old_alerts)




