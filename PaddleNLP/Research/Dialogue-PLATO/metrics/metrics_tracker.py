#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
MetricsTracker class
"""

from collections import defaultdict


class MetricsTracker(object):
    """ Tracking metrics. """

    def __init__(self):
        self.metrics_val = defaultdict(float)
        self.metrics_avg = defaultdict(float)
        self.num_samples = 0

    def update(self, metrics, num_samples):
        for key, val in metrics.items():
            if val is not None:
                val = float(val)
                self.metrics_val[key] = val
                avg_val = (self.metrics_avg.get(key, 0) * self.num_samples +
                           val * num_samples) / (self.num_samples + num_samples)
                self.metrics_avg[key] = avg_val
        self.num_samples += num_samples

    def clear(self):
        self.metrics_val = defaultdict(float)
        self.metrics_avg = defaultdict(float)
        self.num_samples = 0

    def items(self):
        return self.metrics_avg.items()

    def get(self, name):
        if self.num_samples == 0:
            raise ValueError("There is no data in Metrics.")
        return self.metrics_avg.get(name)

    def state_dict(self):
        return {
            "metrics_val": self.metrics_val,
            "metrics_avg": self.metrics_avg,
            "num_samples": self.num_samples,
        }

    def load_state_dict(self, state_dict):
        self.metrics_val = state_dict["metrics_val"]
        self.metrics_avg = state_dict["metrics_avg"]
        self.num_samples = state_dict["num_samples"]

    def value(self):
        metric_strs = []
        for key, val in self.metrics_val.items():
            metric_str = f"{key.upper()}-{val:.3f}"
            metric_strs.append(metric_str)
        metric_strs = "   ".join(metric_strs)
        return metric_strs

    def summary(self):
        metric_strs = []
        for key, val in self.metrics_avg.items():
            metric_str = f"{key.upper()}-{val:.3f}"
            metric_strs.append(metric_str)
        metric_strs = "   ".join(metric_strs)
        return metric_strs
