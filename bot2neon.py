#!/usr/bin/env python
# ----------------------------------------------------------------------------
# Copyright 2016 ooMWoo Inc.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ----------------------------------------------------------------------------
"""
Convert robot training data to Nervana Neon format
    - convert video into series of JPG images
    - convert joystick/motor command to target class for training
        - create list of images and target class for neon
"""

import subprocess

# Extract frames from video and reduce FPS
# avconv -i video0.avi -vf "select=''not(mod(n\,5))'',showinfo,vflip,hflip"  %08d.jpg
datadir = "/home/ilia/neon/bot/data/"
cmd = "rm " + datadir + "*.jpg 2> /dev/null"
subprocess.call(cmd, shell=True)
cmd = "avconv -i " + datadir + "video0.avi -vf \"select=''not(mod(n\,5))'',showinfo,vflip,hflip\"  " +\
      datadir + "%08d.jpg 2>&1 | grep pts"
output = subprocess.check_output(cmd, shell=True)
# Parse output: frame number, pts_time


# Read list of extracted frames and their timestamps

# Read bot commands and their timestamps

# Match frame timestamps to command timestamps

# Write image-and-class list for neon
