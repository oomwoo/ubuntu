#!/usr/bin/env python
# Convert recordings captured by a robot to Nervana Neon suite
# for training a neural network
# The robot runs on Raspberry Pi and ROBOTC for VEX EDR
# See more at https://github.com/oomwoo/
#
# Copyright (C) 2016 oomwoo.com
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License version 3.0
# as published by the Free Software Foundation.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License <http://www.gnu.org/licenses/> for details.
"""
Convert robot training data to Nervana Neon format
    - convert video into series of JPG images
    - convert joystick/motor command to target class for training
        - create list of images and target class for neon
"""

import sys, subprocess, getopt, glob, re, os, random


def usage():
    print "python bot2neon.py"
    print "  Convert video and control recordings into training Neon sets"
    print "  -p rec: log and video file name prefix"
    print "  -f 5: video FPS, default 5"
    print "  -d: display received commands for debug"
    print "  -m: horizontal mirror"
    print "  -v: vertical mirror"
    print "  -?: print usage"


def extract_command_value(s, cmd):
    i = s.find(cmd)
    if i == -1:
        return -1
    return int(s[i+1:i+3], 16)


def extract_time(s):
    i = s.find(' ')
    return float(s[0:i])


def rm_files(file_path_name):
    command = "rm -r " + file_path_name + " 2> /dev/null"
    subprocess.call(command, shell=True)


def create_dir(dir_path_name):
    command = "mkdir " + dir_path_name + " 2> /dev/null"
    subprocess.call(command, shell=True)


hor_flip = False
ver_flip = False
debug = False
fps = 5
file_name_prefix = "rec"
validation_pct = 10

opts, args = getopt.getopt(sys.argv[1:], "dfpmv?")

for opt, arg in opts:
    if opt == '-d':
        debug = True
    elif opt == '-p':
        file_name_prefix = arg
    elif opt == '-f':
        fps = int(arg)
    elif opt == '-m':
        hor_flip = not(hor_flip)
    elif opt == '-v':
        ver_flip = not(ver_flip)
    elif opt == '-?':
        usage()
        sys.exit(2)


# avconv -i video0.avi -vf "select=''not(mod(n\,5))'',showinfo,vflip,hflip"  %08d.jpg
home_dir = os.path.expanduser("~")
rec_dir = home_dir + "/ubuntu/rec/"
dataset_dir = home_dir + "/ubuntu/dataset/"

# Wipe data set directory
rm_files(dataset_dir)
create_dir(dataset_dir)

# TODO process multiple sets of recordings
# Glob recordings
file_names = glob.glob(rec_dir + file_name_prefix + "[0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9].h264")
rec_numbers = [int((re.findall('\d+', s))[0]) for s in file_names]

start_frame_number = 1
csv = []
class_list = []

# Iterate over each recording file set
for rec_num in rec_numbers:
    rec_file_name = file_name_prefix + str(rec_num).zfill(8)

    # Delete old files
    rm_files(rec_dir + "*.jpg")

    # Extract frames from video(s)
    cmd = "avconv -i " + rec_dir + rec_file_name + ".h264 -vf \"showinfo"
    if hor_flip:
        cmd = cmd + ",hflip"
    if ver_flip:
        cmd = cmd + ",vflip"
    cmd = cmd + "\"  -start_number " + str(start_frame_number) + " " + rec_dir + "%08d.jpg 2>&1 | grep pts"
    output = subprocess.check_output(cmd, shell=True)

    # Read list of extracted frames and their timestamps
    file_names = glob.glob(rec_dir + "[0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9].jpg")
    numbers = [int((re.findall('\d+', s))[0]) for s in file_names]

    # Read bot commands and their timestamps
    log_file_name = rec_dir + rec_file_name + ".txt"
    with open(log_file_name) as f:
        lines = f.read().splitlines()

    # note time of the 1st log entry
    log_start_time = extract_time(lines[0])

    # note time of last log entry
    log_end_time = extract_time(lines[-1])

    # One frame time
    frame_period = (log_end_time - log_start_time) / len(numbers)

    frame_times = [(n - start_frame_number) * frame_period for n in numbers]
    print rec_file_name + ".h264 FPS " + "{:.2f}".format(frame_period)

    frame_file_names = []

    for l in lines:
        # line has user command?
        i = extract_command_value(l, 'u')
        if i < 0:
            continue
        # get time in seconds from the moment log started
        t = extract_time(l) - log_start_time
        # find image number best matching that time
        frame_times_diff = [abs(ft - t) for ft in frame_times]
        frame_idx = frame_times_diff.index(min(frame_times_diff))
        frame_no = numbers[frame_idx]

        # Create class sub-directory
        subdir_name = str(i)
        if i not in class_list:
            create_dir(dataset_dir + subdir_name)
            class_list.append(i)

        # Add image name + target class to Neon image list
        file_name = str(frame_no).zfill(8) + ".jpg"
        if file_name in frame_file_names:
            continue
        s = subdir_name + '/' + file_name + ", " + str(i)
        csv.append(s)
        frame_file_names.append(file_name)

        # Move jpg file
        src = rec_dir + file_name
        dst = dataset_dir + subdir_name + '/' + file_name
        os.rename(src, dst)

    start_frame_number = max(numbers) + 1

# Shuffle csv entries
random.shuffle(csv)

# Write training list
k = int(len(csv) * (100 - validation_pct) / 100.0)
with open(dataset_dir + "train_file.csv", 'w') as f:
    f.write("\n".join(csv[0:k]))

# Write validation list
with open(dataset_dir + "val_file.csv", 'w') as f:
    f.write("\n".join(csv[k+1:-1]))

# Delete unused frames
rm_files(rec_dir + "*.jpg")

# gzip csv files
cmd = "gzip  " + dataset_dir + "*.csv 2> /dev/null"
subprocess.call(cmd, shell=True)

# python neon/data/batch_writer.py  --rec_dir /usr/local/data/macrobatch_out \
#                                   --image_dir /location/of/csv_files \
#                                   --set_type csv
