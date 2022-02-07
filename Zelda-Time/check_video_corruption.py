import pdb

import numpy as np
import os
import glob
import pandas as pd
import facemap
from facemap import utils, process
from tqdm import tqdm
import time
import cv2
from collections import defaultdict
# some facemap process stuff
from io import StringIO

# For accessing files on server
# The dependencies are not obvious,
# see: https://askubuntu.com/questions/80448/what-would-cause-the-gi-module-to-be-missing-from-python
# from gi.repository import Gio

# Plotting
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.use('Agg')  # to play nicely with pyqt: see: https://stackoverflow.com/questions/33051790/could-not-find-or-load-the-qt-platform-plugin-xcb

import subprocess as sp

import datetime

def check_file_corrupted(vid_path):
    """
    Checks if vid_path (tested on mj2 videos) is corrupted
    by reading a frame from it and see if anything returns
    Parameters
    ----------
    vid_path (str)
        path to the video file eg. /my/folder/video.mj2

    Returns
    -------

    """
    vid_corrupted = 0
    try:
        vid = cv2.VideoCapture(vid_path)
        if not vid.isOpened():
            vid_corrupted = 1

        # read (next) frame
        ret, frame = vid.read()

        if not ret:
            vid_corrupted = 1
    except:
        vid_corrupted = 1

    return vid_corrupted


def main():
    print('Checking for corrupted files')

    print_summary = True
    main_info_folder_in_server = False
    load_from_server = False
    video_ext = '.mj2'
    mouse_info_folder = '//zserver/Code/AVrig'
    default_server_path = '//zinu/Subjects'  # '//128.40.224.65/subjects/'
    all_video_info = defaultdict(list)
    # subset_mice_to_use = ['TS011', 'SP013']
    subset_mice_to_use = None  # ['CB018', 'CB019', 'CB020']  # None if no subsetting
    subset_date_range = ['2022-02-07', '2022-02-07']

    if main_info_folder_in_server:
        gvfs = Gio.Vfs.get_default()
        mouse_info_folder = gvfs.get_file_for_uri(mouse_info_folder).get_path()


    if subset_mice_to_use is not None:
        mouse_info_csv_paths = []
        for mouse_name in subset_mice_to_use:
            mouse_info_csv_paths.append(
                glob.glob(os.path.join(mouse_info_folder, '%s.csv' % mouse_name))[0]
            )
    else:
        mouse_info_csv_paths = glob.glob(os.path.join(mouse_info_folder, '*.csv'))

    files_to_exclude = ['aMasterMouseList.csv', 'video_corruption_check.csv',
                        'kilosort_queue.csv']

    for path in mouse_info_csv_paths:
        if os.path.basename(path) in files_to_exclude:
            mouse_info_csv_paths.remove(path)

    all_mouse_info = []

    for csv_path in mouse_info_csv_paths:
        mouse_info = pd.read_csv(csv_path)
        mouse_name = os.path.basename(csv_path).split('.')[0]
        mouse_info['subject'] = mouse_name

        if 'path' not in mouse_info.columns:
            mouse_info['server_path'] = default_server_path
        else:
            mouse_info['server_path'] = ['//' + '/'.join(x.split('\\')[2:4]) for x in mouse_info['path']]

        all_mouse_info.append(mouse_info)

    all_mouse_info = pd.concat(all_mouse_info)

    if subset_date_range is not None:
        all_mouse_info = all_mouse_info.loc[
            (all_mouse_info['expDate'] >= subset_date_range[0]) &
            (all_mouse_info['expDate'] <= subset_date_range[1])
        ]

    # loop through the experiments and see if there are videos with no corresponding facemap output
    file_skipped = 0
    tot_video_files = 0
    for row_idx, exp_info in tqdm(all_mouse_info.iterrows()):
        # get list of files from the exp folder
        if load_from_server:
            gvfs = Gio.Vfs.get_default()
            main_folder = exp_info['server_path']
            exp_info['main_folder'] = gvfs.get_file_for_uri(main_folder).get_path()

            if exp_info['main_folder'] is None:
                print('WARNING: main folder not file for server path: %s' % exp_info['server_path'])
        else:
            main_folder = exp_info['server_path']
            exp_info['main_folder'] = main_folder

        if type(exp_info['expNum']) is not int:
            exp_info['expNum'] = int(exp_info['expNum'])

        exp_folder = os.path.join(exp_info['main_folder'], exp_info['subject'],
                                  exp_info['expDate'], str(exp_info['expNum']))
        # look for video files
        video_files = glob.glob(os.path.join(exp_folder, '*%s' % video_ext))

        # remove the *lastFrames.mj2 videos
        video_files = [x for x in video_files if 'lastFrames' not in x]
        video_file_fov_names = [os.path.basename(x).split('_')[3].split('.')[0] for x in video_files]

        # check if video files are corrupted
        for video_fpath, video_fov in zip(video_files, video_file_fov_names):
            vid_corrupted = check_file_corrupted(vid_path=video_fpath)

            # check if lastFrame exist
            fov_files = glob.glob(os.path.join(exp_folder, '*%s*%s' % (video_fov, video_ext)))
            lastFrameVidExist = any(['lastFrames' in x for x in fov_files])
            # Add information to all_video_info
            all_video_info['rigName'].append(exp_info['rigName'])
            all_video_info['subject'].append(exp_info['subject'])
            all_video_info['expNum'].append(exp_info['expNum'])
            all_video_info['expDate'].append(exp_info['expDate'])
            all_video_info['expDuration'].append(exp_info['expDuration'])
            all_video_info['ephys'].append(exp_info['ephys'])
            all_video_info['video_fov'].append(video_fov)
            all_video_info['video_fpath'].append(
                video_fpath
            )
            all_video_info['fileSizeBytes'].append(
                exp_info[video_fov]
            )
            all_video_info['lastFrameVidExist'].append(
                lastFrameVidExist
            )
            all_video_info['vid_corrupted'].append(
                vid_corrupted
            )
            all_video_info['expDef'].append(exp_info['expDef'])

    all_video_info = pd.DataFrame.from_dict(all_video_info)

    if print_summary:
        num_mice = len(np.unique(all_video_info['subject']))
        print('Checked %.f video files in %.f mice' % (len(all_video_info), num_mice))
        print('Found %.f corrupted video files' % (np.sum(all_video_info['vid_corrupted'])))

    save_file_path = os.path.join(
        mouse_info_folder, 'video_corruption_check.csv'
    )
    print('Saving results to %s' % save_file_path)
    all_video_info.to_csv(save_file_path)

if __name__ == '__main__':
    main()