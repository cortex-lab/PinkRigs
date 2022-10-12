import pdb

import deeplabcut
import os
import glob
import pandas as pd
import yaml
import re
import shutil
import socket   # to get computer name
import subprocess as sp
import numpy as np

# For accessing files on server
if ('Zelda' not in socket.gethostname()) & ('zelda' not in socket.gethostname()):
    from gi.repository import Gio

def get_pinkrig_pcs(mouse_info_folder, subset_mice_to_use=None,
                    subset_date_range=None, default_server_path=None):


    if subset_mice_to_use is not None:
        mouse_info_csv_paths = []
        for mouse_name in subset_mice_to_use:
            mouse_info_csv_paths.append(
                glob.glob(os.path.join(mouse_info_folder, '%s.csv' % mouse_name))[0]
            )
    else:
        mouse_info_csv_paths = glob.glob(os.path.join(mouse_info_folder, '*.csv'))

    files_to_exclude = []
    pattern_to_match = re.compile('[A-Z][A-Z][0-9][0-9][0-9]')

    for path in mouse_info_csv_paths:
        if os.path.basename(path) in files_to_exclude:
            mouse_info_csv_paths.remove(path)
        fname_without_ext = '.'.split(path)
        if not pattern_to_match.match(fname_without_ext):
            mouse_info_csv_paths.remove(path)

    all_mouse_info = []

    for csv_path in mouse_info_csv_paths:
        mouse_info = pd.read_csv(csv_path)
        mouse_name = os.path.basename(csv_path).split('.')[0]
        mouse_info['subject'] = mouse_name

        if 'path' not in mouse_info.columns:
            mouse_info['server_path'] = default_server_path
        if 'expFolder' in mouse_info.columns:
            server_paths = ['//%s/%s' % (x.split(os.sep)[2], x.split(os.sep)[3]) for x in
                            mouse_info['expFolder'].values]
            mouse_info['server_path'] = server_paths
        if 'path' in mouse_info.columns:
            mouse_info['server_path'] = \
                ['//' + '/'.join(x.split('\\')[2:4]) for x in mouse_info['path']]

        all_mouse_info.append(mouse_info)

    all_mouse_info = pd.concat(all_mouse_info)

    if subset_date_range is not None:
        all_mouse_info = all_mouse_info.loc[
            (all_mouse_info['expDate'] >= subset_date_range[0]) &
            (all_mouse_info['expDate'] <= subset_date_range[1])
            ]

    return all_mouse_info


def get_crop_coordinates(subset_video_path):




    return crop_coordinates
def run_dlc_on_video(input_video_path):
    """

    Parameters
    ----------
    input_video_path

    Returns
    -------

    """


    return 1

def main():
    """
    Parameters
    ----------
    None
    Returns
    -------
    None

    Usage instructions
    ------------------
    If starting from scratch, then run (1) create_project (2) edit_config, (3) extract_frames, (4) label_frames
    """

    access_file_via_server = False
    check_project_made = False  # Check that project is already made
    # main_folder = 'smb://zinu.local/subjects/'
    # main_folder = '/run/user/1000/gvfs/smb-share:server=zinu.local,share=subjects'  # 'smb://zinu.local/subjects/'
    main_folder = '/home/timothysit/local_pinkrigs_DLC_training_videos/'

    # If gvfs is saying things are not directories (eg. zserver), try relaunching gvfs in the terminal
    # and restarting Files / Thunar
    # projectName = 'bscopeDeepLabCut'  # bscope or pinkrigs
    projectName = 'pinkrigs'
    experimenter = 'Tim'
    supported_steps = ['create_project', 'edit_config', 'extract_frames', 'label_frames',
                       'check_labels', 'create_training_set', 'train_network', 'evaluate_network',
                       'analyze_video', 'create_labeled_video', 'filter_predictions', 'plot_trajectories',
                       'extract_outlier_frames', 'add_video', 'upload_model_to_server', 'download_model_from_server',
                       'resize_video', 'extract_subset_of_video', 'run_dlc_pipeline']

    steps_to_run = ['run_dlc_pipeline']

    process_params = {
        'analyze_video': dict(
           only_use_additional_videos=True,
        ),
        'create_labeled_video': dict(
            only_use_additional_videos=True,
        ),
        'filter_predictions': dict(
            only_use_additional_videos=True,
        ),
        'plot_trajectories': dict(
            only_use_additional_videos=True,
        ),
        'extract_outlier_frames': dict(
            only_use_additional_videos=True,
        ),
        'resize_video': dict(
            downsize_factor=2,
        ),
        'extract_subset_of_video': dict(

        ),
        'run_dlc_pipeline': dict(

        )
    }

    if projectName == 'bscopeDeepLabCut':
        working_directory = '/home/timothysit/'
    elif projectName == 'pinkrigs':
        # working_directory = '/run/user/1000/gvfs/smb-share:server=zserver.local,share=code'
        # working_directory = os.path.join(working_directory, 'AVRigDLC')
        if ('Zelda' in socket.gethostname()) | ('zelda' in socket.gethostname()):
            # working_directory = '//zserver/code/AVRigDLC' # 'C:/Users/Experiment/Desktop'
            remote_working_directory = '//zserver/Code/AVRigDLC'
            working_directory = 'C:/Users/Experiment/Desktop'
            # remote_working_directory = working_directory
        else:
            working_directory = '/home/timothysit/AVRigDLC/'
            remote_working_directory = '/run/user/1000/gvfs/smb-share:server=zserver.local,share=code/AVRigDLC'


    project_custom_configs = {
        'bscopeDeepLabCut': dict(
            bodyparts=['eyeL', 'eyeR', 'eyeU', 'eyeD',
                      'pupilL', 'pupilR', 'pupilU', 'pupilD',
                      'whiskPadL', 'whiskPadR'],
            skeleton=[
                      ['eyeL', 'eyeR'],
                      ['eyeU', 'eyeD'],
                      ['pupilL', 'pupilR'],
                      ['pupilU', 'pupilD'],
                      ['whiskPadL', 'whiskPadR']
                     ],
            dotsize=11,
    ),
        'pinkrigs': dict(
            bodyparts=['eyeL', 'eyeR', 'eyeU', 'eyeD',
                       'pupilL', 'pupilR', 'pupilU', 'pupilD',
                       'whiskPadL', 'whiskPadR'],
            skeleton=[
                ['eyeL', 'eyeR'],
                ['eyeU', 'eyeD'],
                ['pupilL', 'pupilR'],
                ['pupilU', 'pupilD'],
                ['whiskPadL', 'whiskPadR']
            ],
            dotsize=2,
        )
    }

    # if projectName == 'pinkRigs':
    #     all_mouse_info = all_mouse_info

    if access_file_via_server:
        # does not work for me for some reason???
        gvfs = Gio.Vfs.get_default()
        # pdb.set_trace()
        main_folder = gvfs.get_file_for_uri(main_folder).get_path()

    if main_folder is None:
        print('Warning: main_folder not found')

    project_video_paths = {
        'bscopeDeepLabCut': [
            os.path.join(main_folder, 'AH001/2021-10-19/2/2021-10-19_2_AH001_eye.mj2'),
            os.path.join(main_folder, 'AH002/2021-10-29/3/2021-10-29_3_AH002_eye.mj2')
        ],
        'pinkrigs': [
            # os.path.join(main_folder, 'JF039/2021-07-30/2/2021-07-30_2_JF039_eyeCam.mj2'),
            # os.path.join(main_folder, 'AP102/2021-07-21/1/2021-07-21_1_AP102_eyeCam.mj2'),
            os.path.join(main_folder, '2022-06-09_3_AV014_eyeCam.mj2'),
            os.path.join(main_folder, '2022-07-25_1_AV020_eyeCam.mj2'),
            os.path.join(main_folder, '2022-08-02_5_AV020_eyeCam.mj2'),
            os.path.join(main_folder, '2022-08-03_4_AV015_eyeCam.mj2'),
        ]
    }

    add_video_paths = [
        os.path.join(main_folder, '2022-07-25_1_AV020_eyeCam.mj2'),
        os.path.join(main_folder, '2022-08-02_5_AV020_eyeCam.mj2'),
        os.path.join(main_folder, '2022-08-03_4_AV015_eyeCam.mj2'),
    ]

    additional_video_paths = {
        'bscopeDeepLabCut':
                    [
                        os.path.join(main_folder, 'AH003/2021-12-10/3/2021-12-10_3_AH003_eye.mj2'),
                    ],
        'pinkrigs': [
            # '/home/timothysit/passive-av-videos/2022-03-18_2_AV003_eyeCam.mj2'
            # os.path.join(main_folder, '2022-06-09_3_AV014_eyeCam.mj2'),
            # os.path.join(main_folder, '2022-07-25_1_AV020_eyeCam.mj2'),
            # os.path.join(main_folder, '2022-08-02_5_AV020_eyeCam.mj2'),
            # os.path.join(main_folder, '2022-08-03_4_AV015_eyeCam.mj2'),
            '//zaru.cortexlab.net/Subjects/AV015/2022-07-18/2/2022-07-18_2_AV015_eyeCam.mj2'
            # '//zaru.cortexlab.net/Subjects/AV015/2022-07-18/2/2022-07-18_2_AV015_eyeCam_downsampled.mp4'
            # 'C:/Users/Experiment/Desktop/pinkrigs-Tim-2022-09-12/videos/2022-07-18_2_AV015_eyeCam.mj2'
        ]
    }


     # Step 1: Create project
    # TODO: how to add videos to project?
    if 'create_project' in steps_to_run:
        project_made = 0
        if check_project_made:
            project_folder_search = glob.glob(os.path.join(working_directory, '%s*' % projectName))
            if len(project_folder_search) > 0:
                print('Found %.f folder with matching project name, will skip creating project unless '
                      'you set check_project_made = False')
                project_made = 1

        if not project_made:
            print('Creating DLC project')
            video_paths = project_video_paths[projectName]

            for v_path in video_paths:
                file_found = os.path.exists(v_path)
                assert file_found

            # may be a problem creating symlink between zserver and znas...
            # see: https://superuser.com/questions/1337257/clients-cant-create-symlinks-on-samba-share
            # also see this: https://unix.stackexchange.com/questions/145636/symlink-from-one-workstation-to-another-without-mount
            # pdb.set_trace()
            deeplabcut.create_new_project(
                projectName, experimenter, [video_paths[0]], working_directory=working_directory
            )

    # Step 2: Edit config YAML files
    if 'edit_config' in steps_to_run:
        project_folder_search = glob.glob(os.path.join(working_directory, '%s*' % projectName))
        project_folder = project_folder_search[0]
        print('Found project in %s' % project_folder)
        print('Editing yaml file with preset specified in runDeepLabCut.py')
        yaml_file_path = os.path.join(project_folder, 'config.yaml')
        with open(yaml_file_path) as f:
            yaml_data = yaml.load(f, Loader=yaml.FullLoader)
            if projectName in project_custom_configs.keys():
                print('Writing custom parameters to config file')
                for key, val in project_custom_configs[projectName].items():
                    yaml_data[key] = val

        # save config
        print('Saving new config file')
        with open(yaml_file_path, 'w') as f:

            yaml.dump(yaml_data, f)


    # Optional : add video to config
    if 'add_video'in steps_to_run:
        project_folder_search = glob.glob(os.path.join(working_directory, '%s*' % projectName))
        project_folder = project_folder_search[0]
        print('Found project in %s' % project_folder)
        print('Editing yaml file with preset specified in runDeepLabCut.py')
        yaml_file_path = os.path.join(project_folder, 'config.yaml')

        deeplabcut.add_new_videos(yaml_file_path, add_video_paths, copy_videos=False)
        print('Added new videos to the config file')

    if 'extract_frames' in steps_to_run:
        project_folder_search = glob.glob(os.path.join(working_directory, '%s*' % projectName))
        project_folder = project_folder_search[0]
        print('Found project in %s' % project_folder)
        yaml_file_path = os.path.join(project_folder, 'config.yaml')
        print('Extracting files using settings from %s' % yaml_file_path)
        # TODO: default to accept yes when prompted
        deeplabcut.extract_frames(yaml_file_path, 'automatic', 'kmeans')

    if 'label_frames' in steps_to_run:
        # TODO: write code to check frames were already extracted
        project_folder_search = glob.glob(os.path.join(working_directory, '%s*' % projectName))
        project_folder = project_folder_search[0]
        print('Found project in %s' % project_folder)
        yaml_file_path = os.path.join(project_folder, 'config.yaml')
        print('Labelling files using settings from %s' % yaml_file_path)
        # Step 3: label frames
        # Go to labeled data and start working
        deeplabcut.label_frames(yaml_file_path)

    if 'check_labels' in steps_to_run:
        project_folder_search = glob.glob(os.path.join(working_directory, '%s*' % projectName))
        project_folder = project_folder_search[0]
        print('Found project in %s' % project_folder)
        yaml_file_path = os.path.join(project_folder, 'config.yaml')
        print('Checking labels using settings from %s' % yaml_file_path)
        # check labels
        deeplabcut.check_labels(yaml_file_path)

    if 'create_training_set' in steps_to_run:
         # Create training set
         project_folder_search = glob.glob(os.path.join(working_directory, '%s*' % projectName))
         project_folder = project_folder_search[0]
         print('Found project in %s' % project_folder)
         yaml_file_path = os.path.join(project_folder, 'config.yaml')
         print('Creating training using settings from %s' % yaml_file_path)
         deeplabcut.create_training_dataset(yaml_file_path)

    if 'train_network' in steps_to_run:
         # Train network
         project_folder_search = glob.glob(os.path.join(working_directory, '%s*' % projectName))
         project_folder = project_folder_search[0]
         print('Found project in %s' % project_folder)
         yaml_file_path = os.path.join(project_folder, 'config.yaml')
         print('Training network using settings from %s' % yaml_file_path)
         deeplabcut.train_network(yaml_file_path)

    if 'evaluate_network' in steps_to_run:
        project_folder_search = glob.glob(os.path.join(working_directory, '%s*' % projectName))
        project_folder = project_folder_search[0]
        print('Found project in %s' % project_folder)
        yaml_file_path = os.path.join(project_folder, 'config.yaml')
        print('Evaluating network using settings from %s' % yaml_file_path)
        deeplabcut.evaluate_network(yaml_file_path, Shuffles=[1], plotting=True)

    if 'analyze_video' in steps_to_run:
        project_folder_search = glob.glob(os.path.join(working_directory, '%s*' % projectName))
        project_folder = project_folder_search[0]
        yaml_file_path = os.path.join(project_folder, 'config.yaml')
        print('Analyzing videos using settings from %s' % yaml_file_path)
        all_video_paths = []
        all_video_paths.extend(additional_video_paths[projectName])
        if not process_params['analyze_video']['only_use_additional_videos']:
            all_video_paths.extend(project_video_paths[projectName])

        deeplabcut.analyze_videos(yaml_file_path, all_video_paths, dynamic=(False, 0.5, 10),
                                  save_as_csv=True)
        # 2022-09-26: Dynamic cropping seems to decrease stanility of ROIs

    if 'create_labeled_video' in steps_to_run:
        project_folder_search = glob.glob(os.path.join(working_directory, '%s*' % projectName))
        project_folder = project_folder_search[0]
        yaml_file_path = os.path.join(project_folder, 'config.yaml')
        print('Creating labeled video')
        all_video_paths = []
        all_video_paths.extend(additional_video_paths[projectName])
        if not process_params['create_labeled_video']['only_use_additional_videos']:
            all_video_paths.extend(project_video_paths[projectName])
        deeplabcut.create_labeled_video(yaml_file_path, all_video_paths)

    if 'filter_predictions' in steps_to_run:
        print('Filtering predictions')
        project_folder_search = glob.glob(os.path.join(working_directory, '%s*' % projectName))
        project_folder = project_folder_search[0]
        yaml_file_path = os.path.join(project_folder, 'config.yaml')

        all_video_paths = []
        all_video_paths.extend(additional_video_paths[projectName])
        if not process_params['filter_predictions']['only_use_additional_videos']:
            all_video_paths.extend(project_video_paths[projectName])

        deeplabcut.filterpredictions(yaml_file_path, all_video_paths)

    if 'plot_trajectories' in steps_to_run:
        print('Plotting trajectories')
        project_folder_search = glob.glob(os.path.join(working_directory, '%s*' % projectName))
        project_folder = project_folder_search[0]
        yaml_file_path = os.path.join(project_folder, 'config.yaml')
        all_video_paths = []
        all_video_paths.extend(additional_video_paths[projectName])
        if not process_params['plot_trajectories']['only_use_additional_videos']:
            all_video_paths.extend(project_video_paths[projectName])
        deeplabcut.plot_trajectories(yaml_file_path, all_video_paths)

    if 'extract_outlier_frames' in steps_to_run:
        print('Extracting outlier frames')
        project_folder_search = glob.glob(os.path.join(working_directory, '%s*' % projectName))
        project_folder = project_folder_search[0]
        yaml_file_path = os.path.join(project_folder, 'config.yaml')
        all_video_paths = []
        all_video_paths.extend(additional_video_paths[projectName])
        if not process_params['extract_outlier_frames']['only_use_additional_videos']:
            all_video_paths.extend(project_video_paths[projectName])
        deeplabcut.extract_outlier_frames(yaml_file_path, all_video_paths)

    if 'upload_model_to_server' in steps_to_run:

        print('Uploading model to server')
        print('Server location specified: %s' % remote_working_directory)
        project_folder_search = glob.glob(os.path.join(working_directory, '%s*' % projectName))
        project_folder = project_folder_search[0]
        project_folder_name = os.path.basename(project_folder)
        shutil.copytree(project_folder, os.path.join(remote_working_directory, project_folder_name))
        print('Finished copying model to server')

    if 'download_model_from_server' in steps_to_run:

        print('Copying model from server to local computer')
        remote_project_folder_search = glob.glob(os.path.join(remote_working_directory, '%s*' % projectName))
        project_folder_name = os.path.basename(remote_project_folder_search[0])
        local_project_path = os.path.join(working_directory, project_folder_name)
        shutil.copytree(remote_project_folder_search[0], local_project_path)
        print('Finished copying model to local computer')

    if 'resize_video' in steps_to_run:

        print('Resizing video')
        downsize_factor = process_params['resize_video']['downsize_factor']
        ffmpeg_path = 'C:/Users/Experiment/.conda/envs/DEEPLABCUT/Library/bin/ffmpeg.exe'

        for input_video_path in additional_video_paths[projectName]:
            output_video_folder = os.path.dirname(input_video_path)
            input_video_name = os.path.basename(input_video_path)
            input_name_components = input_video_name.split('.')
            output_video_name = input_name_components[0] + '_downsampled' + '.' + 'mp4' # input_name_components[1]
            output_video_path = os.path.join(output_video_folder, output_video_name)
            ffmpeg_args = [ffmpeg_path,
                           '-i', input_video_path,
                           '-vf', "scale='iw/%.f:ih/%.f'" % (downsize_factor, downsize_factor),
                           output_video_path]
            sp.call(ffmpeg_args)
    if 'extract_subset_of_video' in steps_to_run:
        print('Extracting subset of video')
        ffmpeg_path = 'C:/Users/Experiment/.conda/envs/DEEPLABCUT/Library/bin/ffmpeg.exe'

        for input_video_path in additional_video_paths[projectName]:
            output_video_folder = os.path.dirname(input_video_path)
            input_video_name = os.path.basename(input_video_path)
            input_name_components = input_video_name.split('.')
            output_video_name = input_name_components[0] + '_subset' + '.' + 'mp4'  # input_name_components[1]
            output_video_path = os.path.join(output_video_folder, output_video_name)
            ffmpeg_args = [ffmpeg_path,
                           '-ss', '0',
                           '-i', input_video_path,
                           '-c', 'copy',
                           '-t', '10',
                           output_video_path]
            sp.call(ffmpeg_args)
    if 'run_dlc_pipeline' in steps_to_run:

        ffmpeg_path = 'C:/Users/Experiment/.conda/envs/DEEPLABCUT/Library/bin/ffmpeg.exe'

        for input_video_path in additional_video_paths[projectName]:
            output_video_folder = os.path.dirname(input_video_path)
            input_video_name = os.path.basename(input_video_path)
            input_name_components = input_video_name.split('.')

            # subset video
            subset_video_name_without_ext = input_name_components[0] + '_subset'
            subset_video_name = subset_video_name_without_ext + '.' + 'mp4'
            subset_video_path = os.path.join(output_video_folder, subset_video_name)
            ffmpeg_args = [ffmpeg_path,
                           '-ss', '0',
                           '-y',  # overwrite video if existing subset video exists
                           '-i', input_video_path,
                           '-c', 'copy',
                           '-t', '10',
                           subset_video_path]
            sp.call(ffmpeg_args)

            # run deeplabcut on subset video
            project_folder_search = glob.glob(os.path.join(working_directory, '%s*' % projectName))
            project_folder = project_folder_search[0]
            yaml_file_path = os.path.join(project_folder, 'config.yaml')

            deeplabcut.analyze_videos(yaml_file_path, [subset_video_path], dynamic=(False, 0.5, 10),
                                      save_as_csv=True)

            # TODO: read csv file to get the coordinates to crop
            subset_vid_h5_path = glob.glob(os.path.join(
                output_video_folder,
                '*%s*.h5' % (subset_video_name_without_ext)
            ))[0]

            subset_vid_output_df = pd.read_hdf(subset_vid_h5_path) # pandas multindex
            scorer_name = 'DLC_resnet50_pinkrigsSep12shuffle1_50000'

            pad_pixels = 20
            eyeL_xvals = np.array([x[(scorer_name, 'eyeL', 'x')] for (_, x) in subset_vid_output_df.iterrows()])
            eyeR_xvals = np.array([x[(scorer_name, 'eyeR', 'x')] for (_, x) in subset_vid_output_df.iterrows()])

            eyeU_yvals = np.array([x[(scorer_name, 'eyeU', 'y')] for (_, x) in subset_vid_output_df.iterrows()])
            eyeD_yvals = np.array([x[(scorer_name, 'eyeD', 'y')] for (_, x) in subset_vid_output_df.iterrows()])


            eyeL_xvals[eyeL_xvals < 0] = np.mean(eyeL_xvals)
            eyeR_xvals[eyeR_xvals < 0] = np.mean(eyeR_xvals)
            eyeU_yvals[eyeU_yvals < 0] = np.mean(eyeU_yvals)
            eyeD_yvals[eyeD_yvals < 0] = np.mean(eyeD_yvals)

            crop_window = [
                np.mean(eyeL_xvals) - pad_pixels,
                np.mean(eyeR_xvals) + pad_pixels,
                np.mean(eyeU_yvals) - pad_pixels,
                np.mean(eyeD_yvals) + pad_pixels,
            ]

            # pdb.set_trace()

            # update config file with parameters
            yaml_file_path = os.path.join(project_folder, 'config.yaml')
            with open(yaml_file_path) as f:
                yaml_data = yaml.load(f, Loader=yaml.FullLoader)
                yaml_data['cropping'] = True
                yaml_data['x1'] = int(crop_window[0])
                yaml_data['x2'] = int(crop_window[1])
                yaml_data['y1'] = int(crop_window[2])
                yaml_data['y2'] = int(crop_window[3])

            # save config
            print('Saving new config file')
            with open(yaml_file_path, 'w') as f:

                yaml.dump(yaml_data, f)

            #  pdb.set_trace()
            deeplabcut.analyze_videos(yaml_file_path, [input_video_path], dynamic=(False, 0.5, 10),
                                      save_as_csv=True) # cropping=crop_window)
            deeplabcut.create_labeled_video(yaml_file_path, [input_video_path],
                                            displaycropped=True)
            # pdb.set_trace()





if __name__ == '__main__':
    main()


