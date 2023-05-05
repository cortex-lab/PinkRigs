import pdb

import numpy as np
import os
import socket
import glob
import pandas as pd
from facemap import utils, process
from tqdm import tqdm
import time
import cv2
from pathlib import Path  # I have a distinct aversion for os.path.join.

# some facemap process stuff
from io import StringIO

# For matching strings to find mouse csvs (and exclude others)
import re

# For accessing files on server (if running this code on a unix-based machine)
# The dependencies are not obvious,
# see: https://askubuntu.com/questions/80448/what-would-cause-the-gi-module-to-be-missing-from-python
from gi.repository import Gio

# Plotting
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.use('Agg')  # to play nicely with pyqt:
# see: https://stackoverflow.com/questions/33051790/could-not-find-or-load-the-qt-platform-plugin-xcb
import sciplotlib.style as splstyle

import subprocess as sp

import datetime
import natsort


# Run deeplabcut to get anchor points to draw ROI
import deeplabcut
import tensorflow as tf

# other video processing
import skvideo.io

# Pink rig dependencies
from pathlib import Path
import sys
pinkRig_path = glob.glob(r'C:\Users\*\Documents\Github\PinkRigs')
pinkRig_path = Path(pinkRig_path[0])
sys.path.insert(0, (pinkRig_path.__str__()))

from Admin.csv_queryExp import queryCSV, Bunch 

"""
This is a modified version of automatic_facemap.py, combined with batch_process_pinkavrig_videos.py
Instead of running all the time in the background, this script expects to be called 
some time after 8 pm everyday, and will stop running when it is after 8 am the next day.
To call this from matlab: 

(1) Find the path to the python interpreter with the facemap-installed environment 
(2) On a matlab script, do system('path/to/bin/python' evening_facemap.py)

"""


def check_file_corrupted(vid_path):
    """
    Checks if vid_path (tested on mj2 videos) is corrupted
    by reading a frame from it and see if anything returns
    Parameters
    ----------
    vid_path : str
        path to the video to check corruption

    Returns
    -------
    vid_corrupted : int
        1 means the file is corrupted
        0 means the file is not corrupted
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


def run_facemap(video_fpath):
    """
    Copy of the facemap code
    https://github.com/MouseLand/facemap/blob/main/tutorial.ipynb
    Also reference process.run here so that the output format is the same
    https://github.com/MouseLand/facemap/blob/0e8fa78ee7f2c48cc73221ff9f3cfbec43d88957/facemap/process.py#L489
    Parameters
    ----------
    video_fpath : str
        path to the video to be preocessed
    Returns
    -------
    None
    """

    video = pims.Video(video_fpath)
    Ly = video.frame_shape[0]
    Lx = video.frame_shape[1]
    # number of frames in the movie
    nframes = len(video)

    # get subsampled mean across frames
    # grab up to 2000 frames to average over for mean

    nf = min(2000, nframes)

    # load in chunks of up to 200 frames (for speed)
    nt0 = min(200, nframes)
    nsegs = int(np.floor(nf / nt0))

    # what times to sample
    tf = np.floor(np.linspace(0, nframes - nt0, nsegs)).astype(int)

    avgframe = np.zeros((Ly, Lx), np.float32)
    avgmotion = np.zeros((Ly, Lx), np.float32)

    ns = 0
    for n in range(nsegs):
        t = tf[n]

        im = np.array(video[t:t + nt0])
        # im is TIME x Ly x Lx x 3 (3 is RGB)
        if im.ndim > 3:
            im = im[:, :, :, 0]
        # convert im to Ly x Lx x TIME
        im = np.transpose(im, (1, 2, 0)).astype(np.float32)

        # most movies have integer values
        # convert to float to average
        im = im.astype(np.float32)

        # add to averages
        avgframe += im.mean(axis=-1)
        immotion = np.abs(np.diff(im, axis=-1))
        avgmotion += immotion.mean(axis=-1)
        ns += 1

    avgframe /= float(ns)
    avgmotion /= float(ns)

    # compute incremental SVD across frames
    # load chunks of 1000 and take 250 PCs from each
    # then concatenate and take SVD of compilation of 250 PC chunks
    # number of components kept from SVD is ncomps
    ncomps = 500

    nt0 = min(1000, nframes)  # chunk size
    nsegs = int(min(np.floor(25000 / nt0), np.floor(nframes / nt0)))
    nc = 250  # <- how many PCs to keep in each chunk

    # what times to sample
    tf = np.floor(np.linspace(0, nframes - nt0 - 1, nsegs)).astype(int)

    # giant U that we will fill up with smaller SVDs
    U = np.zeros((Ly * Lx, nsegs * nc), np.float32)

    for n in range(nsegs):
        t = tf[n]
        im = np.array(video[t:t + nt0])
        # im is TIME x Ly x Lx x 3 (3 is RGB)
        if im.ndim > 3:
            im = im[:, :, :, 0]
        # convert im to Ly x Lx x TIME
        im = np.transpose(im, (1, 2, 0)).astype(np.float32)

        # most movies have integer values
        # convert to float to average
        im = im.astype(np.float32)

        im = np.abs(np.diff(im, axis=-1))
        im = np.reshape(im, (Ly * Lx, -1))

        # subtract off average motion
        im -= avgmotion.flatten()[:, np.newaxis]

        # take SVD
        usv = utils.svdecon(im, k=nc)

        U[:, n * nc:(n + 1) * nc] = usv[0]

    # take SVD of concatenated spatial PCs
    ### USV = ???
    USV = utils.svdecon(U, k=ncomps)
    U = USV[0]

    ### when do these spatial PCs occur in time?
    # project spatial PCs onto movies (in chunks again)

    ncomps = U.shape[1]
    nt0 = min(1000, nframes)  # chunk size
    nsegs = int(np.ceil(nframes / nt0))

    # time ranges
    itimes = np.floor(np.linspace(0, nframes, nsegs + 1)).astype(int)

    # projection of spatial PCs onto movie
    motSVD = np.zeros((nframes, ncomps), np.float32)

    for n in range(nsegs):
        im = np.array(video[itimes[n]:itimes[n + 1]])
        # im is TIME x Ly x Lx x 3 (3 is RGB)
        if im.ndim > 3:
            im = im[:, :, :, 0]
        # convert im to Ly x Lx x TIME
        im = np.transpose(im, (1, 2, 0)).astype(np.float32)

        im = np.reshape(im, (Ly * Lx, -1))

        # we need to keep around the last frame for the next chunk
        if n > 0:
            im = np.concatenate((imend[:, np.newaxis], im), axis=-1)
        imend = im[:, -1]
        im = np.abs(np.diff(im, axis=-1))

        # subtract off average motion
        im -= avgmotion.flatten()[:, np.newaxis]

        # project U onto immotion
        vproj = im.T @ U
        if n == 0:
            vproj = np.concatenate((vproj[0, :][np.newaxis, :], vproj), axis=0)

        motSVD[itimes[n]:itimes[n + 1], :] = vproj

    # Save output
    proc = {
        'filenames': filenames, 'save_path': savepath, 'Ly': Ly, 'Lx': Lx,
        'sbin': sbin, 'fullSVD': fullSVD, 'save_mat': save_mat,
        'Lybin': Lybin, 'Lxbin': Lxbin,
        'sybin': sybin, 'sxbin': sxbin, 'LYbin': LYbin, 'LXbin': LXbin,
        'avgframe': avgframe, 'avgmotion': avgmotion,
        'avgframe_reshape': avgframe_reshape, 'avgmotion_reshape': avgmotion_reshape,
        'motion': M, 'motSv': S_mot, 'movSv': S_mov,
        'motMask': U_mot, 'movMask': U_mov,
        'motMask_reshape': U_mot_reshape, 'movMask_reshape': U_mov_reshape,
        'motSVD': V_mot, 'movSVD': V_mov,
        'pupil': pups, 'running': runs, 'blink': blinks, 'rois': rois,
        'sy': sy, 'sx': sx
    }
    # save processing
    savename = save(proc, savepath)

    return None


def update_mainwindow(MainWindow, GUIobject, s, prompt):
    """
    Facemap GUI function
    Parameters
    ----------
    MainWindow
    GUIobject
    s
    prompt

    Returns
    -------

    """
    if MainWindow is not None and GUIobject is not None:
        message = s.getvalue().split('\x1b[A\n\r')[0].split('\r')[-1]
        MainWindow.update_status_bar(prompt + message, update_progress=True)
        GUIobject.QApplication.processEvents()


def spatial_bin(im, sbin, Lyb, Lxb):
    """
    Facemap processing function
    Parameters
    ----------
    im
    sbin
    Lyb
    Lxb

    Returns
    -------

    """
    imbin = im.astype(np.float32)
    if sbin > 1:
        imbin = (np.reshape(im[:, :Lyb * sbin, :Lxb * sbin], (-1, Lyb, sbin, Lxb, sbin))).mean(axis=-1).mean(axis=-2)
    imbin = np.reshape(imbin, (-1, Lyb * Lxb))
    return imbin


def imall_init(nfr, Ly, Lx):
    """
    Facemap processing function
    Parameters
    ----------
    nfr
    Ly
    Lx

    Returns
    -------

    """
    imall = []
    for n in range(len(Ly)):
        imall.append(np.zeros((nfr, Ly[n], Lx[n]), 'uint8'))
    return imall


def compute_SVD(containers, cumframes, Ly, Lx, avgframe, avgmotion, motSVD=True, movSVD=False,
                ncomps=500, sbin=3, rois=None, fullSVD=True, GUIobject=None, MainWindow=None):
    """
    Facemap processing function
    # compute the SVD over frames in chunks, combine the chunks and take a mega-SVD
    # number of components kept from SVD is ncomps
    # the pixels are binned in spatial bins of size sbin
    # cumframes: cumulative frames across videos
    # Flags for motSVD and movSVD indicate whether to compute SVD of raw frames and/or
    #   difference of frames over time
    # Return:
    #       U_mot: motSVD
    #       U_mov: movSVD
    Parameters
    ----------
    containers
    cumframes
    Ly
    Lx
    avgframe
    avgmotion
    motSVD
    movSVD
    ncomps
    sbin
    rois
    fullSVD
    GUIobject
    MainWindow

    Returns
    -------

    """

    sbin = max(1, sbin)
    nframes = cumframes[-1]

    # load in chunks of up to 1000 frames
    nt0 = min(1000, nframes)
    nsegs = int(min(np.floor(15000 / nt0), np.floor(nframes / nt0)))
    nc = int(250)  # <- how many PCs to keep in each chunk
    nc = min(nc, nt0 - 1)
    if nsegs == 1:
        nc = min(ncomps, nt0 - 1)
    # what times to sample
    tf = np.floor(np.linspace(0, nframes - nt0 - 1, nsegs)).astype(int)

    # binned Ly and Lx and their relative inds in concatenated movies
    Lyb, Lxb, ir = process.binned_inds(Ly, Lx, sbin)
    if fullSVD:
        U_mot = [np.zeros(((Lyb * Lxb).sum(), nsegs * nc), np.float32)]
        U_mov = [np.zeros(((Lyb * Lxb).sum(), nsegs * nc), np.float32)]
    else:
        U_mot = [np.zeros((0, 1), np.float32)]
        U_mov = [np.zeros((0, 1), np.float32)]
    nroi = 0
    motind = []
    ivid = []

    ni_mot = []
    ni_mot.append(0)
    ni_mov = []
    ni_mov.append(0)

    if rois is not None:
        for i, r in enumerate(rois):
            ivid.append(r['ivid'])
            if r['rind'] == 1:
                nroi += 1
                motind.append(i)
                nyb = r['yrange_bin'].size
                nxb = r['xrange_bin'].size
                U_mot.append(np.zeros((nyb * nxb, nsegs * min(nc, nyb * nxb)), np.float32))
                U_mov.append(np.zeros((nyb * nxb, nsegs * min(nc, nyb * nxb)), np.float32))
                ni_mot.append(0)
                ni_mov.append(0)
    ivid = np.array(ivid).astype(np.int32)
    motind = np.array(motind)

    ns = 0
    w = StringIO()
    for n in tqdm(range(nsegs), file=w):
        img = imall_init(nt0, Ly, Lx)
        t = tf[n]
        utils.get_frames(img, containers, np.arange(t, t + nt0), cumframes)
        if fullSVD:
            imall_mot = np.zeros((img[0].shape[0] - 1, (Lyb * Lxb).sum()), np.float32)
            imall_mov = np.zeros((img[0].shape[0] - 1, (Lyb * Lxb).sum()), np.float32)
        for ii, im in enumerate(img):
            usevid = False
            if fullSVD:
                usevid = True
            if nroi > 0:
                wmot = (ivid[motind] == ii).nonzero()[0]
                if wmot.size > 0:
                    usevid = True
            if usevid:
                if motSVD:  # compute motion energy
                    imbin_mot = spatial_bin(im, sbin, Lyb[ii], Lxb[ii])
                    imbin_mot = np.abs(np.diff(imbin_mot, axis=0))
                    imbin_mot -= avgmotion[ii]
                    if fullSVD:
                        imall_mot[:, ir[ii]] = imbin_mot
                if movSVD:  # for raw frame svd
                    imbin_mov = spatial_bin(im, sbin, Lyb[ii], Lxb[ii])
                    imbin_mov = imbin_mov[1:, :]
                    imbin_mov -= avgframe[ii]
                    if fullSVD:
                        imall_mov[:, ir[ii]] = imbin_mov
                if nroi > 0 and wmot.size > 0:
                    if motSVD:
                        imbin_mot = np.reshape(imbin_mot, (-1, Lyb[ii], Lxb[ii]))
                    if movSVD:
                        imbin_mov = np.reshape(imbin_mov, (-1, Lyb[ii], Lxb[ii]))
                    wmot = np.array(wmot).astype(int)
                    wroi = motind[wmot]
                    for i in range(wroi.size):
                        if motSVD:
                            lilbin = imbin_mot[:, rois[wroi[i]]['yrange_bin'][0]:rois[wroi[i]]['yrange_bin'][-1] + 1,
                                     rois[wroi[i]]['xrange_bin'][0]:rois[wroi[i]]['xrange_bin'][-1] + 1]
                            lilbin = np.reshape(lilbin, (lilbin.shape[0], -1))
                            ncb = min(nc, lilbin.shape[-1])
                            usv = utils.svdecon(lilbin.T, k=ncb)
                            ncb = usv[0].shape[-1]
                            U_mot[wmot[i] + 1][:, ni_mot[wmot[i] + 1]:ni_mot[wmot[i] + 1] + ncb] = usv[0] * usv[
                                1]  # U[wmot[i]+1][:, ni[wmot[i]+1]:ni[wmot[i]+1]+ncb] = usv[0]
                            ni_mot[wmot[i] + 1] += ncb
                        if movSVD:
                            lilbin = imbin_mov[:, rois[wroi[i]]['yrange_bin'][0]:rois[wroi[i]]['yrange_bin'][-1] + 1,
                                     rois[wroi[i]]['xrange_bin'][0]:rois[wroi[i]]['xrange_bin'][-1] + 1]
                            lilbin = np.reshape(lilbin, (lilbin.shape[0], -1))
                            ncb = min(nc, lilbin.shape[-1])
                            usv = utils.svdecon(lilbin.T, k=ncb)
                            ncb = usv[0].shape[-1]
                            U_mov[wmot[i] + 1][:, ni_mov[wmot[i] + 1]:ni_mov[wmot[i] + 1] + ncb] = usv[0] * usv[
                                1]  # U[wmot[i]+1][:, ni[wmot[i]+1]:ni[wmot[i]+1]+ncb] = usv[0]
                            ni_mov[wmot[i] + 1] += ncb
        update_mainwindow(MainWindow, GUIobject, w, "Computing SVD ")

        if fullSVD:
            if motSVD:
                ncb = min(nc, imall_mot.shape[-1])
                usv = utils.svdecon(imall_mot.T, k=ncb)
                ncb = usv[0].shape[-1]
                U_mot[0][:, ni_mot[0]:ni_mot[0] + ncb] = usv[0] * usv[1]
                ni_mot[0] += ncb
            if movSVD:
                ncb = min(nc, imall_mov.shape[-1])
                usv = utils.svdecon(imall_mov.T, k=ncb)
                ncb = usv[0].shape[-1]
                U_mov[0][:, ni_mov[0]:ni_mov[0] + ncb] = usv[0] * usv[1]
                ni_mov[0] += ncb
        ns += 1

    S_mot = np.zeros(500)
    S_mov = np.zeros(500)
    # take SVD of concatenated spatial PCs
    if ns > 1:
        for nr in range(len(U_mot)):
            if nr == 0 and fullSVD:
                if motSVD:
                    U_mot[nr] = U_mot[nr][:, :ni_mot[0]]
                    usv = utils.svdecon(U_mot[nr], k=min(ncomps, U_mot[nr].shape[1] - 1))
                    U_mot[nr] = usv[0] * usv[1]
                    S_mot = usv[1]
                if movSVD:
                    U_mov[nr] = U_mov[nr][:, :ni_mov[0]]
                    usv = utils.svdecon(U_mov[nr], k=min(ncomps, U_mov[nr].shape[1] - 1))
                    U_mov[nr] = usv[0] * usv[1]
                    S_mov = usv[1]
            elif nr > 0:
                if motSVD:
                    U_mot[nr] = U_mot[nr][:, :ni_mot[nr]]
                    usv = utils.svdecon(U_mot[nr], k=min(ncomps, U_mot[nr].shape[1] - 1))
                    U_mot[nr] = usv[0] * usv[1]
                    S_mot = usv[1]
                if movSVD:
                    U_mov[nr] = U_mov[nr][:, :ni_mov[nr]]
                    usv = utils.svdecon(U_mov[nr], k=min(ncomps, U_mov[nr].shape[1] - 1))
                    U_mov[nr] = usv[0] * usv[1]
                    S_mov = usv[1]
    return U_mot, U_mov, S_mot, S_mov


def process_ROIs(containers, cumframes, Ly, Lx, avgframe, avgmotion, U_mot, U_mov, motSVD=True, movSVD=False,
                 sbin=3, tic=None, rois=None, fullSVD=True, GUIobject=None, MainWindow=None):
    """
    Facmeap processing function
        # project U onto each frame in the video and compute the motion energy for motSVD
    # also compute pupil on single frames on non binned data
    # the pixels are binned in spatial bins of size sbin
    # containers is a list of videos loaded with av
    # cumframes are the cumulative frames across videos
    Parameters
    ----------
    containers
    cumframes
    Ly
    Lx
    avgframe
    avgmotion
    U_mot
    U_mov
    motSVD
    movSVD
    sbin
    tic
    rois
    fullSVD
    GUIobject
    MainWindow

    Returns
    -------

    """

    if tic is None:
        tic = time.time()
    nframes = cumframes[-1]

    pups = []
    pupreflector = []
    blinks = []
    runs = []

    motind = []
    pupind = []
    blind = []
    runind = []
    ivid = []
    nroi = 0  # number of motion ROIs

    if fullSVD:
        ncomps_mot = U_mot[0].shape[-1]
        ncomps_mov = U_mov[0].shape[-1]
        V_mot = [np.zeros((nframes, ncomps_mot), np.float32)]
        V_mov = [np.zeros((nframes, ncomps_mov), np.float32)]
        M = [np.zeros((nframes), np.float32)]
    else:
        V_mot = [np.zeros((0, 1), np.float32)]
        V_mov = [np.zeros((0, 1), np.float32)]
        M = [np.zeros((0,), np.float32)]

    if rois is not None:
        for i, r in enumerate(rois):
            ivid.append(r['ivid'])
            if r['rind'] == 0:
                pupind.append(i)
                pups.append({'area': np.zeros((nframes,)), 'com': np.zeros((nframes, 2)),
                             'axdir': np.zeros((nframes, 2, 2)), 'axlen': np.zeros((nframes, 2))})
                if 'reflector' in r:
                    pupreflector.append(utils.get_reflector(r['yrange'], r['xrange'], rROI=None, rdict=r['reflector']))
                else:
                    pupreflector.append(np.array([]))
            elif r['rind'] == 1:
                motind.append(i)
                nroi += 1
                V_mot.append(np.zeros((nframes, U_mot[nroi].shape[1]), np.float32))
                V_mov.append(np.zeros((nframes, U_mov[nroi].shape[1]), np.float32))
                M.append(np.zeros((nframes,), np.float32))
            elif r['rind'] == 2:
                blind.append(i)
                blinks.append(np.zeros((nframes,)))
            elif r['rind'] == 3:
                runind.append(i)
                runs.append(np.zeros((nframes, 2)))

    ivid = np.array(ivid).astype(np.int32)
    motind = np.array(motind).astype(np.int32)

    # compute in chunks of 500
    nt0 = 500
    nsegs = int(np.ceil(nframes / nt0))
    # binned Ly and Lx and their relative inds in concatenated movies
    Lyb, Lxb, ir = process.binned_inds(Ly, Lx, sbin)
    imend = []
    for ii in range(len(Ly)):
        imend.append([])
    t = 0
    nt1 = 0
    s = StringIO()
    for n in tqdm(range(nsegs), file=s):
        t += nt1
        img = imall_init(nt0, Ly, Lx)
        utils.get_frames(img, containers, np.arange(t, t + nt0), cumframes)
        nt1 = img[0].shape[0]

        if len(pupind) > 0:  # compute pupil
            pups = self.process_pupil_ROIs(t, nt, img, ivid, rois, pupind, pups)
        if len(blind) > 0:
            blinks = self.process_blink_ROIs(t, nt, img, ivid, rois, blind, blinks)
        if len(runind) > 0:  # compute running
            runs = self.process_running(t, nt, img, ivid, rois, runind, runs)

        # bin and get motion
        if fullSVD:
            if n > 0:
                imall_mot = np.zeros((img[0].shape[0], (Lyb * Lxb).sum()), np.float32)
                imall_mov = np.zeros((img[0].shape[0], (Lyb * Lxb).sum()), np.float32)
            else:
                imall_mot = np.zeros((img[0].shape[0] - 1, (Lyb * Lxb).sum()), np.float32)
                imall_mov = np.zeros((img[0].shape[0] - 1, (Lyb * Lxb).sum()), np.float32)
        if fullSVD or nroi > 0:
            for ii, im in enumerate(img):
                usevid = False
                if fullSVD:
                    usevid = True
                if nroi > 0:
                    wmot = (ivid[motind] == ii).nonzero()[0]
                    if wmot.size > 0:
                        usevid = True
                if usevid:
                    imbin = spatial_bin(im, sbin, Lyb[ii], Lxb[ii])
                    if n > 0:
                        imbin = np.concatenate((imend[ii][np.newaxis, :], imbin), axis=0)
                    imend[ii] = imbin[-1]
                    if motSVD:  # compute motion energy for motSVD
                        imbin_mot = np.abs(np.diff(imbin, axis=0))
                    if movSVD:  # use raw frames for movSVD
                        imbin_mov = imbin[1:, :]
                    if fullSVD:
                        M[t:t + imbin_mot.shape[0]] += imbin_mot.sum(axis=(-2, -1))
                        if motSVD:
                            imall_mot[:, ir[ii]] = imbin_mot - avgmotion[ii].flatten()
                        if movSVD:
                            imall_mov[:, ir[ii]] = imbin_mov - avgframe[ii].flatten()
                if nroi > 0 and wmot.size > 0:
                    wmot = np.array(wmot).astype(int)
                    if motSVD:
                        imbin_mot = np.reshape(imbin_mot, (-1, Lyb[ii], Lxb[ii]))
                        avgmotion[ii] = np.reshape(avgmotion[ii], (Lyb[ii], Lxb[ii]))
                    if movSVD:
                        imbin_mov = np.reshape(imbin_mov, (-1, Lyb[ii], Lxb[ii]))
                        avgframe[ii] = np.reshape(avgframe[ii], (Lyb[ii], Lxb[ii]))
                    wroi = motind[wmot]
                    for i in range(wroi.size):
                        if motSVD:
                            lilbin = imbin_mot[:, rois[wroi[i]]['yrange_bin'][0]:rois[wroi[i]]['yrange_bin'][-1] + 1,
                                     rois[wroi[i]]['xrange_bin'][0]:rois[wroi[i]]['xrange_bin'][-1] + 1]
                            M[wmot[i] + 1][t:t + lilbin.shape[0]] = lilbin.sum(axis=(-2, -1))
                            lilbin -= avgmotion[ii][rois[wroi[i]]['yrange_bin'][0]:rois[wroi[i]]['yrange_bin'][-1] + 1,
                                      rois[wroi[i]]['xrange_bin'][0]:rois[wroi[i]]['xrange_bin'][-1] + 1]
                            lilbin = np.reshape(lilbin, (lilbin.shape[0], -1))
                            vproj = lilbin @ U_mot[wmot[i] + 1]
                            if n == 0:
                                vproj = np.concatenate((vproj[0, :][np.newaxis, :], vproj), axis=0)
                            V_mot[wmot[i] + 1][t:t + vproj.shape[0], :] = vproj
                        if movSVD:
                            lilbin = imbin_mov[:, rois[wroi[i]]['yrange_bin'][0]:rois[wroi[i]]['yrange_bin'][-1] + 1,
                                     rois[wroi[i]]['xrange_bin'][0]:rois[wroi[i]]['xrange_bin'][-1] + 1]
                            lilbin -= avgframe[ii][rois[wroi[i]]['yrange_bin'][0]:rois[wroi[i]]['yrange_bin'][-1] + 1,
                                      rois[wroi[i]]['xrange_bin'][0]:rois[wroi[i]]['xrange_bin'][-1] + 1]
                            lilbin = np.reshape(lilbin, (lilbin.shape[0], -1))
                            vproj = lilbin @ U_mov[wmot[i] + 1]
                            if n == 0:
                                vproj = np.concatenate((vproj[0, :][np.newaxis, :], vproj), axis=0)
                            V_mov[wmot[i] + 1][t:t + vproj.shape[0], :] = vproj
            if fullSVD:
                if motSVD:
                    vproj = imall_mot @ U_mot[0]
                    if n == 0:
                        vproj = np.concatenate((vproj[0, :][np.newaxis, :], vproj), axis=0)
                    V_mot[0][t:t + vproj.shape[0], :] = vproj
                if movSVD:
                    vproj = imall_mov @ U_mov[0]
                    if n == 0:
                        vproj = np.concatenate((vproj[0, :][np.newaxis, :], vproj), axis=0)
                    V_mov[0][t:t + vproj.shape[0], :] = vproj
            update_mainwindow(MainWindow, GUIobject, s, "Computing projection ")

    return V_mot, V_mov, M, pups, blinks, runs


def run_facemap_mod(filenames, motSVD=True, movSVD=False, GUIobject=None, parent=None,
                    proc=None, savepath=None, verbose=True):
    '''
    Modified version of the facemap function so it works without a GUI
    Parameters
    ----------
    filenames : list of names of video(s) to get
    '''
    ''' uses filenames and processes fullSVD if no roi's specified '''
    ''' parent is from GUI '''
    ''' proc can be a saved ROI file from GUI '''
    ''' savepath is the folder in which to save _proc.npy '''
    start = time.time()
    # grab files
    rois = None
    sy, sx = 0, 0

    if parent is not None:
        filenames = parent.filenames
        _, _, _, containers = utils.get_frame_details(filenames)
        cumframes = parent.cumframes
        sbin = parent.sbin
        rois = utils.roi_to_dict(parent.ROIs, parent.rROI)
        Ly = parent.Ly
        Lx = parent.Lx
        fullSVD = parent.checkBox.isChecked()
        save_mat = parent.save_mat.isChecked()
        sy = parent.sy
        sx = parent.sx
        motSVD, movSVD = parent.motSVD_checkbox.isChecked(), parent.movSVD_checkbox.isChecked(),
    else:
        # This is the line that spills out some warnings
        """
        [ERROR:0] global /tmp/pip-req-build-l1r0y34w/opencv/modules/videoio/src/cap.cpp (160) open VIDEOIO(CV_IMAGES): raised OpenCV exception:
        OpenCV(4.5.3) /tmp/pip-req-build-l1r0y34w/opencv/modules/videoio/src/cap_images.cpp:253: error: (-5:Bad argument) CAP_IMAGES: can't find starting number (in the name of file): a in function 'icvExtractPattern'
        """
        cumframes, Ly, Lx, containers = utils.get_frame_details(filenames)

        if proc is None:
            sbin = 1
            # fullSVD = True
            # fullSVD : whether or not "multivideo SVD" is computed
            # should be set to false normally in this case (single video, single roi SVD)
            fullSVD = False

            save_mat = False
            rois = None
            # TODO: need to specify some type of rois here (just the whole video)
        else:
            sbin = proc['sbin']
            fullSVD = proc['fullSVD']
            save_mat = proc['save_mat']
            rois = proc['rois']
            sy = proc['sy']
            sx = proc['sx']

    Lybin, Lxbin, iinds = process.binned_inds(Ly, Lx, sbin)
    LYbin, LXbin, sybin, sxbin = utils.video_placement(Lybin, Lxbin)
    nroi = 0

    # TODO: may be good to check video corruption around here
    # eg. error on AN022 2021-06-16 experiment 2 eye/side camera
    # pdb.set_trace()

    if rois is not None:
        for r in rois:
            # Tim: added thing here to update yrange and xrange
            if r['yrange'] == 'full':
                r['yrange'] = np.arange(0, Ly[0]).astype(np.int32)
            elif type(r['yrange']) is int:
                if r['yrange'] > 0:
                    r['yrange'] = np.arange(r['yrange'], Ly[0]).astype(np.int32)
                elif r['yrange'] < 0:
                    r['yrange'] = np.arange(0, Ly[0] - r['yrange']).astype(np.int32)

            if r['xrange'] == 'full':
                r['xrange'] = np.arange(0, Lx[0]).astype(np.int32)

            r['ellipse'] = np.zeros((Ly[0], Lx[0])).astype(bool)  # this is useless for now

            if r['rind'] == 1:
                r['yrange_bin'] = np.arange(np.floor(r['yrange'][0] / sbin),
                                            np.floor(r['yrange'][-1] / sbin)).astype(int)
                r['xrange_bin'] = np.arange(np.floor(r['xrange'][0] / sbin),
                                            np.floor(r['xrange'][-1]) / sbin).astype(int)
                nroi += 1

    tic = time.time()
    # compute average frame and average motion across videos (binned by sbin) ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    tqdm.write('Computing subsampled mean...')
    avgframe, avgmotion = process.subsampled_mean(containers, cumframes, Ly, Lx, sbin)
    avgframe_reshape = utils.multivideo_reshape(np.hstack(avgframe)[:, np.newaxis],
                                                LYbin, LXbin, sybin, sxbin, Lybin, Lxbin, iinds)
    avgframe_reshape = np.squeeze(avgframe_reshape)
    avgmotion_reshape = utils.multivideo_reshape(np.hstack(avgmotion)[:, np.newaxis],
                                                 LYbin, LXbin, sybin, sxbin, Lybin, Lxbin, iinds)
    avgmotion_reshape = np.squeeze(avgmotion_reshape)

    # Update user with progress
    tqdm.write('Computed subsampled mean at %0.2fs' % (time.time() - tic))
    if parent is not None:
        parent.update_status_bar("Computed subsampled mean")
    if GUIobject is not None:
        GUIobject.QApplication.processEvents()

    # Compute motSVD and/or movSVD from frames subsampled across videos
    #   and return spatial components                      ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    ncomps = 500
    if fullSVD or nroi > 0:
        tqdm.write('Computing subsampled SVD...')
        print("motSVD", motSVD, "movSVD", movSVD)

        # call a local version of the compute_SVD because for some reason
        # there are some complaints about ncomps being specified twice.

        U_mot, U_mov, S_mot, S_mov = compute_SVD(containers, cumframes, Ly, Lx, avgframe,
                                                 avgmotion, motSVD=motSVD, movSVD=movSVD, ncomps=ncomps,
                                                 sbin=sbin, rois=rois, fullSVD=fullSVD)
        tqdm.write('Computed subsampled SVD at %0.2fs' % (time.time() - tic))

        if parent is not None:
            parent.update_status_bar("Computed subsampled SVD")
        if GUIobject is not None:
            GUIobject.QApplication.processEvents()

        U_mot_reshape = U_mot.copy()
        U_mov_reshape = U_mov.copy()
        if fullSVD:
            U_mot_reshape[0] = utils.multivideo_reshape(U_mot_reshape[0], LYbin, LXbin, sybin, sxbin, Lybin, Lxbin,
                                                        iinds)
            U_mov_reshape[0] = utils.multivideo_reshape(U_mov_reshape[0], LYbin, LXbin, sybin, sxbin, Lybin, Lxbin,
                                                        iinds)
        if nroi > 0:
            k = 1
            for r in rois:
                if r['rind'] == 1:
                    ly = r['yrange_bin'].size
                    lx = r['xrange_bin'].size
                    U_mot_reshape[k] = np.reshape(U_mot[k].copy(), (ly, lx, U_mot[k].shape[-1]))
                    U_mov_reshape[k] = np.reshape(U_mov[k].copy(), (ly, lx, U_mov[k].shape[-1]))
                    k += 1
    else:
        U_mot, U_mov, S_mot, S_mov = [], [], [], []
        U_mot_reshape, U_mov_reshape = [], []

    # Add V_mot and/or V_mov calculation: project U onto all movie frames ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # and compute pupil (if selected)
    tqdm.write('Computing projection for motSVD...')
    V_mot, V_mov, M, pups, blinks, runs = process_ROIs(containers, cumframes, Ly, Lx, avgframe, avgmotion,
                                                       U_mot, U_mov, motSVD, movSVD, sbin=sbin, tic=tic, rois=rois,
                                                       fullSVD=fullSVD,
                                                       GUIobject=GUIobject, MainWindow=parent)
    tqdm.write('Computed motSVD projection at %0.2fs' % (time.time() - tic))

    # smooth pupil and blinks and running  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # for p in pups:
    #    if 'area' in p:
    #        p['area_smooth'],_ = pupil.smooth(p['area'].copy())
    #        p['com_smooth'] = p['com'].copy()
    #        p['com_smooth'][:,0],_ = pupil.smooth(p['com_smooth'][:,0].copy())
    #        p['com_smooth'][:,1],_ = pupil.smooth(p['com_smooth'][:,1].copy())
    # for b in blinks:
    #    b,_ = pupil.smooth(b.copy())

    # if parent is not None:
    #    parent.update_status_bar("Computed projection")
    # if GUIobject is not None:
    #    GUIobject.QApplication.processEvents()

    # Save output  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    proc = {
        'filenames': filenames, 'save_path': savepath, 'Ly': Ly, 'Lx': Lx,
        'sbin': sbin, 'fullSVD': fullSVD, 'save_mat': save_mat,
        'Lybin': Lybin, 'Lxbin': Lxbin,
        'sybin': sybin, 'sxbin': sxbin, 'LYbin': LYbin, 'LXbin': LXbin,
        'avgframe': avgframe, 'avgmotion': avgmotion,
        'avgframe_reshape': avgframe_reshape, 'avgmotion_reshape': avgmotion_reshape,
        'motion': M, 'motSv': S_mot, 'movSv': S_mov,
        'motMask': U_mot, 'movMask': U_mov,
        'motMask_reshape': U_mot_reshape, 'movMask_reshape': U_mov_reshape,
        'motSVD': V_mot, 'movSVD': V_mov,
        'pupil': pups, 'running': runs, 'blink': blinks, 'rois': rois,
        'sy': sy, 'sx': sx
    }

    # TODO: need to deal with these later (wasting space)
    proc['movSVD'] = None
    proc['movSv'] = None
    proc['movMask'] = None
    proc['movMask_reshape'] = None

    # save processing
    savename = facemap_save(proc, savepath)
    utils.close_videos(containers)

    # if parent is not None:
    #    parent.update_status_bar("Output saved in "+savepath)
    # if GUIobject is not None:
    #    GUIobject.QApplication.processEvents()
    tqdm.write('run time %0.2fs' % (time.time() - start))

    return savename


def facemap_save(proc, savepath=None):
    """
    Save function from facemap modified here to allow saving of larger files
    Parameters
    ----------
    proc : dict
        facemap output with motion SVDs, pupil size etc.
    savepath : str
        where to save the facemap output
    Returns
    -------
    savename : str
        name of the file that was saved to
    """
    # save ROIs and traces
    basename, filename = os.path.split(proc['filenames'][0][0])
    filename, ext = os.path.splitext(filename)
    if savepath is not None:
        basename = savepath

    savename = os.path.join(basename, ("%s_proc.npy" % filename))

    np.save(savename, proc)
    if proc['save_mat']:
        if 'save_path' in proc and proc['save_path'] is None:
            proc['save_path'] = basename

        d2 = {}
        if proc['rois'] is None:
            proc['rois'] = 0
        for k in proc.keys():
            if isinstance(proc[k], list) and len(proc[k]) > 0 and isinstance(proc[k][0], np.ndarray):
                for i in range(len(proc[k])):
                    d2[k + '_%d' % i] = proc[k][i]
            else:
                d2[k] = proc[k]
        savenamemat = os.path.join(basename, ("%s_proc.mat" % filename))
        io.savemat(savenamemat, d2)
        del d2
    return savename


def compress_video(video_fpath, ffmpeg_path='/usr/bin/ffmpeg', crf=1, output_video_format='mp4'):
    """
    TODO: note output_vidoe_format 'mj2' does not work with h264 for now... need to find a solution
    Parameters
    ----------
    video_fpath : str
        path to video you want to compress
    ffmpeg_path : str
        where ffmpeg is located on your computer
    crf : int
        how much compression to do, 0 means loseless, and higher means more compression
        Kenneth from says IBL uses crf = 29, but that is actually not loseless
        according to the ffmpeg docs, crf = 0 is loseless
    output_video_format : str
        video file format to ouput
    Returns
    -------
    output_path : str
        path to the compressed video
    """



    if output_video_format is None:
        output_video_format = video_fpath.rpartition('.')[-1]

    output_path = video_fpath.rpartition('.')[0] + '_compressed_yuv420p_crf1_slower.' + output_video_format

    sp.call([
        ffmpeg_path, '-i', video_fpath,
        '-pix_fmt', 'yuv420p',
        '-vcodec', 'h264',
        '-crf', str(crf),
        '-preset', 'slower',
        output_path
    ])

    return output_path


def update_file_list(all_mouse_info, file_list_csv_path=None, load_from_server=False):
    """
    Write a shared file to server to list which facemap files are
    (1) processed
    (2) to be processed
    (3) processing
    Parameters
    ----------
    all_mouse_info : pandas dataframe
        dataframe contained information of each mouse
        this can be obtained using the function get_all_mouse_info()
    file_list_csv_path : str
        path to the csv containing the files to run (?)
    load_from_server : bool
        whether you are loading the data from server
    Returns
    -------
    all_file_info :
    """

    if not os.path.exists(file_list_csv_path):
        print('No file list found, creating one in specified path')

        file_list_df = pd.DataFrame()

        # By default assumes none of the files are being processed at this instant
        file_list_df['running_facemap'] = False

        # Go through each file and check if facemap already ran
        for row_idx, exp_info in all_mouse_info.iterrows():
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

            exp_folder = os.path.join(exp_info['main_folder'], exp_info['subject'],
                                      exp_info['expDate'], str(exp_info['expNum']))
            # look for video files
            video_files = glob.glob(os.path.join(exp_folder, '*%s' % video_ext))

            # remove the *lastFrames.mj2 videos
            video_files = [x for x in video_files if 'lastFrames' not in x]
            video_file_fov_names = [os.path.basename(x).split('_')[3].split('.')[0] for x in video_files]

        for video_fpath, video_fov in zip(video_files, video_file_fov_names):
            # look for facemap processed file
            processed_facemap_path = glob.glob(os.path.join(exp_folder, '*%s*proc.npy') % video_fov)
    else:
        print('Found info about list of files to process')

    return all_file_info


def get_mouse_info_csv_paths(subset_mice_to_use=None, subset_date_range=None):
    """
    Obtains a dataframe with information about each experiment, from the CSVs provided in the
    pink rig folder
    Parameters
    ----------
    subset_mice_to_use
    subset_date_range

    Returns
    -------

    """
    if socket.gethostname() == 'timothysit-cortexlab':  # Tim's Desktop
        main_info_folder_in_server = True
        mouse_info_folder = 'smb://zserver.local/code/AVrig/'
        default_server_path = 'smb://128.40.224.65/subjects/'
    else:  # Zelda rigs
        main_info_folder_in_server = False
        mouse_info_folder = '//zserver/Code/AVrig'
        default_server_path = '//128.40.224.65/subjects/'

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

    files_to_exclude = []

    pattern_to_match = re.compile('[A-Z][A-Z][0-9][0-9][0-9]')

    for path in mouse_info_csv_paths:
        if os.path.basename(path) in files_to_exclude:
            mouse_info_csv_paths.remove(path)
        fname_without_ext = path.split(os.sep)[-1].split('.')[0]
        if not pattern_to_match.match(fname_without_ext):
            mouse_info_csv_paths.remove(path)

    return mouse_info_csv_paths


def update_mouse_csv_record():
    """
    Updates information about the csv on the server
    Returns
    -------

    """

    if socket.gethostname() == 'timothysit-cortexlab':  # Tim's Desktop
        main_info_folder_in_server = True
        mouse_info_folder = 'smb://zserver.local/code/AVrig/'
        default_server_path = 'smb://128.40.224.65/subjects/'
    else: # Zelda rigs
        main_info_folder_in_server = False
        mouse_info_folder = '//zserver/Code/AVrig'
        default_server_path = '//128.40.224.65/subjects/'

    if main_info_folder_in_server:
        gvfs = Gio.Vfs.get_default()
        mouse_info_folder = gvfs.get_file_for_uri(mouse_info_folder).get_path()

    subset_mice_to_use = None  # ['FT030', 'FT031', 'FT032', 'FT035']
    subset_date_range = None  # ['2021-12-01', '2021-12-20']

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
        fname_without_ext = path.split(os.sep)[-1].split('.')[0]
        if not pattern_to_match.match(fname_without_ext):
            mouse_info_csv_paths.remove(path)

    all_mouse_info = []

    for csv_path in mouse_info_csv_paths:
        mouse_info = pd.read_csv(csv_path)
        mouse_name = os.path.basename(csv_path).split('.')[0]

        modifiable_mouse_info = mouse_info.copy()


        # Convert facemap processing column to a single number to see if all cameras are processed
        facemapStatusCode = mouse_info['faceMapFrontSideEye']
        facemapStatusCodeSum = np.zeros(len(facemapStatusCodeSum), ) + np.nan

        for codeStr in facemapStatusCode.values:
            codeInt = [float(x) for x in codeStr.split(',')]
            facemapStatusCodeSum[nCodeStr] = int(np.sum(codeInt))

        modifiable_mouse_info['facemapStatusCodeSum'] = facemapStatusCodeSum

        # Subset columns with facemap not processed
        subset_modifiable_mouse_info = modifiable_mouse_info.loc[
            modifiable_mouse_info['facemapStatusCodeSum'] < 3
        ]

        pdb.set_trace()


        # TODO: for subset, check if facemap processed yet or not


        # mouse_info['subject'] = mouse_name


    return 0


def plot_facemap_results():
    """
    Plots the facemap output
    Returns
    -------

    """
    # TODO: first locate the csv with all the information
    mouse_info_csv_paths = get_mouse_info_csv_paths()




    # Loop through each folder, check if there is a plot already made, if not, load results and make plot
    for csv_path in mouse_info_csv_paths:
        mouse_info = pd.read_csv(csv_path)
        mouse_name = os.path.basename(csv_path).split('.')[0]

        for row_idx, exp_info in mouse_info.iterrows():

            exp_folder = os.path.join(exp_info['main_folder'], exp_info['subject'],
                                      exp_info['expDate'], str(exp_info['expNum']))
            # look for video files
            video_files = glob.glob(os.path.join(exp_folder, '*%s' % video_ext))

            # remove the *lastFrames.mj2 videos
            video_files = [x for x in video_files if 'lastFrames' not in x]
            video_file_fov_names = [os.path.basename(x).split('_')[3].split('.')[0] for x in video_files]

            # load facemap results
            facemap_output_path = glob.glob(os.path.join(exp_folder, '*%s*proc.npy') % video_fov)[0]
            facemap_output = np.load(facemap_output_path, allow_pickle=True).item()



            # Plot average frame
            avgframe_reshape = facemap_output['avgframe_reshape']
            fig, ax = plt.subplots()
            ax.imshow(avgframe_reshape, aspect='auto', cmap='gray')
            ax.set_xlabel('x axis pixel', size=12)
            ax.set_ylabel('y axis pixel', size=12)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            fig_name = 'facemap_avgframe_reshaped.png'
            fig.savefig(os.path.join(exp_folder, fig_name), dpi=300, bbox_inches='tight')


    return 0


def convert_facemap_output_to_ONE_format(facemap_output_file):
    """
    Convert facemap output to ONE format
    Parameters
    ----------
    facemap_output_file : str
        path to facemap output (face proc) npy file

    Returns
    -------
    0 if the function ran succesfully
    """

    facemap_output_file = Path(facemap_output_file)

    # Make folder for the camera
    eye_proc_basename = facemap_output_file.name

    exp_date = eye_proc_basename.split('_')[0]
    exp_num = eye_proc_basename.split('_')[1]
    subject = eye_proc_basename.split('_')[2]
    fov_name = eye_proc_basename.split('_')[3]

    exp_folder = facemap_output_file.parent
    fov_folder = exp_folder / 'ONE_preproc' / fov_name

    if not fov_folder.is_dir():
        fov_folder.mkdir(exist_ok=True,parents=True)

    # Load facemap output
    facemap_output = np.load(facemap_output_file, allow_pickle=True).item()
    facemap_output = Bunch(facemap_output)

    stub = '.%s_%s_%s_%s.npy' % (exp_date, exp_num, subject, fov_name)
    # Save motion SVD
    nRois = len(facemap_output.rois)
    # concatenate all svds
    motion_svd = [m[:,:,np.newaxis] for m in facemap_output.motSVD[1:]]
    minPCs_to_save = np.min([m.shape[1] for m in motion_svd])
    motion_svd = np.concatenate([m[:,:minPCs_to_save,:] for m in motion_svd],axis=2)
    # shape nFrames x nRois x nPCs 
    motion_svd = np.transpose(motion_svd, (0, 2, 1))
    #save
    np.save(fov_folder / ('camera._av_motionPCs' + stub), motion_svd)

    # Save motion SVD masks for the 1st ROI // pretty arbitrary
    motMask_reshape = facemap_output.motMask_reshape[1]  # x pixel X y pixel X nPC
    np.save(fov_folder / ('_av_motionPCs.weights' + stub), motMask_reshape)

    # Save ROI position
    roi_w_h_x_y = np.array([
        facemap_output.Lx[0],
        facemap_output.Ly[0],
        facemap_output.rois[0]['xrange'][0],
        facemap_output.rois[0]['yrange'][0]
    ])
    roi_w_h_x_y_save_path = os.path.join(fov_folder,
                                             'ROIMotionEnergy.position' + stub)
    np.save(roi_w_h_x_y_save_path, roi_w_h_x_y)

    # Save motion Energy
    motion_energy = np.concatenate([m[:,np.newaxis] for m in facemap_output.motion[1:]],axis=1)
    np.save(fov_folder / ('camera.ROIMotionEnergy' + stub), motion_energy)

    # Save average frame
    frame_average = facemap_output['avgframe_reshape']
    np.save(fov_folder / ('camera.ROIAverageFrame' + stub), frame_average)


    # if the pupil analysis exists save that too 

    if facemap_output.pupil:
        pupil_dat = Bunch(facemap_output.pupil[0])            
        [np.save(fov_folder / ('camera.pupil_%s' % k + stub), pupil_dat[k]) for k in pupil_dat.keys()]
    if facemap_output.blink: 
        np.save(fov_folder / ('camera.eyeblink' + stub),facemap_output.blink[0])


    return 0

def cut_video(video_path):
    """
    Cut video for running deeplabcut on subset of video (for ROI cropping)
    Parameters
    ----------
    video_path : str
        path to the video to cut

    Returns
    -------
    subset_video_path : str
        path to the video that was cut
    """

    subset_start_point = 5
    cut_duration = 10
    ffmpeg_path = 'C:/Program Files/ffmpeg/bin/ffmpeg.exe'   # set up by Tim
    # Also note that this doesn't work if ffmpeg is in C:/ffmpeg/ for some reason...
    video_name = os.path.basename(video_path).split('.')[0]
    subset_video_name_without_ext = video_name + '_subset'
    subset_video_name = subset_video_name_without_ext + '.mp4'
    subset_video_path = os.path.join(os.path.dirname(video_path), subset_video_name)
    ffmpeg_args = [ffmpeg_path,
                   '-ss', '%.f' % subset_start_point,
                   '-y',  # overwrite video if existing subset video exists
                   '-i', video_path,
                   '-c', 'copy',
                   '-t', '%.f' % (cut_duration),
                   subset_video_path
                   ]
    sp.call(ffmpeg_args)

    return subset_video_path


def get_dlc_roi_window(vid_path, projectName):
    """
    Get region of interest from deeplabcut anchor points, and outputs rectangle for facemap
    Parameters
    ----------
    vid_path : str
        path to original video (not the cut version)
    projectName : str
        name of deeplabcut project to use
    Returns
    -------

    """
    output_video_folder = os.path.dirname(vid_path)
    subset_video_name_without_ext = os.path.basename(vid_path)[0:-4]
    subset_vid_h5_path = glob.glob(os.path.join(
        output_video_folder,
        '*%s*%s*.h5' % (subset_video_name_without_ext, projectName)
    ))[0]

    if projectName == 'pinkrigs':

        subset_vid_output_df = pd.read_hdf(subset_vid_h5_path)  # pandas multindex
        # scorer_name = 'DLC_resnet50_pinkrigsSep12shuffle1_50000'
        scorer_name = 'DLC_resnet50_pinkrigsSep12shuffle1_1030000'
        body_parts = ['eyeL', 'eyeR', 'eyeU', 'eyeD', 'pupilL', 'pupilR', 'pupilU', 'pupilD', 'whiskPadL', 'whiskPadR']

        # eyeR_xvals = np.array([x[(scorer_name, 'eyeR', 'x')] for (_, x) in subset_vid_output_df.iterrows()])
        # eyeR_yvals = np.array([x[(scorer_name, 'eyeR', 'y')] for (_, x) in subset_vid_output_df.iterrows()])

        # rough imputation of values where eyeR is not found (negative values)
        # eyeR_xvals[eyeR_xvals < 0] = np.mean(eyeR_xvals)
        # eyeR_yvals[eyeR_yvals < 0] = np.mean(eyeR_yvals)

        # eyeR_xvals_mean = np.nanmean(eyeR_xvals)
        # eyeR_yvals_mean = np.nanmean(eyeR_yvals)

    elif projectName == 'pinkrigsFrontCam':

        subset_vid_output_df = pd.read_hdf(subset_vid_h5_path)  # pandas multindex
        # scorer_name = 'DLC_resnet50_pinkrigsFrontCamOct16shuffle1_150000'
        scorer_name = 'DLC_resnet50_pinkrigsFrontCamOct16shuffle1_1030000'  # new model from 2023-04-14
        body_parts = ['eyeL', 'eyeR', 'snoutL', 'snoutR', 'snoutF', 'pawL', 'pawR']


    elif projectName == 'pinkrigsSideCam':

        subset_vid_output_df = pd.read_hdf(subset_vid_h5_path)  # pandas multindex
        scorer_name = 'DLC_resnet50_pinkrigsSideCamOct17shuffle1_300000'

        body_parts = ['eyeL', 'snoutF', 'spout', 'pawL', 'earL', 'earR', 'earU', 'earD']

    body_part_mean_xy = {}
    body_part_median_xy = {}
    body_part_xy = {}

    for b_part in body_parts:
        body_part_yvals = np.array([x[(scorer_name, b_part, 'y')] for (_, x) in subset_vid_output_df.iterrows()])
        body_part_xvals = np.array([x[(scorer_name, b_part, 'x')] for (_, x) in subset_vid_output_df.iterrows()])

        body_part_xy[b_part + '_x'] = body_part_xvals
        body_part_xy[b_part + '_y'] = body_part_yvals

        # rough imputation of values where eyeR is not found (negative values)
        body_part_yvals[body_part_yvals < 0] = np.mean(body_part_yvals)
        body_part_xvals[body_part_xvals < 0] = np.mean(body_part_xvals)

        yvals_mean = np.nanmean(body_part_yvals)
        xvals_mean = np.nanmean(body_part_xvals)

        body_part_mean_xy[b_part + '_y'] = yvals_mean
        body_part_mean_xy[b_part + '_x'] = xvals_mean

        # more robust to outliers, I think can replace mean
        body_part_median_xy[b_part + '_y'] = np.nanmedian(body_part_yvals)
        body_part_median_xy[b_part + '_x'] = np.nanmedian(body_part_xvals)

    # Defining roi_window
    roi_window = dict()

    if projectName == 'pinkrigs':
        rectangle_width = 200

        # eyeR_xvals_mean = body_part_mean_xy['eyeR_x']
        # eyeR_yvals_mean = body_part_mean_xy['eyeR_y']
        eyeR_xvals_mean = body_part_median_xy['eyeR_x']
        eyeR_yvals_mean = body_part_median_xy['eyeR_y']

        eyeL_xvals_mean = body_part_median_xy['eyeL_x']
        eyeL_yvals_mean = body_part_median_xy['eyeL_y']

        if eyeL_yvals_mean - eyeR_yvals_mean < 60:  # threshold for eyeR being not on top of eyeL
            roi_window['mpl_obj'] = mpl.patches.Rectangle(
                (eyeR_xvals_mean - 300, eyeR_yvals_mean),
                rectangle_width, 150, edgecolor='red', facecolor='red', fill=False, lw=1
            )

            x_start = eyeR_xvals_mean - 300
            x_end = x_start + rectangle_width
            y_start = eyeR_yvals_mean
            y_end = y_start + 150
        else:
            print('Eye camera flipped, implementing fix')
            roi_window['mpl_obj'] = mpl.patches.Rectangle(
                (eyeR_xvals_mean - 120, eyeR_yvals_mean + 150),
                rectangle_width, 150, edgecolor='red', facecolor='red', fill=False, lw=1
            )
            x_start = eyeR_xvals_mean - 120
            y_start = eyeR_yvals_mean + 150
            x_end = x_start + rectangle_width
            y_end = y_start + 150

        roi_window['xrange'] = np.arange(x_start, x_end).astype(np.int32)
        roi_window['yrange'] = np.arange(y_start, y_end).astype(np.int32)

        # moving to the median, seems more stable
        roi_window['eyeR_x_mean'] = eyeR_xvals_mean
        roi_window['eyeR_y_mean'] = eyeR_yvals_mean

        roi_window['eyeL_x_mean'] = eyeL_xvals_mean
        roi_window['eyeL_y_mean'] = eyeL_yvals_mean

    elif projectName == 'pinkrigsFrontCam':

        rectangle_width = 50
        # rectangle_height = eyeL_yvals_mean - eyeR_yvals_mean
        rectangle_height = body_part_median_xy['eyeL_y'] - body_part_median_xy['eyeR_y']

        # eye_x_big_diff = np.abs(body_part_median_xy['eyeL_x'] - body_part_median_xy['eyeR_x']) > 100
        # eye_y_small_diff = np.abs(body_part_median_xy['eyeL_y'] - body_part_median_xy['eyeR_y']) < 50
        # eyeL_on_the_leftx = body_part_median_xy['eyeL_x'] > 390

        eyeR_on_top_of_eyeL = body_part_median_xy['eyeR_y'] < body_part_median_xy['eyeL_y']

        if eyeR_on_top_of_eyeL:
            # something is wrong (likely video flipped), implementing some hack

            rectangle_height = 120

            # if not eyeL_on_the_leftx:
            print('FrontCam flipped, using fix')
            roi_window['mpl_obj'] = mpl.patches.Rectangle(
                (body_part_median_xy['eyeL_x'] - 60, body_part_median_xy['eyeL_y']),
                rectangle_width, rectangle_height, edgecolor='red', facecolor='red', fill=False, lw=1
            )

            x_start = body_part_median_xy['eyeL_x'] - 60
            x_end = x_start + rectangle_width

            """
            else:
                print('Something wrong with frontCam, relying on eyeL')
                x_start = body_part_median_xy['eyeL_x'] + 25
                x_end = x_start + rectangle_width
                roi_window['mpl_obj'] = mpl.patches.Rectangle(
                    (x_start, body_part_median_xy['eyeL_y']),
                    rectangle_width, rectangle_height, edgecolor='red', facecolor='red', fill=False, lw=1
                )
            """

            y_start = body_part_median_xy['eyeL_y'] - rectangle_height
            y_end = y_start + rectangle_height

        else:
            roi_window['mpl_obj'] = mpl.patches.Rectangle(
                (body_part_median_xy['eyeR_x'] + 25, body_part_median_xy['eyeR_y']),
                rectangle_width, rectangle_height, edgecolor='red', facecolor='red', fill=False, lw=1
            )

            x_start = body_part_median_xy['eyeR_x'] + 25
            x_end = x_start + rectangle_width
            y_start = body_part_median_xy['eyeR_y']
            y_end = y_start + rectangle_height

            # this is when rectangle_height is negative
            # which means left and right eye positions are switched
            if y_end < y_start:
                y_start, y_end = y_end, y_start

                # this is to do with frontCam being mirrored
                # x_start = body_part_median_xy['eyeR_x'] - 25 - rectangle_width
                # x_end = x_start + rectangle_width

        roi_window['xrange'] = np.arange(x_start, x_end).astype(np.int32)
        roi_window['yrange'] = np.arange(y_start, y_end).astype(np.int32)
        roi_window['eyeL_x_mean'] = body_part_median_xy['eyeL_x']
        roi_window['eyeL_y_mean'] = body_part_median_xy['eyeL_y']
        roi_window['eyeR_x_mean'] = body_part_median_xy['eyeR_x']
        roi_window['eyeR_y_mean'] = body_part_median_xy['eyeR_y']
        roi_window['snoutF_x_mean'] = body_part_median_xy['snoutF_x']
        roi_window['snoutF_y_mean'] = body_part_median_xy['snoutF_y']

    elif projectName == 'pinkrigsSideCam':

        rectangle_width = 150
        rectangle_height = 100
        roi_window['mpl_obj'] = mpl.patches.Rectangle(
            (body_part_mean_xy['eyeL_x'] - 75, body_part_mean_xy['eyeL_y'] + 25),
            rectangle_width, rectangle_height, edgecolor='red', facecolor='red', fill=False, lw=1
        )

        x_start = body_part_mean_xy['eyeL_x'] - 75
        x_end = x_start + rectangle_width
        y_start = body_part_mean_xy['eyeL_y'] + 25
        y_end = y_start + rectangle_height

        roi_window['xrange'] = np.arange(x_start, x_end).astype(np.int32)
        roi_window['yrange'] = np.arange(y_start, y_end).astype(np.int32)
        roi_window['eyeL_x_mean'] = body_part_mean_xy['eyeL_x']
        roi_window['eyeL_y_mean'] = body_part_mean_xy['eyeL_y']

    return roi_window


def batch_process_facemap(output_format='flat', sessions=None,
                          subset_mice_to_use=None, subset_date_range=None,
                          recompute_facemap=False, recompute_ONE=False,
                          run_on_cropped_roi=True, old_date_to_overwrite=None,
                          verbose=False, print_warnings=False,
                          write_to_log=False, skip_svd_step=False):
    """
    Runs facemap SVD processing on videos given dataframe of video paths and information
    Parameters
    ----------
    
    output_format : str
        'flat' : save original facemap output in the main folder
        'ONE' : save facemap output in ONE format; extract outputs into numpy arrays and save as separate .npy,
                with one folder per camera
    sessions : pandas dataframe
        dataframe obtained by calling queryCSV()
        if this is specified, then this code skips over checking for csvs to run

    subset_mice_to_use : list, optional
        DESCRIPTION. The default is None.
        example: 
            subset_mice_to_use = ['FT030', 'FT031', 'FT032', 'FT035']
    subset_date_range : list, optional
        DESCRIPTION. The default is None.
        example:
            subset_date_range = ['2021-12-01', '2021-12-20']
    recompute_facemap : bool
        whether to recompute facemap even if existing file exists
    recompute_ONE : bool
        whether to recompute ONE format conversion even if existing files(s) existed
        or even if facemap is already processed
    Returns
    -------
    None.

    """
    
    num_videos_to_run_per_call = 1
    num_videos_ran = 0
    facemap_roi_selection_mode = 'automatic'
    align_to_timeline = False
    plot_results = True
    run_video_compression = False
    load_from_server = False
    video_ext = '.mj2'

    # spreadsheet to keep track of which files are done and which are not
    file_list_csv_path = '//zserver/Code/AVrig/facemap_file_list.csv'

    print('Running batch processing of pink rig videos...')

    # read main experiment csv
    # main_csv_path = ''
    # main_csv_df = pd.read_csv(main_csv_path)

    # TODO: think about video compression as well
    # On Zelda-4 timeline machine
    if socket.gethostname() == 'timothysit-cortexlab':
        main_info_folder_in_server = True
        mouse_info_folder = 'smb://zserver.local/code/AVrig/'
        default_server_path = 'smb://128.40.224.65/subjects/'
    else:
        main_info_folder_in_server = False
        mouse_info_folder = '//zserver/Code/AVrig'
        default_server_path = '//128.40.224.65/subjects/'

    if main_info_folder_in_server:
        gvfs = Gio.Vfs.get_default()
        mouse_info_folder = gvfs.get_file_for_uri(mouse_info_folder).get_path()

    if sessions is None:
        if subset_mice_to_use is not None:
            mouse_info_csv_paths = []
            for mouse_name in subset_mice_to_use:
                mouse_info_csv_paths.append(
                    glob.glob(os.path.join(mouse_info_folder, '%s.csv' % mouse_name))[0]
                )
        else:
            mouse_info_csv_paths = glob.glob(os.path.join(mouse_info_folder, '*.csv'))

        # TODO: test the re.compile().match() code
        #files_to_exclude = ['aMasterMouseList.csv',
        #                    'kilosort_queue.csv',
        #                    'video_corruption_check.csv',
        #                    '!MouseList.csv']
        files_to_exclude = []

        pattern_to_match = re.compile('[A-Z][A-Z][0-9][0-9][0-9]')

        subset_mouse_info_csv_paths = []

        for n_path, path in enumerate(mouse_info_csv_paths):
            print(path)
            if os.path.basename(path) in files_to_exclude:
                mouse_info_csv_paths.remove(path)
            fname_without_ext = path.split(os.sep)[-1].split('.')[0]
            if not pattern_to_match.match(fname_without_ext):
                mouse_info_csv_paths.remove(path)
                print('File : %s excluded' % path)
                # del mouse_info_csv_paths[n_path]
            else:
                fname = os.path.basename(path)
                fname_without_ext = fname.split('.')[0]
                str_match = re.match(pattern_to_match, fname_without_ext) is not None

                if not str_match:
                    # mouse_info_csv_paths.remove(path)
                    # del mouse_info_csv_paths[n_path]
                    status_str = 'reject'
                else:
                    status_str = 'accept'
                    subset_mouse_info_csv_paths.append(path)

                print('File : %s, status: %s' % (fname, status_str))

        all_mouse_info = []
        mouse_info_csv_paths = subset_mouse_info_csv_paths

        for csv_path in mouse_info_csv_paths:
            mouse_info = pd.read_csv(csv_path)
            mouse_name = os.path.basename(csv_path).split('.')[0]
            mouse_info['subject'] = mouse_name

            if 'path' not in mouse_info.columns:
                mouse_info['server_path'] = default_server_path
            if 'expFolder' in mouse_info.columns:
                server_paths = ['//%s/%s' % (x.split(os.sep)[2], x.split(os.sep)[3]) for x in mouse_info['expFolder'].values]
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
    else:
        all_mouse_info = sessions
        all_mouse_info['server_path'] = [os.path.join('\\\\', *os.path.normpath(x).split(os.sep)[2:4]) for x in sessions['expFolder'].values]
        all_mouse_info['subject'] = all_mouse_info['subject']

    # Tim temp hack to try running this for one experiment
    # all_mouse_info = all_mouse_info.loc[1:2]

    # make a fake main_csv_df for now to test things
    """
    main_csv_df = pd.DataFrame.from_dict({
        'subject': ['TS011'],
        'date': ['2021-07-26'],
        'exp': ['1'],
        'server_path': ['smb://128.40.224.65/subjects/'],
        # 'server_path': ['smb://znas.local/subjects'],
    })
    """

    # Manually specify this rois object
    # color of the ROI seems to be some random thing, does not matter too much I think
    # https://github.com/MouseLand/facemap/blob/adb3b0b18191ddb8b22c2e6800dd7a69a15a0c39/facemap/roi.py
    rig_procs = {
        'zelda-stim4': dict(
            rois=[
                {
                    'rind': 1,
                    'rtype': 'motion_SVD',
                    'iROI': 0,
                    'ivid': 0,
                    'color': (78.32401579229648, 15.276705546609746, 100.90024790442232),
                    'yrange': -1,  # np.arange(0, Ly[0]).astype(np.int32),
                    'xrange': 'full',  # np.arange(0, Lx[0]).astype(np.int32),
                    'saturation': 255,
                    'pupil_sigma': 2,  # again does not matter I think
                    'ellipse': 'full',  # np.zeros((Ly[0], Lx[0])).astype(bool)
                }
            ],
            sbin=1,
            fullSVD=False,
            save_mat=False,
            sy=None,
            sx=None,
        ),
        'zelda-stim3': dict(
            rois=[
                {
                    'rind': 1,
                    'rtype': 'motion_SVD',
                    'iROI': 0,
                    'ivid': 0,
                    'color': (78.32401579229648, 15.276705546609746, 100.90024790442232),
                    'yrange': 'full',  # np.arange(0, Ly[0]).astype(np.int32),
                    'xrange': 'full',  # np.arange(0, Lx[0]).astype(np.int32),
                    'saturation': 255,
                    'pupil_sigma': 2,  # again does not matter I think
                    'ellipse': 'full',  # np.zeros((Ly[0], Lx[0])).astype(bool)
                },
            ],
            sbin=1,
            fullSVD=False,
            save_mat=False,
            sy=None,
            sx=None,
        ),
        'zelda-stim2': dict(
            rois=[
                {
                    'rind': 1,
                    'rtype': 'motion_SVD',
                    'iROI': 0,
                    'ivid': 0,
                    'color': (78.32401579229648, 15.276705546609746, 100.90024790442232),
                    'yrange': 'full',  # np.arange(0, Ly[0]).astype(np.int32),
                    'xrange': 'full',  # np.arange(0, Lx[0]).astype(np.int32),
                    'saturation': 255,
                    'pupil_sigma': 2,  # again does not matter I think
                    'ellipse': 'full',  # np.zeros((Ly[0], Lx[0])).astype(bool)
                },
            ],
            sbin=1,
            fullSVD=False,
            save_mat=False,
            sy=None,
            sx=None,
        ),
        'zelda-stim1': dict(
            rois=[
                {
                    'rind': 1,
                    'rtype': 'motion_SVD',
                    'iROI': 0,
                    'ivid': 0,
                    'color': (78.32401579229648, 15.276705546609746, 100.90024790442232),
                    'yrange': 'full',  # np.arange(0, Ly[0]).astype(np.int32),
                    'xrange': 'full',  # np.arange(0, Lx[0]).astype(np.int32),
                    'saturation': 255,
                    'pupil_sigma': 2,  # again does not matter I think
                    'ellipse': 'full',  # np.zeros((Ly[0], Lx[0])).astype(bool)
                },
            ],
            sbin=1,
            fullSVD=False,
            save_mat=False,
            sy=None,
            sx=None,
        ), 

        'lilrig-stim': dict(
            rois=[
                {
                    'rind': 1,
                    'rtype': 'motion_SVD',
                    'iROI': 0,
                    'ivid': 0,
                    'color': (78.32401579229648, 15.276705546609746, 100.90024790442232),
                    'yrange': 'full',  # np.arange(0, Ly[0]).astype(np.int32),
                    'xrange': 'full',  # np.arange(0, Lx[0]).astype(np.int32),
                    'saturation': 255,
                    'pupil_sigma': 2,  # again does not matter I think
                    'ellipse': 'full',  # np.zeros((Ly[0], Lx[0])).astype(bool)
                },
            ],
            sbin=1,
            fullSVD=False,
            save_mat=False,
            sy=None,
            sx=None,
        )
    }
    # loop through the experiments and see if there are videos with no corresponding facemap output
    file_skipped = 0
    tot_video_files = 0

    for row_idx, exp_info in all_mouse_info.iterrows():
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

        if len(video_files) == 0:
            print('WARNING: no video files found in %s' % exp_folder)
            continue

        # remove the *lastFrames.mj2 videos
        video_files = [x for x in video_files if 'lastFrames' not in x]
        # remove lilrig video files
        video_files = [x for x in video_files if 'eye.mj2' not in x]
        video_files = [x for x in video_files if 'face.mj2' not in x]

        # TODO: use .count('_') instead
        try:
            video_file_fov_names = [os.path.basename(x).split('_')[3].split('.')[0] for x in video_files]
        except:
            print('WARNING: video filename format is strange for %s, skipping' % exp_folder)
            continue

        if verbose:
            print('%.f Candidate videos to look over' % len(video_files))
            print('\n'.join(video_files))

        tot_video_files += len(video_files)

        # Temp hack by Tim to test video
        # exp_folder = os.path.join(exp_info['main_folder'], 'TS011', '2021-07-26', '1')
        # video_files = [os.path.join(exp_folder, '2021-07-26_1_TS011_eyeCam_lastFrames.mj2')]

        # exp_folder = os.path.join(exp_info['main_folder'], 'AH002', '2021-10-29', '2')
        # video_files = [os.path.join(exp_folder, '2021-10-29_2_AH002_eye.mj2')]
        # video_files = [os.path.join(exp_folder, '2021-10-29_2_AH002_eye_compressed_crf0.mp4')]

        for video_fpath, video_fov in zip(video_files, video_file_fov_names):

            if num_videos_ran == num_videos_to_run_per_call:
                # print('Max video run per call (%.f) reached, stopping.' % num_videos_to_run_per_call)
                break

            if run_video_compression:
                start = time.time()
                compressed_video_path = compress_video(video_fpath)
                end = time.time()
                print('Elapsed time for video compression: %.3f' % (end - start))

            # load facemap parameter based on which rig was used
            proc = rig_procs[exp_info['rigName']]

            # look for facemap processed file
            processed_facemap_path = glob.glob(os.path.join(exp_folder, '*%s*proc.npy' % video_fov))

            # look for text file that says that the video is being processed
            processing_facemap_txt_path = glob.glob(os.path.join(exp_folder, '*%s_processing.txt' % video_fov))

            # look for text file that says that the video is processed
            processed_facemap_txt_path = glob.glob(os.path.join(exp_folder, '*%s_processed.txt' % video_fov))

            # processed / processing files
            processed_or_processing_txt_path = processing_facemap_txt_path + processed_facemap_txt_path

            if len(processed_or_processing_txt_path) != 0:
                porp_date_is_old = np.zeros((len(processed_or_processing_txt_path), ))
                for nfile, processed_txt_path in enumerate(processed_or_processing_txt_path):
                    processed_date = os.path.basename(processed_txt_path)[0:10]

                    if old_date_to_overwrite is not None:
                        if len(old_date_to_overwrite) > 0:
                            old_date_to_overwrite_dt = datetime.datetime.strptime(old_date_to_overwrite, '%Y-%m-%d')
                            processed_date_dt = datetime.datetime.strptime(processed_date, '%Y-%m-%d')
                            if old_date_to_overwrite_dt > processed_date_dt:
                                porp_date_is_old[nfile] = 1

                if np.all(porp_date_is_old):
                    # set face proc file path to empty to trigger recomputation and overwrite
                    processed_facemap_path = []

            # check whether file was already marked as corrupted
            vid_corrupted = check_file_corrupted(vid_path=video_fpath)

            if output_format == 'ONE':
                if not os.path.isdir(os.path.join(exp_folder, 'ONE_preproc')):
                    os.makedirs(os.path.join(exp_folder, 'ONE_preprc'))
                if not os.path.isdir(os.path.join(exp_folder, 'ONE_preproc',
                                                  video_fov)):
                    os.makedirs(os.path.join(exp_folder, 'ONE_preproc',
                                             video_fov))

                corrupted_json_file = os.path.join(exp_folder, 'ONE_preproc',
                                                   video_fov, '%s_corrupted.json' % video_fov)

            if vid_corrupted:
                if output_format == 'ONE':
                    open(corrupted_json_file, 'a').close()

            corrupted_txt_file = os.path.join(exp_folder, '%s_corrupted.txt' % video_fov)
            corrupted_txt_file_not_found = len(glob.glob(corrupted_txt_file)) == 0
            if (not corrupted_txt_file_not_found) & (not vid_corrupted):
                # false alarm, delete the corruption files
                os.remove(corrupted_txt_file)

            if output_format == 'ONE':
                # check for false alarm JSON files, and delete them
                corrupted_json_file_found = (len(glob.glob(corrupted_json_file)) >= 1)
                if corrupted_json_file_found & (not vid_corrupted):
                    os.remove(corrupted_json_file)

            if recompute_facemap:
                if len(processed_facemap_path) != 0:
                    print('Found processed facemap path at: %s, '
                          'but going to recompute facemap because '
                          'recompute_facemap is set to True' % processed_facemap_path)

            if ((len(processed_facemap_path) == 0) or recompute_facemap) & (len(processing_facemap_txt_path) == 0) & (
            corrupted_txt_file_not_found):

                # pdb.set_trace()
                print('%s not processed yet, will run facemap on it now' % video_fov)


                # Check file is not corrupted
                vid_corrupted = check_file_corrupted(vid_path=video_fpath)
                if vid_corrupted:
                    open(corrupted_txt_file, 'a').close()

                    # also write a json file to the ONE folder
                    if output_format == 'ONE':
                        if not os.path.isdir(os.path.join(exp_folder, 'ONE_preproc')):
                            os.makedirs(os.path.join(exp_folder, 'ONE_preprc'))
                        if not os.path.isdir(os.path.join(exp_folder, 'ONE_preproc',
                                                          video_fov)):
                            os.makedirs(os.path.join(exp_folder, 'ONE_preproc',
                                                    video_fov))

                        corrupted_json_file = os.path.join(exp_folder, 'ONE_preproc',
                                                           video_fov, '%s_corrupted.json' % video_fov)
                        open(corrupted_json_file, 'a').close()


                    print('%s is corrupted, skipping...' % video_fov)
                    continue

                if write_to_log:
                    log_file_paths = glob.glob(os.path.join('C:/autoRunLog', '*.txt'))
                    log_file_path = natsort.natsorted(log_file_paths)[-1]
                    with open(log_file_path, 'a') as f:
                        f.write('Processing %s %s \n' % (exp_folder, video_fov))

                # Check whether to run on cropped ROI
                if run_on_cropped_roi:

                    # cut video
                    cut_video_fpath = cut_video(video_fpath)

                    # run deeplabcut to get the anchor coordinates
                    working_directory = 'C:/Users/Experiment/Desktop/avRigDLC'
                    if 'eyeCam' in video_fpath:
                        projectName = 'pinkrigs'
                    elif 'frontCam' in video_fpath:
                        projectName = 'pinkrigsFrontCam'
                    elif 'sideCam' in video_fpath:
                        projectName = 'pinkrigsSideCam'

                    project_folder_search = glob.glob(os.path.join(working_directory, '%s-Tim*' % projectName))
                    project_folder = project_folder_search[0]
                    yaml_file_path = os.path.join(project_folder, 'config.yaml')

                    deeplabcut.analyze_videos(yaml_file_path, [cut_video_fpath],
                                               save_as_csv=True)

                    # read the coordinates
                    roi_window = get_dlc_roi_window(video_fpath, projectName)

                    # modify the crop window settings to give to facemap
                    # currently assume just a single ROI

                    # temp fix for negative ranges (and remove zero for now as well)
                    # this is to do with eyeCam flipped
                    """
                    if 'eyeCam' in video_fpath:
                        pdb.set_trace()
                        if np.min(roi_window['xrange']) <= 0:
                            roi_window['xrange'] = roi_window['xrange'][roi_window['xrange'] > 0]
                            roi_window_xrange_width = roi_window['xrange'][-1] - roi_window['xrange'][0]
                            roi_window['xrange'] = roi_window['xrange'] + roi_window_xrange_width

                            roi_window['yrange'] = roi_window['yrange'][roi_window['yrange'] > 0]
                            roi_window_yrange_height = roi_window['yrange'][-1] - roi_window['yrange'][0]
                            roi_window['yrange'] = roi_window['yrange'] + roi_window_yrange_height
                    """

                    proc['rois'][0]['xrange'] = roi_window['xrange']
                    proc['rois'][0]['yrange'] = roi_window['yrange']
                    print(proc['rois'][0]['xrange'])

                    # if plot_results:
                    #     # plot the ROI locations and window just to check
                    #     pdb.set_trace()


                # make an empty text file saying that the facemap file is being processed
                e = datetime.datetime.now()
                dt_string = e.strftime("%Y-%m-%d-%H-%M-%S")
                processing_facemap_txt_file = os.path.join(exp_folder, '%s_%s_processing.txt' % (dt_string, video_fov))
                open(processing_facemap_txt_file, 'a').close()

                video_path_basename = os.path.basename(video_fpath).split('.')[0]
                # save_path = os.path.join(exp_folder, '%s_proc.npy' % video_path_basename)
                # still not quite sure needs this list in a list...
                if not skip_svd_step:
                    run_facemap_mod([[video_fpath]], proc=proc)
                num_videos_ran += 1
                # run_facemap(video_fpath)

                # finished processing, and so rename the text file to processed
                e = datetime.datetime.now()
                dt_string = e.strftime("%Y-%m-%d-%H-%M-%S")
                processed_txt_file_name = os.path.join(exp_folder, '%s_%s_processed.txt' % (dt_string, video_fov))
                os.rename(processing_facemap_txt_file, processed_txt_file_name)

                if write_to_log:
                    log_file_paths = glob.glob(os.path.join('C:/autoRunLog', '*.txt'))
                    log_file_path = natsort.natsorted(log_file_paths)[-1]
                    with open(log_file_path, 'a') as f:
                        f.write('Finished Processing %s %s \n' % (exp_folder, video_fov))

                # Convert things to ONE format
                if (output_format == 'ONE') and (not skip_svd_step):
                    print('Converting files to ONE format')
                    facemap_output_path = glob.glob(os.path.join(exp_folder, '*%s*proc.npy') % video_fov)[0]
                    convert_facemap_output_to_ONE_format(facemap_output_path)

                if skip_svd_step:
                    # some temp code here to plot DLC output

                    # read the subset video
                    cut_video_array = skvideo.io.vread(cut_video_fpath)
                    avgframe = np.mean(cut_video_array, axis=0)

                    with plt.style.context(splstyle.get_style('nature-reviews')):
                        fig, ax = plt.subplots()
                        ax.imshow(np.mean(avgframe, axis=2), aspect='auto', cmap='gray')

                        # Detected body locations
                        if 'frontCam' in video_fpath:
                            ax.scatter(roi_window['eyeL_x_mean'],
                                            roi_window['eyeL_y_mean'], lw=0, s=20, color='blue')
                            ax.scatter(roi_window['eyeR_x_mean'],
                                           roi_window['eyeR_y_mean'], lw=0, s=20, color='red')

                            ax.scatter(roi_window['snoutF_x_mean'],
                                       roi_window['snoutF_y_mean'], lw=0, s=20, color='orange')

                            #axs[0].scatter(roi_window['snoutF_x_mean'],
                            #               roi_window['snoutF_y_mean'], lw=0, s=20, color='green')

                        elif 'eyeCam' in video_fpath:
                            ax.scatter(roi_window['eyeR_x_mean'],
                                           roi_window['eyeR_y_mean'], lw=0, s=20, color='red')
                            ax.scatter(roi_window['eyeL_x_mean'],
                                       roi_window['eyeL_y_mean'], lw=0, s=20, color='blue')




                        ax.set_xlabel('x axis pixel', size=12)
                        ax.set_ylabel('y axis pixel', size=12)
                        ax.spines['top'].set_visible(False)
                        ax.spines['right'].set_visible(False)
                        ax.spines['left'].set_visible(False)
                        ax.spines['bottom'].set_visible(False)

                        fig_name = '%s_%.f_%s_dlc_%s_avgframe.png' % (exp_info['expDate'], exp_info['expNum'], exp_info['subject'], video_fov)
                        fig.savefig(os.path.join(exp_folder, fig_name), dpi=300, bbox_inches='tight')

                if plot_results and (not skip_svd_step):

                    if not skip_svd_step:
                        # load facemap results
                        facemap_output_path = glob.glob(os.path.join(exp_folder, '*%s*proc.npy') % video_fov)[0]
                        facemap_output = np.load(facemap_output_path, allow_pickle=True).item()

                        # Plot average frame
                        avgframe_reshape = facemap_output['avgframe_reshape']
                        fig, ax = plt.subplots()
                        ax.imshow(avgframe_reshape, aspect='auto', cmap='gray')
                        ax.set_xlabel('x axis pixel', size=12)
                        ax.set_ylabel('y axis pixel', size=12)
                        ax.spines['top'].set_visible(False)
                        ax.spines['right'].set_visible(False)
                        ax.spines['left'].set_visible(False)
                        ax.spines['bottom'].set_visible(False)
                        fig_name = '%s_%.f_%s_facemap_%s_avgframe_reshaped.png' % (exp_info['expDate'], exp_info['expNum'], exp_info['subject'], video_fov)
                        fig.savefig(os.path.join(exp_folder, fig_name), dpi=300, bbox_inches='tight')

                        plt.close(fig)


                    # Plot average frame + ROI window + top motion SVDs + DLC anchor points
                    motmask_reshape = facemap_output['motMask_reshape'][1]

                    roi_y_idx = facemap_output['rois'][0]['yrange']
                    roi_x_idx = facemap_output['rois'][0]['xrange']
                    roi_y_idx_range = roi_y_idx[-1] - roi_y_idx[0]
                    roi_x_idx_range = roi_x_idx[-1] - roi_x_idx[0]

                    avgframe_reshape_roi_subset = avgframe_reshape[roi_y_idx, :][:, roi_x_idx]

                    with plt.style.context(splstyle.get_style('nature-reviews')):
                        fig, axs = plt.subplots(1, 5)
                        fig.set_size_inches(15, 3)

                        cmap = 'gray'

                        axs[0].imshow(avgframe_reshape, cmap=cmap)
                        axs[0].set_title('Average frame', size=11)
                        roi_rect = mpl.patches.Rectangle(
                            (roi_x_idx[0], roi_y_idx[0]),
                            roi_x_idx_range, roi_y_idx_range, edgecolor='red', facecolor='red', fill=False, lw=1
                        )

                        axs[0].add_patch(roi_rect)

                        # Detected body locations
                        if 'frontCam' in video_fpath:
                            axs[0].scatter(roi_window['eyeL_x_mean'],
                                            roi_window['eyeL_y_mean'], lw=0, s=20, color='blue')
                            axs[0].scatter(roi_window['eyeR_x_mean'],
                                           roi_window['eyeR_y_mean'], lw=0, s=20, color='red')

                            #axs[0].scatter(roi_window['snoutF_x_mean'],
                            #               roi_window['snoutF_y_mean'], lw=0, s=20, color='green')

                        elif 'eyeCam' in video_fpath:
                            axs[0].scatter(roi_window['eyeR_x_mean'],
                                           roi_window['eyeR_y_mean'], lw=0, s=20, color='red')


                        axs[1].set_title('Average frame cropped ROI', size=11)
                        axs[1].imshow(avgframe_reshape_roi_subset, cmap=cmap)

                        # motion masks
                        n_motmask_to_plot = 3
                        for motmask_idx in np.arange(n_motmask_to_plot):
                            axs[motmask_idx + 2].imshow(motmask_reshape[:, :, motmask_idx], cmap=cmap)
                            axs[motmask_idx + 2].set_title('Motion mask SVD %.f' % (motmask_idx + 1), size=11)

                        fig.suptitle('%s %s exp %.f' %
                                     (exp_info['subject'], exp_info['expDate'], exp_info['expNum']), size=11)
                        fig_name = '%s_%.f_%s_facemap_%s_roi_crop_and_mask.png' % \
                                   (exp_info['expDate'], exp_info['expNum'], exp_info['subject'], video_fov)
                        fig.savefig(os.path.join(exp_folder, fig_name), dpi=300, bbox_inches='tight')


            else:
                file_skipped += 1
                e = datetime.datetime.now()
                dt_string = e.strftime("%Y-%m-%d-%H-%M-%S")
                video_folder = os.path.dirname(video_fpath)
                processed_txt_file = glob.glob(os.path.join(video_folder, '*%s_processed.txt' % video_fov))
                processed_noted = len(processed_txt_file) > 0
                is_processing = len(processing_facemap_txt_path) > 0

                if (not processed_noted) and (not is_processing):
                    print('Filed already processed but not noted, noting it now')
                    processed_txt_file_name = os.path.join(video_folder, '%s_%s_processed.txt' % (dt_string, video_fov))
                    open(processed_txt_file_name, 'a').close()

                if recompute_ONE:
                    print('Facemap processed, but recomputing ONE because recompute_ONE is set to True')
                   
                    facemap_output_path = glob.glob(os.path.join(exp_folder, '*%s*proc.npy') % video_fov)
                    if facemap_output_path:
                        convert_facemap_output_to_ONE_format(facemap_output_path[0])
                    else:
                        print('not processed.')


    if file_skipped == tot_video_files:
        print('Looks like all files are processed or being processed! Taking a break now...')


def get_all_mouse_info():

    # On Zelda-4 timeline machine
    if socket.gethostname() == 'timothysit-cortexlab':
        main_info_folder_in_server = True
        mouse_info_folder = 'smb://zserver.local/code/AVrig/'
        default_server_path = 'smb://128.40.224.65/subjects/'
    else:
        main_info_folder_in_server = False
        mouse_info_folder = '//zserver/Code/AVrig'
        default_server_path = '//128.40.224.65/subjects/'

    if main_info_folder_in_server:
        gvfs = Gio.Vfs.get_default()
        mouse_info_folder = gvfs.get_file_for_uri(mouse_info_folder).get_path()

    subset_mice_to_use = None  # ['FT030', 'FT031', 'FT032', 'FT035']
    subset_date_range = None  # ['2021-12-01', '2021-12-20']

    if subset_mice_to_use is not None:
        mouse_info_csv_paths = []
        for mouse_name in subset_mice_to_use:
            mouse_info_csv_paths.append(
                glob.glob(os.path.join(mouse_info_folder, '%s.csv' % mouse_name))[0]
            )
    else:
        mouse_info_csv_paths = glob.glob(os.path.join(mouse_info_folder, '*.csv'))

    # TODO: test the re.compile().match() code
    #files_to_exclude = ['aMasterMouseList.csv',
    #                    'kilosort_queue.csv',
    #                    'video_corruption_check.csv',
    #                    '!MouseList.csv']
    files_to_exclude = []

    pattern_to_match = re.compile('[A-Z][A-Z][0-9][0-9][0-9]')

    for path in mouse_info_csv_paths:
        if os.path.basename(path) in files_to_exclude:
            mouse_info_csv_paths.remove(path)
        fname_without_ext = path.split(os.sep)[-1].split('.')[0]
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
            if socket.gethostname() == 'timothysit-cortexlab':
                server_paths= []
                for eFolder in mouse_info['expFolder']:
                    if 'zinu' in eFolder:
                        spath =  'smb://zinu.local/subjects/'
                        server_paths.append(spath)
                    elif 'znas' in eFolder:
                        spath = 'smb://znas.local/subjects/'
                        server_paths.append(spath)
                    else:
                        print(eFolder)
                mouse_info['server_path'] = server_paths
            else:
                server_paths = ['//%s/%s' % (x.split(os.sep)[2], x.split(os.sep)[3]) for x in mouse_info['expFolder'].values]
                mouse_info['server_path'] = server_paths
        if 'path' in mouse_info.columns:
            mouse_info['server_path'] = \
                ['//' + '/'.join(x.split('\\')[2:4]) for x in mouse_info['path']]

        all_mouse_info.append(mouse_info)

    all_mouse_info = pd.concat(all_mouse_info)

    return all_mouse_info


def run_summarize_progress(load_from_server=True, video_ext='.mj2'):
    """
    Looks through videos on the server and see which ones are processed / corrupted
    Parameters
    ----------
    load_from_server : bool
        only applies to Linux computers, whether the filepath is a server paths
    video_ext : str
        video extensions to look for
    Returns
    -------
    progress_df : pandas dataframe
        dataframe with information about name of each video file, whether it is processed,
        when it was processed, and whether the video is corrupted
    """

    all_mouse_info= get_all_mouse_info()

    vid_path_list = []
    vid_processed_list = []
    vid_corrupted_list = []
    vid_processed_date_list = []

    for row_idx, exp_info in all_mouse_info.iterrows():
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

        for video_fpath, video_fov in zip(video_files, video_file_fov_names):
            # look for facemap processed file
            processed_facemap_path = glob.glob(os.path.join(exp_folder, '*%s*proc.npy' % video_fov))

            # look for text file that says that the video is being processed
            processing_facemap_txt_path = glob.glob(os.path.join(exp_folder, '*%s_processing.txt' % video_fov))

            # look for text file that says that the video is processed
            processed_facemap_txt_path = glob.glob(os.path.join(exp_folder, '*%s_processed.txt' % video_fov))

            # check whether file was already marked as corrupted
            vid_corrupted = check_file_corrupted(vid_path=video_fpath)
            corrupted_txt_file = os.path.join(exp_folder, '%s_corrupted.txt' % video_fov)
            corrupted_txt_file_not_found = len(glob.glob(corrupted_txt_file)) == 0
            if (not corrupted_txt_file_not_found) & (not vid_corrupted):
                # false alarm, delete the corruption files
                os.remove(corrupted_txt_file)

            if (len(processed_facemap_path) == 0) & (len(processing_facemap_txt_path) == 0) & (
            corrupted_txt_file_not_found):
                vid_processed = 0
                vid_processed_date= np.nan
            else:
                vid_processed = 1

                if len(processed_facemap_txt_path) == 1:
                    processed_txt_path = processed_facemap_txt_path[0]
                    fname = processed_txt_path.split(os.sep)[-1]
                    vid_processed_date = fname[0:10]
                else:
                    vid_processed_date= np.nan

            vid_path_list.append(video_fpath)
            vid_processed_list.append(vid_processed)
            vid_corrupted_list.append(vid_corrupted)
            vid_processed_date_list.append(vid_processed_date)



    progress_df = pd.DataFrame.from_dict({
        'vid_path': vid_path_list,
        'vid_processed_list': vid_processed_list,
        'vid_corrupted_list': vid_corrupted_list,
        'vid_procesed_date_list': vid_processed_date_list
    })

    return progress_df



def main(**csv_kwargs):
    """
    Main script for running facemap
    Parameters
    ----------
    csv_kwargs :
        arguments given to the queryCSV() function to determine which video files to process
    Returns
    -------

    """

    print('run_facemap called')

    how_often_to_check = 3600  # how often to check the time (seconds), currently not used
    override_time_check = True
    override_limit = 30  # how many times to override time checking before stopping
    override_counter = 0
    continue_running = True  # fixed at True at the start
    summarize_progress = False
    update_mouse_csvs = False
    run_plot_facemap_results = False
    output_format = 'ONE'
    process_most_recent = True
    recompute_ONE = False
    recompute_facemap = False
    write_to_log = True
    skip_svd_step = False
    old_date_to_overwrite = '2023-01-09'
    # if facemap processed data is older than this date, then overwrite existing
    # (regardless of recompute_facemap) leave empty '' or None to forego option

    sessions = queryCSV(**csv_kwargs)
    if process_most_recent:
        sessions = sessions.sort_values('expDate')[::-1]

    # Temp for testing
    # test_path = '/Users/timothysit/FT038/2021-11-04/1/2021-11-04_1_FT038_eyeCam_proc.npy'
    # convert_facemap_output_to_ONE_format(test_path)

    if update_mouse_csvs:
        update_mouse_csv_record()

    if summarize_progress:
        run_summarize_progress()

    if run_plot_facemap_results:
        plot_facemap_results()
        
    subset_mice_to_use = None

    while continue_running:
        e = datetime.datetime.now()
        print("The time is now: %s:%s:%s" % (e.hour, e.minute, e.second))

        hour_str = '%s' % e.hour
        hour_int = int(hour_str)

        if override_time_check:
            if override_counter >= override_limit:
                override_time_check = False

        if (hour_int < 8) | (hour_int >= 20) | override_time_check:
            print('It is prime time to run some facemap!')

            if subset_mice_to_use is not None:
                print('Running facemap on specified subset of mice: %s' % subset_mice_to_use)
            
            batch_process_facemap(output_format=output_format, sessions=sessions,
                                  subset_mice_to_use=subset_mice_to_use,
                                  recompute_ONE=recompute_ONE,
                                  recompute_facemap=recompute_facemap,
                                  old_date_to_overwrite=old_date_to_overwrite,
                                  write_to_log=write_to_log,
                                  skip_svd_step=skip_svd_step)


            if override_time_check:
                override_counter += 1



        else:
            print('It is after 8am, will stop running facemap')
            continue_running = False

if __name__ == '__main__':
    main(subject='all', expDate='last1000')