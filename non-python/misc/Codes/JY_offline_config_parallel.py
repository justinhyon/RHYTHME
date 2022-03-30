#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
10/16/2020

@author: jye

VERSION 1.0.0
"""
import sys
import numpy as np
# import multiprocessing as mp
from multiprocessing import Process, Queue
# from multiprocessing import Pool
import os



# insert path to directory where JY_offline_client.py is located
path = '/Users/jessicaye/OneDrive - California Institute of Technology/ShimojoLab/MyPaper/Validation'
sys.path.insert(1, path)
from JY_offline_client_v2 import TeamFlow

# general configurations
filepaths = []
folder = '/Users/jessicaye/Desktop/ShimojoLabData/1D_Trimmed_data_before_FASTER_separate/JoinedData/170819'
savepath = '/Users/jessicaye/Desktop/ShimojoLabData/1D_Trimmed_data_before_FASTER_separate/PythonValidation'

# Get all .set files in folder
for file in os.listdir(folder):
    if file.endswith('.set'):
        filepaths.append(os.path.join(folder, file))
# filepath = '/users/jessicaye/Desktop/T1_Beep.bdf'  # path to the  bdf data file
# filepath2 = '/users/jessicaye/Desktop/T2_PorTm.bdf'
dataport = 1972  # not used
blocksize_sec = 5  # number of seconds per segment
windowsize = 12  # size of the figures
delay = .1  # not used
plotpref = 'none'  # use either 'participant'(teamflow score plot), 'experiment'(detailed plot), 'both', or 'none' to
                    # toggle showing the plots
saving = False  # True or False. toggle saving the plots. Must create a folder called "TF_figures" in path directory
                # (line 13) before running. there is currently a bug that prevents the team flow scores plot from saving
fsample = 256  # samples per second
badchans1 = ['A1', 'B2', 'C1', 'C21', 'C23', 'D14', 'D21', 'D31', 'D32'] # bad channels
badchans2 = []

# ----------------aep configurations-----------------
TrigChan = '256'  # python begins numbering from 0, so this is the channel number of the stim channel - 1 (not the name)
# subject 1
channelofint_aep1 = [1, 2, 3, 33, 34, 52, 65, 66, 87, 97, 98, 110, 111, 112]
epoeventval1 = 5  # stim channel event value desired
pretrig1 = .3  # seconds before event
posttrig1 = .5  # seconds after event
# subject 2
channelofint_aep2 = [129, 130, 131, 161, 162, 180, 193, 194, 215, 225, 226, 238, 239, 240]
# channelofint_aep2 = channelofint_aep1.copy() # SPECIALLY FOR BASELINE, WHICH ONLY NEEDS 1 PARTICIPANT
epoeventval2 = 5
pretrig2 = .3
posttrig2 = .5

# -----------------plv configurations (unused since MNE does everything)-----------------
channelofint_plv = 0
frqofint = 0

# --------------psd configurations. T= temporal, F=frontal, A=occipital. 1 and 2 are the participant #s----------------
# Gamma
channelofint_psdT1 = [116, 117, 119, 126] #[117, 118, 120, 127] actual channels of interest subject 1
foillowT1 = 30
foihighT1 = 119
# channelofint_psdT2 = channelofint_psdT1.copy() # SPECIALLY FOR BASELINE, WHICH ONLY NEEDS 1 PARTICIPANT
channelofint_psdT2 = [244, 245, 247, 254] #[245, 246, 248, 255] actual channels of interest subject 2
foillowT2 = 30
foihighT2 = 119

# Beta
channelofint_psdF1 = [116, 117, 119, 126] #[117, 118, 120, 127] actual channels of interest subject 1
foillowF1 = 12
foihighF1 = 29
# channelofint_psdF2 = channelofint_psdF1.copy() # SPECIALLY FOR BASELINE, WHICH ONLY NEEDS 1 PARTICIPANT
channelofint_psdF2 = [244, 245, 247, 254] #[245, 246, 248, 255] actual channels of interest subject 2
foillowF2 = 12
foihighF2 = 29

# Alpha
channelofint_psdA1 = [116, 117, 119, 126] #[117, 118, 120, 127] actual channels of interest subject 1
foillowA1 = 7
foihighA1 = 11
# channelofint_psdA2 = channelofint_psdA1.copy() # SPECIALLY FOR BASELINE, WHICH ONLY NEEDS 1 PARTICIPANT
channelofint_psdA2 = [244, 245, 247, 254] #[245, 246, 248, 255] actual channels of interest subject 2
foillowA2 = 7
foihighA2 = 11

if __name__ == '__main__':
    TF = TeamFlow(path, savepath, dataport, blocksize_sec, windowsize, delay, plotpref, saving, channelofint_aep1,
                  channelofint_aep2, TrigChan,
                  epoeventval1, pretrig1, posttrig1, epoeventval2, pretrig2, posttrig2, channelofint_plv,
                  frqofint, fsample, channelofint_psdT1, channelofint_psdT2, channelofint_psdF1, channelofint_psdF2,
                  channelofint_psdA1, channelofint_psdA2, foillowT1, foihighT1, foillowT2, foihighT2, foillowF1,
                  foihighF1, foillowF2, foihighF2, foillowA1, foihighA1, foillowA2, foihighA2)

    # filepaths = [filepath, filepath2]
    ncores = 3
    startidx = 0
    endidx = ncores
    while startidx < len(filepaths):
        if endidx > len(filepaths):
            endidx = len(filepaths)

        processes = []
        for i in range(startidx, endidx, 1):
            process = Process(target=TF.receive_data, args=(filepaths[i], badchans1, badchans2))
            processes.append(process)

        for j in processes:
            j.start()

        for j in processes:
            j.join()

        for j in processes:
            j.kill()

        startidx = endidx
        endidx = endidx + ncores


    # If fork() error, run command line: OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES python JY_offline_config_parallel.py



