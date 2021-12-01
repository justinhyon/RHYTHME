#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
10/16/2020

@author: jhyon

VERSION 1.0.0
"""
import sys
import numpy as np
# import multiprocessing.pool

# insert path to directory where JH_offline_client.py is located
path = '/Users/jasonhyon/Documents/GitHub/teamflow/src'
sys.path.insert(1, path)
from JH_realtime_client_2021 import TeamFlow

# general configurations
option = 'realtime'
savepath = '/Users/jasonhyon/Documents/GitHub/teamflow/misc/merge_test/rt'
blocksize_sec = 5  # number of seconds per segment
windowsize = 12  # size of the figures
units = 'uv'  # v for volts, mv for millivolts, uv for microvolts, Mv for megavolts
plotpref = 'both'  # use either 'participant'(teamflow score plot), 'experiment'(detailed plot), 'both', or 'none' to
                    # toggle showing the plots
saving = False  # True or False. toggle saving the plots. Must create a folder called "TF_figures" in path directory
                # (line 13) before running. there is currently a bug that prevents the team flow scores plot from saving
badchans1 = ['A32', 'B2'] # bad channels
badchans2 = []
nchansparticipant = 128 #number of channels per particiapnt
removefirstsample = True
TrigChan = '256'  # python begins numbering from 0, so this is the channel number of the stim channel - 1 (not the name)
numparticipants = 2

# configurations for realtime mode (ignored in offline mode)
exp_name = 'test_2'
dataport = 1972
n_skipped_segs = 1  # max number of segments to skip in between 2 processed segments due to runtime
wait_segs = 2  # How many segments worth of time (blocksize_sec) to wait for new data before quitting the pipeline. does
                # not apply to the initial wait for first segment to arrive
delay = .01  # how long to wait before checking again for new samples in the buffer

# configurations for offline mode (ignored in realtime mode)
filepath = '/Users/jasonhyon/Desktop/JoinedData/170719_TRIAL2__EM_MD.set'  # path to the  bdf data file
fsample = 256  # samples per second

# ----------------aep configurations----------------- erp
# subject 1
channelofint_aep1 = [1, 2, 3, 33, 34, 52, 65, 66, 87, 97, 98, 110, 111, 112]
epoeventval1 = 5  # stim channel event value desired
pretrig1 = .3  # seconds before event
posttrig1 = .5  # seconds after event
# subject 2
channelofint_aep2 = [129, 130, 131, 161, 162, 180, 193, 194, 215, 225, 226, 238, 239, 240]
epoeventval2 = 5
pretrig2 = .3
posttrig2 = .5

# -----------------plv configurations (unused since MNE does everything)----------------- interbrain connectivity ibc
num_plvsimevents = 5 # number of simulated events for each PLV segment
channelofint_plv = 0
frqofint = 0

# --------------psd configurations. T= temporal, F=frontal, A=occipital. 1 and 2 are the participant #s----------------
# Gamma
channelofint_psdT1 = [116, 117, 119, 126] #[117, 118, 120, 127] actual channels of interest subject 1
foillowT1 = 30
foihighT1 = 119
channelofint_psdT2 = [244, 245, 247, 254] #[245, 246, 248, 255] actual channels of interest subject 2
foillowT2 = 30
foihighT2 = 119

# Beta
channelofint_psdF1 = [116, 117, 119, 126] #[117, 118, 120, 127] actual channels of interest subject 1
foillowF1 = 12
foihighF1 = 29
channelofint_psdF2 = [244, 245, 247, 254] #[245, 246, 248, 255] actual channels of interest subject 2
foillowF2 = 12
foihighF2 = 29

# Alpha
channelofint_psdA1 = [116, 117, 119, 126] #[117, 118, 120, 127] actual channels of interest subject 1
foillowA1 = 7
foihighA1 = 11
channelofint_psdA2 = [244, 245, 247, 254] #[245, 246, 248, 255] actual channels of interest subject 2
foillowA2 = 7
foihighA2 = 11



if __name__ == '__main__':
    TF = TeamFlow(path, savepath, dataport, windowsize, delay, plotpref, saving, TrigChan,)


    TF.master_control(filepath, [badchans1, badchans2], fsample, nchansparticipant, TrigChan,
                    ['Gamma1', 'Beta1', 'Alpha1', 'Gamma2', 'Beta2', 'Alpha2'],
                    [channelofint_psdT1, channelofint_psdF1, channelofint_psdA1,
                     channelofint_psdT2, channelofint_psdF2, channelofint_psdA2],
                    [foillowT1, foillowF1, foillowA1], [foihighT1, foihighF1, foihighA1],
                    blocksize_sec, num_plvsimevents, [[channelofint_aep1, epoeventval1, pretrig1, posttrig1],
                    [channelofint_aep2, epoeventval2, pretrig2, posttrig2]], option, delay, units, removefirstsample,
                      exp_name, n_skipped_segs, wait_segs)
    # For PSD:
    # List of FOI low and high must be the same length
    # Length of channels list must be an integer multiple of FOI low/high list length
    # Put all features related to the same subject together, in the same feature order
    # For AEP:
    # AEPs should be stored in a list of lists. The wrapper lists should correspond to each of the participants.
    # the number of wrapper lists should equal the number of participants. The nested list should contain data
    # corresponding to a single participant