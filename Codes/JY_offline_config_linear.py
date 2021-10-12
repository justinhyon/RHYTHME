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
path = '/Users/jasonhyon/Downloads/Codes/JY_offline_client_v4.py'
sys.path.insert(1, path)
from JY_offline_client_v4 import TeamFlow

# general configurations
filepath = '/Users/jasonhyon/Downloads/JoinedData/170719_TRIAL2__EM_MD.set'  # path to the  bdf data file
savepath = '/Users/jasonhyon/Desktop/merge_test'
dataport = 1972  # not used
blocksize_sec = 5  # number of seconds per segment
num_plvsimevents = 5 # number of simulated events for each PLV segment
windowsize = 12  # size of the figures
delay = .1  # not used
plotpref = 'both'  # use either 'participant'(teamflow score plot), 'experiment'(detailed plot), 'both', or 'none' to
                    # toggle showing the plots
saving = False  # True or False. toggle saving the plots. Must create a folder called "TF_figures" in path directory
                # (line 13) before running. there is currently a bug that prevents the team flow scores plot from saving
fsample = 256  # samples per second
badchans1 = ['A32', 'B2'] # bad channels
badchans2 = []
nchansparticipant = 128 #number of channels per particiapnt

# ----------------aep configurations-----------------
TrigChan = '256'  # python begins numbering from 0, so this is the channel number of the stim channel - 1 (not the name)
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

# -----------------plv configurations (unused since MNE does everything)-----------------
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

numparticipants = 2

if __name__ == '__main__':
    TF = TeamFlow(path, savepath, dataport, windowsize, delay, plotpref, saving, TrigChan,)


    TF.receive_data(filepath, [badchans1, badchans2], fsample, nchansparticipant, TrigChan,
                    ['Gamma1', 'Beta1', 'Alpha1', 'Gamma2', 'Beta2', 'Alpha2'],
                    [channelofint_psdT1, channelofint_psdF1, channelofint_psdA1,
                     channelofint_psdT2, channelofint_psdF2, channelofint_psdA2],
                    [foillowT1, foillowF1, foillowA1], [foihighT1, foihighF1, foihighA1],
                    blocksize_sec, num_plvsimevents, [[channelofint_aep1, epoeventval1, pretrig1, posttrig1],
                    [channelofint_aep2, epoeventval2, pretrig2, posttrig2]])
    # For PSD:
    # List of FOI low and high must be the same length
    # Length of channels list must be an integer multiple of FOI low/high list length
    # Put all features related to the same subject together, in the same feature order
    # For AEP:
    # AEPs should be stored in a list of lists. The wrapper lists should correspond to each of the participants.
    # the number of wrapper lists should equal the number of participants. The nested list should contain data
    # corresponding to a single participant