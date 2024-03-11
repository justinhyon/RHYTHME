#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
10/16/2020

@author: jhyon

VERSION 1.0.0
"""
import numpy as np

try:
    from .Client import RHYTHME
except ImportError:
    from Client import RHYTHME
# import multiprocessing.pool
import toml
import argparse
import sys


# parser = argparse.ArgumentParser()
# parser.add_argument("--cf", help="Enter the path to the configuration toml file", required=True)
# args = parser.parse_args()

def run_rhythme(config_filepath, offline_filepath=None):
    # print('1')
    with open(str(config_filepath)) as fileObj:
        content = fileObj.read()
        all_config = toml.loads(content)

    # print(str(path))
    # insert path to directory where JH_offline_client.py is located
    # path = '/Users/justinhyon/Documents/GitHub/teamflow/src'
    # sys.path.insert(1, path)
    # print("all_config", all_config)
    # general configurations
    general_config = all_config['general_config']
    option = general_config['option']
    savepath = general_config['savepath']
    print_output = general_config['print_output']

    blocksize_sec = general_config['blocksize_sec']  # number of seconds per segment
    units = general_config['units']  # v for volts, mv for millivolts, uv for microvolts, Mv for megavolts
    nchansparticipant = general_config['nchansparticipant']  # number of channels per particiapnt
    numparticipants = general_config['numparticipants']
    removefirstsample = general_config['removefirstsample']
    resample_freq = general_config['resample_freq']
    filter_data_range = general_config['filter_data_range']

    # plotting configurations
    plotpref = general_config[
        'plotpref']  # use either 'participant'(teamflow score plot), 'experiment'(detailed plot), 'both', or 'none' to
    # toggle showing the plots
    ex_windowsize = general_config['ex_windowsize']  # size of the figures
    sub_windowsize = general_config['sub_windowsize']  # size of the figures
    ex_plot_dims = general_config[
        'ex_plot_dims']  # The dimensions of the plot grid to be used for the experimenter plot. Be sure to count total
    # number of required plots, or desired plots will not be rendered
    sub_plot_dims = general_config[
        'sub_plot_dims']  # The dimensions of the plot grid to be used for the subject plot. Be sure to count total
    # number of required plots, or desired plots will not be rendered
    saving = general_config[
        'saving']  # True or False. toggle saving the plots. Must create a folder called "TF_figures" in path directory
    # (line 13) before running. there is currently a bug that prevents the team flow scores plot from saving

    # channel configurations
    badchans = general_config['badchans']
    TrigChan = general_config[
        'TrigChan']  # python begins numbering from 0, so this is the channel number of the stim channel - 1 (not the name)
    # can be 'none' if there is no trigger channel, and a trigger channel of all 0s will be added to the end of the data
    # use option 'create' if events are being sent from FieldTrip Buffer, but there is no Trigger Channel in the data (ie.
    # realtime mode). This will create a stim channel with the events, and adds it as the last channel.

    channelnames = all_config['channelnames_dict']['channelnames']

    # configurations for realtime mode (ignored in offline mode)
    start_zero = general_config[
        'start_zero']  # whether the experiment must start from the first sample (empty buffer), or False for the current
    #  sample in the buffer
    exp_name = general_config['exp_name']
    dataport = general_config['dataport']  # use 3113 for eeg machine
    ignore_overflow = general_config['ignore_overflow']  # throw out samples past the desired block size
    n_skipped_segs = general_config[
        'n_skipped_segs']  # max number of segments to skip in between 2 processed segments due
    # to runtime
    wait_segs = general_config[
        'wait_segs']  # How many segments worth of time (blocksize_sec) to wait for new data before quitting the pipeline. does
    # not apply to the initial wait for first segment to arrive
    delay = general_config['delay']  # how long to wait before checking again for new samples in the buffer

    # configurations for offline mode (ignored in realtime mode)
    if offline_filepath:
        filepath = offline_filepath
    else:
        filepath = general_config['filepath']  # path to the  bdf data file
    fsample = general_config['fsample']  # samples per second

    function_dict = all_config['function_dict']

    num_plvsimevents = general_config['num_plvsimevents']  # number of simulated events for each PLV segment
    channelofint_plv = general_config['channelofint_plv']
    frqofint = general_config['frqofint']

    # function_dict['psd2'] = {
    #     'values_band1': [],
    #     'channelofint_band1': list(range(129, 161)),#[244, 245, 247, 254],  # [117, 118, 120, 127] actual channels of interest subject 1
    #     'foilow_band1': 30,
    #     'foihigh_band1': 119,
    #     'plotwv_band1': 'experiment',
    #     'plotidx_band1': 'experiment',
    #
    #     'values_band2': [],
    #     'channelofint_band2': list(range(161, 193)),#[244, 245, 247, 254],  # [117, 118, 120, 127] actual channels of interest subject 1
    #     'foilow_band2': 12,
    #     'foihigh_band2': 29,
    #     'plotwv_band2': 'experiment',
    #     'plotidx_band2': 'experiment',
    #
    #     'values_band3': [],
    #     'channelofint_band3': list(range(193, 225)),#[244, 245, 247, 254],  # [117, 118, 120, 127] actual channels of interest subject 1
    #     'foilow_band3': 7,
    #     'foihigh_band3': 11,
    #     'plotwv_band3': 'experiment',
    #     'plotidx_band3': 'experiment',
    #
    #     'values_band4': [],
    #     'channelofint_band4': list(range(225, 256)),#[116, 117, 119, 126],  # [117, 118, 120, 127] actual channels of interest subject 1
    #     'foilow_band4': 7,
    #     'foihigh_band4': 11,
    #     'plotwv_band4': 'experiment',
    #     'plotidx_band4': 'experiment',
    # }
    #
    # # function_dict['psd3'] = {
    # #     'values_band1': [],
    # #     'channelofint_band1': [116, 117, 119, 126],  # [117, 118, 120, 127] actual channels of interest subject 1
    # #     'foilow_band1': 30,
    # #     'foihigh_band1': 119,
    # #     'plotwv_band1': 'experiment',
    # #     'plotidx_band1': 'experiment',
    # #
    # #     'values_band2': [],
    # #     'channelofint_band2': [116, 117, 119, 126],  # [117, 118, 120, 127] actual channels of interest subject 1
    # #     'foilow_band2': 12,
    # #     'foihigh_band2': 29,
    # #     'plotwv_band2': 'experiment',
    # #     'plotidx_band2': 'experiment',
    # #
    # #     'values_band3': [],
    # #     'channelofint_band3': [116, 117, 119, 126],  # [117, 118, 120, 127] actual channels of interest subject 1
    # #     'foilow_band3': 7,
    # #     'foihigh_band3': 11,
    # #     'plotwv_band3': 'experiment',
    # #     'plotidx_band3': 'experiment',
    # # }
    # #
    # # function_dict['psd4'] = {
    # #     'values_band1': [],
    # #     'channelofint_band1': [244, 245, 247, 254],  # [117, 118, 120, 127] actual channels of interest subject 1
    # #     'foilow_band1': 30,
    # #     'foihigh_band1': 119,
    # #     'plotwv_band1': 'experiment',
    # #     'plotidx_band1': 'experiment',
    # #
    # #     'values_band2': [],
    # #     'channelofint_band2': [244, 245, 247, 254],  # [117, 118, 120, 127] actual channels of interest subject 1
    # #     'foilow_band2': 12,
    # #     'foihigh_band2': 29,
    # #     'plotwv_band2': 'experiment',
    # #     'plotidx_band2': 'experiment',
    # #
    # #     'values_band3': [],
    # #     'channelofint_band3': [244, 245, 247, 254],  # [117, 118, 120, 127] actual channels of interest subject 1
    # #     'foilow_band3': 7,
    # #     'foihigh_band3': 11,
    # #     'plotwv_band3': 'experiment',
    # #     'plotidx_band3': 'experiment',
    # # }
    #
    # function_dict['flow'] = {
    #             'values_Intra 1': [],
    #             'values_Intra 2': [],
    #             'values_Inter': []
    #         }
    #

    # # general configurations
    # option = 'realtime'
    # savepath = '../../non-python/misc/merge_test/rt'
    #
    # blocksize_sec = 5  # number of seconds per segment
    # units = 'uv'  # v for volts, mv for millivolts, uv for microvolts, Mv for megavolts
    # nchansparticipant = 128  # number of channels per particiapnt
    # numparticipants = 2
    # removefirstsample = True
    # resample_freq = 256
    #
    # # plotting configurations
    # plotpref = 'experiment'  # use either 'participant'(teamflow score plot), 'experiment'(detailed plot), 'both', or 'none' to
    #                             # toggle showing the plots
    # ex_windowsize = (24, 12)  # size of the figures
    # sub_windowsize = (36, 12)  # size of the figures
    # ex_plot_dims = (3, 4)  # The dimensions of the plot grid to be used for the experimenter plot. Be sure to count total
    #                         # number of required plots, or desired plots will not be rendered
    # sub_plot_dims = (2, 3)  # The dimensions of the plot grid to be used for the subject plot. Be sure to count total
    #                         # number of required plots, or desired plots will not be rendered
    # saving = False  # True or False. toggle saving the plots. Must create a folder called "TF_figures" in path directory
    #                 # (line 13) before running. there is currently a bug that prevents the team flow scores plot from saving
    #
    # # channel configurations
    # badchans = [ # must have list of badchans or empty list for each participant
    #     # ['A32', 'B2'], # participant 1
    #     [], # participant 2
    #     [],
    #     []
    # ]
    # TrigChan = 256  # python begins numbering from 0, so this is the channel number of the stim channel - 1 (not the name)
    # # can be 'none' if there is no trigger channel, and a trigger channel of all 0s will be added to the end of the data
    # # use option 'create' if events are being sent from FieldTrip Buffer, but there is no Trigger Channel in the data (ie.
    # # realtime mode). This will create a stim channel with the events, and adds it as the last channel.
    #
    # channelnames = [
    #
    # ]
    #
    # # THIS CODE GENERATES THE LIST OF CHANNEL NAMES USED IN OUR EXPERIMENT AND SHOULD NOT REMAIN IN THE FINAL CODE
    # # <<<<<<< Updated upstream
    # channelnum= 1
    # for p in range(2):
    #     thischannelnames = []
    #     for letter in ['A', 'B', 'C', 'D']:
    #         for num in range(1, 33):
    #             thischannelnames.append(letter + str(num))
    #
    #     channelnames.append(thischannelnames)
    # # =======
    # # channelnames = []
    # # >>>>>>> Stashed changes
    # # for p in range(2):
    # #     thischannelnames = []
    # #     for letter in ['A', 'B', 'C', 'D']:
    # #         for num in range(1, 33):
    # # <<<<<<< Updated upstream
    # #             channelnames.append(str(channelnum) + "=" + letter + str(num))
    # #             channelnum+=1
    # #     # channelnames.append(thischannelnames)
    # # print(channelnames)
    # # print ('\n'.join(channelnames))
    # # =======
    # #             thischannelnames.append(letter + str(num))
    # #     channelnames.append(thischannelnames)
    # # >>>>>>> Stashed changes
    #
    #
    # # configurations for realtime mode (ignored in offline mode)
    # start_zero = True  # whether the experiment must start from the first sample (empty buffer), or False for the current
    #                     #  sample in the buffer
    # exp_name = 'test_2'
    # dataport = 1972 # use 3113 for eeg machine
    # ignore_overflow = True # throw out samples past the desired block size
    # n_skipped_segs = 1  # max number of segments to skip in between 2 processed segments due to runtime
    # wait_segs = 2  # How many segments worth of time (blocksize_sec) to wait for new data before quitting the pipeline. does
    # # not apply to the initial wait for first segment to arrive
    # delay = .01  # how long to wait before checking again for new samples in the buffer
    #
    # # configurations for offline mode (ignored in realtime mode)
    # filepath = '/Users/jasonhyon/Desktop/JoinedData/170719_TRIAL2__EM_MD.set'  # path to the  bdf data file
    # fsample = 256  # samples per second
    #
    # function_dict = {}
    # # ----------------ERP configurations----------------- erp
    # #
    # function_dict['ERP1'] = {
    #     # subject 1
    #     'values_ERP': [],
    #     'channelofint': [1, 2, 3, 33, 34, 52, 65, 66, 87, 97, 98, 110, 111, 112],
    #     'epoeventval': 7,  # stim channel event value desired
    #     'pretrig': 1,  # seconds before event
    #     'posttrig': 1,  # seconds after event
    #     'bands': [[.110, .150], [.210, .250], [.310, .350]],
    #     'signs': ['-', '+', '-'], #  + for same sign of the average value in the corresponding band, - to reverse the sign, abs for absolute value
    #     'plotwv': 'experiment',
    #     'plotidx': 'experiment',
    #
    # }
    # function_dict['ERP2'] = {
    #     # subject 2
    #     'values_ERP': [],
    #     'channelofint': [129, 130, 131, 161, 162, 180, 193, 194, 215, 225, 226, 238, 239, 240],
    #     'epoeventval': 5,
    #     'pretrig': .3,
    #     'posttrig': .5,
    #     'bands': [[.110, .150], [.210, .250], [.310, .350]],
    #     'signs': ['-', '+', '-'],
    #     'plotwv': 'experiment',
    #     'plotidx': 'experiment'
    # }
    # # # function_dict['ERP3'] = {
    # # #     # subject 1
    # # #     'values_ERP': [],
    # # #     'channelofint': [1, 2, 3, 33, 34, 52, 65, 66, 87, 97, 98, 110, 111, 112],
    # # #     'epoeventval': 5,  # stim channel event value desired
    # # #     'pretrig': .3,  # seconds before event
    # # #     'posttrig': .5,  # seconds after event
    # # #     'bands': [[.110, .150], [.210, .250], [.310, .350]],
    # # #     'plotwv': 'experiment',
    # # #     'plotidx': 'experiment'
    # # # }
    # # #
    # # # function_dict['ERP4'] = {
    # # #     # subject 1
    # # #     'values_ERP': [],
    # # #     'channelofint': [1, 2, 3, 33, 34, 52, 65, 66, 87, 97, 98, 110, 111, 112],
    # # #     'epoeventval': 5,  # stim channel event value desired
    # # #     'pretrig': .3,  # seconds before event
    # # #     'posttrig': .5,  # seconds after event
    # # #     'bands': [[.110, .150], [.210, .250], [.310, .350]],
    # # #     'plotwv': 'experiment',
    # # #     'plotidx': 'experiment'
    # # # }
    # # # -----------------plv configurations (unused since MNE does everything)----------------- interbrain connectivity ibc
    # num_plvsimevents = 5  # number of simulated events for each PLV segment
    # channelofint_plv = 0
    # frqofint = 0
    # function_dict['plv'] = {
    #     'plot_option': 'experiment', # either tuple or one of 'experiment' or 'participant'. if tuple specified, intrabrain plv
    #                             # indexes will be plotted for the participants desired. if 'experiment' or 'participant' is
    #                             # specified, an average of all the intrabrain plvs is plotted
    #     # interbrain settings must come first
    #     'values_inter': [],
    #     'plot_matrix': 'experiment',
    #     'index_inter': 'experiment',
    #
    #     'values_intra1': [],
    #     'index_intra1': 'experiment',
    #
    #     'values_intra2': [],
    #     'index_intra2': 'experiment',
    #     #
    #     # 'values_intra3': [],
    #     # 'index_intra3': 'experiment',
    #     #
    #     # 'values_intra4': [],
    #     # 'index_intra4': 'experiment',
    #
    # }
    #
    # # # --------------psd configurations. T= temporal, F=frontal, A=occipital. 1 and 2 are the participant #s----------------
    # function_dict['psd1'] = {
    #     'values_band1': [],
    #     'channelofint_band1': list(range(32)),#[116, 117, 119, 126],  # [117, 118, 120, 127] actual channels of interest subject 1
    #     'foilow_band1': 30,
    #     'foihigh_band1': 119,
    #     'plotwv_band1': 'experiment',
    #     'plotidx_band1': 'experiment',
    #
    #     'values_band2': [],
    #     'channelofint_band2': list(range(32, 64)),#[116, 117, 119, 126],  # [117, 118, 120, 127] actual channels of interest subject 1
    #     'foilow_band2': 12,
    #     'foihigh_band2': 29,
    #     'plotwv_band2': 'experiment',
    #     'plotidx_band2': 'experiment',
    #
    #     'values_band3': [],
    #     'channelofint_band3': list(range(64, 97)),#[116, 117, 119, 126],  # [117, 118, 120, 127] actual channels of interest subject 1
    #     'foilow_band3': 7,
    #     'foihigh_band3': 11,
    #     'plotwv_band3': 'experiment',
    #     'plotidx_band3': 'experiment',
    #
    #     'values_band4': [],
    #     'channelofint_band4': list(range(97, 129)),#[116, 117, 119, 126],  # [117, 118, 120, 127] actual channels of interest subject 1
    #     'foilow_band4': 7,
    #     'foihigh_band4': 11,
    #     'plotwv_band4': 'experiment',
    #     'plotidx_band4': 'experiment',
    # }
    #
    # # function_dict['psd2'] = {
    # #     'values_band1': [],
    # #     'channelofint_band1': list(range(129, 161)),#[244, 245, 247, 254],  # [117, 118, 120, 127] actual channels of interest subject 1
    # #     'foilow_band1': 30,
    # #     'foihigh_band1': 119,
    # #     'plotwv_band1': 'experiment',
    # #     'plotidx_band1': 'experiment',
    # #
    # #     'values_band2': [],
    # #     'channelofint_band2': list(range(161, 193)),#[244, 245, 247, 254],  # [117, 118, 120, 127] actual channels of interest subject 1
    # #     'foilow_band2': 12,
    # #     'foihigh_band2': 29,
    # #     'plotwv_band2': 'experiment',
    # #     'plotidx_band2': 'experiment',
    # #
    # #     'values_band3': [],
    # #     'channelofint_band3': list(range(193, 225)),#[244, 245, 247, 254],  # [117, 118, 120, 127] actual channels of interest subject 1
    # #     'foilow_band3': 7,
    # #     'foihigh_band3': 11,
    # #     'plotwv_band3': 'experiment',
    # #     'plotidx_band3': 'experiment',
    # #
    # #     'values_band4': [],
    # #     'channelofint_band4': list(range(225, 256)),#[116, 117, 119, 126],  # [117, 118, 120, 127] actual channels of interest subject 1
    # #     'foilow_band4': 7,
    # #     'foihigh_band4': 11,
    # #     'plotwv_band4': 'experiment',
    # #     'plotidx_band4': 'experiment',
    # # }
    # #
    # # # function_dict['psd3'] = {
    # # #     'values_band1': [],
    # # #     'channelofint_band1': [116, 117, 119, 126],  # [117, 118, 120, 127] actual channels of interest subject 1
    # # #     'foilow_band1': 30,
    # # #     'foihigh_band1': 119,
    # # #     'plotwv_band1': 'experiment',
    # # #     'plotidx_band1': 'experiment',
    # # #
    # # #     'values_band2': [],
    # # #     'channelofint_band2': [116, 117, 119, 126],  # [117, 118, 120, 127] actual channels of interest subject 1
    # # #     'foilow_band2': 12,
    # # #     'foihigh_band2': 29,
    # # #     'plotwv_band2': 'experiment',
    # # #     'plotidx_band2': 'experiment',
    # # #
    # # #     'values_band3': [],
    # # #     'channelofint_band3': [116, 117, 119, 126],  # [117, 118, 120, 127] actual channels of interest subject 1
    # # #     'foilow_band3': 7,
    # # #     'foihigh_band3': 11,
    # # #     'plotwv_band3': 'experiment',
    # # #     'plotidx_band3': 'experiment',
    # # # }
    # # #
    # # # function_dict['psd4'] = {
    # # #     'values_band1': [],
    # # #     'channelofint_band1': [244, 245, 247, 254],  # [117, 118, 120, 127] actual channels of interest subject 1
    # # #     'foilow_band1': 30,
    # # #     'foihigh_band1': 119,
    # # #     'plotwv_band1': 'experiment',
    # # #     'plotidx_band1': 'experiment',
    # # #
    # # #     'values_band2': [],
    # # #     'channelofint_band2': [244, 245, 247, 254],  # [117, 118, 120, 127] actual channels of interest subject 1
    # # #     'foilow_band2': 12,
    # # #     'foihigh_band2': 29,
    # # #     'plotwv_band2': 'experiment',
    # # #     'plotidx_band2': 'experiment',
    # # #
    # # #     'values_band3': [],
    # # #     'channelofint_band3': [244, 245, 247, 254],  # [117, 118, 120, 127] actual channels of interest subject 1
    # # #     'foilow_band3': 7,
    # # #     'foihigh_band3': 11,
    # # #     'plotwv_band3': 'experiment',
    # # #     'plotidx_band3': 'experiment',
    # # # }
    # #
    # # function_dict['flow'] = {
    # #             'values_Intra 1': [],
    # #             'values_Intra 2': [],
    # #             'values_Inter': []
    # #         }
    # #

    # if __name__ == '__main__':
    # if not print_output:
    #     sys.stdout = open(os.devnull, 'w')
    TF = RHYTHME(savepath, dataport, ex_windowsize, sub_windowsize, delay, plotpref, saving, TrigChan, )

    TF.master_control(filepath=filepath,
                      badchans=badchans,
                      fsample=fsample,
                      nchansparticipant=nchansparticipant,
                      trigchan=TrigChan,
                      blocksize_sec=blocksize_sec,
                      num_plvsimevents=num_plvsimevents,
                      option=option,
                      delay=delay,
                      units=units,
                      remfirstsample=removefirstsample,
                      exp_name=exp_name,
                      ignore_overflow=ignore_overflow,
                      n_skipped_segs=n_skipped_segs,
                      wait_segs=wait_segs,
                      function_dict=function_dict,
                      numparticipants=numparticipants,
                      ex_plot_matrix=np.zeros(ex_plot_dims),
                      sub_plot_matrix=np.zeros(sub_plot_dims),
                      channelnames=channelnames,
                      start_zero=start_zero,
                      resample_freq=resample_freq,
                      filter_data_range=filter_data_range),
    sys.stdout = sys.__stdout__