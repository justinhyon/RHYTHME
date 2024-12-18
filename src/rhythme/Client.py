#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  8 10:53:52 2020

@author: justin hyon, jessica ye
"""
try:
    from .psd import psd, psd_plot, psd_idx_plot
    from .plv import plv, plv_plot, plv_idx_plot
    from .ERP import ERP, ERP_plot, ERP_idx_plot
    from .teamflow import calculate_tmflow, teamflow_plot
    from .custom_function import custom_function, custom_plot_1, custom_plot_2
except ImportError:
    from psd import psd, psd_plot, psd_idx_plot
    from plv import plv, plv_plot, plv_idx_plot
    from ERP import ERP, ERP_plot, ERP_idx_plot
    from teamflow import calculate_tmflow, teamflow_plot
    from custom_function import custom_function, custom_plot_1, custom_plot_2
import sys
import struct
from mne_realtime.externals import FieldTrip
import time
import copy
import numpy as np
from statistics import mean
import mne
from mne.io import RawArray, read_raw_bdf, read_raw_eeglab, read_raw_gdf, read_raw
from mne.preprocessing import ICA
from mne_connectivity import spectral_connectivity_epochs
from mne.time_frequency import psd_array_welch, psd_array_multitaper
import matplotlib.pyplot as plt
from timeit import default_timer as timer
# from multiprocessing import Process
from collections import OrderedDict
import pandas as pd
from sklearn import preprocessing
from os import path
from scipy.signal import welch


class RHYTHME:
    def __init__(self, savepath, dataport, ex_windowsize, sub_windowsize, delay, plotpref, show_plots, saving, TrigChan, ):
        # user defined attributes
        # self.path = path
        self.savepath = savepath
        self.dataport = dataport
        self.exwindowsize = ex_windowsize
        self.subwindowsize = sub_windowsize
        self.delay = delay
        self.plotpref = plotpref
        self.show_plots = show_plots
        self.saving = saving
        self.TrigChan = TrigChan

        # not changed by user
        self.intra1finals = []
        self.intra2finals = []
        self.interfinals = []
        self.Events = []
        self.Header = []
        self.Data = []

        self.segment = 1  # segment counter

        # plt.ion()

    def master_control(self, filepath, badchans, fsample, nchansparticipant, trigchan, blocksize_sec, num_plvsimevents,
                       option, delay, units,
                       remfirstsample, exp_name, ignore_overflow, n_skipped_segs, wait_segs, function_dict, numparticipants,
                       ex_plot_matrix, sub_plot_matrix, channelnames, start_zero, resample_freq, filter_data_range):
        self.option = option

        if self.option == 'offline':
            self.offline_process(filepath, badchans, fsample, nchansparticipant, trigchan, blocksize_sec,
                                 num_plvsimevents, units, remfirstsample,
                                 function_dict, numparticipants, channelnames, ex_plot_matrix,
                                 sub_plot_matrix, resample_freq, filter_data_range)
        elif self.option == "realtime":
            self.realtime_process(delay, badchans, nchansparticipant, trigchan, blocksize_sec, num_plvsimevents, units,
                                  remfirstsample,
                                  exp_name, ignore_overflow, n_skipped_segs, wait_segs, function_dict, ex_plot_matrix,
                                  sub_plot_matrix, numparticipants, channelnames, start_zero, resample_freq, filter_data_range)
        else:
            print('invalid setting option selected')
            sys.exit(1)

    def realtime_process(self, delay, badchans, nchansparticipant, trigchan,blocksize_sec, num_plvsimevents, units,
                         remfirstsample, exp_name, ignore_overflow,
                         n_skipped_segs, wait_segs, function_dict, ex_plot_matrix,
                         sub_plot_matrix, numparticipants, channelnames, start_zero, resample_freq, filter_data_range):

        '''
        numparticipants         number of participants
        channelofints           list of lists containing channels corresponding to different regions of the brain
                                for which psd is calculated
        foilows                 list
        foihighs                list
        ERP_data                list of lists [[participant 1 info], [participant 2 info],...]
        '''
        self.ftc = FieldTrip.Client()

        for n, keyname in enumerate([i for i in list(function_dict.keys()) if 'ERP' in i]):
            # function_dict[keyname]['values_ERP'] = []
            function_dict[keyname]['exB_ERPlist'] = []
            function_dict[keyname]['exE_ERPlist'] = []
            function_dict[keyname]['ERPxvallist'] = []
            # values_ERP = [[] for i in range(len(ERP_data))]
            # exB_ERPlist = [[] for i in range(len(ERP_data))]
            # exE_ERPlist = [[] for i in range(len(ERP_data))]
            # ERPxvallist = [[] for i in range(len(ERP_data))]
        # intra1_tm = []
        # intra2_tm = []
        # inter_tm = []

        # data_dict = OrderedDict()
        # data_dict['Intra 1'] = []
        # data_dict['Intra 2'] = []
        # data_dict['Inter'] = []

        if self.plotpref != 'none':
            fig, ax = self.setup_plotting(ex_plot_matrix, sub_plot_matrix)

        print("REALTIME ANALYSIS MODE")
        # print("Waiting for first data segment from port: ", self.dataport)
        loop = True
        initial_wait = True

        if start_zero:
            prevsamp = 0
            print("Waiting for first data segment from port: ", self.dataport)
        else:
            self.ftc.connect('localhost', self.dataport)  # might throw IOError
            Head = self.ftc.getHeader()
            prevsamp = Head.nSamples
            print("Current sample in buffer: {0}. Waiting for new segment from port: {1}".format(prevsamp, self.dataport))

        while True:
            wait_time = 0
            while True:
                self.ftc.connect('localhost', self.dataport)  # might throw IOError
                H = self.ftc.getHeader()
                # print(H.nSamples)
                fsample = H.fSample
                blocksize = round(blocksize_sec * fsample)
                # print(blocksize)
                # print(prevsamp)
                # print(H.nSamp/les)
                currentsamp = H.nSamples

                # segmentdata1 = self.ftc.getData(
                #     index=(prevsamp, currentsamp - 1))
                #
                # # print(segmentdata1)
                # print(type(segmentdata1))
                #
                # print(segmentdata1.copy())

                if H.nSamples == prevsamp + blocksize:
                    print("normal")
                    initial_wait = False
                    break
                elif H.nSamples > prevsamp + blocksize and ignore_overflow:
                    initial_wait = False

                    currentsamp = prevsamp + blocksize
                    print("Skipped Data: ", (H.nSamples - currentsamp))
                    break
                elif (currentsamp - 1 - prevsamp) <= (blocksize * (n_skipped_segs + 1)) and \
                        (H.nSamples - prevsamp) / blocksize in range(1, n_skipped_segs + 2):
                    print('\nCAUTION: SEGMENT SKIPPED')
                    prevsamp = prevsamp + blocksize
                    break
                else:
                    time.sleep(delay)
                    if not initial_wait:
                        wait_time += 1
                    if wait_time > wait_segs * blocksize_sec * (1 / delay):
                        loop = False
                        print("\n", "-" * 60)
                        print("\nEXPERIMENT ENDED: After {0} seconds, not enough data was received to fill one block. "
                              "Number of unprocessed samples: {1}"
                              .format(blocksize_sec * wait_segs, H.nSamples - prevsamp))
                        break

            # Check presence of header
            # if H is None:
            #     print('Failed to retrieve header!')
            #     sys.exit(1)
            # # Check presence of channels
            # if H.nChannels == 0:
            #     print('no channels were selected')
            #     sys.exit(1)

            # time.sleep(self.delay)
            time1 = timer()
            time3 = timer()

            # if currentsamp != prevsamp + blocksize:
            #     print('received inconsistent number of samples: {}. experiment ending'.format(currentsamp - prevsamp))
            #     loop = False

            if loop:
                print("\n", "-" * 60)
                print('\nCONNECTED: Header retrieved for segment: ' + str(self.segment))
                print("Trying to read from sample {0} to {1}. Samples per block: {2}".format(prevsamp,
                                                                                             currentsamp - 1,
                                                                                             blocksize))
                print("Total Samples Received = {}".format(currentsamp))

                segmentdata = self.ftc.getData(
                    index=(prevsamp, currentsamp - 1)).transpose()  # receive data from buffer
                print('Data shape (nChannels, nSamples): {}'.format(segmentdata.shape))

                ev = self.ftc.getEvents()#index=(prevsamp, currentsamp - 1))
                # print("EVENTS: ", ev)
                for this_ev in ev:
                    print(this_ev.sample)
                # print("Last Event: ", ev_last)

                # if trigchan == 'create':
                #
                #     gen_stim_channel = self.create_stim_channel(ev, prevsamp, currentsamp)

                

                print('Samples retrieved for segment: ' + str(self.segment))
                # print(segmentdata)

                if type(trigchan) == int:
                    try:
                        stim = segmentdata[int(trigchan), :]
                    except IndexError:
                        print("FATAL: trigger/stimulus channel index > number of channels. Check parameter TrigChan")
                        exit(1)
                    minval = np.amin(stim)
                    if minval != 0:
                        stimvals = stim - minval  # correct values of the stim channel (stim is always last channel)
                    else:
                        stimvals = stim

                    segmentdata = np.delete(segmentdata, int(trigchan), axis=0)
                    self.stim_idx = str(self.TrigChan)
                    print(stimvals)
                elif trigchan == 'none' or trigchan == 'na':
                    stimvals = np.zeros(segmentdata.shape[1])
                    # trigchan = nchansparticipant * numparticipants
                    # print(segmentdata.shape)
                    np.append(segmentdata,stimvals)
                    self.stim_idx = str(len(segmentdata))
                    segmentdata = segmentdata.copy()
                elif trigchan == 'create':
                    stimvals = self.create_stim_channel(ev, prevsamp, currentsamp)
                    np.append(segmentdata, stimvals)
                    self.stim_idx = str(len(segmentdata))
                    segmentdata = segmentdata.copy()
                else:
                    "INVALID OPTION FOR TrigChan"
                    exit(1)
                print('Data shape (nChannels, nSamples): {}'.format(segmentdata.shape))

                prevsamp = currentsamp
                participant_data = []
                idx = 0
                while idx < segmentdata.shape[0]:
                    participant_data.append(segmentdata[int(idx):idx + nchansparticipant, :])
                    idx += nchansparticipant

                # D1 = fulldata[0:128, prevsamp:currentsamp]
                # D2 = fulldata[128:256, prevsamp:currentsamp]

                participant_raws = []
                for i, participant in enumerate(participant_data):
                    if i < numparticipants:
                        # print(channelnames)
                        print(i)
                        # print(participant)
                        try:
                            raw = self.make_raw(participant, stimvals, fsample, units, channelnames[i])
                        except IndexError:
                            print("FATAL: fewer channel names lists than number of participants. Check channelnames, "
                                  "numparticipants, nchansparticipant")
                            exit(1)
                        raw = self.preprocess_raw(raw, badchans[i], filter_data_range)
                        participant_raws.append(raw)
                        print('Sub Data shape (nChannels, nSamples): {0} for participant {1}'.format(participant.shape,
                                                                                                     i + 1))
                        del raw  # Delete raw variable so it does not interfere with next loop

                # Change channel names so that we can now append the two subjects together
                for i, participantraw in enumerate(participant_raws):
                    chnames = {}
                    for n, name in enumerate(participantraw.ch_names):
                        chnames.setdefault(name, str(n + i * nchansparticipant))
                    participantraw.rename_channels(chnames)
                    del chnames

                raw = participant_raws[0].copy()
                idx = 1
                while idx < len(participant_raws):
                    raw = raw.add_channels([participant_raws[idx]], force_update_info=True)
                    idx += 1

                # Preprocessing got rid of the stim channel, so now we add the stim channel back in
                print(stimvals)
                print(raw._data[-1])
                info_stim = mne.create_info(['stim'], sfreq=fsample, ch_types=['stim'])
                raw_stim = RawArray(np.asarray(stimvals).reshape(1, len(stimvals)), info_stim)
                raw = raw.add_channels([raw_stim], force_update_info=True)

                if fsample != resample_freq:
                    raw = raw.resample(resample_freq)
                    print("RESAMPLED DATA SHAPE", raw.get_data().shape)

                    fsample = resample_freq
                    blocksize = round(blocksize_sec * fsample)
                    stimvals = raw.get_data()[int(self.stim_idx)]

                time2 = timer()
                print("Time to recieve data and create MNE raw: {}".format(time2 - time1))
                # print("\n", "1-" * 60)

                self.stim_values = stimvals
                print('STIM CHANNEL: ', self.stim_idx, len(stimvals))
                # for i in stimvals:
                #     if i != 0:
                #         print("TRIGGER")

                ex_plot_matrix.fill(0)
                sub_plot_matrix.fill(0)
                # Extract features
                ################## PSDs #################################################
                time1 = timer()
                for subject, keyname in enumerate([i for i in list(function_dict.keys()) if 'psd' in i]):
                    for n, psdkey in enumerate([j for j in list(function_dict[keyname].keys()) if 'values' in j]):
                        # print(psdkey)

                        # psdnames=['band1', 'band2']  #placeholder, remove
                        r = str(n + 1)
                        # i = np.mod(idx, len(
                        #     foilows))  # Adjust for the fact that channels are different amongst subjects, but fois are the same
                        psds, this_psd_spec, this_freqs = psd(raw,
                                                              function_dict[keyname]['values_band' + r],
                                                              function_dict[keyname]['channelofint_band' + r],
                                                              function_dict[keyname]['foilow_band' + r],
                                                              function_dict[keyname]['foihigh_band' + r],
                                                              fsample)
                        if self.plotpref != 'none':
                            print('plot mat', ex_plot_matrix)
                            plot_settings = self.plot_settings(function_dict[keyname]['plotwv_band' + r],
                                                               ex_plot_matrix, sub_plot_matrix)
                            if plot_settings:
                                print('PLOTSETTINGS', plot_settings)
                                whichax, loc, ex_plot_matrix, sub_plot_matrix = plot_settings
                                ax[whichax] = psd_plot(ax[whichax], this_psd_spec, psds, this_freqs, n + 1, loc,
                                                       function_dict[keyname]['foilow_band' + r],
                                                       function_dict[keyname]['foihigh_band' + r],
                                                       subject+1)

                            plot_settings = self.plot_settings(function_dict[keyname]['plotidx_band' + r],
                                                               ex_plot_matrix,
                                                               sub_plot_matrix)
                            if plot_settings:
                                whichax, loc, ex_plot_matrix, sub_plot_matrix = plot_settings
                                ax[whichax] = psd_idx_plot(ax[whichax], psds, n + 1, loc, subject+1)
                        # if self.plotpref == 'both' or self.plotpref == 'experiment':
                        #     band = n + 1
                        #     if band % 2 == 0:
                        #         col = 2
                        #     else:
                        #         col = 0
                        #
                        #     if idx < 2:
                        #         row = 2
                        #     elif 2 <= idx <= 3:
                        #         row = 3
                        #     else:
                        #         print('too many PSD bands, band {} not plotted'.format(band))
                        #
                        #     ax[0] = psd_plot(ax[0], this_psd_spec, psds, this_freqs, band, row, col,
                        #                   function_dict[keyname]['foilow_band' + r],
                        #                   function_dict[keyname]['foihigh_band' + r])

                        function_dict[keyname]['values_band' + r] = psds

                time2 = timer()
                print("Time to compute 6 PSDs: {}".format(time2 - time1))
                print("\n", "2-" * 60)

                ############################################################################

                ########################### values_ERP ##########################################
                time1 = timer()
                Lvalues_ERP, LERPxvallist, LexB_ERPlist, LexE_ERPlist, LsegmentERPdata, LERP_peak_locs = \
                        [], [], [], [], [], []
                for n, keyname in enumerate([i for i in list(function_dict.keys()) if 'ERP' in i]):

                    values_ERP, ERPxvallist, exB_ERPlist, exE_ERPlist, segmentERPdata, ERP_peak_locs = \
                        ERP(raw=raw,
                            ERPlist=function_dict[keyname]['values_ERP'],
                            exB_ERPlist=function_dict[keyname]['exB_ERPlist'],
                            exE_ERPlist=function_dict[keyname]['exE_ERPlist'],
                            ERPxvallist=function_dict[keyname]['ERPxvallist'],
                            fsample=fsample,
                            blocksize=blocksize_res,
                            channelofint=function_dict[keyname]['channelofint'],
                            epoeventval=function_dict[keyname]['epoeventval'],
                            pretrig=function_dict[keyname]['pretrig'],
                            posttrig=function_dict[keyname]['posttrig'],
                            stim_values=stimvals,
                            segment=self.segment,
                            bands=function_dict[keyname]['bands'],
                            signs=function_dict[keyname]['signs'],
                            filter_range=function_dict[keyname]['filter_range'],
                            calc_met=function_dict[keyname]['calc_method'],
                            mean=function_dict[keyname]['mean'])
                    Lvalues_ERP.append(values_ERP)
                    LERPxvallist.append(ERPxvallist)
                    LexB_ERPlist.append(exB_ERPlist)
                    LexE_ERPlist.append(exE_ERPlist)
                    LsegmentERPdata.append(segmentERPdata)
                    LERP_peak_locs.append(ERP_peak_locs)
                for n, keyname in enumerate([i for i in list(function_dict.keys()) if 'ERP' in i]):
                    values_ERP, ERPxvallist, exB_ERPlist, exE_ERPlist, segmentERPdata, ERP_peak_locs = \
                        Lvalues_ERP[n], LERPxvallist[n], LexB_ERPlist[n], LexE_ERPlist[n], LsegmentERPdata[n],\
                        LERP_peak_locs[n]

                    if self.plotpref != 'none':
                        plot_settings = self.plot_settings(function_dict[keyname]['plotwv'], ex_plot_matrix,
                                                           sub_plot_matrix)
                        if plot_settings:
                            # print('PLOTSETTINGS', plot_settings)

                            whichax, loc, ex_plot_matrix, sub_plot_matrix = plot_settings
                            ax[whichax] = ERP_plot(ax=ax[whichax], data=segmentERPdata,
                                                   participant=int(keyname[-1]) - 1,
                                                   fsample=fsample,
                                                   ERPlist=function_dict[keyname]['values_ERP'],
                                                   segment=self.segment,
                                                   pretrig=function_dict[keyname]['pretrig'],
                                                   posttrig=function_dict[keyname]['posttrig'],
                                                   location=loc,
                                                   bands=function_dict[keyname]['bands'],
                                                   calc_met=function_dict[keyname]['calc_method'],
                                                   mean=function_dict[keyname]['mean'],
                                                   max_val = np.max(LsegmentERPdata)
                                                   )
                        plot_settings = self.plot_settings(function_dict[keyname]['plotidx'], ex_plot_matrix,
                                                           sub_plot_matrix)
                        if plot_settings:
                            whichax, loc, ex_plot_matrix, sub_plot_matrix = plot_settings
                            ax[whichax] = ERP_idx_plot(ax=ax[whichax],
                                                       participant=int(keyname[-1]) - 1,
                                                       ERPlist=values_ERP,
                                                       ERPxvallist=ERPxvallist, exB_ERPlist=exB_ERPlist,
                                                       exE_ERPlist=exE_ERPlist,
                                                       segment=self.segment, location=loc
                                                       )

                    function_dict[keyname]['values_ERP'] = values_ERP
                    function_dict[keyname]['ERPxvallist'] = ERPxvallist
                    function_dict[keyname]['exB_ERPlist'] = exB_ERPlist
                    function_dict[keyname]['exE_ERPlist'] = exE_ERPlist

                time2 = timer()
                print("Time to compute 2 values_ERP: {}".format(time2 - time1))
                print("\n", "3-" * 60)

                ##################################PLVs#####################################
                time1 = timer()

                # nparticipants = shape(raw.get_data())[0] - 1 / nchansparticipant
                # numblocks = 10
                for n, keyname in enumerate([i for i in list(function_dict.keys()) if 'plv' in i]):
                    intrameans, intermean, con = plv(raw, self.segment, blocksize_res, fsample, num_plvsimevents,
                                                     numparticipants, function_dict[keyname]['frqofint'],
                                                     nchansparticipant)
                    print('intermean ', intermean)
                    print(len(intrameans))
                    for i, intramean in enumerate(intrameans):
                        function_dict[keyname]['values_intra' + str(i + 1)] += [intramean]
                    # function_dict[keyname]['values_intra2'] += [intra2]
                    function_dict[keyname]['values_inter'] += [intermean]

                    # if self.plotpref == 'both' or self.plotpref == 'experiment':
                    #     ax[0] = plv_plot(ax[0], con, plv_intra1, plv_inter, plv_intra2)

                    if self.plotpref != 'none':

                        plot_settings = self.plot_settings(function_dict[keyname]['plot_matrix'], ex_plot_matrix,
                                                           sub_plot_matrix)

                        if plot_settings:
                            # print('PLOTSETTINGS', plot_settings)
                            whichax, loc, ex_plot_matrix, sub_plot_matrix = plot_settings
                            ax[whichax] = plv_plot(ax[whichax], con, loc, numparticipants, nchansparticipant)

                        if type(function_dict[keyname]['plot_option']) == tuple:
                            for r, indexkeyname in enumerate(
                                    [i for i in list(function_dict[keyname].keys()) if 'plot' in i]):
                                # if numparticipants != 2:
                                #     if n > 1:
                                #         break
                                if r in function_dict[keyname]['plot_option'] or r == 0:
                                    plot_settings = self.plot_settings(function_dict[keyname][indexkeyname],
                                                                       ex_plot_matrix,
                                                                       sub_plot_matrix)

                                    if plot_settings:
                                        nameslice = indexkeyname[indexkeyname.index('_') + 1:]
                                        # print('PLOTSETTINGS', plot_settings)
                                        whichax, loc, ex_plot_matrix, sub_plot_matrix = plot_settings

                                        ax[whichax] = plv_idx_plot(ax[whichax],
                                                                   function_dict[keyname]['values_' + nameslice], loc,
                                                                   nameslice)
                        elif type(function_dict[keyname]['plot_option']) == str:
                            allmean = []
                            for i, intramean in enumerate(intrameans):
                                allmean.append(function_dict[keyname]['values_intra' + str(i + 1)])

                            if numparticipants > 1 and function_dict[keyname]['intra_avg']:
                                allmean = zip(*allmean)
                                allmean = list(allmean)
                                print(1, allmean)
                                for i, vals in enumerate(allmean):
                                    print(2, list(vals))
                                    allmean[i] = np.mean(list(vals))
                                print(3,allmean)

                            for i in range(len(allmean)):
                                plot_settings = self.plot_settings(function_dict[keyname]['plot_intra' + str(i + 1)], ex_plot_matrix,
                                                                   sub_plot_matrix)

                                if plot_settings:
                                    # nameslice = keyname[keyname.index('_') + 1:]
                                    # print('PLOTSETTINGS', plot_settings)
                                    whichax, loc, ex_plot_matrix, sub_plot_matrix = plot_settings
                                    if len(allmean) > 1:
                                        name = f'Subject {i+1} PLV mean index'
                                    else:
                                        name = "Avg Intra-brain PLV index"
                                    ax[whichax] = plv_idx_plot(ax[whichax], allmean[i],
                                                               loc, name)

                            plot_settings = self.plot_settings(function_dict[keyname]['plot_inter'], ex_plot_matrix,
                                                               sub_plot_matrix)

                            if plot_settings:
                                # nameslice = keyname[keyname.index('_') + 1:]
                                # print('PLOTSETTINGS', plot_settings)
                                whichax, loc, ex_plot_matrix, sub_plot_matrix = plot_settings
                                name = 'Inter-brain PLV mean index'
                                ax[whichax] = plv_idx_plot(ax[whichax], function_dict[keyname]['values_inter'],
                                                           loc, name)

                time2 = timer()
                print("Time to compute PLV: {}".format(time2 - time1))
                print("\n", "4-" * 60)

                ############################################################################
                ###############custom function###################
                time1 = timer()
                for subject, keyname in enumerate([i for i in list(function_dict.keys()) if 'custom' in i]):
                    print(list(function_dict.keys()))
                    print([i for i in list(function_dict.keys()) if 'custom' in i])

                    # psdnames=['band1', 'band2']  #placeholder, remove
                    r = str(subject)
                    # i = np.mod(idx, len(
                    #     foilows))  # Adjust for the fact that channels are different amongst subjects, but fois are the same
                    function_dict = custom_function(raw,
                                                    r,
                                                  function_dict,
                                                  fsample)
                    if self.plotpref != 'none':

                        plot_settings = self.plot_settings(function_dict[keyname]['custom_plot_1'],
                                                           ex_plot_matrix, sub_plot_matrix)
                        if plot_settings:
                            print('PLOTSETTINGS', plot_settings)
                            whichax, loc, ex_plot_matrix, sub_plot_matrix = plot_settings
                            ax[whichax] = custom_plot_1(ax[whichax], loc, subject + 1,
                                                   function_dict)

                        plot_settings = self.plot_settings(function_dict[keyname]['custom_plot_2'],
                                                           ex_plot_matrix, sub_plot_matrix)
                        if plot_settings:
                            whichax, loc, ex_plot_matrix, sub_plot_matrix = plot_settings
                            ax[whichax] = custom_plot_1(ax[whichax], n + 1, loc, subject + 1,
                                                        function_dict)
                    # if self.plotpref == 'both' or self.plotpref == 'experiment':
                    #     band = n + 1
                    #     if band % 2 == 0:
                    #         col = 2
                    #     else:
                    #         col = 0
                    #
                    #     if idx < 2:
                    #         row = 2
                    #     elif 2 <= idx <= 3:
                    #         row = 3
                    #     else:
                    #         print('too many PSD bands, band {} not plotted'.format(band))
                    #
                    #     ax[0] = psd_plot(ax[0], this_psd_spec, psds, this_freqs, band, row, col,
                    #                   function_dict[keyname]['foilow_band' + r],
                    #                   function_dict[keyname]['foihigh_band' + r])

                    function_dict[keyname]['values_band' + r] = psds

                time2 = timer()
                print("Time to compute 6 PSDs: {}".format(time2 - time1))
                print("\n", "2-" * 60)
                # intra1_tm, intra2_tm, inter_tm = calculate_tmflow(data_dict=function_dict)
                #
                # function_dict['flow']['values_Intra 1'] = (intra1_tm)
                # function_dict['flow']['values_Intra 2'] = (intra2_tm)
                # function_dict['flow']['values_Inter'] = (inter_tm)
                #
                # if (self.segment == 2) & remfirstsample:
                #     for key, value in function_dict['flow'].items():
                #         if key in ['values_Intra 1', 'values_Intra 2', 'values_Inter']:
                #             valuecopy = value
                #             valuecopy[0] = 0.0
                #             function_dict['flow'][key] = valuecopy
                # print(function_dict.values())

                ############################################################################
                if self.plotpref == 'both' or self.plotpref == 'participant':
                    ax[1] = teamflow_plot(ax[1], function_dict)

                if self.plotpref != 'none' and self.show_plots:

                    for fi in fig:
                        if fi is not None:
                            fi.tight_layout()
                    plt.pause(0.005)
                    plt.show()

                # self.final()
                # if self.plotpref == 'participant':
                #     plt.close(fig=self.fig)
                #
                # if self.plotpref != 'none':
                #     plt.pause(.005)
                #     plt.draw()

                if self.saving and self.plotpref != 'none':
                    for n, fi in enumerate(fig):
                        if fi is not None:
                            figsavepath = self.savepath + '/figures/plot_' + str(self.segment) + '_' + str(n) + '.jpg'
                            fi.savefig(figsavepath)

                time2 = timer()
                print("Time to generate plots: {}".format(time2 - time1))
                print("\n", "5-" * 60)

                time4 = timer()
                print("Time ALL: {}".format(time4 - time3))
                print("All operations complete for segment {}".format(self.segment))

                self.segment += 1
            else:
                self.save_csv(function_dict, exp_name)
                break

    def offline_process(self, filepath, badchans, fsample_orig, nchansparticipant, trigchan, blocksize_sec,
                        num_plvsimevents, units, remfirstsample,
                        function_dict, numparticipants, channelnames, ex_plot_matrix, sub_plot_matrix, resample_freq,
                        filter_data_range):

        '''
        numparticipants         number of participants
        channelofints           list of lists containing channels corresponding to different regions of the brain
                                for which psd is calculated
        foilows                 list
        foihighs                list
        ERP_data                list of lists [[participant 1 info], [participant 2 info],...]
        '''
        for n, keyname in enumerate([i for i in list(function_dict.keys()) if 'ERP' in i]):
            # function_dict[keyname]['values_ERP'] = []
            function_dict[keyname]['exB_ERPlist'] = []
            function_dict[keyname]['exE_ERPlist'] = []
            function_dict[keyname]['ERPxvallist'] = []
        if self.plotpref != 'none':
            fig, ax = self.setup_plotting(ex_plot_matrix, sub_plot_matrix)


        if filepath.endswith('.bdf'):
            fullraw = read_raw_bdf(filepath, preload=True)
        elif filepath.endswith('.set'):
            fullraw = read_raw_eeglab(filepath, preload=True)
        elif filepath.endswith('.gdf'):
            fullraw = read_raw_gdf(filepath, preload=True)
        elif filepath.endswith('.fif'):
            fullraw = read_raw(filepath, preload=True)
        else:
            print('Unsupported EEG file format')
            exit(1)

        fulldata = fullraw._data
        if type(trigchan) == int:
            fulldatanostim = np.delete(fulldata, int(trigchan), axis=0)
        elif trigchan == 'none':
            fulldatanostim = fulldata
        else:
            print('For offline analysis, TrigChan must be either an integer representing a channel, or a '
                  'string "none"')
            exit(1)

        blocksize = round(blocksize_sec * fsample_orig)
        self.filepath = filepath
        prevsamp = 0
        print("OFFLINE ANALYSIS MODE")
        print("Total Samples Loaded = {}".format(fulldata.shape))

        # Initiate list of lists to store psd values for one participant
        # psds = [[] for i in range(len(channelofints))]
        # plv_intra1 = []
        # plv_inter = []
        # plv_intra2 = []
        #
        # values_ERP = [[] for i in range(len(ERP_data))]
        # exB_ERPlist = [[] for i in range(len(ERP_data))]
        # exE_ERPlist = [[] for i in range(len(ERP_data))]
        # ERPxvallist = [[] for i in range(len(ERP_data))]
        # intra1_tm = []
        # intra2_tm = []
        # inter_tm = []

        # data_dict = OrderedDict()
        # data_dict['Intra 1'] = []
        # data_dict['Intra 2'] = []
        # data_dict['Inter'] = []

        # For plotting
        # if self.plotpref != 'none':
        #     if self.plotpref == 'participant' or self.plotpref == 'both':
        #         # plt.close(self.fig)
        #         fig, ax = plt.subplots(1, 3, squeeze=False,
        #                                figsize=(self.windowsize * 3, self.windowsize))

        while True:
            # time.sleep(self.delay)
            time1 = timer()
            time3 = timer()
            currentsamp = prevsamp + blocksize
            if currentsamp < fulldata.shape[1]:
                print("\n", "-" * 60)
                print('\nProcessing data from segment: ' + str(self.segment))
                print("Trying to read from sample {0} to {1}. Samples per block: {2}".format(prevsamp, currentsamp,
                                                                                             blocksize))

                ex_plot_matrix[ex_plot_matrix == 1.] = 0.
                sub_plot_matrix[sub_plot_matrix == 1.] = 0.

                segmentdata = fulldata[:, prevsamp:currentsamp]
                # D1 = fulldata[0:128, prevsamp:currentsamp]
                # D2 = fulldata[128:256, prevsamp:currentsamp]
                if type(trigchan) == int:
                    stim = segmentdata[int(trigchan), :]
                    minval = np.amin(stim)
                    if minval != 0:
                        stimvals = stim - minval  # correct values of the stim channel (stim is always last channel)
                    else:
                        stimvals = stim

                    segmentdata = np.delete(segmentdata, int(trigchan), axis=0)
                    self.stim_idx = str(self.TrigChan)

                    if any(stimvals):
                        stimvals = stimvals / 1000000
                        if units == "mv":
                            stimvals = stimvals * 1000
                        elif units == "uv":
                            stimvals = stimvals * 1000000000000
                        elif units == "nv":
                            stimvals = stimvals * 1000000000000
                        elif units == "Mv":
                            stimvals = stimvals / 1000
                        elif units != "v":
                            print("invalid units selected")
                            sys.exit(1)
                    for i in stimvals:
                        if i!=0:
                            print('event ', i)
                elif trigchan == 'none':
                    stimvals = np.zeros(segmentdata.shape[1])
                    # trigchan = nchansparticipant * numparticipants
                    # print(segmentdata.shape)
                    np.append(segmentdata, stimvals)
                    self.stim_idx = str(len(segmentdata))
                    segmentdata = segmentdata.copy()
                else:
                    "INVALID OPTION FOR TrigChan"
                    exit(1)
                print('Data shape (nChannels, nSamples): {}'.format(segmentdata.shape))
                participant_data = []
                idx = 0
                while idx < segmentdata.shape[0]:
                    participant_data.append(segmentdata[int(idx):idx + nchansparticipant, :])
                    idx += nchansparticipant
                prevsamp = currentsamp
                # print(D)
                print('Samples loaded for segment: ' + str(self.segment))

                participant_raws = []
                for i, participant in enumerate(participant_data):
                    if i < numparticipants:
                        # print(channelnames)
                        print(i)
                        # print(participant)
                        try:
                            raw = self.make_raw(participant, stimvals, fsample_orig, units, channelnames[i])
                        except IndexError:
                            print("FATAL: fewer channel names lists than number of participants. Check channelnames, "
                                  "numparticipants, nchansparticipant")
                            exit(1)
                        raw = self.preprocess_raw(raw, badchans[i], filter_data_range)
                        participant_raws.append(raw)
                        print('Sub Data shape (nChannels, nSamples): {0} for participant {1}'.format(participant.shape,
                                                                                                     i + 1))
                        del raw  # Delete raw variable so it does not interfere with next loop

                # Change channel names so that we can now append the two subjects together
                for i, participant in enumerate(participant_raws):
                    chnames = {}
                    for n, name in enumerate(participant.ch_names):
                        chnames.setdefault(name, str(n + i * nchansparticipant))
                    participant.rename_channels(chnames)

                    del chnames

                raw = participant_raws[0].copy()
                idx = 1
                while idx < len(participant_raws):
                    raw = raw.add_channels([participant_raws[idx]], force_update_info=True)
                    idx += 1
                # print(raw._channel_names)
                # Preprocessing got rid of the stim channel, so now we add the stim channel back in
                info_stim = mne.create_info(['stim'], sfreq=fsample_orig, ch_types=['stim'])
                raw_stim = RawArray(np.asarray(stimvals).reshape(1, len(stimvals)), info_stim)
                raw = raw.add_channels([raw_stim], force_update_info=True)

                if fsample_orig != resample_freq:
                    raw = raw.resample(resample_freq)
                    print("RESAMPLED DATA SHAPE", raw.get_data().shape)

                    fsample = resample_freq
                    blocksize_res= round(blocksize_sec * fsample)
                    stimvals = raw.get_data()[int(self.stim_idx)]
                else:
                    fsample = fsample_orig
                    blocksize_res = blocksize
                time2 = timer()
                print("Time to preprocess data and create MNE raw: {}".format(time2 - time1))
                # print("\n", "1-" * 60)

                self.stim_idx = self.TrigChan
                self.stim_values = stimvals
                # print('STIM CHANNEL: ', self.stim_idx, self.stim_values)

                # Extract features
                ################## PSDs #################################################
                time1 = timer()

                for subject, keyname in enumerate([i for i in list(function_dict.keys()) if 'psd' in i]):
                    for n, psdkey in enumerate([j for j in list(function_dict[keyname].keys()) if 'values' in j]):
                        # print(psdkey)

                        # psdnames=['band1', 'band2']  #placeholder, remove
                        r = str(n + 1)
                        # i = np.mod(idx, len(
                        #     foilows))  # Adjust for the fact that channels are different amongst subjects, but fois are the same
                        psds, this_psd_spec, this_freqs = psd(raw,
                                                              function_dict[keyname]['values_band' + r],
                                                              function_dict[keyname]['channelofint_band' + r],
                                                              function_dict[keyname]['foilow_band' + r],
                                                              function_dict[keyname]['foihigh_band' + r],
                                                              fsample)
                        if self.plotpref != 'none':
                            print(ex_plot_matrix)
                            plot_settings = self.plot_settings(function_dict[keyname]['plotwv_band' + r],
                                                               ex_plot_matrix, sub_plot_matrix)
                            if plot_settings:
                                print('PLOTSETTINGS', plot_settings)
                                whichax, loc, ex_plot_matrix, sub_plot_matrix = plot_settings
                                ax[whichax] = psd_plot(ax[whichax], this_psd_spec, psds, this_freqs, n + 1, loc,
                                                       function_dict[keyname]['foilow_band' + r],
                                                       function_dict[keyname]['foihigh_band' + r],
                                                       subject + 1)

                            plot_settings = self.plot_settings(function_dict[keyname]['plotidx_band' + r],
                                                               ex_plot_matrix,
                                                               sub_plot_matrix)
                            if plot_settings:
                                whichax, loc, ex_plot_matrix, sub_plot_matrix = plot_settings
                                ax[whichax] = psd_idx_plot(ax[whichax], psds, n + 1, loc, subject + 1)
                        # if self.plotpref == 'both' or self.plotpref == 'experiment':
                        #     band = n + 1
                        #     if band % 2 == 0:
                        #         col = 2
                        #     else:
                        #         col = 0
                        #
                        #     if idx < 2:
                        #         row = 2
                        #     elif 2 <= idx <= 3:
                        #         row = 3
                        #     else:
                        #         print('too many PSD bands, band {} not plotted'.format(band))
                        #
                        #     ax[0] = psd_plot(ax[0], this_psd_spec, psds, this_freqs, band, row, col,
                        #                   function_dict[keyname]['foilow_band' + r],
                        #                   function_dict[keyname]['foihigh_band' + r])

                        function_dict[keyname]['values_band' + r] = psds

                time2 = timer()
                print("Time to compute 6 PSDs: {}".format(time2 - time1))
                print("\n", "2-" * 60)

                ############################################################################

                ########################### values_ERP ##########################################
                time1 = timer()
                Lvalues_ERP, LERPxvallist, LexB_ERPlist, LexE_ERPlist, LsegmentERPdata, LERP_peak_locs = \
                        [], [], [], [], [], []
                for n, keyname in enumerate([i for i in list(function_dict.keys()) if 'ERP' in i]):

                    values_ERP, ERPxvallist, exB_ERPlist, exE_ERPlist, segmentERPdata, ERP_peak_locs = \
                        ERP(raw=raw,
                            ERPlist=function_dict[keyname]['values_ERP'],
                            exB_ERPlist=function_dict[keyname]['exB_ERPlist'],
                            exE_ERPlist=function_dict[keyname]['exE_ERPlist'],
                            ERPxvallist=function_dict[keyname]['ERPxvallist'],
                            fsample=fsample,
                            blocksize=blocksize_res,
                            channelofint=function_dict[keyname]['channelofint'],
                            epoeventval=function_dict[keyname]['epoeventval'],
                            pretrig=function_dict[keyname]['pretrig'],
                            posttrig=function_dict[keyname]['posttrig'],
                            stim_values=stimvals,
                            segment=self.segment,
                            bands=function_dict[keyname]['bands'],
                            signs=function_dict[keyname]['signs'],
                            filter_range=function_dict[keyname]['filter_range'],
                            calc_met=function_dict[keyname]['calc_method'],
                            mean=function_dict[keyname]['mean'])
                    Lvalues_ERP.append(values_ERP)
                    LERPxvallist.append(ERPxvallist)
                    LexB_ERPlist.append(exB_ERPlist)
                    LexE_ERPlist.append(exE_ERPlist)
                    LsegmentERPdata.append(segmentERPdata)
                    LERP_peak_locs.append(ERP_peak_locs)
                for n, keyname in enumerate([i for i in list(function_dict.keys()) if 'ERP' in i]):
                    values_ERP, ERPxvallist, exB_ERPlist, exE_ERPlist, segmentERPdata, ERP_peak_locs = \
                        Lvalues_ERP[n], LERPxvallist[n], LexB_ERPlist[n], LexE_ERPlist[n], LsegmentERPdata[n],\
                        LERP_peak_locs[n]

                    if self.plotpref != 'none':
                        plot_settings = self.plot_settings(function_dict[keyname]['plotwv'], ex_plot_matrix,
                                                           sub_plot_matrix)
                        if plot_settings:
                            # print(len(LsegmentERPdata))

                            whichax, loc, ex_plot_matrix, sub_plot_matrix = plot_settings
                            ax[whichax], heatmap = ERP_plot(ax=ax[whichax], data=segmentERPdata,
                                                   participant=int(keyname[-1]) - 1,
                                                   fsample=fsample,
                                                   ERPlist=function_dict[keyname]['values_ERP'],
                                                   segment=self.segment,
                                                   pretrig=function_dict[keyname]['pretrig'],
                                                   posttrig=function_dict[keyname]['posttrig'],
                                                   location=loc,
                                                   bands=function_dict[keyname]['bands'],
                                                   calc_met=function_dict[keyname]['calc_method'],
                                                   mean=function_dict[keyname]['mean'],
                                                   max_val = np.max(LsegmentERPdata)
                                                   )
                        plot_settings = self.plot_settings(function_dict[keyname]['plotidx'], ex_plot_matrix,
                                                           sub_plot_matrix)
                        if plot_settings:
                            whichax, loc, ex_plot_matrix, sub_plot_matrix = plot_settings
                            ax[whichax] = ERP_idx_plot(ax=ax[whichax],
                                                       participant=int(keyname[-1]) - 1,
                                                       ERPlist=values_ERP,
                                                       ERPxvallist=ERPxvallist, exB_ERPlist=exB_ERPlist,
                                                       exE_ERPlist=exE_ERPlist,
                                                       segment=self.segment, location=loc
                                                       )
                        plot_settings = self.plot_settings(function_dict[keyname]['plot_heatmap_colorbar'], ex_plot_matrix,
                                                           sub_plot_matrix)
                        if plot_settings and heatmap:
                            whichax, loc, ex_plot_matrix, sub_plot_matrix = plot_settings
                            # print(type(loc))
                            plt.colorbar(heatmap, cax=ax[whichax][loc[0], loc[1]], label=units, orientation='horizontal')

                    function_dict[keyname]['values_ERP'] = values_ERP
                    function_dict[keyname]['ERPxvallist'] = ERPxvallist
                    function_dict[keyname]['exB_ERPlist'] = exB_ERPlist
                    function_dict[keyname]['exE_ERPlist'] = exE_ERPlist

                time2 = timer()
                print("Time to compute 2 values_ERP: {}".format(time2 - time1))
                print("\n", "3-" * 60)

                ##################################PLVs#####################################
                time1 = timer()

                # raw.save('plv_test_raw_eeg.fif', overwrite=True)
                # numblocks = 10
                for n, keyname in enumerate([i for i in list(function_dict.keys()) if 'plv' in i]):
                    intrameans, intermean, con = plv(raw, self.segment, blocksize_res, fsample, num_plvsimevents,
                                                     numparticipants, function_dict[keyname]['frqofint'],
                                                     nchansparticipant)
                    print('intermean ', intermean)
                    print(len(intrameans))
                    for i, intramean in enumerate(intrameans):
                        function_dict[keyname]['values_intra' + str(i + 1)] += [intramean]
                    # function_dict[keyname]['values_intra2'] += [intra2]
                    function_dict[keyname]['values_inter'] += [intermean]

                    # if self.plotpref == 'both' or self.plotpref == 'experiment':
                    #     ax[0] = plv_plot(ax[0], con, plv_intra1, plv_inter, plv_intra2)

                    if self.plotpref != 'none':

                        plot_settings = self.plot_settings(function_dict[keyname]['plot_matrix'], ex_plot_matrix,
                                                           sub_plot_matrix)

                        if plot_settings:
                            # print('PLOTSETTINGS', plot_settings)
                            whichax, loc, ex_plot_matrix, sub_plot_matrix = plot_settings
                            ax[whichax] = plv_plot(ax[whichax], con, loc, numparticipants, nchansparticipant)

                        if type(function_dict[keyname]['plot_option']) == tuple:
                            for r, indexkeyname in enumerate(
                                    [i for i in list(function_dict[keyname].keys()) if 'plot' in i]):
                                # if numparticipants != 2:
                                #     if n > 1:
                                #         break
                                if r in function_dict[keyname]['plot_option'] or r == 0:
                                    plot_settings = self.plot_settings(function_dict[keyname][indexkeyname],
                                                                       ex_plot_matrix,
                                                                       sub_plot_matrix)

                                    if plot_settings:
                                        nameslice = indexkeyname[indexkeyname.index('_') + 1:]
                                        # print('PLOTSETTINGS', plot_settings)
                                        whichax, loc, ex_plot_matrix, sub_plot_matrix = plot_settings

                                        ax[whichax] = plv_idx_plot(ax[whichax],
                                                                   function_dict[keyname]['values_' + nameslice], loc,
                                                                   nameslice)
                        elif type(function_dict[keyname]['plot_option']) == str:
                            allmean = []
                            for i, intramean in enumerate(intrameans):
                                allmean.append(function_dict[keyname]['values_intra' + str(i + 1)])

                            if numparticipants > 1 and function_dict[keyname]['intra_avg']:
                                allmean = zip(*allmean)
                                allmean = list(allmean)
                                print(1, allmean)
                                for i, vals in enumerate(allmean):
                                    print(2, list(vals))
                                    allmean[i] = np.mean(list(vals))
                                print(3,allmean)

                            for i in range(len(allmean)):
                                plot_settings = self.plot_settings(function_dict[keyname]['plot_intra' + str(i + 1)], ex_plot_matrix,
                                                                   sub_plot_matrix)

                                if plot_settings:
                                    # nameslice = keyname[keyname.index('_') + 1:]
                                    # print('PLOTSETTINGS', plot_settings)
                                    whichax, loc, ex_plot_matrix, sub_plot_matrix = plot_settings
                                    if len(allmean) > 1:
                                        name = f'Subject {i+1} PLV mean index'
                                    else:
                                        name = "Avg Intra-brain PLV index"
                                    ax[whichax] = plv_idx_plot(ax[whichax], allmean[i],
                                                               loc, name)

                            plot_settings = self.plot_settings(function_dict[keyname]['plot_inter'], ex_plot_matrix,
                                                               sub_plot_matrix)

                            if plot_settings:
                                # nameslice = keyname[keyname.index('_') + 1:]
                                # print('PLOTSETTINGS', plot_settings)
                                whichax, loc, ex_plot_matrix, sub_plot_matrix = plot_settings
                                name = 'Inter-brain PLV mean index'
                                ax[whichax] = plv_idx_plot(ax[whichax], function_dict[keyname]['values_inter'],
                                                           loc, name)

                time2 = timer()
                print("Time to compute PLV: {}".format(time2 - time1))
                print("\n", "4-" * 60)

                ############################################################################

                ###############Make Datastructure to make plotting easier###################
                # values_ERP
                # for idx, participant in enumerate(values_ERP):
                #     name = 'ERP ' + str(idx + 1)
                #     data_dict[name] = participant
                #     # namenorm = 'ERP ' + str(idx + 1) + ' norm'
                #     # data_dict[namenorm] = self.moving_average(participant, norm=True, rmzero=False)
                #
                # # PSDs, band names are passed from config fil2e
                # for idx, name in enumerate(featurenames):
                #     data_dict[name] = psds[idx]
                #
                # data_dict['PLV 1'] = plv_intra1
                # data_dict['PLV 2'] = plv_intra2
                # data_dict['PLV inter'] = plv_inter
                #
                # intra1_tm, intra2_tm, inter_tm = self.calculate_tmflow(data_dict)
                #
                # data_dict['Intra 1'] = (intra1_tm)
                # data_dict['Intra 2'] = (intra2_tm)
                # data_dict['Inter'] = (inter_tm)
                #
                # print(data_dict.values())
                # if (self.segment == 2) & remfirstsample:
                #     for key, value in data_dict.items():
                #         if key in ['Intra 1', 'Intra 2', 'Inter']:
                #             print("yote")
                #             valuecopy = value
                #             valuecopy[0] = 0.0
                #             print(value)
                #             print(valuecopy)
                #             data_dict[key] = valuecopy
                # print(data_dict.values())
                # ############################################################################
                #
                # if self.plotpref != 'none':
                #     print("\nGenerating Plots...")
                #     time1 = timer()
                #     if self.plotpref == 'participant' or self.plotpref == 'both':
                #         ax[0, 0].cla()
                #         ax[0, 1].cla()
                #         ax[0, 2].cla()
                #
                #         x = np.arange(1, len(data_dict['Intra 1']) + 1)
                #
                #         ax[0, 0].bar(x, data_dict['Intra 1'], width=0.4, color='red')  # index for flow intra 1
                #         ax[0, 1].bar(x, data_dict['Inter'], width=0.4, color='blue')  # index for plv inter
                #         ax[0, 2].bar(x, data_dict['Intra 2'], width=0.4, color='green')  # index for plv intra 2
                #
                #         # ax[0, 0].plot(x, data_dict['Intra 1'], color='red')  # index for flow intra 1
                #         # ax[0, 1].plot(x, data_dict['Inter'], color='blue')  # index for plv inter
                #         # ax[0, 2].plot(x, data_dict['Intra 2'], color='green')  # index for plv intra 2
                #
                #         ax[0, 0].set_ylim(0, 10)
                #         ax[0, 1].set_ylim(0, 10)
                #         ax[0, 2].set_ylim(0, 10)
                #
                #         ax[0, 0].set(title='Intra 1', xlabel='Segment #', ylabel='Score')
                #         ax[0, 1].set(title='Inter', xlabel='Segment #', ylabel='Score')
                #         ax[0, 2].set(title='Intra 2', xlabel='Segment #', ylabel='Score')
                #         # self.fig.suptitle('Data Segment {}'.format(self.segment), fontsize=16)
                #         plt.pause(0.05)
                        # plt.show()

                # self.final()
                # if self.plotpref == 'participant':
                #     plt.close(fig=self.fig)
                #
                # if self.plotpref != 'none':
                #     plt.pause(.005)
                #     plt.draw()
                ###############custom function###################
                time1 = timer()
                for subject, keyname in enumerate([i for i in list(function_dict.keys()) if 'custom' in i]):
                    print(list(function_dict.keys()))
                    print([i for i in list(function_dict.keys()) if 'custom' in i])

                    # psdnames=['band1', 'band2']  #placeholder, remove
                    r = str(subject)
                    # i = np.mod(idx, len(
                    #     foilows))  # Adjust for the fact that channels are different amongst subjects, but fois are the same
                    function_dict = custom_function(raw,
                                                    r,
                                                    function_dict,
                                                    fsample)
                    if self.plotpref != 'none':

                        plot_settings = self.plot_settings(function_dict[keyname]['custom_plot_1'],
                                                           ex_plot_matrix, sub_plot_matrix)
                        if plot_settings:
                            print('PLOTSETTINGS', plot_settings)
                            whichax, loc, ex_plot_matrix, sub_plot_matrix = plot_settings
                            ax[whichax] = custom_plot_1(ax[whichax], loc, subject + 1,
                                                        function_dict)

                        plot_settings = self.plot_settings(function_dict[keyname]['custom_plot_2'],
                                                           ex_plot_matrix, sub_plot_matrix)
                        if plot_settings:
                            whichax, loc, ex_plot_matrix, sub_plot_matrix = plot_settings
                            ax[whichax] = custom_plot_1(ax[whichax], n + 1, loc, subject + 1,
                                                        function_dict)
                    # if self.plotpref == 'both' or self.plotpref == 'experiment':
                    #     band = n + 1
                    #     if band % 2 == 0:
                    #         col = 2
                    #     else:
                    #         col = 0
                    #
                    #     if idx < 2:
                    #         row = 2
                    #     elif 2 <= idx <= 3:
                    #         row = 3
                    #     else:
                    #         print('too many PSD bands, band {} not plotted'.format(band))
                    #
                    #     ax[0] = psd_plot(ax[0], this_psd_spec, psds, this_freqs, band, row, col,
                    #                   function_dict[keyname]['foilow_band' + r],
                    #                   function_dict[keyname]['foihigh_band' + r])

                    function_dict[keyname]['values_band' + r] = psds

                time2 = timer()
                print("Time to compute 6 PSDs: {}".format(time2 - time1))
                print("\n", "2-" * 60)
                # intra1_tm, intra2_tm, inter_tm = calculate_tmflow(data_dict=function_dict)
                #
                # function_dict['flow']['values_Intra 1'] = (intra1_tm)
                # function_dict['flow']['values_Intra 2'] = (intra2_tm)
                # function_dict['flow']['values_Inter'] = (inter_tm)
                #
                # if (self.segment == 2) & remfirstsample:
                #     for key, value in function_dict['flow'].items():
                #         if key in ['values_Intra 1', 'values_Intra 2', 'values_Inter']:
                #             valuecopy = value
                #             valuecopy[0] = 0.0
                #             function_dict['flow'][key] = valuecopy
                # print(function_dict.values())

                ############################################################################
                # if self.plotpref == 'both' or self.plotpref == 'participant':
                #     ax[1] = teamflow_plot(ax[1], function_dict)

                if self.plotpref != 'none' and self.show_plots:
                    for fi in fig:
                        if fi is not None:
                            fi.tight_layout()
                    # plt.pause(0.005)
                    plt.pause(3)
                    plt.show()

                if self.saving and self.plotpref != 'none':
                    for n, fi in enumerate(fig):
                        if fi is not None:
                            figsavepath = self.savepath + '/figures/plot_' + str(self.segment) + '_' + str(n) + '.jpg'
                            fi.savefig(figsavepath)

                time2 = timer()
                print("Time to generate plots: {}".format(time2 - time1))
                print("\n", "5-" * 60)

                time4 = timer()
                print("Time ALL: {}".format(time4 - time3))
                print("All operations complete for segment {}".format(self.segment))

                self.segment += 1
            else:
                self.save_csv(function_dict)
                break

    def setup_plotting(self, ex_plot_matrix, sub_plot_matrix):
        # For plotting
        # if self.plotpref != 'none':
        #     if self.plotpref == 'participant' or self.plotpref == 'both':
        #         # plt.close(self.fig)
        #         fig, ax = plt.subplots(1, 3, squeeze=False,
        #                                figsize=(self.windowsize * 3, self.windowsize))
        # creates subplots where all plots are added later
        fig = None
        fig2 = None
        ax = None
        ax2 = None

        sub_dims = sub_plot_matrix.shape
        ex_dims = ex_plot_matrix.shape

        subspacevalh = sub_dims[0]
        subspacevalw = sub_dims[1]
        exspacevalh = sub_dims[0]
        exspacevalw = sub_dims[1]

        if self.plotpref != 'none':
            if self.plotpref == 'participant':
                # plt.close(self.fig)
                fig2, ax2 = plt.subplots(sub_dims[0], sub_dims[1], squeeze=False,
                                         figsize=self.subwindowsize)
                fig2.subplots_adjust(left=0.05, right=.975, top=.92, bottom=0.05, hspace=subspacevalh, wspace=subspacevalw)
            elif self.plotpref == 'experiment':
                fig, ax = plt.subplots(ex_dims[0], ex_dims[1], figsize=self.exwindowsize)
                fig.subplots_adjust(left=0.05, right=.975, top=.92, bottom=0.05, hspace=.25, wspace=.25)
                # fig.tight_layout()

            elif self.plotpref == 'both':
                fig, ax = plt.subplots(ex_dims[0], ex_dims[1], figsize=self.exwindowsize)
                fig.subplots_adjust(left=0.05, right=.975, top=.92, bottom=0.05, hspace=exspacevalh, wspace=exspacevalw)

                fig2, ax2 = plt.subplots(sub_dims[0], sub_dims[1], squeeze=False,
                                         figsize=self.subwindowsize)
                fig2.subplots_adjust(left=0.05, right=.975, top=.92, bottom=0.05, hspace=subspacevalh, wspace=subspacevalw)
        plt.ion()

        return [fig, fig2], [ax, ax2]

    def plot_settings(self, this_plot_setting, ex_plot_matrix, sub_plot_matrix):
        if self.plotpref == 'none' or this_plot_setting == 'none':
            return False

        if type(this_plot_setting) == tuple:
            if this_plot_setting[0] == 'participant':
                if self.plotpref == 'participant' or self.plotpref == 'both':
                    sub_plot_matrix[this_plot_setting[1][0], this_plot_setting[1][1]] = 1
                    x, y = this_plot_setting[1][0], this_plot_setting[1][1]
                    return 1, (x, y), ex_plot_matrix, sub_plot_matrix
            elif this_plot_setting[0] == 'experiment':
                if self.plotpref == 'experiment' or self.plotpref == 'both':
                    ex_plot_matrix[this_plot_setting[1][0], this_plot_setting[1][1]] = 1
                    x, y = this_plot_setting[1][0], this_plot_setting[1][1]
                    return 0, (x, y), ex_plot_matrix, sub_plot_matrix

        elif type(this_plot_setting) == str:
            if this_plot_setting == 'participant':
                if self.plotpref == 'participant' or self.plotpref == 'both':
                    cont, sub_plot_matrix, x, y = self._fill_next_matrix_spot(sub_plot_matrix)
                    if cont:
                        return 1, (x, y), ex_plot_matrix, sub_plot_matrix
                    else:
                        return False
            elif this_plot_setting == 'experiment':
                if self.plotpref == 'experiment' or self.plotpref == 'both':
                    cont, ex_plot_matrix, x, y = self._fill_next_matrix_spot(ex_plot_matrix)
                    if cont:
                        return 0, (x, y), ex_plot_matrix, sub_plot_matrix
                    else:
                        return False
        return False

    def _fill_next_matrix_spot(self, plot_matrix):
        if np.all(plot_matrix) == 1:
            print("ALL PLOTS IN GRID FILLED, PLOT NOT RENDERED")
            i = np.inf
            j = np.inf
            return False, plot_matrix, i, j
        for i, x in enumerate(plot_matrix):
            if x.ndim > 0:
                for j, val in enumerate(x):
                    if val == 0:
                        plot_matrix[i, j] = 1
                        return True, plot_matrix, i, j
            else:
                if x == 0:
                    plot_matrix[i] = 1
                    j = 1
                    return True, plot_matrix, i, j

    def save_csv(self, data, exp_name=None):
        final_dict = {}
        for n, keyname in enumerate(list(data.keys())):
            for m, valuekeyname in enumerate([i for i in list(data[keyname].keys()) if 'values' in i]):
                final_dict[keyname + '_' + valuekeyname] = data[keyname][valuekeyname]


        try:
            df = pd.DataFrame.from_dict(data=final_dict)
        except:
            print('WARNING: NOT ALL ARRAYS ARE THE SAME LENGTH')
            df = pd.DataFrame.from_dict(data=final_dict, orient='index')
            print(df)
            df = df.transpose()
        if self.option == 'offline':
            base = path.basename(self.filepath)
            base = path.splitext(base)[0]
        else:
            base = exp_name

        print("SAVING CSV TO " + self.savepath + '/' + base + '_results.csv')
        df.to_csv(path_or_buf=self.savepath + '/' + base + '_results.csv', index=False)
        print(df)

    def make_raw(self, data, stim, fsample, units, channelnames):
        D = data

        # Make channel names for biosemi128 EEG reader (move to config file?)

        # biosemi128 = list(range(len(data)))
        # biosemi128 = [str(int) for int in biosemi128]
        # Specify channel types (list with length equal to total number of channels)
        print("\nStarting MNE Operations...")
        channel_types = []
        # print(channelnames)
        if type(channelnames) != list:
            print("FATAL: channel names must be a list within a list, even if there is only 1 participant")
            exit(1)
        channel_names = channelnames.copy()
        # Assume data is from biosemi 128
        for idx in range(len(channelnames)):
            channel_types.append('eeg')  # fill all channel types as eeg
        # print(len(D))
        # channel_types[self.TrigChan] = 'stim'
        # Add in stim channel
        # self.stim_idx = channel_names.index(self.TrigChan)
        # stim_idx = len(D)
        D[np.isnan(D)] = 0
        # minval = np.amin(stim)
        # if minval != 0:
        #     stim_values = stim - minval  # correct values of the stim channel (stim is always last channel)
        # else:
        #     stim_values = stim

        # Adjust units to uv
        # print(D)
        if units == "mv":
            D = D / 1000
        elif units == "uv":
            D = D * 1000000
        elif units == "Mv":
            D = D * 1000
        elif units == "nv":
            D = D * 1000000000
        elif units != "v":
            print("invalid units selected")
            sys.exit(1)
        # if units == "mv":
        #     D = D * 1000
        # elif units == "v":
        #     D = D * 1000000
        # elif units == "Mv":
        #     D = D * 1000
        # elif units != "v":
        #     print("invalid units selected")
        #     sys.exit(1)
        # print(D)

        # D = np.vstack([D, stim_values])
        # self.stim_values = stim_values



        info = mne.create_info(channel_names, fsample, channel_types)
        # print(channel_names)
        raw = RawArray(D, info)  # generates MNE raw array from data
        # print("raw from make_raw", raw.get_data())
        # ICA
        # ica = ICA(n_components=15, random_state=97)
        # ica.fit(raw)

        return raw

    def preprocess_raw(self, raw_data, badchans, filter_range):
        raw = raw_data
        raw.set_eeg_reference(ref_channels='average', ch_type='eeg', projection=False)
        raw.info['bads'] = badchans

        raw = raw.copy().filter(l_freq=filter_range[0], h_freq=filter_range[1], picks=None, filter_length='auto',
                                l_trans_bandwidth='auto',
                                h_trans_bandwidth='auto', n_jobs=1, method='iir', iir_params=None, phase='zero',
                                fir_window='hamming', fir_design='firwin',
                                skip_by_annotation=('edge', 'bad_acq_skip'), pad='reflect_limited', verbose=None)

        montage = mne.channels.make_standard_montage('biosemi128')
        raw_montage = raw.copy().set_montage(montage, match_case=False)
        interp = raw_montage.copy().pick_types(eeg=True, exclude=()).interpolate_bads(reset_bads=True)

        # # ICA
        # interp.pick_types(meg=False, eeg=True, exclude='bads', stim=True).load_data()
        # ica = ICA(n_components=15, random_state=97) # 15 components for ICA was pre-determined in jupyter notebook
        # ica.fit(interp)
        # ica.exclude = [11] # ICA 11 is bad when there are 15 ICAs
        # ica.apply(interp)

        return interp

    # def get_real_events(self, raw):
    #     print('\nTrying to read events...')
    #     E = mne.find_events(raw, stim_channel='Status', initial_event=True)
    #     print("events present: {}".format(E))
    #     return E

    # def generate_sim_events(self, raw, blocksize, blocksize_sec, fsample):
    #     seg = blocksize / blocksize_sec - 1
    #     print('Generating {0} simulated events {1} samples apart...'.format(blocksize_sec, seg))
    #     sec = 0.5
    #     intervals = []
    #     while sec <= blocksize_sec:
    #         intervals.append(int(seg * sec))
    #         sec += 1
    #     stim = []
    #     for idx in np.arange(blocksize):
    #         if idx in intervals:
    #             stim.append(1.)
    #         else:
    #             stim.append(0.)
    #     # raw[self.stim_idx] = stim
    #     # print('raw_len', len(raw))
    #     # raw[len(raw.ch_names)] = stim
    #     # E = mne.find_events(raw, stim_channel=self.TrigChan)
    #     info_stim = mne.create_info(['PLVstim'], sfreq=fsample, ch_types=['stim'])
    #     raw_stim = RawArray(np.asarray(stim).reshape(1, len(stim)), info_stim)
    #     PLV_raw = raw.add_channels([raw_stim], force_update_info=True)
    #     E = mne.find_events(PLV_raw, stim_channel='PLVstim')
    #     print("Simulated events in segment {0}: {1}".format(self.segment, E))
    #     return E

    def create_stim_channel(self, ev_arr, seg_start, seg_end):
        event_locations = []
        event_values = []

        for event in reversed(ev_arr):
            if seg_start < event.sample < seg_end:
                event_locations.append(event.sample - seg_start)
                event_values.append(event.value)
            if event.sample < seg_start:
                break

        stim_chan = np.zeros(seg_end-seg_start)

        for n, loc in enumerate(event_locations):
            stim_chan[loc-1] = event_values[n]

        print('stim ', event_locations, event_values, stim_chan)
        return stim_chan

    def moving_average(self, l, avg=True, norm=False, p=False, rmzero=True, initval=10):
        a = []
        if isinstance(l, list):
            if len(l) > 1:
                # Generate list of indexes of non-zero values
                a = [e for i, e in enumerate(l) if e != 0]

                # Generate list of indexes with zero-values
                zero_idxs = [i for i, e in enumerate(l) if e == 0]

                # Make list of non-zero values
                # for i in ls:
                #     a.append(l[i])
                if len(a) == 0:
                    a.append(initval)

                if p == True: print(a)

                if norm:
                    a = pd.DataFrame(data=a)
                    a = a.values  # returns a numpy array
                    min_max_scaler = preprocessing.MinMaxScaler()
                    a = min_max_scaler.fit_transform(a)

                if avg:
                    a = pd.DataFrame(data=a)
                    # a = a.ewm(alpha=0.5, adjust=False).mean()
                    a = a.expanding().mean()
                    if p == True: print(a)
                    a = a.values  # returns a numpy array

                a = np.asarray(a)
                flat = []
                for sublist in a:
                    if a.ndim > 1:
                        for item in sublist:
                            flat.append(item)
                    else:
                        flat.append(sublist)
                if p == True: print(flat)

                if rmzero == False:
                    for i in zero_idxs:
                        flat.insert(i, 0)
                # print("list length: ", len(l))
                # print("flat length:", len(flat))
                return flat
            else:
                return l

    # def calculate_tmflow(self, data_dict):
    #
    #     new_dict = data_dict.copy()
    #
    #     for key, value in new_dict.items():
    #         new_dict[key] = self.moving_average(value, norm=True)
    #
    #     # for idx, ERP in enumerate(values_ERP):
    #     #     values_ERP_norm_plot[idx] = self.moving_average(ERP, norm=True, initval=0.55997824)
    #     #     values_ERP_raw_plot[idx] = self.moving_average(ERP, norm=False)
    #     #
    #     # for idx, psd in enumerate(psds):
    #     #     psds_raw_plot[idx] = self.moving_average(psd, norm=False)
    #     #     psds_norm_plot[idx] = self.moving_average(psd, norm=True)
    #     #
    #     # PLV_inter_plot = self.moving_average(plv_inter, norm=True)
    #     # PLV_intra1_plot = self.moving_average(plv_intra1, norm=True)
    #     # PLV_intra2_plot = self.moving_average(plv_intra2, norm=True)
    #
    #     intra1 = data_dict['Intra 1']
    #     intra2 = data_dict['Intra 2']
    #     inter = data_dict['Inter']
    #
    #     if new_dict['ERP 1'][-1] != 0:
    #         intra1.append((1 / new_dict['ERP 1'][-1]) + new_dict['Alpha1'][-1] + \
    #                       (1 / new_dict['Beta1'][-1]))
    #     else:
    #         intra1.append((1 / 0.55997824) + new_dict['Alpha1'][-1] + \
    #                       (1 / new_dict['Beta1'][-1]))
    #
    #     if new_dict['ERP 2'][-1] != 0:
    #         intra2.append((1 / new_dict['ERP 2'][-1]) + new_dict['Alpha2'][-1] + (1 / new_dict['Beta2'][-1]))
    #     else:
    #         intra2.append((1 / 0.55997824) + new_dict['Alpha2'][-1] + (1 / new_dict['Beta2'][-1]))
    #
    #     inter.append(new_dict['PLV inter'][-1] + ((new_dict['PLV 1'][-1] + new_dict['PLV 2'][-1]) / 2) +
    #                  ((new_dict['Gamma1'][-1] + new_dict['Gamma2'][-1]) / 2))
    #
    #     return intra1, intra2, inter

    # interfinal = float(interfinal)
    # intra1_tm.append(intras_plot[0])
    # intra2_tm.append(intras_plot[1])
    # inter_tm.append(interfinal)
    # intra1_plot = self.moving_average(intra1_tm, p=True, norm=True)
    # intra2_plot = self.moving_average(intra2_tm, p=True, norm=True)
    # inter_plot = self.moving_average(inter_tm, p=True, norm=True)
    #
    # print('Subject 1 intra brain score: ', intra1_plot[-1])
    # print('Subject 2 intra brain score: ', intra2_plot[-1])
    # print('Inter-brain score: ', inter_plot[-1])

    # self.ax2[0, 0].set(title='Intrabrain scores for Subject 1', xlabel='Segment #', ylabel='Score')
    #                   # ylim=(0, max(self.intra1finals) + .2))
    # self.ax2[0, 1].set(title='Inter-brain teamflow scores', xlabel='Segment #', ylabel='Score')
    #                   # ylim=(0, max(self.interfinals) + .2))
    # self.ax2[0, 2].set(title='Intrabrain scores for subject 2', xlabel='Segment #', ylabel='Score')
    #                   # ylim=(0, max(self.intra2finals) + .2))

    # self.ax2[0, 0].text(0.1, max(intra1_plot) - .1, 'Current score: {}'.format(round(intras_plot[0], 2)),
    #                     ha='center', va='center', transform=self.ax[0, 0].transAxes, size=12)
    # self.ax2[0, 1].text(0.1, max(inter_plot) - .1, 'Current score: {}'.format(round(interfinal, 2)), ha='center',
    #                     va='center', transform=self.ax[0, 1].transAxes, size=12)
    # self.ax2[0, 2].text(0.1, max(intra2_plot) - .1, 'Current score: {}'.format(round(intras_plot[1], 2)),
    #                     ha='center', va='center', transform=self.ax[0, 2].transAxes, size=12)

    # plt.show()
