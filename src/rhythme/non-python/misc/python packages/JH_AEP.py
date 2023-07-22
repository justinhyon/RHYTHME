#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  8 10:53:52 2020

@author: justin hyon
"""

import sys
from mne_realtime.externals import FieldTrip
import time
import numpy as np
from statistics import mean
import mne
from mne.io import RawArray
from mne.connectivity import spectral_connectivity
from mne.time_frequency import psd_welch, psd_multitaper
import matplotlib.pyplot as plt
from timeit import default_timer as timer
from multiprocessing import Process


class TeamFlow:
    def __init__(self, path, dataport, blocksize_sec, windowsize, delay, plotpref, channelofint_aep1, channelofint_aep2,
                 TrigChan,
                 epoeventval1, pretrig1, posttrig1, epoeventval2, pretrig2, posttrig2, frqofint, fsample):
        # user defined attributes
        self.path = path
        self.dataport = dataport
        self.blocksize_sec = blocksize_sec
        self.windowsize = windowsize
        self.delay = delay
        self.plotpref = plotpref
        self.TrigChan = TrigChan
        self.epoeventval1 = epoeventval1
        self.pretrig1 = pretrig1
        self.posttrig1 = posttrig1
        self.epoeventval2 = epoeventval2
        self.pretrig2 = pretrig2
        self.posttrig2 = posttrig2
        self.channelofint_aep1 = channelofint_aep1
        self.channelofint_aep2 = channelofint_aep2

        self.frqofint = frqofint
        self.fsample = fsample

        # not changed by user
        self.AEPs1 = []
        self.exB_AEPs1 = []
        self.exE_AEPs1 = []
        self.AEP1xval = []
        self.AEPs2 = []
        self.exB_AEPs2 = []
        self.exE_AEPs2 = []
        self.AEP2xval = []
        self.Events = []
        self.Header = []
        self.Data = []

        # creates subplots where all plots are added later

        self.fig, self.ax = plt.subplots(1, 2, figsize=(self.windowsize * 2, self.windowsize), squeeze=False)
        self.fig.subplots_adjust(left=0.05, right=.975, top=.92, bottom=0.05, hspace=.4, wspace=.2)
        plt.ion()

        self.segment = 1  # segment counter
        self.ftc = FieldTrip.Client()
        self.prevsamp = 0
        print("Waiting for first data segment from port: ", self.dataport)

    def receive_data(self):
        #  RECEIVES DATA FROM PORT AND CONVERTS TO MNE, THEN SENDS TO AEP FUNCTION.
        while True:
            time.sleep(self.delay)
            time1 = timer()
            time3 = timer()
            self.ftc.connect('localhost', self.dataport)  # might throw IOError
            H = self.ftc.getHeader()

            # Check presence of header
            if H is None:
                print('Failed to retrieve header!')
                sys.exit(1)
            # Check presence of channels
            if H.nChannels == 0:
                print('no channels were selected')
                sys.exit(1)

            self.fSamp = H.fSample
            self.blocksize = round(self.blocksize_sec * self.fSamp)
            currentsamp = H.nSamples

            if currentsamp == self.prevsamp + self.blocksize:

                self.prevsamp = currentsamp
                # self.segment = int(H.nSamples / self.blocksize)  # segment counter

                print("\n", "-" * 60)
                print('\nCONNECTED: Header retrieved for segment: ' + str(self.segment))

                # determine the size of blocks and start and stop samples to process
                print("Total Samples Received = {}".format(H.nSamples))
                firstSamp = H.nSamples - self.blocksize
                lastSamp = H.nSamples - 1

                print("Trying to read from sample {0} to {1}. Samples per block: {2}".format(firstSamp, lastSamp,
                                                                                             self.blocksize))
                D = self.ftc.getData(index=(firstSamp, lastSamp)).transpose()  # receive data from buffer
                print('Samples retrieved for segment: ' + str(self.segment))
                print('Data shape (nChannels, nSamples): {}'.format(D.shape))

                # generate MNE info array
                print("\nStarting MNE Operations...")
                channel_types = []
                for idx in range(H.nChannels):
                    channel_types.append('eeg')  # fill all channel types as eeg
                channel_names = H.labels
                # info = mne.create_info(channel_names, self.fSamp, channel_types)

                # make adjustments to data to correspond with MNE raw format
                self.stim_idx = channel_names.index(self.TrigChan)
                minval = np.amin(D[self.stim_idx])
                D = D / (1000000)  # Changes units to Volts from uV
                self.stim_values = 1000000 * (D[self.stim_idx]) - minval  # correct values of the stim channel
                D[self.stim_idx] = self.stim_values
                channel_types[self.stim_idx] = 'stim'
                info = mne.create_info(channel_names, self.fSamp, channel_types)
                raw = RawArray(D, info)  # generates MNE raw array from data

                raw = raw.copy().resample(sfreq=256)  # downsampling to 256
                self.stim_values = raw._data[self.stim_idx]
                self.fSamp = 256
                self.blocksize = round(self.blocksize_sec * self.fSamp)

                time2 = timer()
                print("Time to recieve data and create MNE raw: {}".format(time2 - time1))
                print("\n", "1-" * 60)

                self.aep(raw)

                print("\nGenerating Plots...")
                time1 = timer()

                self.fig.suptitle('Data Segment {}'.format(self.segment), fontsize=16)
                plt.pause(.005)
                plt.draw()
                figsavepath = self.path + '/AEP_figures/AEP_plot_' + str(self.segment) + '.jpg'
                plt.savefig(figsavepath)

                time2 = timer()
                print("Time to generate plots: {}".format(time2 - time1))
                print("\n", "5-" * 60)

                time4 = timer()
                print("Time ALL: {}".format(time4 - time3))
                print("All operations complete for segment {}".format(self.segment))

                self.Header.append(H)
                self.Data.append(D)
                # self.Events.append(E)
                self.segment += 1

    def aep(self, raw, mode=0):
        #  CALCULATES AEP
        if mode == 0:
            channelofint = self.channelofint_aep1
            epoeventval = self.epoeventval1
            pretrig = self.pretrig1
            posttrig = self.posttrig1

        if epoeventval in self.stim_values:
            indexes = [i for i, e in enumerate(self.stim_values) if e == epoeventval]
            print(
                "\nAEP Subject {3}:\n{0} event(s) with value {1} found in this segment at location(s) {2}, " \
                "performing AEP calculation..."
                    .format(len(indexes), epoeventval, indexes, mode + 1))

            # FILTERING LINE WHICH HAS DISCREPANCY FROM MATLAB
            raw = raw.copy().filter(l_freq=2.0, h_freq=7.0, picks=None, filter_length='auto', l_trans_bandwidth='auto',
                                    h_trans_bandwidth='auto', n_jobs=1, method='iir', iir_params=None, phase='zero',
                                    fir_window='hamming', fir_design='firwin',
                                    skip_by_annotation=('edge', 'bad_acq_skip'), pad='reflect_limited', verbose=None)
            raw._data[self.stim_idx] = self.stim_values  # restores events destroyed by filtering
            data = raw.get_data()

            goodevent = False
            for testevent in indexes:
                testfirstSamp = round(testevent - (pretrig * self.fSamp))
                testlastSamp = round(testevent + (posttrig * self.fSamp))
                if (0 < testfirstSamp) & (testlastSamp < self.blocksize):
                    goodevent = True

            firstloop = True
            for event in indexes:
                terminate = False

                firstSamp = round(event - (pretrig * self.fSamp))

                print(event, self.fSamp, posttrig)
                lastSamp = round(event + (posttrig * self.fSamp))
                print(lastSamp)

                if firstSamp < 0:
                    print(
                        "Event with value {} is at extreme beginning of segment. No calculation performed, "
                        "marker added.".format(epoeventval))
                    if mode == 0:
                        if firstloop and not goodevent:
                            self.AEPs1.append(0)
                            self.AEP1xval.append(self.segment)
                        self.exB_AEPs1.append(self.segment)
                    elif mode == 1:
                        if firstloop and not goodevent:
                            self.AEPs2.append(0)
                            self.AEP2xval.append(self.segment)
                        self.exB_AEPs2.append(self.segment)
                    self.aep_plot(data=np.zeros(round((posttrig + pretrig) * self.fSamp)), mode=mode)
                    terminate = True

                if lastSamp > self.blocksize:
                    print("Event with value {} is at extreme end of segment. No calculation performed, marker added."
                          .format(epoeventval))
                    if mode == 0:
                        if firstloop and not goodevent:
                            self.AEPs1.append(0)
                            self.AEP1xval.append(self.segment)
                        self.exE_AEPs1.append(self.segment)
                    terminate = True

                if not terminate:
                    dat = data
                    dat = np.delete(dat, np.arange(lastSamp - 1, self.blocksize), axis=1)
                    dat = np.delete(dat, np.arange(0, firstSamp - 1), axis=1)
                    row_idx = np.array(channelofint)  # picks; add 128 to each number for participant 2
                    print("using {0} channels: {1}".format(len(row_idx), row_idx))
                    dat = dat[row_idx, :]

                    # baseline correction
                    temp_dat = dat[:, 0:int(pretrig * self.fSamp)]
                    for idx, chan in enumerate(temp_dat):
                        ch_mean = np.mean(chan)
                        dat[idx] -= ch_mean

                    # Define peaks of interest: N1 (110 ? 150 ms), P1 (210 ? 250 ms), and N2 (310 ? 350 ms).
                    times = np.array([[.110, .150], [.210, .250], [.310, .350]])
                    for a, b in enumerate(times):
                        for c, d in enumerate(b):
                            times[a, c] = round(d * self.fSamp + pretrig * self.fSamp)
                    times = times.astype(int)
                    # calculate AEP peak amplitude
                    AEPamps = []
                    for ch in dat:
                        N1 = sum(ch[times[0, 0]:times[0, 1]]) / (times[0, 1] - times[0, 0])
                        P1 = sum(ch[times[1, 0]:times[1, 1]]) / (times[1, 1] - times[1, 0])
                        N2 = sum(ch[times[2, 0]:times[2, 1]]) / (times[2, 1] - times[2, 0])
                        average = (N1 + P1 + N2) / 3
                        AEPamps.append(average)
                    mAEPamp = sum(AEPamps) / len(AEPamps)
                    print("Average AEP peak amplitude: ", mAEPamp)
                    if mode == 0:
                        self.AEP1xval.append(self.segment)
                        self.AEPs1.append(mAEPamp)
                    elif mode == 1:
                        self.AEP2xval.append(self.segment)
                        self.AEPs2.append(mAEPamp)

                    self.aep_plot(data=dat, mode=mode)
                firstloop = False

        else:
            print("\nAEP Subject {0}:\nno events with value {1} found in this segment, AEP calculation not performed"
                  .format(mode + 1, epoeventval))
            if mode == 0:
                self.AEPs1.append(0.)
                self.AEP1xval.append(self.segment)
            elif mode == 1:
                self.AEPs2.append(0.)
                self.AEP2xval.append(self.segment)
            self.aep_plot(data=np.zeros(round((posttrig + pretrig) * self.fSamp)), mode=mode)

    def aep_plot(self, data, mode=0):
        #  PLOTS AEP AND PEAK INDEX. NO NEED TO CHANGE CODE HERE.

        if mode == 0:
            y = 0
            AEPs = self.AEPs1
            exB_AEPs = self.exB_AEPs1
            exE_AEPs = self.exE_AEPs1
            AEPxval = self.AEP1xval
            pretrig = self.pretrig1
            posttrig = self.posttrig1
        dat = data.transpose()
        self.ax[0, y].cla()  # clears the axes to prepare for new data
        self.ax[0, y + 1].cla()
        x = np.arange((-1 * pretrig), posttrig, (posttrig + pretrig) / int(
            (posttrig + pretrig) * self.fSamp))  # generate x axis values (time)
        x = np.append(x, [x[-1]+1])
        print(x.shape, dat.shape)
        # plots standard deviation if data is not 0
        if data.ndim > 1:
            sdAEPamp = dat.std(axis=1)
            dat = data.mean(axis=0)
            self.ax[0, y].fill_between(x, dat + sdAEPamp, dat - sdAEPamp, facecolor='red', alpha=0.3)

        # plots the data against time
        self.ax[0, y].plot(x, dat, color='black')

        # format AEP vs time plot
        self.ax[0, y].axvline(x=0, color='red')
        self.ax[0, y].axvspan(.110, .150, alpha=0.3, color='green')
        self.ax[0, y].axvspan(.210, .250, alpha=0.3, color='green')
        self.ax[0, y].axvspan(.310, .350, alpha=0.3, color='green')
        self.ax[0, y].hlines(0, -1 * pretrig, posttrig)
        self.ax[0, y].set_title('Subject {0} AEP'.format(mode + 1))
        self.ax[0, y].set_xlabel('Time (s)')
        self.ax[0, y].set_ylabel('Volts')

        # generate plot of all AEP peak indexes
        x = AEPxval
        print(self.AEPs1, len(x), len(AEPs))
        self.ax[0, y + 1].bar(x, AEPs, width=0.4)
        self.ax[0, y + 1].bar(exB_AEPs, .000001, width=0.2, alpha=.5)
        self.ax[0, y + 1].bar(exE_AEPs, -.000001, width=0.2, alpha=.5)
        # self.ax[0, y + 1].axis(xmin=-3)
        for i, v in zip(AEPxval, AEPs):
            if v != 0.:
                self.ax[0, y + 1].text(i - .5, v, str(round(v, 10)))
        self.ax[0, y + 1].hlines(0, 0, self.segment)
        self.ax[0, y + 1].set_title('Subject {} AEP Peak Amplitude Index'.format(mode + 1))
        self.ax[0, y + 1].set_xlabel('Segment #')
        self.ax[0, y + 1].set_ylabel('AEP Peak Index (V)')
