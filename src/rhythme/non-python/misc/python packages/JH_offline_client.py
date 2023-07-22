#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
10/16/2020

@author: justin hyon

VERSION 1.0.0
"""

import sys
from mne_realtime.externals import FieldTrip
import time
import numpy as np
from statistics import mean
import mne
from mne.io import RawArray, read_raw_bdf, read_raw_eeglab
from mne.connectivity import spectral_connectivity
from mne.time_frequency import psd_welch, psd_multitaper
import matplotlib.pyplot as plt
from timeit import default_timer as timer
from multiprocessing import Process
import pandas as pd
from collections import OrderedDict
from sklearn import preprocessing
from os import path


class TeamFlow:
    def __init__(self, path, dataport, blocksize_sec, windowsize, delay, plotpref, saving, channelofint_aep1,
                 channelofint_aep2, TrigChan,
                 epoeventval1, pretrig1, posttrig1, epoeventval2, pretrig2, posttrig2, channelofint_plv,
                 frqofint, fsample, channelofint_psdT1, channelofint_psdT2, channelofint_psdF1, channelofint_psdF2,
                 channelofint_psdA1, channelofint_psdA2, foillowT1, foihighT1, foillowT2, foihighT2, foillowF1,
                 foihighF1, foillowF2, foihighF2, foillowA1, foihighA1, foillowA2, foihighA2):
        # user defined attributes
        self.path = path
        self.fSamp = fsample
        self.dataport = dataport
        self.blocksize_sec = blocksize_sec
        self.windowsize = windowsize
        self.delay = delay
        self.plotpref = plotpref
        self.saving = saving
        self.TrigChan = TrigChan
        self.epoeventval1 = epoeventval1
        self.pretrig1 = pretrig1
        self.posttrig1 = posttrig1
        self.epoeventval2 = epoeventval2
        self.pretrig2 = pretrig2
        self.posttrig2 = posttrig2
        self.channelofint_aep1 = channelofint_aep1
        self.channelofint_aep2 = channelofint_aep2

        self.channelofint_plv = channelofint_plv
        self.frqofint = frqofint

        self.channelofint_psdT1 = channelofint_psdT1
        self.channelofint_psdT2 = channelofint_psdT2
        self.channelofint_psdF1 = channelofint_psdF1
        self.channelofint_psdF2 = channelofint_psdF2
        self.channelofint_psdA1 = channelofint_psdA1
        self.channelofint_psdA2 = channelofint_psdA2
        self.foilowT1 = foillowT1
        self.foihighT1 = foihighT1
        self.foilowT2 = foillowT2
        self.foihighT2 = foihighT2
        self.foilowF1 = foillowF1
        self.foihighF1 = foihighF1
        self.foilowF2 = foillowF2
        self.foihighF2 = foihighF2
        self.foilowA1 = foillowA1
        self.foihighA1 = foihighA1
        self.foilowA2 = foillowA2
        self.foihighA2 = foihighA2

        # not changed by user
        self.AEPs1 = []
        self.exB_AEPs1 = []
        self.exE_AEPs1 = []
        self.AEP1xval = []
        self.AEPs2 = []
        self.exB_AEPs2 = []
        self.exE_AEPs2 = []
        self.AEP2xval = []
        self.PLV_intra1 = []
        self.PLV_inter = []
        self.PLV_intra2 = []
        self.PSDsT1 = []
        self.PSDsT2 = []
        self.PSDsF1 = []
        self.PSDsF2 = []
        self.PSDsA1 = []
        self.PSDsA2 = []
        self.intra1finals = []
        self.intra2finals = []
        self.interfinals = []
        self.Events = []
        self.Header = []
        self.Data = []

        # creates subplots where all plots are added later
        # if self.plotpref == 'participant' or self.plotpref == 'both':
            # plt.close(self.fig)
        self.fig2, self.ax2 = plt.subplots(4, 4, squeeze=False,
                                           figsize=(self.windowsize * 3, self.windowsize))

        self.fig, self.ax = plt.subplots(5, 4, figsize=(self.windowsize * 2, self.windowsize))
        self.fig.subplots_adjust(left=0.05, right=.975, top=.92, bottom=0.05, hspace=.4, wspace=.2)
        plt.ion()

        self.segment = 1  # segment counter
        self.ftc = FieldTrip.Client()
        self.prevsamp = 0
        # print("Waiting for first data segment from port: ", self.dataport)

    def receive_data(self, filepath):
        base = path.basename(filepath)
        base = path.splitext(base)[1]
        if base == '.bdf':
            fullraw = read_raw_bdf(filepath, preload=True)
        elif base == '.set':
            fullraw = read_raw_eeglab(filepath, preload=True)
        self.filepath = filepath
        fulldata = fullraw._data
        self.blocksize = round(self.blocksize_sec * self.fSamp)
        print("Total Samples Received = {}".format(fulldata.shape[1]))

        while True:
            # time.sleep(self.delay)
            time1 = timer()
            time3 = timer()

            currentsamp = self.prevsamp + self.blocksize
            if currentsamp < fulldata.shape[1]:
                print("\n", "-" * 60)
                print('\nCONNECTED: Header retrieved for segment: ' + str(self.segment))
                print("Trying to read from sample {0} to {1}. Samples per block: {2}".format(self.prevsamp, currentsamp,
                                                                                             self.blocksize))

                D = fulldata[:, self.prevsamp:currentsamp]
                self.prevsamp = currentsamp
                print(D)
                print('Samples retrieved for segment: ' + str(self.segment))
                print('Data shape (nChannels, nSamples): {}'.format(D.shape))

                # generate MNE info array
                print("\nStarting MNE Operations...")
                channel_types = []
                channel_names = []
                for idx in range(D.shape[0]):
                    channel_types.append('eeg')  # fill all channel types as eeg
                    channel_names.append(str(idx))

                # make adjustments to data to correspond with MNE raw format
                self.stim_idx = channel_names.index(self.TrigChan)
                minval = np.amin(D[self.stim_idx])
                D = D / (1000000)  # Changes units to Volts from uV
                self.stim_values = 1000000 * (D[self.stim_idx]) - minval  # correct values of the stim channel
                D[self.stim_idx] = self.stim_values
                channel_types[self.stim_idx] = 'stim'
                info = mne.create_info(channel_names, self.fSamp, channel_types)

                D = D * 1000000
                D[self.stim_idx] = self.stim_values

                print(D)

                raw = RawArray(D, info)  # generates MNE raw array from data
                time2 = timer()

                # raw.save('/Users/jasonhyon/Desktop/1_segment_test.fif', overwrite=True)

                print("Time to recieve data and create MNE raw: {}".format(time2 - time1))
                print("\n", "1-" * 60)
                self.plv_par(raw)  # plv function
                self.psd_par(raw)  # psd function
                self.aep_par(raw)  # aep function


                print("\nGenerating Plots...")

                time1 = timer()
                self.fig.suptitle('Data Segment {}'.format(self.segment), fontsize=16)

                # if self.plotpref == 'participant' or self.plotpref == 'both':
                self.final()
                if self.plotpref == 'experiment' or self.plotpref == 'none':
                    plt.close(fig=self.fig2)
                if self.plotpref == 'participant' or self.plotpref == 'none':
                    plt.close(fig=self.fig)
                if self.plotpref != 'none':
                    plt.pause(.005)
                    plt.draw()
                if self.saving:
                    figsavepath = self.path + '/TF_figures/TF_plot_' + str(self.segment) + '.jpg'
                    plt.savefig(figsavepath)

                time2 = timer()
                print("Time to generate plots: {}".format(time2 - time1))
                print("\n", "5-" * 60)

                time4 = timer()
                print("Time ALL: {}".format(time4 - time3))
                print("All operations complete for segment {}".format(self.segment))

                self.segment += 1

            else:
                self.save_csv()
                break

    def psd_par(self, raw):
        time1 = timer()

        self.psd(raw, 0)  # PSD function
        self.psd(raw, 1)  # PSD function
        self.psd(raw, 2)  # PSD function
        self.psd(raw, 3)  # PSD function
        self.psd(raw, 4)  # PSD function
        self.psd(raw, 5)  # PSD function
        time2 = timer()
        print("Time to compute 6 PSDs: {}".format(time2 - time1))
        print("\n", "2-" * 60)

    def aep_par(self, raw):
        time1 = timer()
        self.aep(raw, 0)  # AEP function
        self.aep(raw, 1)  # AEP function
        time2 = timer()
        print("Time to compute 2 AEPs: {}".format(time2 - time1))
        print("\n", "3-" * 60)

    def plv_par(self, raw):
        time1 = timer()
        self.plv(raw)  # PLV function
        time2 = timer()
        print("Time to compute PLV: {}".format(time2 - time1))
        print("\n", "4-" * 60)

    def get_real_events(raw):
        print('\nTrying to read events...')
        E = mne.find_events(raw, stim_channel='Status', initial_event=True)
        print("events present: {}".format(E))
        return E

    def generate_sim_events(self, raw):
        seg = self.blocksize / self.blocksize_sec - 1
        print('Generating {0} simulated events {1} samples apart...'.format(self.blocksize_sec, seg))
        sec = 0.5
        intervals = []
        while sec <= self.blocksize_sec:
            intervals.append(int(seg * sec))
            sec += 1
        stim = []
        for idx in np.arange(self.blocksize):
            if idx in intervals:
                stim.append(1.)
            else:
                stim.append(0.)
        print(raw[self.stim_idx])
        raw[self.stim_idx] = stim
        E = mne.find_events(raw, stim_channel=self.TrigChan)
        print("Simulated events in segment {0}: {1}".format(self.segment, E))
        return E

    def aep(self, raw, mode):
        if mode == 0:
            channelofint = self.channelofint_aep1
            epoeventval = self.epoeventval1
            pretrig = self.pretrig1
            posttrig = self.posttrig1
        elif mode == 1:
            channelofint = self.channelofint_aep2
            epoeventval = self.epoeventval2
            pretrig = self.pretrig2
            posttrig = self.posttrig2

        if epoeventval in self.stim_values:
            indexes = [i for i, e in enumerate(self.stim_values) if e == epoeventval]
            print(
                "\nAEP Subject {3}:\n{0} event(s) with value {1} found in this segment at location(s) {2}, " \
                "performing AEP calculation..."
                    .format(len(indexes), epoeventval, indexes, mode + 1))

            # band pass filter
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
                lastSamp = round(event + (posttrig * self.fSamp))

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
                    elif mode == 1:
                        if firstloop and not goodevent:
                            self.AEPs2.append(0)
                            self.AEP2xval.append(self.segment)
                        self.exE_AEPs2.append(self.segment)
                    self.aep_plot(data=np.zeros(round((posttrig + pretrig) * self.fSamp)), mode=mode)
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

    def aep_plot(self, data, mode):
        if mode == 0:
            y = 0
            AEPs = self.AEPs1
            exB_AEPs = self.exB_AEPs1
            exE_AEPs = self.exE_AEPs1
            AEPxval = self.AEP1xval
            pretrig = self.pretrig1
            posttrig = self.posttrig1
        elif mode == 1:
            y = 2
            AEPs = self.AEPs2
            exB_AEPs = self.exB_AEPs2
            exE_AEPs = self.exE_AEPs2
            AEPxval = self.AEP2xval
            pretrig = self.pretrig2
            posttrig = self.posttrig2
        dat = data.transpose()
        self.ax[0, y].cla()  # clears the axes to prepare for new data
        self.ax[0, y + 1].cla()
        x = np.arange((-1 * pretrig), posttrig, (posttrig + pretrig) / int(
            (posttrig + pretrig) * self.fSamp))  # generate x axis values (time)

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
        print(AEPs, len(x), len(AEPs))
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

    def plv(self, raw):
        print("\nCalculating PLV:")

        E = self.generate_sim_events(raw)  # Replaces events with equally spaced dummy events to calculate PLV
        epochs = mne.Epochs(raw, E, tmin=-.5, tmax=.5)

        con, freqs, times, n_epochs, n_tapers = spectral_connectivity(epochs, method='plv', sfreq=self.fSamp, fmin=13.,
                                                                      fmax=50., n_jobs=1)
        con = con.mean(axis=2)
        print('Connectivity Matrix dimensions: {}'.format(con.shape))
        np.delete(con, self.stim_idx, axis=1)
        np.delete(con, self.stim_idx, axis=0)
        print('Connectivity Matrix dimensions: {}'.format(con.shape))

        intra1 = con[0:int(con.shape[0] / 2), 0:int(con.shape[1] / 2)]
        inter = con[int(con.shape[0] / 2):int(con.shape[0] / 2) * 2, 0:int(con.shape[1] / 2)]
        intra2 = con[int(con.shape[0] / 2):int(con.shape[0] / 2) * 2, int(con.shape[1] / 2):int(con.shape[1] / 2) * 2]
        plv1, plv2, plv3 = [], [], []

        plv2.append(np.mean(inter[0, :-1]))  # 2 triangular sections have no first row, while rectangular section does
        idx = 1
        while idx < int(con.shape[0] / 2):
            plv1.append(np.mean(intra1[idx, 0:idx]))
            plv2.append(np.mean(inter[idx, :]))
            plv3.append(np.mean(intra2[idx, 0:idx]))
            idx += 1
        plv1 = np.mean(plv1)
        plv2 = np.mean(plv2)
        plv3 = np.mean(plv3)
        self.PLV_intra1.append(plv1)
        self.PLV_inter.append(plv2)
        self.PLV_intra2.append(plv3)
        print("PLV value for intra-brain subject 1: ", plv1)
        print("PLV value for inter-brain: ", plv2)
        print("PLV value for intra-brain subject 2: ", plv3)

        self.plv_plot(con)

    def plv_plot(self, data):
        self.ax[1, 0].cla()  # clears the axes to prepare for new data
        self.ax[1, 1].cla()
        self.ax[1, 2].cla()
        self.ax[1, 3].cla()

        # connectivity matrix
        self.ax[1, 0].axvline(x=data.shape[0] / 2, color='red')  # horizontal and vertical lines for quadrants
        self.ax[1, 0].hlines(int(data.shape[0] / 2), 0, data.shape[0], color='red')
        self.ax[1, 0].imshow(data)  # plots the data

        # average PLV indexes
        x = np.arange(1, len(self.PLV_intra1) + 1)
        self.ax[1, 1].bar(x, self.PLV_intra1, width=0.4, color='red')  # index for plv intra 1
        self.ax[1, 2].bar(x, self.PLV_inter, width=0.4, color='blue')  # index for plv inter
        self.ax[1, 3].bar(x, self.PLV_intra2, width=0.4, color='green')  # index for plv intra 2

        for i, v in enumerate(self.PLV_intra1):  # adds labels to bars
            if v != 0.:
                self.ax[1, 1].text(i + 1 - .2, v, str(round(v, 2)))
        for i, v in enumerate(self.PLV_inter):
            if v != 0.:
                self.ax[1, 2].text(i + 1 - .2, v, str(round(v, 2)))
        for i, v in enumerate(self.PLV_intra2):
            if v != 0.:
                self.ax[1, 3].text(i + 1 - .2, v, str(round(v, 2)))

        self.ax[1, 0].set(title='Connectivity Matrix', xlabel='Channel #', ylabel='Channel #')
        self.ax[1, 1].set(title='PLV values for Subject 1', xlabel='Segment #', ylabel='PLV Value', ylim=(0, 1))
        self.ax[1, 2].set(title='PLV Values for Inter-Brain', xlabel='Segment #', ylabel='PLV Value', ylim=(0, 1))
        self.ax[1, 3].set(title='PLV Values for subject 2', xlabel='Segment #', ylabel='PLV Value', ylim=(0, 1))

    def psd(self, raw, mode):
        if mode == 0:
            x = 2
            y = 0
            channelofint = self.channelofint_psdT1
            foilow = self.foilowT1
            foihigh = self.foihighT1
            text = 'Subject 1 Temporal'
            band = 'BetaGamma'
        elif mode == 1:
            x = 2
            y = 2
            channelofint = self.channelofint_psdF1
            foilow = self.foilowF1
            foihigh = self.foihighF1
            text = 'Subject 1 Frontal'
            band = 'BetaGamma'
        elif mode == 2:
            x = 3
            y = 0
            channelofint = self.channelofint_psdT2
            foilow = self.foilowT2
            foihigh = self.foihighT2
            text = 'Subject 2 Temporal'
            band = 'BetaGamma'
        elif mode == 3:
            x = 3
            y = 2
            channelofint = self.channelofint_psdF2
            foilow = self.foilowF2
            foihigh = self.foihighF2
            text = 'Subject 2 Frontal'
            band = 'BetaGamma'
        elif mode == 4:
            x = 4
            y = 0
            channelofint = self.channelofint_psdA1
            foilow = self.foilowA1
            foihigh = self.foihighA1
            text = 'Subject 1 Occipital'
            band = 'Alpha'
        elif mode == 5:
            x = 4
            y = 2
            channelofint = self.channelofint_psdA2
            foilow = self.foilowA2
            foihigh = self.foihighA2
            text = 'Subject 2 Occipital'
            band = 'Alpha'

        print('\nPlotting Power Spectrum Density {}...'.format(text))

        self.ax[x, y].cla()

        # plot psd
        raw.info['bads'].append(raw.info['ch_names'][self.stim_idx])
        fig, psd, freqs = self.calc_psd(raw, fmax=120, n_fft=int(self.fSamp), picks=channelofint,
                                        ax=self.ax[x, y], area_mode='std', average=True)

        print('PSDPSDPSDPSDPSD', psd, psd.shape)
        # find average psd value in betagamma frequency band for channels of interest
        print('Averaging PSD values between frequencies {0} and {1} Hz and in {2} channels: {3}'
              .format(foilow, foihigh, len(channelofint), channelofint))
        psd = np.delete(psd, np.arange(foihigh - 1, freqs.shape[0]), axis=1)
        psd = np.delete(psd, np.arange(0, foilow - 1), axis=1)
        psd = np.mean(psd, axis=0)
        psd = np.mean(psd, axis=0)
        print(psd)
        if mode == 0:
            self.PSDsT1.append(psd)
            PSDs = self.PSDsT1
        elif mode == 1:
            self.PSDsF1.append(psd)
            PSDs = self.PSDsF1
        elif mode == 2:
            self.PSDsT2.append(psd)
            PSDs = self.PSDsT2
        elif mode == 3:
            self.PSDsF2.append(psd)
            PSDs = self.PSDsF2
        elif mode == 4:
            self.PSDsA1.append(psd)
            PSDs = self.PSDsA1
        elif mode == 5:
            self.PSDsA2.append(psd)
            PSDs = self.PSDsA2
        print('Average {0} Band PSD: {1}dB'.format(band, psd))

        # format psd plot
        self.ax[x, y].text(0.1, .9, 'Average {0} Band PSD: {1}dB'.format(band, round(psd, 2)), ha='left', va='center',
                           transform=self.ax[x, y].transAxes)
        self.ax[x, y].set(title='Power Spectrum Density {}'.format(text), xlabel='Frequency',
                          ylabel='Power Density (dB)')
        self.ax[x, y].axvspan(foilow, foihigh, alpha=0.3, color='green')

        # plot psd averages index
        self.ax[x, y + 1].cla()
        xvals = np.arange(1, len(PSDs) + 1)
        self.ax[x, y + 1].bar(xvals, PSDs, width=0.4, color='orange')
        for i, v in enumerate(PSDs):
            if v != 0.:
                self.ax[x, y + 1].text(i + 1 - .2, v - .4, str(round(v, 2)))
        self.ax[x, y + 1].set(title='Average PSD Index {}'.format(text), xlabel='Segment #', ylabel='Power Density (dB')

    def calc_psd(self, raw, fmin=0, fmax=np.inf, tmin=None, tmax=None, proj=False,
                 n_fft=None, n_overlap=0, reject_by_annotation=True,
                 picks=None, ax=None, color='black', xscale='linear',
                 area_mode='std', area_alpha=0.33, dB=True, estimate='auto',
                 show=True, n_jobs=1, average=False, line_alpha=None,
                 spatial_colors=True, sphere=None, verbose=None):
        """%(calc_psd_doc)s.
        Parameters
        ----------
        raw : instance of Raw
            The raw object.
        fmin : float
            Start frequency to consider.
        fmax : float
            End frequency to consider.
        tmin : float | None
            Start time to consider.
        tmax : float | None
            End time to consider.
        proj : bool
            Apply projection.
        n_fft : int | None
            Number of points to use in Welch FFT calculations.
            Default is None, which uses the minimum of 2048 and the
            number of time points.
        n_overlap : int
            The number of points of overlap between blocks. The default value
            is 0 (no overlap).
        reject_by_annotation : bool
            Whether to omit bad segments from the data while computing the
            PSD. If True, annotated segments with a description that starts
            with 'bad' are omitted. Has no effect if ``inst`` is an Epochs or
            Evoked object. Defaults to True.
        %(plot_psd_picks_good_data)s
        ax : instance of Axes | None
            Axes to plot into. If None, axes will be created.
        %(plot_psd_color)s
        %(plot_psd_xscale)s
        %(plot_psd_area_mode)s
        %(plot_psd_area_alpha)s
        %(plot_psd_dB)s
        %(plot_psd_estimate)s
        %(show)s
        %(n_jobs)s
        %(plot_psd_average)s
        %(plot_psd_line_alpha)s
        %(plot_psd_spatial_colors)s
        %(topomap_sphere_auto)s
        %(verbose)s
        Returns
        -------
        fig : instance of Figure
            Figure with frequency spectra of the data channels.
        psd
        freq

        """
        from mne.viz.utils import _set_psd_plot_params, _plot_psd, _check_psd_fmax, plt_show
        fig, picks_list, titles_list, units_list, scalings_list, ax_list, \
        make_label, xlabels_list = _set_psd_plot_params(raw.info, proj, picks, ax, area_mode)
        _check_psd_fmax(raw, fmax)
        del ax
        psd_list = list()
        if n_fft is None:
            if tmax is None or not np.isfinite(tmax):
                tmax = raw.times[-1]
            tmin = 0. if tmin is None else tmin
            n_fft = min(np.diff(raw.time_as_index([tmin, tmax]))[0] + 1, 2048)
        for picks in picks_list:
            psd, freqs = psd_welch(raw, tmin=tmin, tmax=tmax, picks=picks,
                                   fmin=fmin, fmax=fmax, proj=proj, n_fft=n_fft,
                                   n_overlap=n_overlap, n_jobs=n_jobs,
                                   reject_by_annotation=reject_by_annotation)
            psd_list.append(psd)
        fig = _plot_psd(raw, fig, freqs, psd_list, picks_list, titles_list,
                        units_list, scalings_list, ax_list, make_label, color,
                        area_mode, area_alpha, dB, estimate, average,
                        spatial_colors, xscale, line_alpha, sphere, xlabels_list)
        plt_show(show)

        return fig, psd, freqs

    def moving_average(self, l, avg=True, norm=False, p=False):
        a = []
        if isinstance(l, list):
            if len(l) > 1:
                ls = [i for i, e in enumerate(l) if e != 0]
                for i in ls:
                    a.append(l[i])
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

                return flat
            else:
                return l

    def save_csv(self):
        data = OrderedDict()
        data['segment #'] = np.arange(1, len(self.interfinals)+1)
        data['AEP 1'] = self.AEPs1
        data['AEP 2'] = self.AEPs2
        data['PLV 1'] = self.PLV_intra1
        data['PLV 2'] = self.PLV_intra2
        data['PLV inter'] = self.PLV_inter
        data['PSDa1'] = self.PSDsA1
        data['PSDa2'] = self.PSDsA2
        data['PSDt1'] = self.PSDsT1
        data['PSDt2'] = self.PSDsT2
        data['PSDf1'] = self.PSDsF1
        data['PSDf2'] = self.PSDsF2
        data['flow1'] = self.moving_average(self.intra1finals, norm=True)
        data['flow2'] = self.moving_average(self.intra2finals, norm=True)
        data['teamflow'] = self.moving_average(self.interfinals, norm=True)

        data = pd.DataFrame.from_dict(data=data)

        base = path.basename(self.filepath)
        base = path.splitext(base)[0]
        data.to_csv(path_or_buf=self.path + '/' + base + '_teamflow.csv', index=False)
        print(data)

    def final(self):
        AEPs1 = self.moving_average(self.AEPs1, norm=True)
        AEPs2 = self.moving_average(self.AEPs2, norm=True)

        PSDsA1 = self.moving_average(self.PSDsA1, norm=True)
        PSDsA1raw = self.moving_average(self.PSDsA1, avg=False)
        PSDsA1avg = self.moving_average(self.PSDsA1)
        PSDsA1norm = self.moving_average(self.PSDsA1, avg=False, norm=True)

        PLV_inter = self.moving_average(self.PLV_inter, norm=True)
        PLV_interraw = self.moving_average(self.PLV_inter, avg=False)
        PLV_interavg = self.moving_average(self.PLV_inter)
        PLV_internorm = self.moving_average(self.PLV_inter, avg=False, norm=True)

        AEPs1raw = self.moving_average(self.AEPs1, avg=False)
        AEPs1avg = self.moving_average(self.AEPs1)
        AEPs1norm = self.moving_average(self.AEPs1, avg=False, norm=True)
        PSDsA2 = self.moving_average(self.PSDsA2, norm=True)
        PSDsT1 = self.moving_average(self.PSDsT1, norm=True)
        PSDsT2 = self.moving_average(self.PSDsT2, norm=True)
        PSDsF1 = self.moving_average(self.PSDsF1, norm=True)
        PSDsF2 = self.moving_average(self.PSDsF2, norm=True)
        PLV_intra1 = self.moving_average(self.PLV_intra1, norm=True)
        PLV_intra2 = self.moving_average(self.PLV_intra2, norm=True)

        intra1final = (1 / AEPs1[-1]) + PSDsA1[-1] + (1 / PSDsF1[-1])
        intra2final = (1 / AEPs2[-1]) + PSDsA2[-1] + (1 / PSDsF2[-1])

        interfinal = PLV_inter[-1] + ((PLV_intra1[-1] + PLV_intra2[-1]) / 2) + \
                     ((PSDsT1[-1] + PSDsT2[-1]) / 2)

        interfinal = float(interfinal)
        self.intra1finals.append(intra1final)
        self.intra2finals.append(intra2final)
        self.interfinals.append(interfinal)
        intra1finals = self.intra1finals.copy()
        intra2finals = self.intra2finals.copy()
        interfinals = self.interfinals.copy()
        if self.segment > 1:
            intra1finals.pop(0)
            intra2finals.pop(0)
            interfinals.pop(0)
        intra1finals = self.moving_average(intra1finals, p=True, norm=True)
        intra2finals = self.moving_average(intra2finals, p=True, norm=True)
        interfinals = self.moving_average(interfinals, p=True, norm=True)
        print('Subject 1 intra brain score: ', intra1finals[-1])
        print('Subject 2 intra brain score: ', intra2finals[-1])
        print('Inter-brain score: ', interfinals[-1])

        self.ax2[0, 0].cla()
        self.ax2[0, 1].cla()
        self.ax2[0, 2].cla()
        self.ax2[0, 3].cla()
        self.ax2[1, 0].cla()
        self.ax2[1, 1].cla()
        self.ax2[1, 2].cla()
        self.ax2[1, 3].cla()
        self.ax2[2, 0].cla()
        self.ax2[2, 1].cla()
        self.ax2[2, 2].cla()
        self.ax2[2, 3].cla()

        self.ax2[3, 0].cla()
        self.ax2[3, 1].cla()
        self.ax2[3, 2].cla()
        self.ax2[3, 3].cla()

        x = np.arange(1, len(PSDsA1) + 1)
        self.ax2[0, 0].bar(x, PSDsA1raw, width=0.4, color='red')  # index for plv intra 1
        self.ax2[0, 1].bar(x, PSDsA1norm, width=0.4, color='blue')  # index for plv inter
        self.ax2[0, 2].bar(x, PSDsA1avg, width=0.4, color='green')  # index for plv intra 2
        self.ax2[0, 3].bar(x, PSDsA1, width=0.4, color='yellow')  # index for plv intra 2
        self.ax2[1, 0].bar(x, PLV_interraw, width=0.4, color='red')  # index for plv intra 1
        self.ax2[1, 1].bar(x, PLV_internorm, width=0.4, color='blue')  # index for plv inter
        self.ax2[1, 2].bar(x, PLV_interavg, width=0.4, color='green')  # index for plv intra 2
        self.ax2[1, 3].bar(x, PLV_inter, width=0.4, color='yellow')  # index for plv intra 2
        xa = np.arange(1, len(AEPs1) + 1)
        self.ax2[2, 0].bar(xa, AEPs1raw, width=0.4, color='red')  # index for plv intra 1
        self.ax2[2, 1].bar(xa, AEPs1norm, width=0.4, color='blue')  # index for plv inter
        self.ax2[2, 2].bar(xa, AEPs1avg, width=0.4, color='green')  # index for plv intra 2
        self.ax2[2, 3].bar(xa, AEPs1, width=0.4, color='yellow')  # index for plv intra 2

        xf = np.arange(1, len(interfinals) + 1)
        self.ax2[3, 0].bar(xf, intra1finals, width=0.4, color='red')  # index for plv intra 1
        self.ax2[3, 1].bar(xf, interfinals, width=0.4, color='blue')  # index for plv inter
        self.ax2[3, 2].bar(xf, intra2finals, width=0.4, color='green')  # index for plv intra 2

        self.ax2[0, 0].set(title='RAW PSD', xlabel='Segment #', ylabel='Score')
        self.ax2[0, 1].set(title='Only normalized', xlabel='Segment #', ylabel='Score')
        self.ax2[0, 2].set(title='only moving average', xlabel='Segment #', ylabel='Score')
        self.ax2[0, 3].set(title='normalized and moving average', xlabel='Segment #', ylabel='Score')

        self.ax2[3, 0].set(title='Flow subject 1', xlabel='Segment #', ylabel='Score')
        self.ax2[3, 1].set(title='Team Flow', xlabel='Segment #', ylabel='Score')
        self.ax2[3, 2].set(title='Flow subject 2', xlabel='Segment #', ylabel='Score')
        self.ax2[3, 3].set(title='normalized and moving average', xlabel='Segment #', ylabel='Score')

        self.ax2[0, 0].text(0.1, max(intra1finals) - .1, 'Current score: {}'.format(round(intra1final, 2)),
                            ha='center', va='center', transform=self.ax[0, 0].transAxes, size=12)
        self.ax2[0, 1].text(0.1, max(interfinals) - .1, 'Current score: {}'.format(round(interfinal, 2)), ha='center',
                            va='center', transform=self.ax[0, 1].transAxes, size=12)
        self.ax2[0, 2].text(0.1, max(intra2finals) - .1, 'Current score: {}'.format(round(intra2final, 2)),
                            ha='center', va='center', transform=self.ax[0, 2].transAxes, size=12)

    def plot(self, raw):
        print('\nPlotting Averaged Potencial...')
        data = raw.get_data()
        self.ax[0, 3].cla()  # clears the axes to prepare for new data
        x = np.arange(0, 5, 1 / 2048)  # generate x axis scale (from 0 to 1 second)
        row_idx = np.array(
            [1, 2, 3, 33, 34, 52, 65, 66, 87, 97, 98, 110, 111, 112])  # picks add 128 to each number for participant 2
        print("averaging {0} channels: {1}".format(len(row_idx), row_idx))
        data = data[row_idx, :]
        dat = data.mean(axis=0)
        dat = dat[:10240]
        dat = dat.transpose()
        # plots the data against time
        self.ax[0, 3].plot(x, dat)
        self.ax[0, 3].set(title='Placeholder Plot', xlabel='Time', ylabel='Volts')
