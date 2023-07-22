def aep(raw, aeplist, exB_AEPlist, exE_AEPlist, aepxvallist, fsample, blocksize, channelofint, epoeventval,
        pretrig, posttrig, stim_values, segment):

    import numpy as np

    if epoeventval in stim_values:
        indexes = [i for i, e in enumerate(stim_values) if e == epoeventval]
        # print(
        #     "\nAEP Subject {3}:\n{0} event(s) with value {1} found in this segment at location(s) {2}, " \
        #     "performing AEP calculation..."
        #         .format(len(indexes), epoeventval, indexes, mode + 1))
        print(
            "\nAEP :\n{0} event(s) with value {1} found in this segment at location(s) {2}, " \
            "performing AEP calculation..."
                .format(len(indexes), epoeventval, indexes))

        # band pass filter
        raw = raw.copy().filter(l_freq=3.0, h_freq=7.0, picks=None, filter_length='auto', l_trans_bandwidth='auto',
                                h_trans_bandwidth='auto', n_jobs=1, method='iir', iir_params=None, phase='zero',
                                fir_window='hamming', fir_design='firwin',
                                skip_by_annotation=('edge', 'bad_acq_skip'), pad='reflect_limited', verbose=None)
        # Add stim channel back in
        # info_stim = mne.create_info(['stim'], sfreq=self.fSamp, ch_types=['stim'])
        # raw_stim = RawArray(np.asarray(self.stim_values).reshape(1, len(self.stim_values)), info_stim)
        # raw = raw.add_channels([raw_stim], force_update_info=True)
        # raw._data[self.stim_idx] = self.stim_values  # restores events destroyed by filtering
        data = raw.get_data()

        goodevent = False
        for testevent in indexes:
            testfirstSamp = round(testevent - (pretrig * fsample))
            testlastSamp = round(testevent + (posttrig * fsample))
            if (0 < testfirstSamp) & (testlastSamp < blocksize):
                goodevent = True

        firstloop = True
        append_aep = False
        mAEP1amp = []


        for event in indexes:
            terminate = False
            firstSamp = round(event - (pretrig * fsample))
            lastSamp = round(event + (posttrig * fsample))

            if firstSamp < 0:
                print(
                    "Event with value {} is at extreme beginning of segment. No calculation performed, "
                    "marker added.".format(epoeventval))
                if firstloop and not goodevent:
                    aeplist.append(0)
                    aepxvallist.append(segment)
                exB_AEPlist.append(segment)
                # self.aep_plot(data=np.zeros(round((posttrig + pretrig) * fsample)), mode=mode)
                terminate = True

            if lastSamp > blocksize:
                print("Event with value {} is at extreme end of segment. No calculation performed, marker added."
                      .format(epoeventval))
                if firstloop and not goodevent:
                    aeplist.append(0)
                    aepxvallist.append(segment)
                exE_AEPlist.append(segment)
                terminate = True

            if not terminate:
                append_aep = True
                dat = data
                dat = np.delete(dat, np.arange(lastSamp - 1, blocksize), axis=1)
                dat = np.delete(dat, np.arange(0, firstSamp - 1), axis=1)
                row_idx = np.array(channelofint)  # picks; add 128 to each number for participant 2
                print("using {0} channels: {1}".format(len(row_idx), row_idx))
                dat = dat[row_idx, :]

                # baseline correction
                temp_dat = dat[:, 0:int(pretrig * fsample)]
                for idx, chan in enumerate(temp_dat):
                    ch_mean = np.mean(chan)
                    dat[idx] -= ch_mean

                # Define peaks of interest: N1 (110 ? 150 ms), P1 (210 ? 250 ms), and N2 (310 ? 350 ms).
                times = np.array([[.110, .150], [.210, .250], [.310, .350]])
                for a, b in enumerate(times):
                    for c, d in enumerate(b):
                        times[a, c] = round(d * fsample + pretrig * fsample)
                times = times.astype(int)
                # calculate AEP peak amplitude
                AEPamps = []
                for ch in dat:
                    N1 = sum(ch[times[0, 0]:times[0, 1]]) / (times[0, 1] - times[0, 0])
                    P1 = sum(ch[times[1, 0]:times[1, 1]]) / (times[1, 1] - times[1, 0])
                    N2 = sum(ch[times[2, 0]:times[2, 1]]) / (times[2, 1] - times[2, 0])
                    average = (-N1 + P1 - N2) / 3  # Confirm this equation
                    AEPamps.append(average)
                mAEPamp = sum(AEPamps) / len(AEPamps)
                print("Average AEP peak amplitude: ", mAEPamp)
                aepxvallist.append(segment)
                mAEP1amp.append(mAEPamp)

            firstloop = False

        if append_aep and len(mAEP1amp) > 0:
            aeplist.append(np.mean(mAEP1amp))

        return aeplist, aepxvallist, exB_AEPlist, exE_AEPlist

    else:
        # print("\nAEP Subject {0}:\nno events with value {1} found in this segment, AEP calculation not performed"
        #       .format(mode + 1, epoeventval))
        print("\nno events with value", epoeventval, "found in this segment, AEP calculation not performed")
        aeplist.append(0.)
        aepxvallist.append(segment)
        return aeplist, aepxvallist, exB_AEPlist, exE_AEPlist



# def aep_plot(self, data, mode, fsample):
#     if mode == 0:
#         y = 0
#         AEPs = self.AEPs1
#         exB_AEPs = self.exB_AEPs1
#         exE_AEPs = self.exE_AEPs1
#         AEPxval = self.AEP1xval
#         pretrig = self.pretrig1
#         posttrig = self.posttrig1
#     elif mode == 1:
#         y = 2
#         AEPs = self.AEPs2
#         exB_AEPs = self.exB_AEPs2
#         exE_AEPs = self.exE_AEPs2
#         AEPxval = self.AEP2xval
#         pretrig = self.pretrig2
#         posttrig = self.posttrig2
#     dat = data.transpose()
#
#     if self.plotpref != 'none':
#         self.ax[0, y].cla()  # clears the axes to prepare for new data
#         self.ax[0, y + 1].cla()
#         x = np.arange((-1 * pretrig), posttrig, (posttrig + pretrig) / int(
#             (posttrig + pretrig) * fsample))  # generate x axis values (time)
#
#         # plots standard deviation if data is not 0
#         if data.ndim > 1:
#             sdAEPamp = dat.std(axis=1)
#             dat = data.mean(axis=0)
#             self.ax[0, y].fill_between(x, dat + sdAEPamp, dat - sdAEPamp, facecolor='red', alpha=0.3)
#
#         # plots the data against time
#         self.ax[0, y].plot(x, dat, color='black')
#
#         # format AEP vs time plot
#         self.ax[0, y].axvline(x=0, color='red')
#         self.ax[0, y].axvspan(.110, .150, alpha=0.3, color='green')
#         self.ax[0, y].axvspan(.210, .250, alpha=0.3, color='green')
#         self.ax[0, y].axvspan(.310, .350, alpha=0.3, color='green')
#         self.ax[0, y].hlines(0, -1 * pretrig, posttrig)
#         self.ax[0, y].set_title('Subject {0} AEP'.format(mode + 1))
#         self.ax[0, y].set_xlabel('Time (s)')
#         self.ax[0, y].set_ylabel('Volts')
#
#         # generate plot of all AEP peak indexes
#         x = AEPxval
#         print(AEPs, len(x), len(AEPs))
#         self.ax[0, y + 1].bar(x, AEPs, width=0.4)
#         self.ax[0, y + 1].bar(exB_AEPs, .000001, width=0.2, alpha=.5)
#         self.ax[0, y + 1].bar(exE_AEPs, -.000001, width=0.2, alpha=.5)
#         # self.ax[0, y + 1].axis(xmin=-3)
#         for i, v in zip(AEPxval, AEPs):
#             if v != 0.:
#                 self.ax[0, y + 1].text(i - .5, v, str(round(v, 10)))
#         self.ax[0, y + 1].hlines(0, 0, self.segment)
#         self.ax[0, y + 1].set_title('Subject {} AEP Peak Amplitude Index'.format(mode + 1))
#         self.ax[0, y + 1].set_xlabel('Segment #')
#         self.ax[0, y + 1].set_ylabel('AEP Peak Index (V)')
