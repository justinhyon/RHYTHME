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
        # info_stim = mne.create_info(['stim'], sfreq=fSamp, ch_types=['stim'])
        # raw_stim = RawArray(np.asarray(stim_values).reshape(1, len(stim_values)), info_stim)
        # raw = raw.add_channels([raw_stim], force_update_info=True)
        # raw._data[stim_idx] = stim_values  # restores events destroyed by filtering
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
                # aep_plot(data=np.zeros(round((posttrig + pretrig) * fsample)), mode=mode)
                terminate = True

                dat = np.zeros((len(channelofint), int(np.round((pretrig * fsample) + (posttrig * fsample)))))

            if lastSamp > blocksize:
                print("Event with value {} is at extreme end of segment. No calculation performed, marker added."
                      .format(epoeventval))
                if firstloop and not goodevent:
                    aeplist.append(0)
                    aepxvallist.append(segment)
                exE_AEPlist.append(segment)
                terminate = True

                dat = np.zeros((len(channelofint), int(np.round((pretrig * fsample) + (posttrig * fsample)))))

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

        print('aepexblist for seg ', exB_AEPlist)
        return aeplist, aepxvallist, exB_AEPlist, exE_AEPlist, dat

    else:
        # print("\nAEP Subject {0}:\nno events with value {1} found in this segment, AEP calculation not performed"
        #       .format(mode + 1, epoeventval))
        print("\nno events with value", epoeventval, "found in this segment, AEP calculation not performed")
        aeplist.append(0.)
        aepxvallist.append(segment)
        return aeplist, aepxvallist, exB_AEPlist, exE_AEPlist, np.zeros((len(channelofint), int(np.round((pretrig * fsample) + (posttrig * fsample)))))



def aep_plot(ax, data, participant, fsample, aeplist, aepxvallist, exB_AEPlist, exE_AEPlist, pretrig, posttrig, segment):
    import numpy as np

    print(participant)
    if participant == 1:
        y = 0
    elif participant == 2:
        y = 2

    AEPs = aeplist


    pretrig = pretrig
    posttrig = posttrig
    # if narticipant == 1:
    #     y = 0
    #     AEPs = aeplist[0]
    #     exB_AEPs = exB_AEPs1
    #     exE_AEPs = exE_AEPs1
    #     AEPxval = AEP1xval
    #     pretrig = pretrig1
    #     posttrig = posttrig1
    # elif participant == 2:
    #     y = 2
    #     AEPs = AEPs2
    #     exB_AEPs = exB_AEPs2
    #     exE_AEPs = exE_AEPs2
    #     AEPxval = AEP2xval
    #     pretrig = pretrig2
    #     posttrig = posttrig2

    # data = raw.get_data()
    dat = data.transpose()


    ax[0, y].cla()  # clears the axes to prepare for new data
    ax[0, y + 1].cla()
    x = np.arange((-1 * pretrig), posttrig, (posttrig + pretrig) / int(
        (posttrig + pretrig) * fsample))  # generate x axis values (time)

    # plots standard deviation if data is not 0
    if data.ndim > 1:

        sdAEPamp = data.std(axis=0)[:-1]

        dat = data.mean(axis=0)[:-1]

        ax[0, y].fill_between(x, dat + sdAEPamp, dat - sdAEPamp, facecolor='red', alpha=0.3)

    # plots the data against time
    ax[0, y].plot(x, dat, color='black')

    # format AEP vs time plot
    ax[0, y].axvline(x=0, color='red')
    ax[0, y].axvspan(.110, .150, alpha=0.3, color='green')
    ax[0, y].axvspan(.210, .250, alpha=0.3, color='green')
    ax[0, y].axvspan(.310, .350, alpha=0.3, color='green')
    ax[0, y].hlines(0, -1 * pretrig, posttrig)
    ax[0, y].set_title('Subject {0} AEP'.format(participant + 1))
    ax[0, y].set_xlabel('Time (s)')
    ax[0, y].set_ylabel('Volts')

    # generate plot of all AEP peak indexes
    x = aepxvallist

    ax[0, y + 1].bar(x, AEPs, width=0.4)
    ax[0, y + 1].bar(exB_AEPlist, .000001, width=0.2, alpha=.5)
    ax[0, y + 1].bar(exE_AEPlist, -.000001, width=0.2, alpha=.5)
    # ax[0, y + 1].axis(xmin=-3)
    for i, v in zip(aepxvallist, AEPs):
        if v != 0.:
            ax[0, y + 1].text(i - .5, v, str(round(v, 10)))
    ax[0, y + 1].hlines(0, 0, segment)
    ax[0, y + 1].set_title('Subject {} AEP Peak Amplitude Index'.format(participant + 1))
    ax[0, y + 1].set_xlabel('Segment #')
    ax[0, y + 1].set_ylabel('AEP Peak Index (V)')

    return ax
