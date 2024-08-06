import statistics
from scipy.signal import find_peaks
def ERP(raw, ERPlist, exB_ERPlist, exE_ERPlist, ERPxvallist, fsample, blocksize, channelofint, epoeventval,

        pretrig, posttrig, stim_values, segment, bands, signs, filter_range, calc_met, mean):

    import numpy as np

    if epoeventval in stim_values:
        indexes = [i for i, e in enumerate(stim_values) if e == epoeventval]
        # print(
        #     "\nERP Subject {3}:\n{0} event(s) with value {1} found in this segment at location(s) {2}, " \
        #     "performing ERP calculation..."
        #         .format(len(indexes), epoeventval, indexes, mode + 1))
        print(
            "\nERP :\n{0} event(s) with value {1} found in this segment at location(s) {2}, " \
            "performing ERP calculation..."
                .format(len(indexes), epoeventval, indexes))

        # band pass filter
        # print(raw.get_data)
        if filter_range:
            raw = raw.copy().filter(l_freq=filter_range[0], h_freq=filter_range[1], picks=None, filter_length='auto', l_trans_bandwidth='auto',
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
        append_ERP = False
        mERP1amp = []
        peak_locs=[]

        for event in indexes:
            terminate = False
            firstSamp = round(event - (pretrig * fsample))
            lastSamp = round(event + (posttrig * fsample))

            if firstSamp < 0:
                print(
                    "Event with value {} is at extreme beginning of segment. No calculation performed, "
                    "marker added.".format(epoeventval))
                if firstloop and not goodevent:
                    ERPlist.append(0)
                    ERPxvallist.append(segment)
                exB_ERPlist.append(segment)
                # ERP_plot(data=np.zeros(round((posttrig + pretrig) * fsample)), mode=mode)
                terminate = True

                dat = np.zeros((len(channelofint), int(np.round((pretrig * fsample) + (posttrig * fsample)))))

            if lastSamp > blocksize:
                print("Event with value {} is at extreme end of segment. No calculation performed, marker added."
                      .format(epoeventval))
                print(lastSamp, blocksize, event, posttrig, fsample)
                if firstloop and not goodevent:
                    ERPlist.append(0)
                    ERPxvallist.append(segment)
                exE_ERPlist.append(segment)
                terminate = True

                dat = np.zeros((len(channelofint), int(np.round((pretrig * fsample) + (posttrig * fsample)))))

            if not terminate:
                append_ERP = True
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
                times = np.array(bands)
                for i, band in enumerate(times):
                    for j, time in enumerate(band):
                        times[i, j] = round(time * fsample + pretrig * fsample)
                times = times.astype(int)  # now in terms of samples
                # print(times)
                peak_locs = []
                # calculate ERP peak amplitude

                if calc_met == 'bands':
                    ERPamps = []
                    for ch in dat:
                        bandavgs = []
                        for i in range(len(times)):
                            thisavg = sum(ch[times[i, 0]:times[i, 1]]) / (times[i, 1] - times[i, 0])
                            if signs[i] == '-':
                                thisavg = -1 * thisavg
                            elif signs[i] == 'abs':
                                thisavg = np.abs(thisavg)
                            bandavgs.append(thisavg)
                            # P1 = sum(ch[times[i, 0]:times[i, 1]]) / (times[i, 1] - times[i, 0])
                            # N2 = sum(ch[times[i, 0]:times[i, 1]]) / (times[i, 1] - times[i, 0])
                        average = np.mean(bandavgs)
                        ERPamps.append(average)
                    mERPamp = sum(ERPamps) / len(ERPamps)
                elif calc_met == 'peaks' and mean:
                    peak_amps = []
                    mdat = np.mean(dat, 0)
                    # for mdat in dat:
                    peaks, props = find_peaks(mdat, height=0)
                    amps = props['peak_heights']
                    # peak_amps.append(amps.tolist())
                    peak_locs.append(np.transpose([peaks, amps]).tolist())

                    negpeaks, negprops = find_peaks(-mdat, height=0)
                    negamps = negprops['peak_heights']
                    peak_amps += (amps.tolist() + negamps.tolist())
                    for i, a in enumerate(negamps):
                        negamps[i] = -a
                    peak_locs.append(np.transpose([negpeaks, negamps]).tolist())
                    # print('test', peak_locs)

                    mERPamp = np.mean(peak_amps)
                    print("Average ERP peak amplitude: ", mERPamp)
                elif calc_met == 'peaks' and not mean:
                    peak_amps = []
                    mdat = np.mean(dat, 0)
                    # for mdat in dat:
                    peaks, props = find_peaks(mdat, height=0)
                    amps = props['peak_heights']
                    # peak_amps.append(amps.tolist())
                    peak_locs.append(np.transpose([peaks, amps]).tolist())

                    negpeaks, negprops = find_peaks(-mdat, height=0)
                    negamps = negprops['peak_heights']
                    peak_amps += (amps.tolist() + negamps.tolist())
                    for i, a in enumerate(negamps):
                        negamps[i] = -a
                    peak_locs.append(np.transpose([negpeaks, negamps]).tolist())
                    # print('test', peak_locs)

                    mERPamp = np.mean(peak_amps)
                    print("Average ERP peak amplitude: ", mERPamp)
                else:
                    print("Invalid settings for ERP")
                    exit(1)
                ERPxvallist.append(segment)
                mERP1amp.append(mERPamp)

                ERPlist.append(mERPamp)

            firstloop = False

        # if append_ERP and len(mERP1amp) > 0:
        #     ERPlist.append(np.mean(mERP1amp))

        print('ERPexblist for seg ', exB_ERPlist)
        print('ERPexElist for seg ', exE_ERPlist)
        # print(ERPxvallist)
        # print(ERPlist)
        return ERPlist, ERPxvallist, exB_ERPlist, exE_ERPlist, dat, peak_locs

    else:
        # print("\nERP Subject {0}:\nno events with value {1} found in this segment, ERP calculation not performed"
        #       .format(mode + 1, epoeventval))
        print("\nno events with value", epoeventval, "found in this segment, ERP calculation not performed")
        ERPlist.append(0.)
        ERPxvallist.append(segment)
        return ERPlist, ERPxvallist, exB_ERPlist, exE_ERPlist, np.zeros(
            (len(channelofint), int(np.round((pretrig * fsample) + (posttrig * fsample))))), []


def ERP_plot(ax, data, participant, fsample, ERPlist, pretrig, posttrig, segment,
             location, bands, calc_met, mean, max_val):
    import numpy as np

    x = location[0]
    y = location[1]
    # print(participant)
    # if participant == 1:
    #     y = 0
    # elif participant == 2:
    #     y = 2

    # ERPs = ERPlist
    #
    # pretrig = pretrig
    # posttrig = posttrig
    # if narticipant == 1:
    #     y = 0
    #     ERPs = ERPlist[0]
    #     exB_ERPs = exB_ERPs1
    #     exE_ERPs = exE_ERPs1
    #     ERPxval = ERP1xval
    #     pretrig = pretrig1
    #     posttrig = posttrig1
    # elif participant == 2:
    #     y = 2
    #     ERPs = ERPs2
    #     exB_ERPs = exB_ERPs2
    #     exE_ERPs = exE_ERPs2
    #     ERPxval = ERP2xval
    #     pretrig = pretrig2
    #     posttrig = posttrig2

    # data = raw.get_data()
    # if mean:
    #     dat = mean(data)
    # dat = 1000 * dat

    ax[x, y].cla()  # clears the axes to prepare for new data
    # ax[x, y + 1].cla()
    xval = np.arange((-1 * pretrig), posttrig, (posttrig + pretrig) / np.round(
        (posttrig + pretrig) * fsample))  # generate x axis values (time)
    heatmap = None
    if mean:
        # plots standard deviation if data is not 0
        if data.ndim > 1:
            sdERPamp = data.std(axis=0)#[:-1]

            dat = data.mean(axis=0)#[:-1]
            # print(data.shape,xval.shape,dat.shape,sdERPamp.shape)
            ax[x, y].fill_between(xval, dat + sdERPamp, dat - sdERPamp, facecolor='red', alpha=0.3)

        # plots the data against time
        ax[x, y].plot(xval, dat, color='black')
        if calc_met == 'bands':
            for band in bands:
                ax[x, y].axvspan(band[0], band[1], alpha=0.3, color='green')

        # format ERP vs time plot
        ax[x, y].axvline(x=0, color='red')

            # ax[x, y].axvspan(.210, .250, alpha=0.3, color='green')
            # ax[x, y].axvspan(.310, .350, alpha=0.3, color='green')
        ax[x, y].hlines(0, -1 * pretrig, posttrig)
        ax[x, y].set_title('Subject {0} ERP'.format(participant + 1))
        ax[x, y].set_xlabel('Time (s)')
        ax[x, y].set_ylabel('uV')
    elif not mean:
        # from mpl_toolkits.axes.grid1 import make_axes_locatable
        heatmap = ax[x, y].imshow(data, cmap='hot', interpolation='nearest', aspect='auto', extent=[xval[0],
                                                                    xval[-1], 1, len(data)], vmin=0, vmax=max_val)

        # ax[x, y].colorbar()

    # # generate plot of all ERP peak indexes
    # x = ERPxvallist
    #
    # ax[0, y + 1].bar(x, ERPs, width=0.4)
    # ax[0, y + 1].bar(exB_ERPlist, .000001, width=0.2, alpha=.5)
    # ax[0, y + 1].bar(exE_ERPlist, -.000001, width=0.2, alpha=.5)
    # # ax[0, y + 1].axis(xmin=-3)
    # for i, v in zip(ERPxvallist, ERPs):
    #     if v != 0.:
    #         ax[0, y + 1].text(i - .5, v, str(round(v, 10)))
    # ax[0, y + 1].hlines(0, 0, segment)
    # ax[0, y + 1].set_title('Subject {} ERP Peak Amplitude Index'.format(participant + 1))
    # ax[0, y + 1].set_xlabel('Segment #')
    # ax[0, y + 1].set_ylabel('ERP Peak Index (V)')

    return ax, heatmap


def ERP_idx_plot(ax, participant, ERPlist, ERPxvallist, exB_ERPlist, exE_ERPlist, segment, location):
    # if participant == 1:
    #     y = 0
    # elif participant == 2:
    #     y = 2

    x = location[0]
    y = location[1]

    ax[x,y].cla()
    ERPs = ERPlist
    # generate plot of all ERP peak indexes
    # xval = ERPxvallist

    if ERPs == []:
        markheight = .000001
    else:
        markheight = statistics.mean(ERPs)

    ax[x, y ].bar(ERPxvallist, ERPs, width=0.4)
    ax[x, y ].bar(exB_ERPlist, markheight, width=0.2, alpha=.5)
    ax[x, y ].bar(exE_ERPlist, -1*markheight, width=0.2, alpha=.5)
    # ax[x, y].plot(ERPxvallist, ERPs)
    # ax[x, y].plot(exB_ERPlist, markheight, width=0.2, alpha=.5)
    # ax[x, y].bar(exE_ERPlist, -1 * markheight, width=0.2, alpha=.5)
    # ax[0, y + 1].axis(xmin=-3)
    # for i, v in zip(ERPxvallist, ERPs):
    #     if v != 0.:
    #         ax[x, y ].text(i - .5, v, str(round(v, 10)))
    ax[x, y ].hlines(0, 0, segment)
    ax[x, y ].set_title('Subject {} ERP Peak Amplitude Index'.format(participant + 1))
    ax[x, y ].set_xlabel('Segment #')
    ax[x, y ].set_ylabel('ERP Peak Index (uV)')

    return ax
