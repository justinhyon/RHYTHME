def realtime_process(self, delay, badchans, nchansparticipant, trigchan, featurenames, channelofints,
                     foilows, foihighs, blocksize_sec, num_plvsimevents, aep_data, units, remfirstsample, exp_name):
    '''
    numparticipants         number of participants
    channelofints           list of lists containing channels corresponding to different regions of the brain
                            for which psd is calculated
    foilows                 list
    foihighs                list
    aep_data                list of lists [[participant 1 info], [participant 2 info],...]
    '''
    self.ftc = FieldTrip.Client()

    self.prevsamp = 0

    # if filepath.endswith('.bdf'):
    #     fullraw = read_raw_bdf(filepath, preload=True)
    # if filepath.endswith('.set'):
    #     fullraw = read_raw_eeglab(filepath, preload=True)
    #
    #
    # fulldata = fullraw._data

    # self.filepath = filepath
    # print("Total Samples Received = {}".format(fulldata.shape[1]))

    # Initiate list of lists to store psd values for one participant
    psds = [[] for i in range(len(channelofints))]
    plv_intra1 = []
    plv_inter = []
    plv_intra2 = []
    aeps = [[] for i in range(len(aep_data))]
    exB_AEPlist = [[] for i in range(len(aep_data))]
    exE_AEPlist = [[] for i in range(len(aep_data))]
    aepxvallist = [[] for i in range(len(aep_data))]
    # intra1_tm = []
    # intra2_tm = []
    # inter_tm = []

    data_dict = OrderedDict()
    data_dict['Intra 1'] = []
    data_dict['Intra 2'] = []
    data_dict['Inter'] = []

    # For plotting
    if self.plotpref != 'none':
        if self.plotpref == 'participant' or self.plotpref == 'both':
            # plt.close(self.fig)
            fig, ax = plt.subplots(1, 3, squeeze=False,
                                   figsize=(self.windowsize * 3, self.windowsize))
    print("REALTIME ANALYSIS MODE")
    print("Waiting for first data segment from port: ", self.dataport)
    loop = True
    initial_wait = True

    while True:
        wait_time = 0
        while True:
            self.ftc.connect('localhost', self.dataport)  # might throw IOError
            H = self.ftc.getHeader()
            fsample = H.fSample
            blocksize = round(blocksize_sec * fsample)
            currentsamp = H.nSamples

            if H.nSamples == self.prevsamp + blocksize:
                initial_wait = False
                break
            else:
                time.sleep(delay)
                if not initial_wait:
                    wait_time += 1
                if wait_time > blocksize_sec * (1 / delay):
                    loop = False
                    print("\n", "-" * 60)
                    print("\nEXPERIMENT ENDED: After {0} seconds, not enough data was received to fill one block. "
                          "Number of unprocessed samples: {1}"
                          .format(blocksize_sec, H.nSamples - self.prevsamp))
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

        # if currentsamp != self.prevsamp + blocksize:
        #     print('received inconsistent number of samples: {}. experiment ending'.format(currentsamp - self.prevsamp))
        #     loop = False

        if loop:
            print("\n", "-" * 60)
            print('\nCONNECTED: Header retrieved for segment: ' + str(self.segment))
            print("Trying to read from sample {0} to {1}. Samples per block: {2}".format(self.prevsamp, currentsamp - 1,
                                                                                         blocksize))
            print("Total Samples Received = {}".format(currentsamp))

            segmentdata = self.ftc.getData(
                index=(self.prevsamp, currentsamp - 1)).transpose()  # receive data from buffer

            self.prevsamp = currentsamp

            print('Samples retrieved for segment: ' + str(self.segment))
            print('Data shape (nChannels, nSamples): {}'.format(segmentdata.shape))
            print(segmentdata)

            stimvals = segmentdata[int(trigchan), :] * 1000000
            segmentdata = np.delete(segmentdata, int(trigchan), axis=0)

            participant_data = []
            idx = 0
            while idx < segmentdata.shape[0]:
                participant_data.append(segmentdata[int(idx):idx + nchansparticipant, :])
                idx += nchansparticipant

            # D1 = fulldata[0:128, self.prevsamp:currentsamp]
            # D2 = fulldata[128:256, self.prevsamp:currentsamp]

            participant_raws = []
            for i, participant in enumerate(participant_data):
                raw = self.make_raw(participant, stimvals, fsample, units)
                raw = self.preprocess_raw(raw, badchans[i])
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
            info_stim = mne.create_info(['stim'], sfreq=fsample, ch_types=['stim'])
            raw_stim = RawArray(np.asarray(stimvals).reshape(1, len(stimvals)), info_stim)
            raw = raw.add_channels([raw_stim], force_update_info=True)

            time2 = timer()
            print("Time to recieve data and create MNE raw: {}".format(time2 - time1))
            # print("\n", "1-" * 60)
            self.stim_idx = self.TrigChan
            self.stim_values = stimvals
            print('STIM CHANNEL: ', self.stim_idx, self.stim_values)

            # Extract features
            ################## PSDs #################################################
            time1 = timer()

            for idx, chans in enumerate(channelofints):
                i = np.mod(idx, len(
                    foilows))  # Adjust for the fact that channels are different amongst subjects, but fois are the same
                psds[idx] = psd(raw, psds[idx], chans, foilows[i], foihighs[i], fsample)

            time2 = timer()
            print("Time to compute 6 PSDs: {}".format(time2 - time1))
            print("\n", "2-" * 60)

            ############################################################################

            ########################### AEPs ##########################################
            time1 = timer()

            for idx, participant in enumerate(aep_data):
                channelofint = participant[0]
                epoeventval = participant[1]
                pretrig = participant[2]
                posttrig = participant[3]
                aeps[idx], exB_AEPlist[idx], exE_AEPlist[idx], aepxvallist[idx], = aep(raw, aeps[idx],
                                                                                       exB_AEPlist[idx],
                                                                                       exE_AEPlist[idx],
                                                                                       aepxvallist[idx], fsample,
                                                                                       blocksize, channelofint,
                                                                                       epoeventval, pretrig,
                                                                                       posttrig, stimvals,
                                                                                       self.segment)
                print(aeps[idx])
            time2 = timer()
            print("Time to compute 2 AEPs: {}".format(time2 - time1))
            print("\n", "3-" * 60)

            ##################################PLVs#####################################
            time1 = timer()

            # numblocks = 10
            intra1, inter, intra2 = plv(raw, self.segment, blocksize, fsample, num_plvsimevents)
            plv_intra1.append(intra1)
            plv_inter.append(inter)
            plv_intra2.append(intra2)

            time2 = timer()
            print("Time to compute PLV: {}".format(time2 - time1))
            print("\n", "4-" * 60)

            ############################################################################

            ###############Make Datastructure to make plotting easier###################
            # AEPs
            for idx, participant in enumerate(aeps):
                name = 'AEP ' + str(idx + 1)
                data_dict[name] = participant
                # namenorm = 'AEP ' + str(idx + 1) + ' norm'
                # data_dict[namenorm] = self.moving_average(participant, norm=True, rmzero=False)

            # PSDs, band names are passed from config file
            for idx, name in enumerate(featurenames):
                data_dict[name] = psds[idx]

            data_dict['PLV 1'] = plv_intra1
            data_dict['PLV 2'] = plv_intra2
            data_dict['PLV inter'] = plv_inter

            intra1_tm, intra2_tm, inter_tm = self.calculate_tmflow(data_dict)

            data_dict['Intra 1'] = (intra1_tm)
            data_dict['Intra 2'] = (intra2_tm)
            data_dict['Inter'] = (inter_tm)

            print(data_dict.values())
            if (self.segment == 2) & remfirstsample:
                for key, value in data_dict.items():
                    if key in ['Intra 1', 'Intra 2', 'Inter']:
                        print("yote")
                        valuecopy = value
                        valuecopy[0] = 0.0
                        print(value)
                        print(valuecopy)
                        data_dict[key] = valuecopy
            print(data_dict.values())
            ############################################################################

            if self.plotpref != 'none':
                print("\nGenerating Plots...")
                time1 = timer()
                if self.plotpref == 'participant' or self.plotpref == 'both':
                    ax[0, 0].cla()
                    ax[0, 1].cla()
                    ax[0, 2].cla()

                    x = np.arange(1, len(data_dict['Intra 1']) + 1)

                    ax[0, 0].bar(x, data_dict['Intra 1'], width=0.4, color='red')  # index for flow intra 1
                    ax[0, 1].bar(x, data_dict['Inter'], width=0.4, color='blue')  # index for plv inter
                    ax[0, 2].bar(x, data_dict['Intra 2'], width=0.4, color='green')  # index for plv intra 2

                    # ax[0, 0].plot(x, data_dict['Intra 1'], color='red')  # index for flow intra 1
                    # ax[0, 1].plot(x, data_dict['Inter'], color='blue')  # index for plv inter
                    # ax[0, 2].plot(x, data_dict['Intra 2'], color='green')  # index for plv intra 2

                    ax[0, 0].set_ylim(0, 10)
                    ax[0, 1].set_ylim(0, 10)
                    ax[0, 2].set_ylim(0, 10)

                    ax[0, 0].set(title='Intra 1', xlabel='Segment #', ylabel='Score')
                    ax[0, 1].set(title='Inter', xlabel='Segment #', ylabel='Score')
                    ax[0, 2].set(title='Intra 2', xlabel='Segment #', ylabel='Score')
                    # self.fig.suptitle('Data Segment {}'.format(self.segment), fontsize=16)
                    plt.pause(0.05)
                    # plt.show()

            # self.final()
            # if self.plotpref == 'participant':
            #     plt.close(fig=self.fig)
            #
            # if self.plotpref != 'none':
            #     plt.pause(.005)
            #     plt.draw()

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
            self.save_csv(data_dict, exp_name)
            break