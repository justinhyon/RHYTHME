def psd(raw, psd_list, chan, foilow, foihigh, fSamp):
    import numpy as np
    from scipy.signal import welch
    # if self.plotpref != 'none':
    #     print('\nPlotting Power Spectrum Density {}...'.format(text))
    #     self.ax[x, y].cla()
    #     # Plot PSD
    #     raw.info['bads'].append(raw.info['ch_names'][self.stim_idx])
    #     fig, psd, freqs = self.calc_psd(raw, fmax=120, n_fft=int(self.fSamp), picks=channelofint,
    #                                     ax=self.ax[x, y], area_mode='std', average=True)
    # else:
    #     raw.info['bads'].append(raw.info['ch_names'][self.stim_idx])
    #     fig, psd, freqs = self.calc_psd(raw, fmax=120, n_fft=int(self.fSamp), picks=channelofint,
    #                                     area_mode='std', average=True)

    data = raw._data
    data = data[chan, :]
    print("chan: ", chan)
    # data = data * 1000000
    print('Data shape (nChannels, nSamples): {}'.format(data.shape))

    f, pxx1 = welch(data, fs=fSamp, window='hamming', nperseg=fSamp,
                    noverlap=0, nfft=fSamp, detrend=False)
    psd = np.transpose(pxx1)
    psd_db = np.multiply(10, np.log10(psd))
    freqs = range(foilow, foihigh, 1)
    sub_psd_db = psd_db[freqs, :]
    psd_list.append(np.mean(np.mean(sub_psd_db, axis=-1), axis=-1))

    return psd_list

    # if self.plotpref != 'none':
    #     # format psd plot
    #     self.ax[x, y].text(0.1, .9, 'Average {0} Band PSD: {1}dB'.format(band, round(psd, 2)), ha='left', va='center',
    #                        transform=self.ax[x, y].transAxes)
    #     self.ax[x, y].set(title='Power Spectrum Density {}'.format(text), xlabel='Frequency',
    #                       ylabel='Power Density (dB)')
    #     self.ax[x, y].axvspan(foilow, foihigh, alpha=0.3, color='green')
    #
    #     # plot psd averages index
    #     self.ax[x, y + 1].cla()
    #     xvals = np.arange(1, len(PSDs) + 1)
    #     self.ax[x, y + 1].bar(xvals, PSDs, width=0.4, color='orange')
    #     for i, v in enumerate(PSDs):
    #         if v != 0.:
    #             self.ax[x, y + 1].text(i + 1 - .2, v - .4, str(round(v, 2)))
    #     self.ax[x, y + 1].set(title='Average PSD Index {}'.format(text), xlabel='Segment #', ylabel='Power Density (dB')

    # if self.plotpref != 'none':
    #     del ax
    #     # fig = _plot_psd(raw, fig, freqs, psd_list, picks_list, titles_list,
    #     #                 units_list, scalings_list, ax_list, make_label, color,
    #     #                 area_mode, area_alpha, dB, estimate, average,
    #     #                 spatial_colors, xscale, line_alpha, sphere, xlabels_list)
    #
    #     # NEED TO UPDATE
    #     plt.plot(freq, psd_list)
    #     plt.xlabel('frequency [Hz]')
    #     plt.ylabel('PSD [db/Hz]')
    #     plt_show(show)