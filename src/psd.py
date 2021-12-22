import numpy as np


def psd(raw, psd_list, chan, foilow, foihigh, fSamp):

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
    print('TEST', psd_list)
    plot_psd = psd_db.transpose().mean(axis=0)
    return psd_list, plot_psd, f

def psd_plot(ax, psd, PSDs, freq, band, location, foilow, foihigh, subject):
    x, y = location
    # format psd plot
    ax[x, y].cla()
    ax[x, y].text(0.1, .9, 'Average Band PSD: {}dB'.format(np.round(PSDs[-1], 2)), ha='left', va='center',
                       transform=ax[x, y].transAxes)
    ax[x, y].set(title='Power Spectrum Density Subject {0} Band {1}'.format(subject, band), xlabel='Frequency',
                      ylabel='Power Density (dB)')
    ax[x, y].axvspan(foilow, foihigh, alpha=0.3, color='green')
    ax[x, y].plot(freq, psd)
    return ax

def psd_idx_plot(ax, PSDs, band, location, subject):
    x, y = location
    # plot psd averages index
    ax[x, y].cla()
    xvals = np.arange(1, len(PSDs) + 1)
    ax[x, y].bar(xvals, PSDs, width=0.4, color='orange')
    for i, v in enumerate(PSDs):
        if v != 0.:
            ax[x, y].text(i + 1 - .2, v - .4, str(round(v, 2)))
    ax[x, y].set(title='Average PSD Index Band Subject {0} Band {1}'.format(subject, band), xlabel='Segment #', ylabel='Power Density (dB)')
    return ax

    # fig = _plot_psd(raw, fig, freqs, psd_list, picks_list, titles_list,
    #                 units_list, scalings_list, ax_list, make_label, color,
    #                 area_mode, area_alpha, dB, estimate, average,
    #                 spatial_colors, xscale, line_alpha, sphere, xlabels_list)

    # NEED TO UPDATE

