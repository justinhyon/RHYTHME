from copy import deepcopy

def custom_function(raw_data, subject, data_dict, fsample):

    #  this is a customizable function which can be coded to perform any desired analyses on the raw time series data,
    #  and on the ERP, PSD, and PSD values. the raw data is stored in raw_data, as an array of dimensions [channels,
    #  time series data]. Note that only data for the current segment.
    new_dict = deepcopy(data_dict)

    intra1 = data_dict['custom']['values_intra1']
    intra2 = data_dict['custom']['values_intra2']
    inter = data_dict['custom']['values_inter']

    ERPlist = []
    for n, keyname in enumerate([i for i in list(new_dict.keys()) if 'ERP' in i]):
        ERPlist.append(new_dict[keyname]['values_ERP'])

    psd1 = {}
    psd2 = {}
    for keyname in [i for i in list(new_dict.keys()) if 'psd' in i]:
        print(new_dict[keyname].items())
        for n, psdkey in enumerate([j for j in list(new_dict[keyname].keys()) if 'values' in j]):
            if '1' in keyname:
                psd1[psdkey] = new_dict[keyname][psdkey]
            elif '2' in keyname:
                psd2[psdkey] = new_dict[keyname][psdkey]

    plv1 = new_dict['plv']['values_intra1']
    plv2 = new_dict['plv']['values_intra2']
    plvinter = new_dict['plv']['values_inter']

    # print(ERPlist)
    # print(psd1)
    if ERPlist[0][-1] != 0:
        intra1.append((1 / ERPlist[0][-1]) + psd1['values_band3'][-1] + \
                      (1 / psd1['values_band2'][-1]))
    else:
        intra1.append( psd1['values_band3'][-1] + \
                      (1 / psd1['values_band2'][-1]))

    if ERPlist[1][-1] != 0:
        intra2.append((1 / ERPlist[1][-1]) + psd2['values_band3'][-1] + (1 / psd2['values_band2'][-1]))
    else:
        intra2.append( psd2['values_band3'][-1] + (1 / psd2['values_band2'][-1]))

    inter.append(plvinter[-1] + ((plv1[-1] + plv2[-1]) / 2) +
                 ((psd1['values_band1'][-1] + psd2['values_band1'][-1]) / 2))

    new_dict['custom']['values_intra1'] = intra1
    new_dict['custom']['values_intra2'] = intra2
    new_dict['custom']['values_inter'] = inter

    return new_dict


def custom_plot_1(ax, location, subject, data_dict):
    x, y = location
    # format psd plot
    ax[x, y].cla()
    ax[x, y].set(title='Custom Plot Subject {}'.format(subject), xlabel='X axis title',
                 ylabel='Y axis title')
    ax[x, y].plot(freq, psd)
    return ax

def custom_plot_2(ax, location, subject, data_dict):
    x, y = location
    # format psd plot
    ax[x, y].cla()
    ax[x, y].set(title='Custom Plot Subject {}'.format(subject), xlabel='X axis title',
                 ylabel='Y axis title')
    ax[x, y].plot(freq, psd)
    return ax
