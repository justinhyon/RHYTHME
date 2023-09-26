from copy import deepcopy

def custom_function(raw_data, subject, data_dict, fsample):

    #  this is a customizable function which can be coded to perform any desired analyses on the raw time series data,
    #  and on the ERP, PSD, and PSD values. the raw data is stored in raw_data, as an array of dimensions [channels,
    #  time series data]. Note that only data for the current segment.
    new_dict = deepcopy(data_dict)

    print(list(new_dict.keys()))
    print(raw_data)

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
