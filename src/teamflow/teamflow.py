from copy import deepcopy
def calculate_tmflow(data_dict):
    new_dict = deepcopy(data_dict)

    for key, value in new_dict.items():
        print(key, ':', value)
        for innerkey, innervalue in value.items():
            if 'values' in innerkey:
                new_dict[key][innerkey] = moving_average(innervalue, norm=True)

    # for idx, ERP in enumerate(ERPlist):
    #     ERPlist_norm_plot[idx] = self.moving_average(ERP, norm=True, initval=0.55997824)
    #     ERPlist_raw_plot[idx] = self.moving_average(ERP, norm=False)
    #
    # for idx, psd in enumerate(psds):
    #     psds_raw_plot[idx] = self.moving_average(psd, norm=False)
    #     psds_norm_plot[idx] = self.moving_average(psd, norm=True)
    #
    # PLV_inter_plot = self.moving_average(plv_inter, norm=True)
    # PLV_intra1_plot = self.moving_average(plv_intra1, norm=True)
    # PLV_intra2_plot = self.moving_average(plv_intra2, norm=True)


    print(data_dict.items())
    intra1 = data_dict['flow']['values_Intra 1']
    intra2 = data_dict['flow']['values_Intra 2']
    inter = data_dict['flow']['values_Inter']

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

    print(ERPlist)
    print(psd1)
    if ERPlist[0][-1] != 0:
        intra1.append((1 / ERPlist[0][-1]) + psd1['values_band3'][-1] + \
                      (1 / psd1['values_band2'][-1]))
    else:
        intra1.append((1 / 0.55997824) + psd1['values_band3'][-1] + \
                      (1 / psd1['values_band2'][-1]))

    if ERPlist[1][-1] != 0:
        intra2.append((1 / ERPlist[1][-1]) + psd2['values_band3'][-1] + (1 / psd2['values_band2'][-1]))
    else:
        intra2.append((1 / 0.55997824) + psd2['values_band3'][-1] + (1 / psd2['values_band2'][-1]))

    inter.append(plvinter[-1] + ((plv1[-1] + plv2[-1]) / 2) +
                 ((psd1['values_band1'][-1] + psd2['values_band1'][-1]) / 2))

    return intra1, intra2, inter


def teamflow_plot(ax, data_dict):
    import numpy as np
    print("\nGenerating Plots...")
    # time1 = timer()

    ax[0, 0].cla()
    ax[0, 1].cla()
    ax[0, 2].cla()

    x = np.arange(1, len(data_dict['flow']['values_Intra 1']) + 1)

    ax[0, 0].bar(x, data_dict['flow']['values_Intra 1'], width=0.4, color='red')  # index for flow intra 1
    ax[0, 1].bar(x, data_dict['flow']['values_Inter'], width=0.4, color='blue')  # index for plv inter
    ax[0, 2].bar(x, data_dict['flow']['values_Intra 2'], width=0.4, color='green')  # index for plv intra 2

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
    return ax

def moving_average(l, avg=True, norm=False, p=False, rmzero=True, initval=10):
    import pandas as pd
    from sklearn import preprocessing
    import numpy as np
    a = []
    if isinstance(l, list):
        if len(l) > 1:
            # Generate list of indexes of non-zero values
            a = [e for i, e in enumerate(l) if e != 0]

            # Generate list of indexes with zero-values
            zero_idxs = [i for i, e in enumerate(l) if e == 0]

            # Make list of non-zero values
            # for i in ls:
            #     a.append(l[i])
            if len(a) == 0:
                a.append(initval)

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

            if rmzero == False:
                for i in zero_idxs:
                    flat.insert(i, 0)
            # print("list length: ", len(l))
            # print("flat length:", len(flat))
            return flat
        else:
            return l