def calculate_tmflow(data_dict):
    new_dict = data_dict.copy()

    for key, value in new_dict.items():
        new_dict[key] = moving_average(value, norm=True)

    # for idx, aep in enumerate(aeps):
    #     aeps_norm_plot[idx] = self.moving_average(aep, norm=True, initval=0.55997824)
    #     aeps_raw_plot[idx] = self.moving_average(aep, norm=False)
    #
    # for idx, psd in enumerate(psds):
    #     psds_raw_plot[idx] = self.moving_average(psd, norm=False)
    #     psds_norm_plot[idx] = self.moving_average(psd, norm=True)
    #
    # PLV_inter_plot = self.moving_average(plv_inter, norm=True)
    # PLV_intra1_plot = self.moving_average(plv_intra1, norm=True)
    # PLV_intra2_plot = self.moving_average(plv_intra2, norm=True)

    intra1 = data_dict['Intra 1']
    intra2 = data_dict['Intra 2']
    inter = data_dict['Inter']

    if new_dict['AEP 1'][-1] != 0:
        intra1.append((1 / new_dict['AEP 1'][-1]) + new_dict['Alpha1'][-1] + \
                      (1 / new_dict['Beta1'][-1]))
    else:
        intra1.append((1 / 0.55997824) + new_dict['Alpha1'][-1] + \
                      (1 / new_dict['Beta1'][-1]))

    if new_dict['AEP 2'][-1] != 0:
        intra2.append((1 / new_dict['AEP 2'][-1]) + new_dict['Alpha2'][-1] + (1 / new_dict['Beta2'][-1]))
    else:
        intra2.append((1 / 0.55997824) + new_dict['Alpha2'][-1] + (1 / new_dict['Beta2'][-1]))

    inter.append(new_dict['PLV inter'][-1] + ((new_dict['PLV 1'][-1] + new_dict['PLV 2'][-1]) / 2) +
                 ((new_dict['Gamma1'][-1] + new_dict['Gamma2'][-1]) / 2))

    return intra1, intra2, inter


def teamflow_plot(ax, data_dict):
    import numpy as np
    print("\nGenerating Plots...")
    # time1 = timer()

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