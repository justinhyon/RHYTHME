import sys
import struct
from psd import psd, psd_plot, psd_idx_plot
from plv import plv, plv_plot, plv_idx_plot
from ERP import ERP, ERP_plot, ERP_idx_plot
from teamflow import calculate_tmflow, teamflow_plot
from mne_realtime.externals import FieldTrip
import time
import copy
import numpy as np
from statistics import mean
import mne
from mne.io import RawArray, read_raw_bdf, read_raw_eeglab
from mne.preprocessing import ICA
from mne_connectivity import spectral_connectivity_epochs
from mne.time_frequency import psd_welch, psd_multitaper
import matplotlib.pyplot as plt
from timeit import default_timer as timer
# from multiprocessing import Process
from collections import OrderedDict
import pandas as pd
from sklearn import preprocessing
from os import path

segmentdata = self.ftc.getData(
                    index=(prevsamp, currentsamp - 1)).transpose()  # receive data from buffer
                print('Data shape (nChannels, nSamples): {}'.format(segmentdata.shape))

                for i, c in enumerate(segmentdata[64:128]):
                    print(c, i)