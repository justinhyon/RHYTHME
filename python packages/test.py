import mne


raw = mne.io.read_raw_bdf(input_fname='/Users/jasonhyon/Desktop/T1_Beep.bdf')
raw.crop(35, 40)
raw.save('/Users/jasonhyon/Desktop/MD_MNE/T1_Beep_cropped_2.fif', overwrite = True)