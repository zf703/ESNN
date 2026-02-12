import os
import sys
print('Current working path is %s' % str(os.getcwd()))
sys.path.insert(0, os.getcwd())
import platform
from scipy.io import *
import dill as pickle
import collections
from preprocess.preprocessing_library import FFT, Slice, Magnitude, Log10
from preprocess.lib_extension import Substract_average_plus_P_2_3x3, IFFT, Center_surround_diff, Normalise, RGB_0_255, Concatenation, Substract_average_plus_P_2_5x5, Substract_average_plus_P_2_7x7, Smooth_Gaussian_3x3, Smooth_Gaussian_5x5, Smooth_Gaussian_7x7
from utils.pipeline import Pipeline
import numpy as np
from joblib import Parallel, delayed
import warnings


eeg_type_data = collections.namedtuple('eeg_type_data', ['patient_id', 'label', 'data', 's1', 's2', 's3', 'd'])
def convert_to_fft(window_length, overlap, sampling_frequency, fft_min_freq, fft_max_freq_actual, file_path, label):

    warnings.filterwarnings("ignore")
    time_series_data = loadmat(file_path)['data']
    pipeline = Pipeline([FFT(), Slice(fft_min_freq, fft_max_freq_actual), Magnitude(), Log10()])
    # pipeline = Pipeline([FFT(), Magnitude(), Log10()])
    window_size = window_length * sampling_frequency
    m, n = time_series_data.shape
    if m > n:
        time_series_data = time_series_data.transpose()
    len = (time_series_data.shape[1] - window_size) // (window_size - int(window_size * overlap)) + 1
    fft_data = []
    for i in range(int(len)):
        begin = int((window_size - int(window_size * overlap)) * i)
        signal_window = time_series_data[:, begin:begin + window_size]
        fft_window = pipeline.apply(signal_window)
        # print(fft_window.shape)
        fft_data.append(fft_window)

    id = file_path.split('\\')[3].split('.')[0]
    named_data = eeg_type_data(patient_id=id, label=label, data=fft_data, s1=[], s2=[], s3=[], d=[])

    return named_data

def create_s1(file_data):

    warnings.filterwarnings("ignore")
    pipeline = Pipeline([Substract_average_plus_P_2_3x3(), IFFT(), Smooth_Gaussian_3x3()])
    time_series_data = file_data.data
    s1_data = []

    for i in time_series_data:
        window = pipeline.apply(i)
        s1_data.append(window)

    s1_data = np.array(s1_data)
    named_data = eeg_type_data(patient_id=file_data.patient_id, label=file_data.label, data=file_data.data, s1=s1_data, s2=[], s3=[], d=[])

    return named_data

#Saliency map S2
def create_s2(file_data):

    warnings.filterwarnings("ignore")
    pipeline = Pipeline([Substract_average_plus_P_2_5x5(), IFFT(), Smooth_Gaussian_5x5()])
    time_series_data = file_data.data
    s2_data = []

    for i in time_series_data:
        window = pipeline.apply(i)
        s2_data.append(window)

    s2_data = np.array(s2_data)
    named_data = eeg_type_data(patient_id=file_data.patient_id, label=file_data.label, data=file_data.data, s1=file_data.s1, s2=s2_data, s3=[], d=[])

    return named_data

#Saliency map S3
def create_s3(file_data):

    warnings.filterwarnings("ignore")
    pipeline = Pipeline([Substract_average_plus_P_2_7x7(), IFFT(), Smooth_Gaussian_7x7()])
    time_series_data = file_data.data
    s3_data = []

    for i in time_series_data:
        window = pipeline.apply(i)
        s3_data.append(window)

    s3_data = np.array(s3_data)
    named_data = eeg_type_data(patient_id=file_data.patient_id, label=file_data.label, data=file_data.data, s1=file_data.s1, s2=file_data.s2, s3=s3_data, d=[])

    return named_data
#RGB-encoded spectogram D
def create_d(file_data):
    warnings.filterwarnings("ignore")
    # Three of these pipelines are needed, as concatenation takes a different kind of parameter (three maps)
    pipeline1 = Pipeline([Normalise()])
    pipeline2 = Pipeline([Concatenation()])
    pipeline3 = Pipeline([RGB_0_255()])

    # The three feature maps
    data_s1 = file_data.s1
    data_s2 = file_data.s2
    data_s3 = file_data.s3

    d_data = []
    l = len(data_s1)
    for i in range(0, l):
        # Window definitions, the maps are of same size & shape so 1 looper can be used for all
        window_s1 = data_s1[i]
        window_s2 = data_s2[i]
        window_s3 = data_s3[i]
        # Normalise each window value
        window_s1_norm = pipeline1.apply(window_s1)
        window_s2_norm = pipeline1.apply(window_s2)
        window_s3_norm = pipeline1.apply(window_s3)
        # Concatenate normalised values
        d_norm = pipeline2.apply1(window_s1_norm, window_s2_norm, window_s3_norm)
        # RGB 0-255 conversion
        d_rgb = pipeline3.apply(d_norm)

        d_data.append(d_rgb)

    d_data = np.array(d_data)
    named_data = eeg_type_data(patient_id=file_data.patient_id, label=file_data.label, data=file_data.data, s1=file_data.s1, s2=file_data.s2, s3=file_data.s3, d=d_data)

    return named_data

def main():

    path = 'D:\dataset_19'
    hc_save_d_path = 'D:\hotspot4\HC'
    mdd_save_d_path = 'D:\hotspot4\MDD'
    hc_path = path + '\\' + 'HC'
    mdd_path = path + '\\' + 'MDD'
    hc_list = os.listdir(hc_path)
    mdd_list = os.listdir(mdd_path)
    hc_list_path = [os.path.join(hc_path, f) for f in hc_list]
    mdd_list_path = [os.path.join(mdd_path, f) for f in mdd_list]
    # print(hc_list_path)
    # print(mdd_list_path)

    sampling_frequency = 256  # Hz
    fft_min_freq = 1  # Hz

    window_lengths = [2]#[0.25, 0.5, 1]#[1, 2, 4, 8, 16]
    # fft_max_freqs = [12, 24, 48, 64, 96]#[12, 24]

    overlaps = [0]

    # fft_max_freqs = [64, 96, 128, 192, 256, 384, 512]
    fft_max_freqs = [128, 256, 512]

    for window_length in window_lengths:
        # window_steps = list(np.arange(window_length / 4, window_length / 2 + window_length / 4, window_length / 4))
        # window_steps = list(np.arange(window_length / 8, window_length / 2 + window_length / 8, window_length / 8))
        for overlap in overlaps:
            for fft_max_freq_actual in fft_max_freqs:

                save_d_data_dir = os.path.join(hc_save_d_path, 'd_HC_' + 'wl_' + str(window_length) + '_o_' + str(overlap)) + '_mf_' + str(fft_max_freq_actual)

                if not os.path.exists(save_d_data_dir):
                    os.makedirs(save_d_data_dir)
                else:
                    exit('Pre-processed data already exists!')

                for file_path in sorted(hc_list_path):
                    converted_fft_data = convert_to_fft(window_length, overlap, sampling_frequency, fft_min_freq, fft_max_freq_actual, file_path, 0)
                    base_dir = file_path.split('\\')[3].split('.')[0] + '.pkl'
                    converted_s1_data = create_s1(converted_fft_data)
                    converted_s2_data = create_s2(converted_s1_data)
                    converted_s3_data = create_s3(converted_s2_data)
                    converted_d_data = create_d(converted_s3_data)
                    pickle.dump(converted_d_data, open(os.path.join(save_d_data_dir, base_dir), 'wb'))

    for window_length in window_lengths:
        # window_steps = list(np.arange(window_length/4, window_length/2 + window_length/4, window_length/4))
        #window_steps = list(np.arange(window_length / 8, window_length / 2 + window_length / 8, window_length / 8))
        for overlap in overlaps:
            for fft_max_freq_actual in fft_max_freqs:
                save_d_data_dir = os.path.join(mdd_save_d_path, 'd_MDD_' + 'wl_' + str(window_length) + '_o_' + str(overlap)) + '_mf_' + str(fft_max_freq_actual)

                if not os.path.exists(save_d_data_dir):
                    os.makedirs(save_d_data_dir)
                else:
                    exit('Pre-processed data already exists!')

                for file_path in sorted(mdd_list_path):
                    converted_fft_data = convert_to_fft(window_length, overlap, sampling_frequency, fft_min_freq, fft_max_freq_actual, file_path, 1)
                    base_dir = file_path.split('\\')[3].split('.')[0] + '.pkl'
                    converted_s1_data = create_s1(converted_fft_data)
                    converted_s2_data = create_s2(converted_s1_data)
                    converted_s3_data = create_s3(converted_s2_data)
                    converted_d_data = create_d(converted_s3_data)
                    pickle.dump(converted_d_data, open(os.path.join(save_d_data_dir, base_dir), 'wb'))








if __name__ == '__main__':
    main()
    # # fft_data = pickle.load(open('D:\FFT\HC\\fft_HC_wl2_ws_0.5\\1.pkl', 'rb'))
    # # s1_data = pickle.load(open('D:\S1\HC\s1_HC_wl2_ws_0.5\\1.pkl', 'rb'))
    # # s2_data = pickle.load(open('D:\S2\HC\s2_HC_wl2_ws_0.5\\1.pkl', 'rb'))
    # d_data = pickle.load(open('D:\spectral\D\HC\d_HC_wl2_ws_0.5\\1.pkl', 'rb'))
    # # data1 = fft_data.data
    # # data2 = s1_data.data
    # # data3 = s2_data.data
    # data4 = d_data.data
    # # label1 = fft_data.label
    # # label2 = s1_data.label
    # # label3 = s2_data.label
    # label4 = d_data.label
    # # id1 = fft_data.patient_id
    # # id2 = s1_data.patient_id
    # # id3 = s2_data.patient_id
    # id4 = d_data.patient_id
    # print(data4[0].shape)
    # # print(data4)
    # print(type(label4))
    # print(id4)
    # d_data = pickle.load(open('D:\hotspot\HC\d_HC_wl_2_o_0_mf_12\\1.pkl', 'rb'))
    # data1 = d_data.data
    # print(data1[0].shape)

