# data preprocessing for wav files
import glob
import numpy as np
from scipy.io import wavfile

def preprocessing(directory, window_size=100, target_std=10):
    """
    read all wav files in directory, normalize the entire dataset by a standard deviation and fragment all of them with a certian window_size

    Parameters
    ----------
    directory: str
        directory to read wav files from
    window_size: int, default=100
        window size for each fragment
    target_std: double, default=10
        standard deviation to normalize to
    
    Returns
        numpy array containing fragmented data
    """
    # get file names of all wav files in directory
    files = glob.glob(directory)

    entire_dataset = np.array([])
    wav_list = []
    for f in files:
        # read wav files
        wav = wavfile.read(f)[1]
        wav_list.append(wav)
        entire_dataset = np.concatenate((entire_dataset, wav))
    # mean and standard deviation of the entire dataset to be used for normalization later
    mean = np.mean(entire_dataset)
    std = np.std(entire_dataset)
    
    input_data = np.array([])
    input_size = 0
    for wav in wav_list:
        # normalize input to target standard deviation
        wav = np.divide(np.subtract(wav, mean), std/target_std)
        # fregment wav file according to window_size
        wav_fragment_length = int(wav.size/window_size)
        wav = wav[:wav_fragment_length * window_size]
        input_size += wav_fragment_length
        input_data = np.append(input_data, wav)

    input_data = input_data.reshape((input_size, window_size))


    # input_data = np.array([])
    # input_size = 0
    # for f in files:
    #     wav = wavfile.read(f)[1]
    #     wav_fragment_length = int(wav.size/window_size)
    #     wav = wav[:wav_fragment_length * window_size]
    #     input_size += wav_fragment_length
    #     input_data = np.append(input_data, wav)

    # input_data = input_data.reshape((input_size, window_size))

    # for i in range(0, input_data.shape[0]):
    #     input_data[i] = np.divide(np.subtract(input_data[i], np.mean(input_data[i])), np.std(input_data[i])/target_std)

    return input_data