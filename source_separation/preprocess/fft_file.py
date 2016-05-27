import os
import glob
from wav_fft import readFFT, writeFFT

def createdir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def extractAllFFTfromWAV(src_path, destination, percent=0.05):
    """
        Read all the wav files in src_path
        Convert it to spectrums
        Writes the STFT into destination, splitting data into train/test
    """
    
    sound_type = ['clean', 'noise', 'noisy']
    data_type  = ['train', 'test']

    createdir(src_path)
    print src_path

    # Going through all the subfolders (clean/train, srcclean/test, ...)
    for s_type in sound_type:
        createdir(destination + '/tidigits_' + s_type)
        print src_path + '/tidigits_' + s_type
        for d_type in data_type:
            createdir(destination + '/tidigits_' + s_type + '/' + d_type)

            # Get all the wav files recursively (used glob)
            path = src_path + '/tidigits_' + s_type + '/' + d_type
            listwavfiles = [y for x in os.walk(path) for y in glob.glob(os.path.join(x[0], '*.wav'))]
            i = 0
            total = int(len(listwavfiles)*percent)

            for filepath in listwavfiles:
                print s_type, "-", d_type, i, "/", total
                # Don't read too much files
                if i > total:
                    break
                i += 1
                # Read file, convert it
                filename = os.path.basename(filepath)
                stft_data = readFFT(filepath)
                # Save STFT to e.g. wav/tidigits_clean/train/
                filename_npz = filename[:-4]
                writeFFT(destination + '/tidigits_' + s_type + '/' + d_type, filename_npz, stft_data)

