# convert NIST files to wav files
from sphfile import SPHFile
import glob

def NIST_to_wav(input_directory, output_directory):
    """
    convert files in directory to wav files

    Parameters
    ----------
    input_directory: str
        directory matching all files to be converted
    output_directory: str
        directory to output all converted files
    """
    files = glob.glob(input_directory)

    for f in files:
        sph = SPHFile(f)
        name = f[f.rfind('\\') + 1: f.rfind('.')]
        sph.write_wav(output_directory + name + ".wav")