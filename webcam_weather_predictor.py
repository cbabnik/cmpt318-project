# Author:
#  Curtis Babnik
#  cbabnik@sfu.ca
#  301235515

import sys
if len(sys.argv) < 4:
    print("Improper call to python file  webcam_weather_predictor.py")
    print("Try the form: python3 webcam_weather_predictor.py " +
          "<images dir> <weather dir> <output>")
    sys.exit(1)
images_dir  = sys.argv[1]
weather_dir = sys.argv[2]
output      = sys.argv[3]

# standard
import numpy as np
import pandas as pd
# imaging
from matplotlib import pyplot as plt
import seaborn as sns
# misc
from glob import glob # getting paths
from os.path import basename # get base filenames from path format
from scipy.misc import imread # read image

# constants
HEADER_LINE = 14 # For Columns in weather columns

#TODO make this more efficient!
# just read in all relevent data, not caring about format
def readFiles():
    # read weather_data
    paths = glob(weather_dir + "/*.csv")
    dfs = [pd.read_csv(f,header=HEADER_LINE) for f in paths]
    weather_data = pd.concat(dfs).reset_index()
    # read pictures_data
    paths = glob(images_dir + "/*.jpg")
    dfs = [pd.DataFrame([[basename(f), imread(f).reshape(-1,3)]]) for f in paths]
    picture_data = pd.concat(dfs).reset_index()
    # return
    return weather_data, picture_data

def main():
    weather, pictures = readFiles()

if __name__=="__main__":
    main()
