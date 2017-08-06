# Author:
#  Curtis Babnik
#  cbabnik@sfu.ca
#  301235515

import sys
if len(sys.argv) < 4:
    print("Improper call to python file  webcam_weather_predictor.py")
    print("Try the form: python3 webcam_weather_predictor.py " +
          "<images dir> <weather dir> <output>")
    print("(<output> is optional. Without it results are printed but not written.)")
    sys.exit(1)
images_dir  = sys.argv[1]
weather_dir = sys.argv[2]
output      = ""
if len(sys.argv) >= 4:
    output  = sys.argv[3]

# my files
import image_feature_extraction as ife
# standard
import numpy as np
import pandas as pd
# imaging
from matplotlib import pyplot as plt
import seaborn as sns
# regular expressions
import re
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
    dfs = [pd.read_csv(f,header=HEADER_LINE,keep_default_na=False) for f in paths]
    weather_data = pd.concat(dfs).reset_index()
    # read pictures_data
    paths = glob(images_dir + "/*.jpg")
    dfs = [pd.DataFrame([[basename(f), imread(f).reshape(-1,3)]]) for f in paths]
    picture_data = pd.concat(dfs).reset_index()
    picture_data.columns = ["index", "Title", "Pixels"]
    # return
    return weather_data, picture_data

# Helper method to get date infromation from picture filename
picture_fileName_pattern = re.compile(r"katkam-((\d{4})" + r"(\d{2})"*5 + r").jpg")
def extract_time(name):
    m = picture_fileName_pattern.match(name)
    # time   = m.group(1)
    year   = m.group(2)
    month  = m.group(3)
    day    = m.group(4)
    hour   = m.group(5)
    minute = m.group(6)
    # second = m.group(7)
    return "%s-%s-%s %s:%s" % (year, month, day, hour, minute)

# Join weather and pictures on Date/Time. (Add Date/Time to pictures)
def join(weather, pictures):
    pictures["Date/Time"] = pictures["Title"].apply(extract_time)
    pictures.set_index('Date/Time', inplace=True)
    weather.set_index('Date/Time', inplace=True)
    return weather.join(pictures, how="inner", rsuffix="_photo")

# Both the cleaning and the shaping
# Remove bad data (missing values, etc)
# Remove unneccessary columns, and name how I like.
# Don't do anything with Pixels, that gets it's own function
def clean(data):
    data = data[data["Data Quality"] == "‡"] # no incomplete or estimated data
    data = data[data["Weather"] != "NA"]
    # I only need Pixels and Weather, but I may want to care about the other four
    data = data[["Pixels",
                 "Weather",
                 #"Temp (°C)",
                 #"Stn Press (kPa)",
                 #"Rel Hum (%)",
                 #"Visibility (km)",
    ]]
    data.columns = ["Pixels",
                    "Weather",
                    #"Temp",
                    #"Pressure",
                    #"Humidity",
                    #"Visibility",
    ]
    return data

def main():
    # The level of coupling is pretty awful, but I might be ok with that.
    # I'm only breaking the code into stages with the functions anyway.
    weather, pictures = readFiles()
    raw_data = join(weather, pictures)
    raw_data = clean(raw_data)
    data = ife.select(raw_data, "Brightness", "Colours")

if __name__=="__main__":
    main()
