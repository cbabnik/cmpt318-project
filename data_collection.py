# Author:
#  Curtis Babnik
#  cbabnik@sfu.ca
#  301235515

# standard
import numpy as np
import pandas as pd
# regular expressions
import re
# misc
from glob import glob # getting paths
from os.path import basename # get base filenames from path format
from scipy.misc import imread # read image

# This file includes methods to transform a list of pixels to features.

# constants
HEADER_LINE = 14 # For Columns in weather columns
DATE_PATTERN = re.compile(r"(\d{4})-(\d{2})-(\d{2}) (\d{2}):00")
PICTURE_FORMAT = r"katkam-%s%s%s%s0000.jpg"

def readWeather(dir_path):
    paths = glob(dir_path + "/*.csv")
    dfs = [pd.read_csv(f,header=HEADER_LINE,keep_default_na=False) for f in paths]
    weather_data = pd.concat(dfs)
    # remove superfluous info and unusable data here too
    weather_data = weather_data[
            (weather_data["Weather"] != "NA") &
            (weather_data["Weather"] != "Thunderstorms") & # 2 ignored
            (weather_data["Weather"] != "Drizzle") & # 16 ignored
            (weather_data["Data Quality"] == "‡")]
    weather_data = weather_data[[
            "Weather",
            "Date/Time",
            # The following could be potentially useful and/or predictable
            # "Temp (°C)", "Stn Press (kPa)", "Rel Hum (%)", "Visibility (km)",
    ]]
    return weather_data

def readImage(full_path):
    try:
        pixels = imread(full_path).reshape(-1,3) # row major order
    except:
        pixels = None
    return pixels

def dateToFileName(date_str):
    m = DATE_PATTERN.match(date_str)
    year   = m.group(1)
    month  = m.group(2)
    day    = m.group(3)
    hour   = m.group(4)
    return PICTURE_FORMAT % (year, month, day, hour)

def readImageFromDate(date_str, dir_path):
    return readImage(dir_path +'/'+ dateToFileName(date_str))
