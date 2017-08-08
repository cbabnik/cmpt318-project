# Author:
#  Curtis Babnik
#  cbabnik@sfu.ca
#  301235515

# collect command line arguments
import sys
if len(sys.argv) < 4:
    print("Improper call to python file  webcam_weather_predictor.py")
    print("Try the form: python3 webcam_weather_predictor.py " +
          "<images dir> <weather dir> <output dir>")
    print("(<output dir> is optional. Without it results are not written to files)")
    sys.exit(1)
images_dir  = sys.argv[1]
weather_dir = sys.argv[2]
output_dir  = None
if len(sys.argv) >= 4:
    output_dir = sys.argv[3]

# standard
import numpy as np
import pandas as pd
# my files
import image_feature_extraction as ife
import modelling as models
import data_collection as reader
# imaging
from matplotlib import pyplot as plt
import seaborn as sns

# overwrite ife with these
RES_X = 64
RES_Y = 48
PCA_F = 100

def getSky(string):
    if string.find("Rain") != -1:           return "Rain"
    if string.find("Snow") != -1:           return "Snow"
    elif string.find("Drizzle") != -1:      return None # too confusing
    if string.find("Mostly Cloudy") != -1:  return "Somewhat Cloudy"
    elif string.find("Mainly Clear") != -1: return "Somewhat Cloudy"
    elif string.find("Cloudy") != -1:       return "Very Cloudy"
    elif string.find("Clear") != -1:        return "Clear"
    else:                                   return None
def getFog(string):
    if string.find("Fog") != -1: return "Fog"
    else:                        return "No Fog"

def splitWeather(weather_data):
    weather_data["Sky"] = weather_data["Weather"].apply(getSky)
    weather_data["Fog"] = weather_data["Weather"].apply(getFog)
    weather_data.drop("Weather", axis=1, inplace=True)

def allFeatures(df):
    features = df.columns
    features = features.drop(["Sky", "Fog"])
    return features

def main():
    print()
    # collect weather
    print("Reading Weather data...")
    raw_data = reader.readWeather(weather_dir)
    splitWeather(raw_data)
    raw_data = raw_data[pd.notnull(raw_data["Sky"])] # drop ~40 values

    # get associated pictures as pixels
    print("Reading in matching images... (with %sx%s res)" % (RES_X, RES_Y))
    raw_data['Pixels'] = raw_data['Date/Time'].apply(
            reader.readImageFromDate, dir_path=images_dir, resize=(RES_X,RES_Y))
    raw_data = raw_data[pd.notnull(raw_data['Pixels'])]  # remove times with no pic
    raw_data.set_index('Date/Time', inplace=True)

    # extract features from pixels (and toss pixels)
    print("Extracting features from pixels...")
    ife.RESOLUTION_W = RES_X
    ife.RESOLUTION_H = RES_Y
    data = ife.select(raw_data, "Brightness", "Colours", "Brightness Pix")

    # setup modelling data
    print("Using PCA to reduce features... (%s features)" % PCA_F)
    print("...please be patient...")
    models.X_labels = allFeatures(data)
    models.y_labels = "Sky"
    models.PCA_FEATURES = PCA_F
    models.feed(data)

    print("results:")
    # try, fit, and extract data from models
    models.bayes(post=True)
    models.knn(n=10, post=True)
    models.svm(C=2, gamma=1e-6, post=True)

    # output some results
    if output_dir is not None:
        pass

if __name__=="__main__":
    main()
