# Author:
#  Curtis Babnik
#  cbabnik@sfu.ca
#  301235515

# collect command line arguments
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

def getSnow(string):
    if string.find("Snow") != -1: return "Snow"
    else:                         return "No Snow"
def getRain(string):
    if string.find("Rain") != -1:      return "Rain"
    elif string.find("Drizzle") != -1: return None
    else:                              return "No Rain"
def getClouds(string):
    if string.find("Mostly Cloudy") != -1:  return "Somewhat Cloudy"
    elif string.find("Mainly Clear") != -1: return "Somewhat Cloudy"
    elif string.find("Cloudy") != -1:       return "Very Cloudy"
    elif string.find("Clear") != -1:        return "Clear"
    else:                                   return None
def getFog(string):
    if string.find("Fog") != -1: return "Fog"
    else:                        return "No Fog"

def splitWeather(weather_data):
    weather_data["Rain"] = weather_data["Weather"].apply(getRain)
    weather_data["Snow"] = weather_data["Weather"].apply(getSnow)
    weather_data["Clouds"] = weather_data["Weather"].apply(getClouds)
    weather_data["Fog"] = weather_data["Weather"].apply(getFog)
    weather_data.drop("Weather", axis=1, inplace=True)

def main():
    raw_data = reader.readWeather(weather_dir)
    raw_data['Pixels'] = raw_data['Date/Time'].apply(
            reader.readImageFromDate, dir_path=images_dir)
    raw_data = raw_data[pd.notnull(raw_data['Pixels'])]
    raw_data.set_index('Date/Time', inplace=True)

    splitWeather(raw_data)

    data = ife.select(raw_data, "Brightness", "Colours")

    models.X_labels = ["Pixels", 0, 1, 2]
    models.y_labels = "Clouds"
    models.feed(data)

    print()
    models.svm(C=100, gamma=0.0001, post=True)
    models.bayes(post=True)
    models.knn(post=True)
    models.knn(n=5, post=True)

if __name__=="__main__":
    main()
