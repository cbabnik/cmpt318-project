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
# writing files
from glob import glob
import os
import random
# imaging
from matplotlib import pyplot as plt
from matplotlib.image import imread
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

def toPercent(fl):
    return str(int(np.round(fl*100))) + "%"

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
    print("...please be patient (takes about 20s on my machine)...")
    models.X_labels = allFeatures(data)
    models.y_labels = "Sky"
    models.PCA_FEATURES = PCA_F
    models.feed(data)

    print()
    print("results:")
    # try, fit, and extract data from models
    m1 = models.bayes(post=True)
    m2 = models.knn(n=10, post=True)
    m3 = models.svm(C=2, gamma=1e-6, post=True) # return best

    # ==========
    #  Analysis
    # ==========
    print()
    print("(SVM model will be used for results, since it usually does best)")
    # pair predictions with original data
    predictions = pd.DataFrame(m3.predict(models.X_test), index=models.test_index)
    predictions = predictions.join(data["Sky"])
    predictions.columns = ["Prediction", "Reality"]
    # count all
    prediction_counts = predictions.copy()
    prediction_counts["Occurrences"] = 1
    counts = prediction_counts.groupby(["Reality", "Prediction"]).count()
    # now shape nicely in some grids
    counts2 = counts.reset_index()
    counts2.set_index(counts2["Reality"]+counts2["Prediction"], inplace=True)
    counts2 = counts2["Occurrences"]
    grid_counts = []
    OPTS = ["Clear", "Somewhat Cloudy", "Very Cloudy", "Rain", "Snow"]
    for x in OPTS:
        for y in OPTS:
            grid_counts.append(counts2.get(x+y, 0))
    grid_counts = np.array(grid_counts).reshape(5,5)
    grid = pd.DataFrame(grid_counts, index=OPTS, columns=OPTS)
    totals = grid.sum(axis=1)
    pcgrid = grid.div(totals, axis=0)
    grid["total"] = totals
    pcgridstr = pcgrid.applymap(toPercent)
    pcgridstr["correct"] = np.diag(pcgridstr)

    print("Predictions as columns, Reality as index")
    print()
    print("_Totals_")
    print(grid)
    print()
    print("_Percents_")
    print(pcgridstr)

    # output some results
    if output_dir is not None:
        print()
        print("Writing output files...")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        if not os.path.exists(output_dir + "/images"):
            os.makedirs(output_dir + "/images")
        else:
            for img in glob(output_dir + "/images/*"):
                os.remove(img)
        # summary
        f = open(output_dir + "/summary.txt", "w")
        f.write("Predictions as columns, Reality as index\n")
        f.write("\n_Totals_\n" + str(grid) + "\n")
        f.write("\n_Percents\n" + str(pcgridstr) + "\n")
        f.write("\nOverall got %.2f accuracy with Support Machine Vector\n"
                % (m3.score(models.X_test,models.y_test)))
        f.write("\nIn comparison:\n")
        f.write("Bayes scored %.2f\n"
                % (m1.score(models.X_test,models.y_test)))
        f.write("K Nearest Neighbours scored %.2f\n"
                % (m2.score(models.X_test,models.y_test)))
        f.close()
        # predictions.csv
        predictions.index = predictions.index.map(reader.dateToFileName)
        predictions.to_csv(output_dir + "/predictions.csv",index_label="Picture")
        # graphs
        plt.figure(figsize=(6,4.5))
        arr = [-2,-1,0,1,2]
        plt.plot(arr,arr,linewidth=0)
        arr = [.8*a for a in arr]
        plt.xticks(arr,OPTS,rotation="vertical",fontsize=8)
        plt.yticks(arr,OPTS,fontsize=8)
        plt.xlabel("predictions")
        plt.ylabel("reality")
        plt.gcf().subplots_adjust(bottom=0.35,left=0.28)
        pcgrid2 = pcgrid.applymap(np.sqrt)
        for i in [0,1,2,3,4]:
            for j in [0,1,2,3,4]:
                x = OPTS[i]
                y = OPTS[j]
                if x == y:  style = "go"
                else:       style = "ro"
                dotsize = 30*pcgrid2[y].get(x,0)
                plt.plot(arr[j],arr[i],style,markersize=dotsize)
        plt.savefig(output_dir + "/graph.png")
        plt.clf()
        # images
        f = open(output_dir + "/images/note.txt", "w")
        f.write("There are only 30 pictures in this folder.\n")
        f.write("The 30 were chosen randomly from the test data.\n")
        f.write("The text written on each is the prediction.\n")
        f.write("Correct predictions are written in green.\n")
        f.write("Incorrect predictions are written in red.\n")
        f.close()
        sample = predictions.sample(30)
        sample["correct"] = sample["Prediction"] == sample["Reality"]
        for x in sample.index:
            plt.imshow(imread(images_dir + "/" + x))
            if sample["correct"].get(x): color = "green"
            else:                        color = "red"
            plt.text(128,96,sample["Prediction"].get(x),
                     fontsize=20, color=color,
                     horizontalalignment="center",
                     verticalalignment="center")
            plt.savefig(output_dir + "/images/" + x)
            plt.clf()

    print()
    print("Done.")
    if output_dir is not None:
        print("Have a look at %s directory!" % output_dir)
    else:
        print("Try passing an output_dir command line arg next time! :)")

if __name__=="__main__":
    main()
