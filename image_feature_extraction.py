# Author:
#  Curtis Babnik
#  cbabnik@sfu.ca
#  301235515

import numpy as np
import pandas as pd
import sys

# This file includes methods to transform a list of pixels to features.

RESOLUTION_W = 256 # override this if needed, all images should match
RESOLUTION_H = 192 # override this if needed, all images should match

# pixels expected in format [W*H][3], with row major order (~order not matter)

def overall_brightness(pixels):
    return np.mean(pixels)
overall_brightness_vec = np.vectorize(overall_brightness)

# gives 3 features
def overall_colours(pixels):
    return np.mean(pixels, axis=0)
overall_colours_vec = np.vectorize(overall_brightness)

# gives a feature for each pixel
def brightness(pixels):
    return np.mean(pixels, axis=1)
brightness_vec = np.vectorize(brightness)

# TODO this is still kind of inefficient. Think about it
# ( The issue is that: series.of.arrays->np.matrix->dataframe->concat )
def select(df, *args):
    pix = df["Pixels"]
    # build return frame from a concatenation
    frames = [df.drop('Pixels',1)]
    for arg in args:
        if arg == "Brightness Pix":
            values = np.matrix(pix.apply(brightness).tolist())
            frame = pd.DataFrame(values, index=df.index)
            frames.append(frame)
        elif arg == "Brightness":
            series = pix.apply(overall_brightness).rename("Brightness")
            frames.append(series)
        elif arg == "Colours":
            values = np.matrix(pix.apply(overall_colours).tolist())
            frame = pd.DataFrame(values, index=df.index)
            frame.columns = ['Red', 'Green', 'Blue']
            frames.append(frame)
        else:
            print("Invalid select: " + arg)
            sys.exit(1)
    return pd.concat(frames, axis=1)
