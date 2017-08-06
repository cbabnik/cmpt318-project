# Author:
#  Curtis Babnik
#  cbabnik@sfu.ca
#  301235515

import numpy as np
import pandas as pd
import sys

# This file includes methods to transform a list of pixels to features.

RESOLUTION_W = 256
RESOLUTION_H = 192

# pixels expected in format [W*H][3], with row major order

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

# TODO this is inefficient, work on it.
def select(df, *args):
    pix = df["Pixels"]
    # build return frame from a concatenation
    frames = [df.drop('Pixels',1)]
    for arg in args:
        if arg == "Brightness Pix":
            pass
            # this is serving to be a bit of a problem, I'll come back to it
        elif arg == "Brightness":
            frames.append(pix.apply(overall_brightness))
        elif arg == "Colours":
            frames.append(pix.apply(overall_colours).apply(pd.Series))
        else:
            print("Invalid select: " + arg)
            sys.exit(1)
    return pd.concat(frames, axis=1)
