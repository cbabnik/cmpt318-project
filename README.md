Webcams, Predictions, and Weather (CMPT 318 Project)

#### Student & Author : Curtis Babnik (cbabnik@sfu.ca)

## What is it?

This python program takes as input a folder of weather data, and a folder of images, 
and creates a model to try to predict the weather from the images.
The best result I have gotten is 70% accuracy on the 5 categories:
"Clear", "Somewhat Cloudy", "Very Cloudy", "Rain", "Snow"

## How do I use it?

#### First prepare some data.

If you're using data provided to CMPT 318 students, that is great! do that. Skip ahead past this section. If not, I'm sorry. I allowed my program to accept a specific filetype. There are example inputs, so please look at them.

The weather data must be in csv format. To match the given data __I ignore the first 14 lines of text__ in the file.
It is important that if you are providing your own data, the first 14 lines OF TEXT are not part of the data.
Each csv file should be together in a folder. 
The important part is that your data must have a Weather column, a Date/Time column, and a Data Quality column.
Weather should be a string including some of these keywords.
"__Clear__", "__Mainly Clear__", "__Mostly Cloudy__", "__Somewhat Cloudy__", "__Rain__", "__Snow__".
Date/Time should be in the format "__YYYY-MM-DD HH:00__"
Lastly Data Quality must be set to "__‡__" for each row you care to include.

The images must be named a particular way: __katkam-aaaabbccdd0000.jpg__ . aaaa is the year, bb is month, cc is day, dd is hour.
For example katkam-20170504230000.jpg. The pictures will only be used if the date expressed is also in the weather data.
Gather these images into a folder. You can use any resolution of image, but it will always be read in at 64x48 resolution.

#### Next Prepare the libraries.

you will need:
- pandas
- numpy
- sklearn
- scipy

#### Now run the file.

If you want output files generated, use the form

_python3 webcam_weather_predictor.py image_dir weather_dir output_dir_

If you don't want files generated, exclude output_dir

#### Finally wait patiently for about 30 seconds

Some output examples:

![](example_output/graph.png?raw=true "Predictions vs Reality")
![](example_output/images/katkam-20160625160000.jpg?raw=true "Good prediction")
![](example_output/images/katkam-20160726160000.jpg?raw=true "Bad prediction (apparently)")
