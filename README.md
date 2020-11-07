# Object Detection

DMY1411T Assignment 2 - Object Detection.

## Description
Makes use of Tensorflow's pre-trained model mobilenet to detect and draw bounding boxes around 2 precise classes "Man" 
and "Woman".

A GUI or Command Line Tool can be used (see below for Usage)

## Setup
`pip install -r requirements.txt`

*Note: activate a virtual environment before running pip install*

## Usage

### GUI Tool
`python main.py`

### Command Line Tool
###### Local file path
`python object_detector.py -i example/example1.jpg -t local`

###### Online file path
`python object_detector.py -i "https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcSXrpeNub8d6eoVdvLJXJX3ffO2Qicrmez_UA&usqp=CAU" -t online`

## Sample Output
![photo_name](output/local/example1.jpg)

![photo_name](output/online/1edd3166-e302-42f2-8946-a72ec7638290.jpg)

