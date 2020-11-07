# Object Detection

DMY1411T Assignment 2 - Object Detection.

## Description
Makes use of Tensorflow's pre-trained model mobilenet to detect and draw bounding boxes around 2 precise classes "Man" 
and "Woman".

A GUI or Command Line Tool can be used (see below for Usage).

## Setup
`pip install -r requirements.txt`

*Note: activate a virtual environment before running pip install*

## Usage

### GUI Tool
Launches a basic GUI for file navigation and uploading.

`python main.py`

### Command Line Tool
The command line tool makes use of 2 required arguments, -i which specifies the local/online file path to the image file
and -t which specifies the type (either online or local).

###### Local file path

`python object_detector.py -i example/example1.jpg -t local`

###### Online file path

`python object_detector.py -i "https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcSXrpeNub8d6eoVdvLJXJX3ffO2Qicrmez_UA&usqp=CAU" -t online`

### Flask App
Launches a basic Flask App. The detect route requires users to specify ?url= which specifies the local/online file
 path to the image file and ?url_type= which specifies the type (either online or local).

###### To launch the Flask app

`python app.py`

###### Local file path

`localhost:3000/detect?url=example/example1.jpg&url_type=local`

###### Online file path

`localhost:3000/detect?url=https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcSXrpeNub8d6eoVdvLJXJX3ffO2Qicrmez_UA&usqp=CAU&url_type=online`

## Sample Output

![photo_name](output/local/example1.jpg)

![photo_name](output/online/1edd3166-e302-42f2-8946-a72ec7638290.jpg)



