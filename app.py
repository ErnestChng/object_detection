from flask import Flask, request, send_file

from object_detector import ObjectDetector

model = ObjectDetector()
app = Flask(__name__)

DISPLAY_MSG = "DMY1401TT DYOM Assignment 2 - Object Detection REST service"


@app.route('/')
def home():
    return DISPLAY_MSG


@app.route('/detect')
def detect():
    arguments = request.args
    url, url_type = arguments.get('url'), arguments.get('url_type')

    if url is None or url_type is None:
        raise AssertionError('Please specify ?url= and ?url_type= properly in the route')

    im, path_out = model.detect_objects(url, url_type=url_type, is_flask=True)

    return send_file(path_out, attachment_filename='output_image.jpg')


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=3000)
