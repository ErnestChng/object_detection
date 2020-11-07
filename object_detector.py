import argparse
import os
import time
import uuid
from pathlib import Path
from typing import Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from PIL import Image
from PIL import ImageColor
from PIL import ImageDraw
from PIL import ImageFont
from PIL import ImageOps
from six import BytesIO
from six.moves.urllib.request import urlopen

# Disable Tensorflow debugging information
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

##### Typing Classes
TENSOR = Union[tf.python.framework.ops.Tensor,
               tf.python.framework.sparse_tensor.SparseTensor,
               tf.python.ops.ragged.ragged_tensor.RaggedTensor]

DETECTOR = tf.python.eager.wrap_function.WrappedFunction

IMAGE = Image.Image

#### Constants
MODULES = {
    'resnet': "https://tfhub.dev/google/faster_rcnn/openimages_v4/inception_resnet_v2/1",
    'mobilenet': "https://tfhub.dev/google/openimages_v4/ssd/mobilenet_v2/1"
}

INTERESTED_CLASSES = [
    b'Man',
    b'Woman'
]


class ObjectDetector:
    """
    Object Detector used to identify objects in an image.
    """

    def __init__(self):
        """
        Constructor for the ObjectDetector. Runs a basic setup.
        """
        print('Checking setup...\n')
        self.check_setup()

        print('\nLoading module from TF hub...')
        module_handle = MODULES['mobilenet']
        self.detector = hub.load(module_handle).signatures['default']
        print('Module successfully loaded\n')

    @staticmethod
    def check_setup() -> None:
        """
        Checks the Tensorflow version installed and the number of GPUs available on the user's system.

        :return: None
        """
        # print Tensorflow version
        print(f"Tensorflow Version: {tf.__version__}")

        # check available GPUs in the system
        gpu = tf.test.gpu_device_name()
        print('No GPU devices are available') if gpu == '' else print(f"The following GPU devices are available: {gpu}")

    @staticmethod
    def display_image(image: Image) -> None:
        """
        Displays the image for visualisation using Matplotlib.

        :param image: PIL Image Object
        :return: None
        """
        plt.figure(figsize=(12, 8))
        plt.grid(False)
        plt.imshow(image)
        plt.show()

    def download_and_resize_image(self,
                                  url: str,
                                  url_type: str,
                                  new_width: int = 256,
                                  new_height: int = 256,
                                  display: bool = False) -> str:
        """
        Downloads the image according to the given url and resizes the image based on the new_width and new_height.

        Displays the image if display is True.

        :param url: String representing the file path of the image
        :param url_type: String representing the type of image url. Can be either 'online' or 'local'
        :param new_width: Integer representing the width of the resized PIL image
        :param new_height: Integer representing the height of the resized PIL image
        :param display: Boolean value. If True, displays the image.
        :return: String representing the file path of the image downloaded in the temp folder
        """
        print("Processing image...")
        start_time = time.time()

        if url_type == 'online':
            response = urlopen(url)
            image_data = BytesIO(response.read())
        else:
            image_data = url

        pil_image = Image.open(image_data)
        pil_image = ImageOps.fit(pil_image, (new_width, new_height), Image.ANTIALIAS)
        pil_image_rgb = pil_image.convert("RGB")

        filename = str(uuid.uuid4()) if url_type == 'online' else Path(url).stem
        path_in = f"input/{url_type}/{filename}.jpg"

        pil_image_rgb.save(path_in, format="JPEG", quality=90)
        print(f"Image downloaded to {path_in}")

        end_time = time.time()
        print(f"Download time: {end_time - start_time}")

        if display:
            self.display_image(pil_image)

        print("Image successfully processed\n")

        return path_in

    @staticmethod
    def draw_bounding_box_on_image(image: Image,
                                   y_min: float,
                                   x_min: float,
                                   y_max: float,
                                   x_max: float,
                                   color: str,
                                   font: ImageFont.ImageFont,
                                   thickness: int = 4,
                                   display_str_list: Union[tuple, list] = ()) -> None:
        """
        Draws a bounding box to an image.

        :param image: TensorFlow Image Object of the image
        :param y_min: Float representing the bottom y-coordinates of the box
        :param x_min: Float representing the left x-coordinates of the box
        :param y_max: Float representing the top y-coordinates of the box
        :param x_max: Float representing the right x-coordinates of the box
        :param color: String representing the colour to be used
        :param font: ImageFont representing the font to be used in the header
        :param thickness: Integer representing the thickness of the box. Default is 4
        :param display_str_list: Default is a tuple
        :return: None
        """
        draw = ImageDraw.Draw(image)
        im_width, im_height = image.size

        (left, right, top, bottom) = (x_min * im_width,
                                      x_max * im_width,
                                      y_min * im_height,
                                      y_max * im_height)

        draw.line([(left, top), (left, bottom), (right, bottom), (right, top), (left, top)],
                  width=thickness,
                  fill=color)

        # If the total height of the display strings added to the top of the bounding
        # box exceeds the top of the image, stack the strings below the bounding box
        # instead of above.
        display_str_heights = [font.getsize(ds)[1] for ds in display_str_list]

        # Each display_str has a top and bottom margin of 0.05x.
        total_display_str_height = (1 + 2 * 0.05) * sum(display_str_heights)

        if top > total_display_str_height:
            text_bottom = top
        else:
            text_bottom = top + total_display_str_height

        # Reverse list and print from bottom to top.
        for display_str in display_str_list[::-1]:
            text_width, text_height = font.getsize(display_str)
            margin = np.ceil(0.05 * text_height)

            draw.rectangle([(left, text_bottom - text_height - 2 * margin), (left + text_width, text_bottom)],
                           fill=color)

            draw.text((left + margin, text_bottom - text_height - margin),
                      display_str,
                      fill="black",
                      font=font)

            text_bottom -= text_height - 2 * margin

    def draw_boxes(self,
                   image: np.ndarray,
                   boxes: np.ndarray,
                   class_names: np.ndarray,
                   scores: np.ndarray,
                   max_boxes: int = 10,
                   min_score: float = 0.1) -> np.ndarray:
        """
        Overlay labeled boxes on an image with formatted scores and label names.

        :param image: Array representation of the image
        :param boxes: Array representation of the image box
        :param class_names: Array representation of the name of classes
        :param scores: Array representation of the prediction scores
        :param max_boxes: Integer representing the maximum number of boxes to be drawn. Default is 10.
        :param min_score: Float representing the minimum threshold score required for a box to be drawn.
        :return: Array representation of the image
        """
        colors = list(ImageColor.colormap.values())

        font = ImageFont.load_default()

        for i in range(min(boxes.shape[0], max_boxes)):
            if class_names[i] in INTERESTED_CLASSES:
                if scores[i] >= min_score:
                    y_min, x_min, y_max, x_max = tuple(boxes[i])
                    display_str = "{}: {}%".format(class_names[i].decode("ascii"), int(100 * scores[i]))
                    color = colors[hash(class_names[i]) % len(colors)]
                    image_pil = Image.fromarray(np.uint8(image)).convert("RGB")

                    self.draw_bounding_box_on_image(image_pil,
                                                    y_min,
                                                    x_min,
                                                    y_max,
                                                    x_max,
                                                    color,
                                                    font,
                                                    display_str_list=[display_str])

                    np.copyto(image, np.array(image_pil))

        return image

    @staticmethod
    def load_img(path: str) -> TENSOR:
        """
        Loads the image from a file path and converts it into a Tensor.

        :param path: String representing the file path of the image
        :return: Tensor representation of the image
        """
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3)

        return img

    def run_detector(self, detector: DETECTOR, path: str) -> Tuple[IMAGE, str]:
        """
        Runs the detector process.

        :param detector: Detector model
        :param path: String representing the file path of the image
        :return: None
        """
        img = self.load_img(path)

        converted_img = tf.image.convert_image_dtype(img, tf.float32)[tf.newaxis, ...]
        start_time = time.time()
        print("Starting object detection...")
        result = detector(converted_img)
        end_time = time.time()

        print("Found %d objects." % len(result["detection_scores"]))
        print("Inference time: ", end_time - start_time)

        result = {key: value.numpy() for key, value in result.items()}

        image_with_boxes = self.draw_boxes(img.numpy(),
                                           result["detection_boxes"],
                                           result["detection_class_entities"],
                                           result["detection_scores"])

        im = Image.fromarray(image_with_boxes)
        path_out = path.replace('input', 'output')
        im.save(path_out, format="JPEG", quality=90)
        print(f"Image downloaded to {path_out}")

        self.display_image(image_with_boxes)

        return im, path_out

    def detect_objects(self, image_url: str, *, url_type: str, is_flask: bool = False) -> Optional[Tuple[IMAGE, str]]:
        """
        Entry point for the Object Detection where user specifies the image url and url type.

        :param image_url: String representing the file path of the image
        :param url_type: String representing the type of image url. Can be either 'online' or 'local'
        :param is_flask: Boolean representing if Image and file path should be returned for the Flask app
        :return: Image with bounding boxes and file path to Image if is_flask is True, else None
        """
        url_types = ['online', 'local']

        if url_type not in url_types:
            raise AssertionError("url_type specified is not recognised. Please use either 'online' or 'local'")

        image_path = self.download_and_resize_image(image_url, url_type, 640, 480)

        im, path_out = self.run_detector(self.detector, image_path)

        return im, path_out if is_flask else None


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=True, help="path to input image")
    ap.add_argument("-t", "--type", required=True, help="image type (online or local)")
    args = vars(ap.parse_args())

    model = ObjectDetector()
    model.detect_objects(args['image'], url_type=args['type'])
