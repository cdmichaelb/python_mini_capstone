"""
PDX Code Guild Full Stack Bootcamp
->Mini Capstone Project
Michael B
"""

import multiprocessing
import cv2
import numpy as np
from PIL import Image
import win32gui
import win32ui
from ctypes import windll
# I have tensorflow installed, but for some reason it isn't working???
import tensorflow as tf


class ImageVisualizer:
    def __init__(self, window_name="Image Visualizer") -> None:
        """
        This function will initialize the class and set the window name and template image to be used for the visualizer function.
        :param window_name: The name of the window to be used for the visualizer
        :param template_img: The template image to be used for the visualizer
        """
        self.window_name = window_name
        self.hwnd = win32gui.FindWindow(None, self.window_name)
        self.template_img = cv2.imread("template.png", cv2.IMREAD_GRAYSCALE)
        self.processed_image = None
        self.unprocessed_img = None

    def __str__(self) -> str:
        pass

    def process_img(self, original_image, template_img) -> np.ndarray:
        """
        This function will process the image to find the template image and return the processed image and the unprocessed image with the template image highlighted.
        :param original_image: The original image to be processed
        :param template_img: The template image to be used for the visualizer
        :return: The processed image and the unprocessed image with the template image highlighted
        :rtype: numpy.ndarray
        """
        # Convert to grayscale
        processed_img = cv2.cvtColor(np.array(original_image), cv2.COLOR_BGR2GRAY)
        unprocessed_img = cv2.cvtColor(np.array(original_image), cv2.COLOR_BGR2RGB)

        # Find the template
        processed_img = cv2.matchTemplate(
            processed_img, template_img, cv2.TM_CCOEFF_NORMED
        )

        return processed_img, unprocessed_img

    def take_screenshot(self) -> np.ndarray:
        """
        This function will take a screenshot of the screen and return the image as a numpy array.
        :return: The image as a numpy array
        :rtype: numpy.ndarray
        """
        # Get the window handle
        left, top, right, bottom = win32gui.GetClientRect(self.hwnd)
        # Get the window size
        width = right - left
        height = bottom - top
        # Create a window area
        hwndDC = win32gui.GetWindowDC(self.hwnd)
        mfcDC = win32ui.CreateDCFromHandle(hwndDC)
        saveDC = mfcDC.CreateCompatibleDC()
        # Create a bitmap
        saveBitMap = win32ui.CreateBitmap()
        saveBitMap.CreateCompatibleBitmap(mfcDC, width, height)
        # Select the bitmap into the saveDC
        saveDC.SelectObject(saveBitMap)
        # Copy the screen into the bitmap
        result = windll.user32.PrintWindow(self.hwnd, saveDC.GetSafeHdc(), 1)
        # Save the bitmap
        bmpinfo = saveBitMap.GetInfo()
        bmpstr = saveBitMap.GetBitmapBits(True)
        # Convert the bitmap to a numpy array
        im = Image.frombuffer(
            "RGB",
            (bmpinfo["bmWidth"], bmpinfo["bmHeight"]),
            bmpstr,
            "raw",
            "BGRX",
            0,
            1,
        )

        # im = np.array(im)
        # im = im[:, :, ::-1].copy()

        win32gui.DeleteObject(saveBitMap.GetHandle())
        saveDC.DeleteDC()
        mfcDC.DeleteDC()
        win32gui.ReleaseDC(self.hwnd, hwndDC)

        if result == 1:
            return im

    def visualizer(self) -> None:
        """
        This function will show the image and take a screenshot of the screen and process the image to find the template image.
        The template image will be highlighted in the image.
        """
        while True:
            """
            This loop will run until the user presses the 'q' key.
            The image will be shown and the template image will be highlighted.
            """
            self.processed_image, self.unprocessed_img = self.process_img(
                np.array(self.take_screenshot()), self.template_img
            )

            w = self.template_img.shape[1]
            h = self.template_img.shape[0]

            threshold = 0.65
            yloc, xloc = np.where(self.processed_image > threshold)

            rectangles = []
            for (x, y) in zip(xloc, yloc):
                rectangles.append((x, y, w, h))
                rectangles.append((x, y, w, h))

            rectangles, weights = cv2.groupRectangles(rectangles, 1, 0.2)

            for (x, y, w, h) in rectangles:
                cv2.rectangle(
                    self.unprocessed_img, (x, y), (x + w, y + h), (0, 255, 0), 2
                )

            cv2.imshow(self.window_name, self.unprocessed_img)

            if cv2.waitKey(25) & 0xFF == ord("q"):
                cv2.destroyAllWindows()
                break


def viz1():
    """
    This function will initialize the class and set the window name and template image to be used for the visualizer function.
    """
    # Initialize the class
    viewer = ImageVisualizer("AFK Arena")
    # Run the visualizer function
    viewer.visualizer()


def viz2():
    """
    This function will initialize the class and set the window name and template image to be used for the visualizer function.
    """
    # Initialize the class
    viewer2 = ImageVisualizer("poring")
    # Run the visualizer function
    viewer2.visualizer()


class MultiProcess:
    """
    This class will handle multiple processes.
    """

    def __init__(self, process_names: list):
        """
        This function will initialize the class and set the process name.
        :param process_name: The name of the process
        """
        self.process_names = process_names

    def __str__(self) -> str:
        pass

    def start_process(self):
        """
        This function will start the process.
        """
        # Start the process
        for process in self.process_names:
            po = multiprocessing.Pool(processes=12)
            po.map(viz1, range(10))
            po.map(viz2, range(10))
            po.close()
            po.join


# Need to create a model.
class ImageClassifier:
    """
    This class will handle the image classification.
    """

    def __init__(self, model_name: str, model_path: str):
        """
        This function will initialize the class and set the model name and model path.
        :param model_name: The name of the model
        :param model_path: The path of the model
        """
        self.model_name = model_name
        self.model_path = model_path

    def __str__(self) -> str:
        pass

    def load_model(self):
        """
        This function will load the model.
        """
        # Load the model
        self.model = tf.keras.models.load_model(self.model_path)

    def classify_image(self, image: np.ndarray) -> str:
        """
        This function will classify the image and return the class name.
        :param image: The image to be classified
        :return: The class name
        :rtype: str
        """
        # Convert the image to a numpy array
        image = np.array(image)
        # Resize the image
        image = cv2.resize(image, (64, 64))
        # Convert the image to a numpy array
        image = np.array(image)
        # Normalize the image
        image = image / 255.0
        # Add a fourth dimension to the image
        image = np.expand_dims(image, axis=0)
        # Classify the image
        prediction = self.model.predict(image)
        # Return the class name
        return self.model_name + ": " + str(np.argmax(prediction))


if __name__ == "__main__":
    """
    This is the main function.
    It will run the visualizer function for the first window and the second window.
    Should make this into a class.
    """
    process_list = []
    # process_list.append(viz1)
    # process_list.append(viz2)

    # multithreading = MultiProcess()
    # multithreading.start_process()

    # Remove Multithreading until I can escape it properly.

    viz1()

    ...  # Add more code here
