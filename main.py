"""
PDX Code Guild Full Stack Bootcamp
->Mini Capstone Project
Michael B
"""

import multiprocessing # Multiprocessing
import cv2 # Computer Vision Functions
import numpy as np # Math Functions
from PIL import Image # Image Functions
import win32gui # Window Functions
import win32ui # Window Functions
from ctypes import windll # Window Functions
import os # File System Functions


def load_images(folder_path) -> list:
    """
    This function will load all the images from a folder into an array.
    :param folder_path: The path to the folder containing the images
    :return: An array containing all the images
    """
    # Initialize the array
    images = []
    # Load all the images from the folder into the array
    for filename in os.listdir(folder_path):
        images.append(cv2.imread(folder_path + filename, cv2.IMREAD_GRAYSCALE))
    # Return the array
    return images


class ImageVisualizer:
    def __init__(self, window_name="Image Visualizer") -> None:
        """
        This function will initialize the class and set the window name and template image to be used for the visualizer function.
        :param window_name: The name of the window to be used for the visualizer
        :param template_img: The template image to be used for the visualizer
        """
        self.window_name = window_name
        self.hwnd = win32gui.FindWindow(None, self.window_name)
        self.template_imgs = load_images("templates/")
        # Array to hold the template images
        self.processed_image = None
        self.unprocessed_img = None
        self.threshold = 0.89

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
        self.processed_img = cv2.cvtColor(np.array(original_image), cv2.COLOR_BGR2GRAY)
        self.unprocessed_img = cv2.cvtColor(np.array(original_image), cv2.COLOR_BGR2RGB)

        # Find the template
        self.processed_img = cv2.matchTemplate(
            self.processed_img, template_img, cv2.TM_CCOEFF_NORMED
        )

        return self.processed_img, self.unprocessed_img

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

        # Delete the bitmap
        win32gui.DeleteObject(saveBitMap.GetHandle())
        # Delete the saveDC
        saveDC.DeleteDC()
        # Delete the mfcDC
        mfcDC.DeleteDC()
        # Release the window handle
        win32gui.ReleaseDC(self.hwnd, hwndDC)

        # If the window is visible
        if result == 1:
            # Return the image
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
            rectangles = []

            # Take a screenshot of the screen
            screenshot = self.take_screenshot()
            # Process the image

            for template_img in self.template_imgs:
                self.processed_image, self.unprocessed_img = self.process_img(
                    np.array(screenshot), template_img
                )
                # Get the width of the template image
                w = template_img.shape[1]
                # Get the height of the template image
                h = template_img.shape[0]
                # Find the template image
                yloc, xloc = np.where(self.processed_image > self.threshold)

                # Loop through the template image
                for (x, y) in zip(xloc, yloc):
                    # Add the rectangle to the list twice (for grouping purposes)
                    rectangles.append((x, y, w, h))
                    rectangles.append((x, y, w, h))
                # Group the rectangles
                rectangles2, weights = cv2.groupRectangles(rectangles, 1, 0.2)
                # Loop through the rectangles
                for (x, y, w, h) in rectangles2:
                    # Draw the rectangle on the image
                    cv2.rectangle(
                        self.unprocessed_img, (x, y), (x + w, y + h), (0, 255, 0), 2
                    )
                    # Increase the threshold
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    cv2.destroyAllWindows()
                    quit()
            # Show the image
            cv2.imshow(self.window_name, self.unprocessed_img)


def main(process_name) -> None:
    """
    This function will initialize the class and set the window name and template image to be used for the visualizer function.
    :param process_name: The name of the process to be used for the visualizer
    """
    # Initialize the class
    viewer = ImageVisualizer(process_name)
    # Run the visualizer function
    viewer.visualizer()


class MultiProcess:
    """
    This class will handle multiple processes.
    """

    def __init__(self, process_names: list) -> None:
        """
        This function will initialize the class and set the process name.
        :param process_names: The names of the processes to be used
        """
        self.process_names = process_names

    def __str__(self) -> str:
        pass

    def start_process(self) -> None:
        """
        This function will start the process.
        """
        # Start the process
        for process in self.process_names:
            po = multiprocessing.Process(target=main, args=(process,))
            po.start()
            po.join


if __name__ == "__main__":
    """
    This is the main function.
    It will run the visualizer function for the first window and the second window.
    Should make this into a class.
    """
    process_list = []
    process_list.append("poring")
    process_list.append("AFK Arena")
    process_list.append("AfkArenaTest")
    process_list.append("Ragnarok")
    # Create a new instance of the class
    multithreading = MultiProcess(process_list)
    # Start the process
    multithreading.start_process()
    ...  # Add more code here
