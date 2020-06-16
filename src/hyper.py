import os
import time
import cv2
import numpy as np
from Intersection.intersection import Intersection
import configparser

config = configparser.ConfigParser()
config.read('config.ini')

def convert_to_gray(img, name, mode='debug', run='Run1'):
    """
    Gray Scaling the image for enhancing edge detection
    :param img: input image (numpy array)
    :param name: used for saving the results
    :param mode: save if in debug
    :return: Grayscaled image with Guassian Blur
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if mode == 'debug':
        cv2.imwrite(f"../res/{name}/gray_noisy.jpg", gray)
    # implement Gussian Blur to reduce noise (https://en.wikipedia.org/wiki/Gaussian_blur)
    # kernel_size=7 gave the best result
    kernel_size = int(config[run]['blur_kernel'])
    gray = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)
    if mode == 'debug':
        cv2.imwrite(f"../res/{name}/gray.jpg", gray)
    return gray


def get_edges(img, name, mode='debug', run='Run1'):
    """
    Detect edges in the input image; thresholds will be calculated automatically
    Having edges will help Hough Transform to easily detect lines
    Threshold setting:https://www.pyimagesearch.com/2015/04/06/zero-parameter-automatic-canny-edge-detection-with-python-and-opencv/
    """
    # Calculating thresholds automatically
    sigma = float(config[run]['sigma'])
    median = np.median(img)
    low_threshold = int(max(0, (1.0 - sigma) * median))
    high_threshold = int(min(255, (1.0 + sigma) * median))

    edges = cv2.Canny(img, low_threshold, high_threshold)
    if mode == 'debug':
        cv2.imwrite(f"../res/{name}/edges.jpg", edges)
    return edges


def get_hough_lines(img, name, threshold=0, min_line_length=0, max_line_gap=0, mode='debug', run='Run1'):
    """
    Get the dominant lines in the image using Probabilistic Hough Transform
    :param threshold: if it is 0 will calculate automatically
    :param min_line_length: if it is 0 will calculate automatically
    :param max_line_gap: if it is 0 will calculate automatically
    :return: Array of lines each element containing endpoints of line
    """
    # optimize
    # //Set threshold as a fraction of the image diagonal size. Still needs tweaking for each image
    # float hough_threshold = (sqrt(((image.rows*image.rows) + (image.cols*image.cols))) / houghlines_divisor);
    # Parameter setting
    threshold_dividend = int(config[run]['threshold_dividend'])
    min_dividend = int(config[run]['min_dividend'])
    max_dividend = min_dividend * float(config[run]['max_dividend'])
    if threshold == 0:  # if not given
        threshold = int((img.shape[0] + img.shape[1]) / threshold_dividend)
    if min_line_length == 0:  # if not given
        min_line_length = int(np.max(img.shape[0]) / min_dividend)
    if max_line_gap == 0:  # if not given
        max_line_gap = int(np.max(img.shape[0]) / max_dividend)
    # print(f"{name} ::: {threshold} ::: {min_line_length} ::: {max_line_gap}")

    lines = cv2.HoughLinesP(img,
                            rho=1,  # Distance resolution of the accumulator in pixelsl
                            theta=np.pi / 180,  # Angle resolution of the accumulator in radians.
                            threshold=threshold,  # Accumulator threshold parameter. (min vote)
                            lines=np.array([]),  # Output vector of lines.
                            minLineLength=min_line_length,  # Minimum line length.
                            maxLineGap=max_line_gap)  # Maximum allowed gap between points on the same line to link them.

    if mode == 'debug' and lines is not None:
        line_image = img.copy()
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 5)
        cv2.imwrite(f"../res/{name}/lines.jpg", line_image)
    return lines


def get_dominant_lines(img, name, mode='debug', run='Run1'):
    """
    This line will get dominant lines of given image in 3 steps:
    1. Convert image to grayscale and implement Gussian Blur to reduce noise
    2. Detect edges of grayscaled image using Canny Edge Detection
    3. Extract dominant lines of image using Probabilistic Hough Transform
    :param img: Input raw image
    :param name: For debug and logging porpuses
    :param mode: If debug it save each step
    :return: List of lines that contains each lines endpoints
    """
    img = convert_to_gray(img, name, mode, run)
    # if False:  # test if dilation or erosion
    #    kernel = np.ones((15, 15), np.uint8)
    #    img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)  # Open (erode, then dilate)
    img = get_edges(img, name, mode, run)
    lines = get_hough_lines(img, name, 0, 0, 0, mode, run)
    if lines is not None:
        return [line[0] for line in lines]
    else:
        return []


def find_intersections(lines, img, run='Run1'):
    """
    Function to find all intersection between lines.
    :param lines: List of lines in [x1, y1, x2, y2] format
    :param img: Input image
    :return: List of Intercetion objects that contains two lines and the point they will intersect
    """
    intersections = []
    for i, line1 in enumerate(lines):
        for line2 in lines[i + 1:]:
            if True:  # not line_1 == line_2:
                try:
                    min_angle = float(config[run]['min_angle'])
                    max_angle = float(config[run]['max_angle'])
                    intersections.append(Intersection(line1, line2, img.shape, min_angle, max_angle))
                except TypeError:
                    # There would be lots of lines that will never intersect. Just ignore them
                    pass
                except Exception as e:
                    print(e)
    return intersections


def pad_image(img, name, pad_ratio=1, mode='debug'):
    """
    We need to find and visualize vanishing points even outside of image with a limit.
    For this purpose I am padding the image by white (255, 255, 255) pixels.
    The requested limit is 1 image shape from each side, it could be set by pad_ratio.
    :param img: Input image
    :param pad_ratio: Ratio of image to pad
    :return: Padded image
    """
    border_type = cv2.BORDER_CONSTANT
    top = int(pad_ratio * img.shape[0])
    bottom = top
    left = int(pad_ratio * img.shape[1])
    right = left
    padded_img = cv2.copyMakeBorder(img, top, bottom, left, right, border_type, None, (255, 255, 255))
    if mode == 'debug':
        cv2.imwrite(f"../res/{name}/padded.jpg", padded_img)
    return padded_img


def shift_select_intersections(intersections, pad_ratio=1.0):
    """
    This function select all intersections that suit inside our padded area and also shift points and lines to be
    possible to visualize properly.
    :param intersections: All possible intersections
    :param pad_ratio: Ratio of image that we accept Vanishing Points there
    :return: Accepted and shifted intersections.
    """
    selected_intersections = []
    for intersection in intersections:
        if intersection.shift(pad_ratio):
            selected_intersections.append(intersection)
    return selected_intersections


def run(file_name, mode='debug', run='Run1'):
    img = cv2.imread(file_name)
    # timestr = time.strftime("%Y%m%d-%H%M%S")
    timestr = time.strftime("_%H_%M")
    name = file_name.split('/')[-1].split('.')[0] + timestr

    try:
        os.mkdir(f"../res/{name}")
    except FileExistsError as fe:
        pass
    if mode == 'debug':
        cv2.imwrite(f"../res/{name}/original.jpg", img)

    lines = get_dominant_lines(img, name, mode, run)
    intersections = find_intersections(lines, img, run)
    padded_image = pad_image(img, name, 0.1)
    selected_intersections = shift_select_intersections(intersections, 0.1)
    with open(f'../res/{name}/intersections.txt', 'w') as f:
        for inter in selected_intersections:
            # print(np.degrees(inter.angle))
            f.writelines(str(int(np.degrees(inter.angle))) + '\n')
    print(f"{name} :: {len(lines)} :: {len(intersections)} ::  {len(selected_intersections)}")
    result = padded_image.copy()
    # print(f"len(selected_intersections): {len(selected_intersections)}")
    try:
        os.mkdir(f"../res/{name}/details/")
    except FileExistsError as fe:
        pass
    try:
        os.mkdir(f"../res/{name.split('_')[0]}/")
    except FileExistsError as fe:
        pass
    for inter, i in zip(selected_intersections, range(len(selected_intersections))):

        result = cv2.line(result, (inter.line1[0], inter.line1[1]), (inter.line1[2], inter.line1[3]), (255, 0, 0), 5)
        result = cv2.line(result, (inter.line2[0], inter.line2[1]), (inter.line2[2], inter.line2[3]), (255, 0, 0), 5)
        result = cv2.circle(result, (inter.x, inter.y), radius=10, color=(0, 0, 255), thickness=-1)
        # inter.print_shifts()

        if mode == 'debug' and False:
            detailed_res = padded_image.copy()
            detailed_res = cv2.line(detailed_res, (inter.line1[0], inter.line1[1]), (inter.line1[2], inter.line1[3]),
                                    (255, 0, 0), 5)
            detailed_res = cv2.line(detailed_res, (inter.line2[0], inter.line2[1]), (inter.line2[2], inter.line2[3]),
                                    (255, 0, 0), 5)
            detailed_res = cv2.circle(detailed_res, (inter.x, inter.y), radius=10, color=(0, 0, 255), thickness=-1)
            cv2.imwrite(f'../res/{name}/details/inter{i}.jpg', detailed_res)

    cv2.imwrite(f'../res/{name}/result.jpg', result)
    cv2.imwrite(f'../res/{name.split("_")[0]}/result_{run}.jpg', result)


if __name__ == '__main__':
    # file_name = '../data/cyclomedia1.jpg'

    for f in os.listdir('../data/'):
        for con in config.sections():
            run(f'../data/{f}', 'debug', con)
            """
            try:
                run(f'../data/{f}', 'debug', con)
            except Exception as e:
                print(f)
                print(e)
            """
