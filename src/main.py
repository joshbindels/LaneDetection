import matplotlib.pylab as plt
import cv2
import numpy as np

def makeCoordinates(image, line_parameters):
    try:
        slope, intercept = line_parameters
        y1 = image.shape[0]
        y2 = int(y1 * (4/5))
        x1 = int((y1 - intercept) / slope)
        x2 = int((y2 - intercept) / slope)
        return np.array([x1, y1, x2, y2], np.int32)
    except:
        return np.array([0, 0, 0, 0], np.int32)

def averageSlopeIntersect(image, lines):
    left_fit = []
    right_fit = []
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        params = np.polyfit((x1, x2), (y1, y2), 1)
        slope = params[0]
        intercept = params[1]
        if slope < -0.2:
            left_fit.append((slope, intercept))
        elif slope > 0.2:
            right_fit.append((slope, intercept))
    left_fit_avg = np.average(left_fit, axis=0)
    right_fit_avg = np.average(right_fit, axis=0)
    left_line = makeCoordinates(image, left_fit_avg)
    right_line = makeCoordinates(image, right_fit_avg)
    return np.array([left_line, right_line], np.int32)

def region_of_interest(image):
    height = image.shape[0]
    width = image.shape[1]
    polygons = np.array([
        [(75, height), (width-75, height), (width/2, height-150)]
    ], np.int32)
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, polygons, 255)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image


def DisplayLines(image, lines):
    line_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 5)
    return line_image

def GetLanesImage(image):
    lane_image = np.copy(image)
    gray = cv2.cvtColor(lane_image, cv2.COLOR_RGB2GRAY)
    canny = cv2.Canny(gray, 100, 300)
    return canny

cap = cv2.VideoCapture("resource/video.mp4")

while True:
    _, frame = cap.read()

    lanes_image = GetLanesImage(frame)
    try:
        cropped_image = region_of_interest(lanes_image)
        # Houghlines params: image, pixel precision, angle precision, threshold (min number of intersections), ...
        lines = cv2.HoughLinesP(cropped_image, 2, np.pi/180, 100, np.array([]), minLineLength=20, maxLineGap=50)
        averaged_lines = averageSlopeIntersect(lanes_image, lines)
        line_image = DisplayLines(frame, averaged_lines)
        combo_image = cv2.addWeighted(frame, 0.6, line_image, 1, 1)

        cv2.imshow("Video", combo_image)
    except: pass

    if cv2.waitKey(5) == ord('q'):
        break


cv2.destroyAllWindows()
cap.release()

"""
cv2.imshow("image", roi_image)
cv2.waitKey()
cv2.destroyAllWindows()
"""
