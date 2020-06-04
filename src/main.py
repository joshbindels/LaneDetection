import cv2
import numpy as np

def CreateLineFromSlopeAndIntercept(image, line_parameters):
    slope, intercept = line_parameters
    y1 = image.shape[0]
    y2 = int(y1 * (0.7))
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    return np.array([x1, y1, x2, y2], np.int32)

def FilterLanesFromLines(lines):
    left = []
    right = []
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        params = np.polyfit((x1, x2), (y1, y2), 1)
        slope = params[0]
        intercept = params[1]
        if slope < -0.2:
            left.append((slope, intercept))
        elif slope > 0.2:
            right.append((slope, intercept))
    return left, right

def GetAverageLineFromLines(image, lines):
    if len(lines) == 0: return np.array([0, 0, 0, 0], np.int32)
    line_avg = np.average(lines, axis=0)
    return CreateLineFromSlopeAndIntercept(image, line_avg)

def GetAverageLaneLine(image, lines):
    left_fit, right_fit = FilterLanesFromLines(lines)
    left_line = GetAverageLineFromLines(image, left_fit)
    right_line = GetAverageLineFromLines(image, right_fit)
    return np.array([left_line, right_line], np.int32)

def RegionOfInterest(image):
    height = image.shape[0]
    width = image.shape[1]
    polygons = np.array([
        [(75, height), (width-75, height), (width/2, height-150)]
    ], np.int32)
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, polygons, 255)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image

def GetLaneHighlightImage(image, lines):
    lane_highlight_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            cv2.line(lane_highlight_image, (x1, y1), (x2, y2), (255, 0, 0), 5)
    return lane_highlight_image

def GetCannyImage(image):
    lane_image = np.copy(image)
    gray = cv2.cvtColor(lane_image, cv2.COLOR_RGB2GRAY)
    canny = cv2.Canny(gray, 100, 255)
    return canny

def GetFinalImage(frame):
    canny_image = GetCannyImage(frame)
    cropped_image = RegionOfInterest(canny_image)

    # Houghlines params:
    #   image, pixel precision, angle precision, threshold (min number of intersections)
    lines = cv2.HoughLinesP(cropped_image, 2, np.pi/180, 100, np.array([]), minLineLength=10, maxLineGap=20)

    if lines is not None:
        averaged_lines = GetAverageLaneLine(canny_image, lines)
        lane_highlight_image = GetLaneHighlightImage(frame, averaged_lines)
        return cv2.addWeighted(frame, 0.6, lane_highlight_image, 1, 1)
    else:
        return cv2.addWeighted(frame, 0.6, np.zeros_like(frame), 1, 1)


if __name__ == "__main__":
    cap = cv2.VideoCapture("resource/video.mp4")

    while True:
        _, frame = cap.read()
        cv2.imshow("Video", GetFinalImage(frame))
        if cv2.waitKey(5) == ord('q'):
            break

    cv2.destroyAllWindows()
    cap.release()
