import cv2
import numpy as np

IMAGE_PATH = "whiteballssample.jpg"


def count_balls(image_path):
    img = cv2.imread(image_path)
    output = img.copy()
    b = img[:, :, 2]
    bw = cv2.merge([b] * 3)
    bw = cv2.cvtColor(bw, cv2.COLOR_BGR2GRAY)

    morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
    # kernel1 = np.ones((5, 5), np.uint)
    _, thresh = cv2.threshold(bw, 77, 255, cv2.THRESH_BINARY)
    morphology = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, morph_kernel)

    circles = cv2.HoughCircles(morphology, cv2.HOUGH_GRADIENT, 2, 40, param1=100, param2=40, minRadius=0, maxRadius=100)
    circles = np.uint16(np.around(circles))
    for (x, y, r) in circles[0, :]:
        cv2.circle(output, (x, y), r, (0, 0, 255), 2)
        cv2.circle(output, (x, y), 3, (0, 0, 255), 3)

    radius_list = circles[0, :, 2]
    radius_mean = radius_list.mean()
    dispersion = round(((radius_list - radius_mean) ** 2).mean(), 2)

    cv2.imshow(f"count = {len(radius_list)}, radius_mean = {radius_mean}, dispersion = {dispersion}", output)
    cv2.waitKey(0)

    return {"count": len(radius_list), "radius_mean": radius_mean, "dispersion": dispersion}


if __name__ == '__main__':
    print(count_balls(IMAGE_PATH))
