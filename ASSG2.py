import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt


def plot_side_by_side(first, second, plot_type, filter_name):
    if plot_type == 'image':
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        ax1.imshow(first)
        ax2.imshow(second)
        f.savefig(filter_name + '.png', bbox_inches='tight')
        ax1.set_title('Original')
        ax2.set_title(filter_name)
    elif plot_type == 'hist':
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        (_, _, _) = ax1.hist(first.flatten(), bins=255)
        (_, _, _) = ax2.hist(second.flatten(), bins=255)
        ax1.set_title('Original')
        ax2.set_title(filter_name)
    plt.show()


def q1(img):
    kernel1 = np.ones((15, 15), np.uint8)
    kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    square_circle1 = cv2.dilate(img, kernel1, iterations=1)
    square_circle2 = cv2.dilate(img, kernel2, iterations=1)
    plot_side_by_side(img, square_circle1, "image", "square-circle-1")
    plot_side_by_side(img, square_circle2, "image", "square-circle-2")


def q2(img):
    kernel = np.ones((3, 3), np.uint8)
    closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    plot_side_by_side(img, closing, "image", "cameraman-denoised")


def q3(img):
    kernel = np.ones((5, 5), np.uint8)
    dilating = cv2.dilate(img, kernel, iterations=1)
    edge = dilating - img
    plot_side_by_side(img, edge, "image", "lady-edge")


def q4(img):
    kernel1 = np.ones((15, 15), np.uint8)
    kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    invert = cv2.bitwise_not(img)
    erosion1 = cv2.erode(invert, kernel1, iterations=1)
    erosion1_invert = cv2.bitwise_not(erosion1)
    erosion2 = cv2.erode(invert, kernel2, iterations=1)
    erosion2_invert = cv2.bitwise_not(erosion2)
    plot_side_by_side(img, erosion1_invert, "image", "circle-square-erode")
    plot_side_by_side(img, erosion2_invert, "image", "circle-ellipse-erode")


def q5(img):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    invert = cv2.bitwise_not(img)
    blur = cv2.GaussianBlur(invert, (5, 5), 0)
    ret, threshold = cv2.threshold(blur, 127, 255, cv2.THRESH_BINARY)
    erosion = cv2.erode(threshold, kernel, iterations=1)
    plot_side_by_side(img, erosion, "image", "circle-erode")


def q6(img):
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    plot_side_by_side(img, opening, "image", "cameraman-erode")


def q7(img):
    kernel1 = np.ones((3, 3), np.uint8)
    kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (12, 12))
    opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel2)
    plot_side_by_side(img, opening, "image", "circles")

    lines = img - opening
    lines = cv2.bitwise_not(lines)
    lines = cv2.morphologyEx(lines, cv2.MORPH_CLOSE, kernel1)
    lines = cv2.bitwise_not(lines)

    kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    lines = cv2.erode(lines, kernel2, iterations=2)
    # kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    lines = cv2.dilate(lines, kernel2, iterations=2)
    kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
    img3 = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel2)
    img3 = cv2.bitwise_not(img3)
    img3 = cv2.morphologyEx(img3, cv2.MORPH_CLOSE, kernel1)
    img3 = cv2.bitwise_not(img3)
    plot_side_by_side(img, img3, "image", "lines")
    q7_b(opening, img3)


def q7_b(circles, lines):
    x, y, z = circles.shape
    circles = cv2.cvtColor(circles, cv2.COLOR_BGR2GRAY)
    circles_out = cv2.connectedComponentsWithStats(circles, 8, cv2.CV_32S)
    circles = cv2.cvtColor(circles, cv2.COLOR_GRAY2BGR)
    circles2 = np.copy(circles)

    print("Number of circles : " + str(circles_out[0]))
    count = 1
    for i in circles_out[3][1:]:
        j = int(i[0])
        k = int(i[1])
        # if j >= x//2:
        #     j -= 3
        # else:
        #     j += 3
        # if k >= y//2:
        #     k -= 3
        # else:
        #     k += 3
        cv2.putText(circles,
                    str(count),
                    (j, k),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 0, 0),
                    2)
        count += 1
    plot_side_by_side(circles2, circles, "image", "circles_with_count")

    lines = cv2.cvtColor(lines, cv2.COLOR_BGR2GRAY)
    lines_out = cv2.connectedComponentsWithStats(lines)
    lines = cv2.cvtColor(lines, cv2.COLOR_GRAY2BGR)
    lines2 = np.copy(lines)

    print("Number of circles : " + str(lines_out[0]))
    count = 1
    for i in lines_out[3][1:]:
        j = int(i[0])
        k = int(i[1])
        cv2.putText(lines,
                    str(count),
                    (j, k),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 0, 0),
                    2)
        count += 1
    plot_side_by_side(lines2, lines, "image", "lines_with_count")


def q13(img):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    ret, threshold = cv2.threshold(blur, 127, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    threshold = cv2.cvtColor(threshold, cv2.COLOR_GRAY2BGR)
    threshold = cv2.bitwise_not(threshold)
    threshold = cv2.morphologyEx(threshold, cv2.MORPH_CLOSE, kernel)
    threshold = cv2.bitwise_not(threshold)
    # cv2.imshow("3", threshold)
    # cv2.waitKey()
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (30, 30))
    threshold = cv2.morphologyEx(threshold, cv2.MORPH_CLOSE, kernel)
    threshold = cv2.bitwise_not(threshold)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
    erosion = cv2.erode(threshold, kernel, iterations=5)
    dilation = cv2.dilate(erosion, kernel, iterations=5)
    small_coins = cv2.bitwise_xor(dilation, threshold)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    small_coins = cv2.morphologyEx(small_coins, cv2.MORPH_OPEN, kernel)
    big_coins = cv2.bitwise_xor(threshold, small_coins)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    big_coins = cv2.morphologyEx(big_coins, cv2.MORPH_OPEN, kernel)
    # plot_side_by_side(small_coins, big_coins, "image", "circles")
    q13_cont(img, small_coins, big_coins)


def q13_cont(img, small_coins, big_coins):
    small_coins = cv2.cvtColor(small_coins, cv2.COLOR_BGR2GRAY)
    small_coins_out = cv2.connectedComponentsWithStats(small_coins, 4, cv2.CV_32S)
    original = np.copy(img)

    small_contours, hierarchy = cv2.findContours(small_coins,
                                           cv2.RETR_CCOMP,
                                           cv2.CHAIN_APPROX_SIMPLE)
    small_count = 0
    for i in small_coins_out[3][1:]:
        (x, y), radius = cv2.minEnclosingCircle(small_contours[small_count])
        j = int(x)
        k = int(y)
        cv2.circle(img, (j, k), int(radius), (0, 0, 255), 3)
        small_count += 1

    big_coins = cv2.cvtColor(big_coins, cv2.COLOR_BGR2GRAY)
    big_coins_out = cv2.connectedComponentsWithStats(big_coins, 4, cv2.CV_32S)

    big_contours, hierarchy = cv2.findContours(big_coins,
                                                 cv2.RETR_CCOMP,
                                                 cv2.CHAIN_APPROX_SIMPLE)
    big_count = 0
    for i in big_coins_out[3][1:]:
        (x, y), radius = cv2.minEnclosingCircle(big_contours[big_count])
        j = int(x)
        k = int(y)
        cv2.circle(img, (j, k), int(radius), (255, 0, 0), 3)
        big_count += 1

    print("You have : " + str(small_count*25 + big_count*50))
    plot_side_by_side(original, img, "image", "coins_detection")


def q13_cont_2(img, circles):
    original = np.copy(img)

    contours, hierarchy = cv2.findContours(circles, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    small_count = 0
    large_count = 0
    radiuses = []

    for i in range(len(contours)):
        (x, y), radius = cv2.minEnclosingCircle(contours[i])
        radius = int(radius)
        radiuses.append(radius)
    radiuses_avg = sum(radiuses) // len(radiuses)

    for i in range(len(contours)):
        (x, y), radius = cv2.minEnclosingCircle(contours[i])
        center = (int(x), int(y))
        radius = int(radius)
        if radiuses_avg <= radius < (radiuses_avg + 20):
            large_count += 1
            cv2.circle(img, center, radius, (255, 0, 0), 3)
        elif radiuses_avg > radius > (radiuses_avg - 20):
            small_count += 1
            cv2.circle(img, center, radius, (0, 0, 255), 3)
    print("You have : " + str(small_count*25 + large_count*50))
    plot_side_by_side(original, img, "image", "general coins")


def q13_general(img):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    ret, threshold = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    if ret < 150.0:
        threshold = cv2.bitwise_not(threshold)
    threshold = cv2.bitwise_not(threshold)
    threshold = cv2.morphologyEx(threshold, cv2.MORPH_OPEN, kernel)
    threshold = cv2.bitwise_not(threshold)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (35, 35))
    threshold = cv2.morphologyEx(threshold, cv2.MORPH_OPEN, kernel)
    q13_cont_2(img, threshold)


def q12(img1, img2):
    original = np.copy(img1)
    cv2.imshow("", original)
    cv2.waitKey()
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    blur1 = cv2.GaussianBlur(gray1, (3, 3), 0)
    ret, threshold1 = cv2.threshold(blur1, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = np.ones((3, 3), dtype=np.uint8)
    threshold1 = cv2.morphologyEx(threshold1, cv2.MORPH_OPEN, kernel)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (45, 45))
    threshold1 = cv2.morphologyEx(threshold1, cv2.MORPH_CLOSE, kernel)
    threshold1 = cv2.bitwise_not(threshold1)
    threshold1 = cv2.cvtColor(threshold1, cv2.COLOR_GRAY2BGR)

    x1, y1, z1 = img1.shape
    img22 = cv2.resize(img2, (y1, x1))
    mask = cv2.bitwise_and(threshold1, img22)
    mask[np.where((mask == [0, 0, 0]).all(axis=2))] = [255, 255, 255]

    img1 = cv2.bitwise_or(img1, threshold1)
    out = cv2.bitwise_and(mask, img1)
    plot_side_by_side(original, out, "image", "Morning with mask")


def main():
    images_folder = "images/"
    # img = cv2.imread(images_folder + "square-circle.png")
    # q1(img)
    # img = cv2.imread(images_folder + "cameraman.png")
    # q2(img)
    # img = cv2.imread(images_folder + "lady.png")
    # q3(img)
    img = cv2.imread(images_folder + "Circle.png")
    q4(img)
    # img = cv2.imread(images_folder + "Circle.png")
    # q5(img)
    # img = cv2.imread(images_folder + "cameraman.png")
    # q6(img)
    # img = cv2.imread(images_folder + "Circle_and_Lines.png")
    # q7(img)
    # img = cv2.imread(images_folder + "coins.png")
    # q13(img)
    # img = cv2.imread(images_folder + "coins_color.jpeg")
    # q13_general(img)
    # img1 = cv2.imread(images_folder + "morning.jpg")
    # img2 = cv2.imread(images_folder + "evening.jpg")
    # q12(img1, img2)


if __name__ == '__main__':
    main()
