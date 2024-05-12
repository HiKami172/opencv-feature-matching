import cv2
import numpy as np
import matplotlib.pyplot as plt


def detect_corners(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print("Error: Image not loaded. Check the file path.")
        return

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if gray.dtype != np.uint8:
        print("Image format is not correct. It should be 8-bit unsigned.")
        return

    # Harris corner detection
    dst = cv2.cornerHarris(gray, blockSize=2, ksize=3, k=0.04)
    dst = cv2.dilate(dst, None)
    img[dst > 0.01 * dst.max()] = [0, 0, 255]

    corners = cv2.goodFeaturesToTrack(dst, 4, 0.01, 10)
    if corners is not None:
        for corner in corners:
            x, y = corner.ravel()
            cv2.circle(img, (int(x), int(y)), 5, (255, 255, 0), -1)

    # SIFT key points
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    img_sift = cv2.drawKeypoints(img, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    plt.subplot(121), plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)), plt.title('Top 4 Corners')
    plt.subplot(122), plt.imshow(cv2.cvtColor(img_sift, cv2.COLOR_BGR2RGB)), plt.title('SIFT Key Points')
    plt.show()


if __name__ == '__main__':
    detect_corners('data/photo_1.jpg')
