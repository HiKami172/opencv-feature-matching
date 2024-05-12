import cv2
import numpy as np


def video_homography(image_path, video_path):
    img_query = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img_query is None:
        print("Error: Query image not loaded. Check the file path.")
        return

    # SIFT feature detector
    sift = cv2.SIFT_create()
    kp_query, desc_query = sift.detectAndCompute(img_query, None)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Video file not opened. Check the file path.")
        return

    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        kp_frame, desc_frame = sift.detectAndCompute(gray_frame, None)

        matches = bf.match(desc_query, desc_frame)
        matches = sorted(matches, key=lambda x: x.distance)
        img_matches = cv2.drawMatches(img_query, kp_query, gray_frame, kp_frame, matches[:10], None,
                                      flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

        if len(matches) > 10:
            src_pts = np.float32([kp_query[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp_frame[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            h, w = img_query.shape
            pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
            dst = cv2.perspectiveTransform(pts, M)

            frame = cv2.polylines(frame, [np.int32(dst)], True, (255, 0, 0), 3, cv2.LINE_AA)

        cv2.imshow('Feature Matching and Object Tracking', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    video_homography('data/photo_3_query.jpg', 'data/video_3_train.mp4')
