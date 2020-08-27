import cv2
import numpy as np


def three_frame_differencing(videopath):
    cap = cv2.VideoCapture(videopath)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    b_frame = np.zeros((height, width), dtype=np.uint8)
    two_frame = np.zeros((height, width), dtype=np.uint8)
    three_frame = np.zeros((height, width), dtype=np.uint8)
    cnt = 0
    while cap.isOpened():
        ret, frame = cap.read()
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if not ret:
            break
        one_frame, two_frame, three_frame = two_frame, three_frame, frame_gray
        abs1 = cv2.absdiff(one_frame, two_frame)  # 相减
        # _, thresh1 = cv2.threshold(abs1, 20, 255, cv2.THRESH_BINARY)  # 二值，大于40的为255，小于0

        abs2 = cv2.absdiff(two_frame, three_frame)

        for row in range(frame.shape[0]):
            for col in range(frame.shape[1]):
                b_frame[row][col] = min(abs1[row][col],abs2[row][col])
        # _, thresh2 = cv2.threshold(abs2, 20, 255, cv2.THRESH_BINARY)

        # binary = cv2.bitwise_and(thresh1, thresh2)  # 与运算
        # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        # erode = cv2.erode(binary, kernel)  # 腐蚀
        # dilate = cv2.dilate(erode, kernel)  # 膨胀
        # dilate = cv2.dilate(dilate, kernel)  # 膨胀

        # img, contours, hei = cv2.findContours(dilate.copy(), mode=cv2.RETR_EXTERNAL,
        #                                       method=cv2.CHAIN_APPROX_SIMPLE)  # 寻找轮廓
        # for contour in contours:
        #     if 100 < cv2.contourArea(contour) < 40000:
        #         x, y, w, h = cv2.boundingRect(contour)  # 找方框
        #         cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255))
        cv2.namedWindow("binary", cv2.WINDOW_NORMAL)
        # cv2.namedWindow("dilate", cv2.WINDOW_NORMAL)
        cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
        cv2.imshow("binary", b_frame)
        # cv2.imshow("dilate", dilate)
        cv2.imshow("frame", frame)

        cv2.imwrite(str(cnt)+".png", b_frame)
        cnt+=1
        if cv2.waitKey(50) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    three_frame_differencing(r'D:\\download\\lensless_system\\selected_videos\\44_blood2_good.avi')