import cv2
import numpy as np

cap = cv2.VideoCapture('D:\\download\\lensless_system\\selected_videos\\44_blood2_good.avi')
# params for ShiTomasi corner detection 特征点检测
feature_params = dict( maxCorners = 10000000,
                       qualityLevel = 0.1,
                       minDistance = 90,
                       blockSize = 30 )

# Lucas kanade params
lk_params = dict(winSize = (50, 50),
                 maxLevel = 2,
                 criteria = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 10, 0.03))

# Create some random colors 画轨迹
color = np.random.randint(0,255,(100,3))

# #  Mouse function
# def select_point(event, x, y, flags, params):
#     global point, point_selected, old_points
#     if event == cv2.EVENT_LBUTTONDOWN:
#         point = (x, y)
#         point_selected = True
#         old_points = np.array([[x, y]], dtype=np.float32)

#Create old frame
_, frame = cap.read()
old_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
# roi = np.zeros_like(old_gray)
# x,y,w,h = 266,143,150,150
# roi[y:y+h, x:x+w] = 255
old_points = cv2.goodFeaturesToTrack(old_gray, mask = old_gray, **feature_params)


cv2.namedWindow("Frame")
# cv2.setMouseCallback("Frame", select_point)
point_selected = False
point = ()
# old_points = np.array([[]])

# Create a mask image for drawing purposes
mask = np.zeros_like(frame)

while True:
    _, frame = cap.read()
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # if point_selected is True:
    # cv2.circle(frame, point, 5, (0, 0, 255), 2)
    new_points, status, error = cv2.calcOpticalFlowPyrLK(old_gray, gray_frame, old_points, None, **lk_params)
    # x, y = new_points.ravel()
    # cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

    # Select good points
    good_new = new_points[status == 1]
    good_old = old_points[status == 1]

    # draw the tracks
    for i,(new,old) in enumerate(zip(good_new,good_old)):
        a,b = new.ravel()
        c,d = old.ravel()
        mask = cv2.line(mask, (a,b),(c,d), color[i].tolist(), 2)
        frame = cv2.circle(frame,(a,b),3,color[i].tolist(),-1)

    img = cv2.add(frame,mask)
    # cv2.imshow('frame',img)
    l = cv2.pyrDown(mask)
    cv2.imshow('frame', l)

    first_level = cv2.pyrDown(frame)
    # cv2.imshow("Frame", frame)
    cv2.imshow("First level", first_level)

    old_gray = gray_frame.copy()
    old_points = new_points

    key = cv2.waitKey(1)
    if key == 27:
        break
    elif key == ord(' '):  # 按下空格键时，暂停
        cv2.waitKey(0)

cap.release()
cv2.destroyAllWindows()