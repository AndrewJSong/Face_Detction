import numpy as np
import cv2
import dlib
import time

Drow_Box = True
Drow_Text = False


def main(INPUT=0):
    cap = cv2.VideoCapture(INPUT)
    FPS = 0
    while 1:
        start_time = time.time()
        ret, img = cap.read()

        # 取灰度，提高处理速度
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        # rects：人脸数量rects
        rects = detector(img_gray, 0)

        for index, rect in enumerate(rects):
            if Drow_Box:
                left = rect.left()
                top = rect.top()
                right = rect.right()
                bottom = rect.bottom()
                cv2.rectangle(img, (left, top),
                              (right, bottom), (255, 255, 0), 2)
            landmarks = np.matrix([[p.x, p.y]
                                   for p in predictor(img, rect).parts()])
            for idx, point in enumerate(landmarks):
                # 68点的坐标
                pos = (point[0, 0], point[0, 1])
                # 利用cv2.circle给每个特征点画一个圈，共68个
                cv2.circle(img, pos, 2, color=(0, 255, 0))
                if Drow_Text:
                    # 利用cv2.putText输出1-68
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    cv2.putText(img, str(idx + 1), pos, font,
                                0.5, (0, 0, 255), 1, cv2.LINE_AA)

        end_time = time.time()
        use_time = end_time-start_time

        FPS = int(1/use_time)

        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, str(FPS), (10, 20), font, 0.8, (0, 0, 255), 1)

        cv2.namedWindow("img", 2)
        cv2.imshow("img", img)
        c = cv2.waitKey(1)

        if c == 27:  # use ESC to quit
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(
        "dlib/shape_predictor_68_face_landmarks.dat")
    main(0)
