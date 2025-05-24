import cv2
import cv2.data
import numpy as np

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def cap_setup(cam_index=0, window_name="Camera", width=640, height=480, show_window=True):
    if show_window:
        cv2.namedWindow(window_name)
    cap = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("Error: could not open camera.")
        exit()

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width) # width
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height) # height
    print("Camera {0} in use !!!".format(cam_index))
    return cap

def frame_read(cap):
    ret, frame = cap.read()
    if not ret:
        print("Error: could not read frame.")
        cap.release()
        exit()
    frame = cv2.flip(frame, 1)
    return frame

def release_cv_window(cap, closing_msg):
    cap.release()
    cv2.destroyAllWindows()
    print(closing_msg)
    print("---------------------------------------------------------------\n")

def save_image(is_face_detected, face_img, img_counter):
    if is_face_detected:
        cv2.imwrite("data/images/image_{0}.jpg".format(img_counter), face_img)
        print("Image {0} captured".format(img_counter))
        img_counter += 1

def detect_face(img, crop_it=True):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_rect = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    is_face_detected = isinstance(face_rect, np.ndarray)
    if crop_it:
        for (x, y, w, h) in face_rect:
            return is_face_detected, img[y:y+h, x:x+w]
    return is_face_detected, face_rect

def capture_image(cam_index=0):
    img_counter = 1
    cap = cap_setup(cam_index=cam_index, window_name="Capture")
    while True:
        frame = frame_read(cap)
        is_face_detected, face_img = detect_face(frame.copy(), crop_it=True)
        if is_face_detected:
            cv2.putText(frame, "Face detected !!!", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
        else:
            cv2.putText(frame, "Face not detected !!!", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.imshow("Capture", frame)
        
        k = cv2.waitKey(1)
        if k%256 == 27:
            print("Escape hit, closing...")
            break
        elif k%256 == 32:
            save_image(is_face_detected=is_face_detected, face_img=face_img, img_counter=img_counter)
    
    release_cv_window(cap=cap, closing_msg="Image capturing done...")
