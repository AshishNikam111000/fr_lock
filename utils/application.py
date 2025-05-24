import cv2
import time
import torch
from utils.models import FaceCNN
from torchvision import transforms
from utils.win import is_system_locked, lock_system, unlock_system
from utils.capture import cap_setup, frame_read, release_cv_window, detect_face


def run_application_with_model(device, cam_index=0, interval=15):
    num_classes = 1
    input_size = 160
    prev_time = time.time()
    label_map = {0: "Recognized"}

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor()
    ])

    model = FaceCNN(num_classes=num_classes, input_shape=(3, input_size, input_size))
    model.load_state_dict(torch.load("face_recognition_model.pth", map_location=device, weights_only=True))
    model.to(device=device)
    model.eval()

    cap = cap_setup(cam_index=cam_index, window_name="Face recognition", show_window=False)
    while True:
        frame = frame_read(cap)
        curr_time = time.time()
        elapsed_time = curr_time - prev_time

        if elapsed_time > interval:
            prev_time = curr_time

            is_face_detected, face_rect = detect_face(frame.copy(), crop_it=False)
            if is_face_detected:
                for (x, y, w, h) in face_rect:
                    # cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    face_img = frame[y:y+h, x:x+w]
                    # cv2.putText(frame, "Face detected !!!", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
                    try:
                        face_tensor = transform(face_img).unsqueeze(0).to(device)
                        with torch.no_grad():
                            output = model(face_tensor)
                            _, predicted = torch.max(output, 1)
                            label = label_map.get(predicted.item(), "Unknown")
                            if predicted.item() == 0 and is_system_locked():
                                print("{0} - Unlocking system...".format(label))
                                unlock_system()
                            # cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                    except Exception as err:
                        print("Error processing face:", err)
            else:
                if not is_system_locked():
                    print("Locking system...")
                    lock_system()
                # cv2.putText(frame, "Face not detected !!!", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
            # cv2.imshow("Face recognition", frame)
            
        k = cv2.waitKey(1)
        if k%256 == 27:
            print("Escape hit, closing...")
            break

    release_cv_window(cap=cap, closing_msg="Closing webcam...")