import os
import sys
import shutil
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from utils.capture import capture_image
from utils.file_ops import create_directory
from utils.models import train_eval_save_model
from utils.meta_info import torch_info, get_device
from utils.prepare import prepare_dataset, split_dataset
from utils.application import run_application_with_model

def init():
    create_directory("data")
    create_directory("data/images")
    print("---------------------------------------------------------------\n")

def setup(should_capture_image=True, cam_index=0):
    init()
    if should_capture_image:
        capture_image(cam_index=cam_index)
    dataset = prepare_dataset()
    num_classes = len(dataset.classes)
    train, validation = split_dataset(dataset)
    train_eval_save_model(num_classes=num_classes, train=train, validation=validation, device=get_device(), epochs=10)



from utils.win import unlock_system

if __name__ == "__main__":
    torch_info()
    unlock_system()
    exit()
    # 
    cam_index = int(input("Camera to use: "))
    while True:
        print("1. Use application\n2. Setup\n3. Clear data\n4. Exit")
        option = input("Choose: ").lower()

        if option == "1":
            # interval = int(input("Provide time interval (in seconds): "))
            run_application_with_model(device=get_device(), cam_index=cam_index)
        elif option == "2":
            should_capture_image = input("Images already present? (Yes/No):").lower()
            if should_capture_image == "yes" or should_capture_image == "y":
                setup(should_capture_image=False, cam_index=cam_index)
            elif should_capture_image == "no" or should_capture_image == "n":
                setup(should_capture_image=True, cam_index=cam_index)
            else:
                print("Invalid value passed !!!")
                exit()
        elif option == "3":
            print("Clearing data !!!")
            path = "data/images"
            if os.path.exists(path):
                shutil.rmtree(path)
                print("Data cleared !!!")
            else:
                print("Skipping: directory doesn't exists !!!")
            exit()
        elif option == "4":
            print("Exiting !!!")
            exit()
        else:
            print("Invalid option selected !!!")
            exit()