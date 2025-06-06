# 🔓 Face Recognition Automatic Lock
A Python-based facial recognition system that allows users to lock your system using their face. Built using OpenCV and PyTorch, this project demonstrates the application of computer vision for biometric automatic locking mechanism. This program automatically locks the system when you move away from it.

It captures faces, train the model, save the model and then use that model to detect face. Once you run the program, you will have to capture your face first (more the data, better the model), then run the program.

## 🧰 Tech Stack
- Python 3.8+
- OpenCV
- PyTorch
- NumPy
- Face cascade

## 🚀 Getting Started
### 1. Clone the Repository
```bash
$> git clone https://github.com/AshishNikam111000/fr_lock.git
$> cd fr_lock
```
### 2. Create an environment, activate and deactivate environment
```bash
$> python -m venv <environment_name>
$> <environment_name>\Scripts\activate
$> deactivate
```
### 2. After activating environment, simply run the main.py
```bash
$> python main.py
```