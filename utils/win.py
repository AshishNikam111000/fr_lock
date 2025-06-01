import ctypes

def is_system_locked():
    user32 = ctypes.windll.User32
    return user32.GetForegroundWindow() == 0

def lock_system():    
    ctypes.windll.user32.LockWorkStation()

def unlock_system():
    print("There is no way to programmitically unlock system.")
