import os

def create_directory(dir_name):
    try:
        os.mkdir(dir_name)
        print("OK: {0} directory created.".format(dir_name))
    except FileExistsError:
        print("Skipping: {0} directory already exists.".format(dir_name))
    except PermissionError:
        print("Permission denied: unable to create {0} directory.".format(dir_name))
        exit()
    except Exception as err:
        print("Unexpected error: {0}.".format(err))
        exit()
