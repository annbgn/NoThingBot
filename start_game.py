import ctypes
import datetime
import time

from numpy import *
from PIL import ImageGrab, ImageOps
from pywinauto.application import Application
from pywinauto.keyboard import send_keys

from edge_detection import detect_edge

GAMEPATH = r"D:/Steam/steamapps/common/NO THING/no_thing.exe"

SCREENSIZE = (
    ctypes.windll.user32.GetSystemMetrics(0),
    ctypes.windll.user32.GetSystemMetrics(1),
)


def start_app():
    app = Application(backend="uia").start(GAMEPATH)
    dlg = app.top_window()
    play_button = dlg.child_window(title="Play!", auto_id="1", control_type="Button")
    play_button.click()
    time.sleep(5)


def start_game():
    send_keys("{SPACE}")
    time.sleep(3)
    send_keys("{SPACE}")


def exit_game():
    time.sleep(15)
    send_keys("{ESC}")


def restart_game():
    time.sleep(1)
    send_keys("{SPACE}")


def check_failure() -> bool:
    red = ImageOps.grayscale(ImageGrab.grab(bbox=(0, 700, 350, 800)))
    yellow = ImageOps.grayscale(ImageGrab.grab(bbox=(1820, 700, 1920, 800)))
    cyan = ImageOps.grayscale(ImageGrab.grab(bbox=(0, 1020, 350, 1060)))
    red_array = array(red.getcolors())
    yellow_array = array(yellow.getcolors())
    cyan_array = array(cyan.getcolors())

    r = red_array.sum()
    y = yellow_array.sum()
    c = cyan_array.sum()

    if r == 35076 and y == 10191 and c == 14179:
        print("fail! -- {}".format(datetime.datetime.now()))
        return True
    return False


def main():
    start_app()
    start_game()
    time.sleep(5)

    for _ in range(10 * 60):  # todo replace with while
        is_fail = check_failure()
        if is_fail:
            restart_game()
        else:
            current_screen = ImageGrab.grab(bbox=(0, 107, 1919, 755))
            try:
                detect_edge(current_screen)
            except Exception as exc:
                print(exc)
                import winsound
                winsound.Beep(500, 1000)
                raise Exception
        time.sleep(0.3)  # todo 0.1 or less

    exit_game()


if __name__ == "__main__":
    main()
