from pywinauto.keyboard import send_keys
import time
from pywinauto.application import Application
from pprint import pprint

GAMEPATH = r"D:/Steam/steamapps/common/NO THING/no_thing.exe"

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


def main():
    start_app()
    start_game()
    exit_game()


if __name__ == '__main__':
    main()
