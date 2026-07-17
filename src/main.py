import time
import cv2
import numpy as np
import mss
import pydirectinput
import pyautogui

from selenium import webdriver
from selenium.webdriver.chrome.options import Options


# Browser Setup
options = Options()
options.add_experimental_option("detach", True)

driver = webdriver.Chrome(options=options)


def setup_game():

    driver.get("https://chromedino.com/dina/")
    driver.maximize_window()

    time.sleep(2)

    screen_w, screen_h = pyautogui.size()

    pyautogui.click(screen_w // 2, screen_h // 2)

    time.sleep(0.5)

    pydirectinput.press("space")


"""def get_monitor():

    # Capture whole monitor once
    with mss.MSS() as sct:

        img = np.array(sct.grab(sct.monitors[1]))

        cv2.imshow("Select ROI", img)

        # Select the game area manually ONCE
        x, y, w, h = cv2.selectROI("Select ROI", img)

        print(x,y,w,h)
        cv2.destroyAllWindows()

    return {
        "left": int(x),
        "top": int(y),
        "width": int(w),
        "height": int(h),
    }
"""

def load_monitor():
    import json
    with open("config.json", "r") as f:
        monitor = json.load(f)
        return monitor

def main():

    setup_game()

    monitor = load_monitor()

    print(monitor)

    with mss.MSS() as sct:

        last = time.time()
        frame_count = 0
        while True:

            frame = np.array(sct.grab(monitor))

            gray = cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY)

            
            gray = cv2.resize(gray, (84, 84))

            cv2.imshow("Dino", gray)
            if frame_count % 50 == 0 and frame_count <= 400:
                cv2.imwrite(f"frame_{frame_count}.jpg",gray)
                

            frame_count += 1
            fps = 1 / (time.time() - last)
            last = time.time()

            print(f"FPS : {fps:.2f}")

            if cv2.waitKey(1) == ord("q"):
                break

        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()