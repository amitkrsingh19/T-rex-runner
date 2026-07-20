import time

import pydirectinput
import pyautogui

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options

## game control class - launch and send commands to the game
class GameControl:
    def __init__(self):
        self.driver = self._launch_browser()
        pydirectinput.PAUSE = 0.07

    ## launch chrome browser and return driver
    def _launch_browser(self):
        options = Options()
        ## start with full screen
        options.add_argument("--start-maximized")
        ## then detach the window
        options.add_experimental_option("detach", True)
        ## forcefully disable darkmode
        options.add_argument("--disable-features=DarkMode")
        ## chrome webdriver 
        driver = webdriver.Chrome(options = options)
        time.sleep(2)
        return driver
    
    ## open dino game using search bar
    def open_dino_game(self):
        pyautogui.hotkey("fn", "f6")
        time.sleep(0.3)
        pyautogui.typewrite("chrome://dino", interval = 0.05)
        pyautogui.press('enter')
        time.sleep(2)

    ## confirm the game has loaded correctly
    def wait_for_game_ready(self):
        print("Waiting for Dino game canvas to appear...")
        while True:
            try:
                canvas = self.driver.find_element(By.CLASS_NAME, "runner-canvas")
                if canvas.is_displayed():
                    print("Game is ready.")
                    print("---" *20)
                    break
            except:
                pass

    ## start game at once    
    def start_game(self):
        print("---"*20)
        print("Starting game...")

        self.open_dino_game()
        self.wait_for_game_ready()

        pydirectinput.press('space')
        time.sleep(1)
         