import time
import json 
from collections import deque

import cv2
import numpy as np
from mss import MSS
import pydirectinput
import pyautogui
import pytesseract

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options

import gymnasium as gym
from gymnasium.spaces import Box, Discrete

## game control class - launch and send commands to the game
class GameControl:
    def __init__(self):
        self.driver = self._launch_browser()

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
         
## Create a Enviroment for dino game by extenfing env Class
class DinoEnv(gym.Env):
    """Gymnasium environment for Chrome Dino game."""

    ACTIONS = {0: "noop", 1: "jump", 2: "duck"}
    
    def __init__(self):
        ## call super to get access to all the available methods of Env Class
        super().__init__()
        print("Initialized Game Enviroment...")
        ## create observation_space - game enviroment box
        self.observation_space = Box(low=0, high=255, shape=(1,83,100), dtype=np.uint8)
        ## create action space of all actions that can be executed in enviroment
        self.action_space = Discrete(3) ## actions - (jump, duck, do-nothing)

        self.frames = deque(maxlen= 4) ## 4 frames at a time
        self.sct = MSS()

        config = self.load_config()
        self.game_location = config["game_location"]
        self.finish_location = config["finish_location"]
        self.score_location = config['score_location']

        self.game_over_template = cv2.imread("game_over_template.png", cv2.IMREAD_GRAYSCALE)
        print("Enviroment Created...")

    def _capture(self, region: dict) -> np.ndarray:
        """Capture a screen region and return grayscale image."""
        frame = np.array(self.sct.grab(region))[:, :, :3]

        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    ## load all the config at once
    def load_config(self):
        with open("config.json", "r") as f:
            config = json.load(f)

            print("config.json loaded for game-frame info")
            return config
        
    ## get the screen which lies inside game region
    def get_observation(self):
        gray = self._capture(self.game_location)

        # Resize
        resized = cv2.resize(gray, (100, 83))
        self.last_frame = resized
        ## append after resizing img to frames
        self.frames.append(resized)
        while len(self.frames) < 4:
            self.frames.append(resized)

        # Add channel dimension
        #observation = resized[np.newaxis, :, :]

        return np.stack(self.frames, axis=0)
    
    ## model will take a step on an action taken 
    def step(self, action):
        match action:
            case 1:
                pydirectinput.press('space')

            case 2:
                pydirectinput.keyDown("down")
                time.sleep(0.05)
                pydirectinput.keyUp("down")
        
        terminated = self.is_done()
        obs = self.get_observation()
        reward = -10 if terminated else 1

        truncated = False

        info = {}

        ## if the game ended get score, write in info
        if terminated:
            info['score'] = self.get_episode_score()

        return obs, reward, terminated, False, info
    
    ## check for game over
    def is_done(self) -> bool:
       screen = self._capture(self.finish_location)


       template = cv2.resize(self.game_over_template, (screen.shape[1], screen.shape[0])) #type: ignore

       diff = cv2.absdiff(screen, template)
       match_ratio = np.mean(diff < 15)
       return True if match_ratio > 0.95 else False
    
    ## restart enviroment from start
    def reset(self, seed=None, options=None): #type: ignore
        time.sleep(0.5)
        super().reset(seed=seed)
        ## restart the game
        pydirectinput.click(x=150, y=150)
        pydirectinput.press("space")
        time.sleep(0.5)

        obs = self.get_observation()
        info = {}
        return obs, info
    
    # visualize the game
    def render(self):
        cv2.imshow("Game", self.last_frame)
        cv2.waitKey(1)

    def close(self):
        cv2.destroyAllWindows()

    ## get each episodes score for logging 
    def get_episode_score(self):
        """Scrapes the high score digits directly from the canvas area"""
        screen = self._capture(self.score_location)

        # Preprocess digits: Invert colors to guarantee black digits on pure white paper space
        processed = cv2.threshold(screen, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

        # Setup Tesseract Digit Config
        custom_config = r'--psm 7 outputbase digits'
        raw_text = pytesseract.image_to_string(processed, config=custom_config).strip()

        # Extract only numeric digits to filter out random punctuation artifacts
        score_digits = "".join([char for char in raw_text if char.isdigit()])
        
        return int(score_digits) if score_digits else None
        

def main():

    game = GameControl()
    game.start_game()
    
    env = DinoEnv()

    #obs, info = env.reset()

    for episode in range(10):
        print()
        print(f"Starting Episode {episode + 1}")
        print("---" * 20)
        obs, info = env.reset()
        done = False
        total_reward = 0

        while not done:
            env.render()

            action = env.action_space.sample() 

            obs, reward, terminated, truncated, info = env.step(action)

            total_reward += reward
            done = terminated or truncated

            if done:
                print(f"Finished Episode with {info}")
                print(f"Total Reward for this Episode is {total_reward}")
                print("---" * 20)
    
    env.close()
if __name__ == "__main__":
    main()