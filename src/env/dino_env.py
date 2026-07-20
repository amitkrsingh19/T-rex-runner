import time
import json 
from collections import deque

import cv2
import numpy as np
from mss import MSS
import pydirectinput
import pytesseract

import gymnasium as gym
from gymnasium.spaces import Box, Discrete

## Create a Enviroment for dino game by extenfing env Class
class DinoEnv(gym.Env):
    """Gymnasium environment for Chrome Dino game."""

    ACTIONS = {0: "noop", 1: "jump", 2: "duck"}
    
    def __init__(self):
        ## call super to get access to all the available methods of Env Class
        super().__init__()
        print("Initialized Game Enviroment...")
        ## Enviroment shape
        self.obs_height = 83
        self.obs_width = 100
        self.frame_stack_size = 4

        ## create observation_space - game enviroment box
        self.observation_space = Box(low=0, high=255, shape=(4,83,100), dtype=np.uint8)
        ## create action space of all actions that can be executed in enviroment
        self.action_space = Discrete(3) ## actions - (jump, duck, do-nothing)

        self.frames = deque(maxlen= 4) ## 4 frames at a time
        self.sct = MSS() ## instanciate MSS - screen capturing

        ## count on current_Step
        self.current_step = 0
        self.min_steps_before_done_check = 5

        ## fps capture
        self.target_fps = 20  
        self.step_duration = 1.0 / self.target_fps 

        ## load all the screen_data points
        config = self.load_config()
        self.active_box = config['active_capture_box']
        self.game_location = config["game_location"]
        self.finish_location = config["finish_location"]
        self.score_location = config['score_location']
        
        self.last_frame = np.zeros((self.obs_height, self.obs_width), dtype=np.uint8)
        
        ## load the game over template
        self.game_over_template = cv2.imread("assets/game_over_template.png", cv2.IMREAD_GRAYSCALE)
        
        print("Enviroment Created...")

    ## compute each sub-area's slice relative to full_region's top-left
    def _slice_region(self,full_gray: np.ndarray, region) -> np.ndarray:
        """Slices a region safely out of the base screen frame array."""
        top, left = region["top"], region["left"]
        height, width = region["height"], region["width"]

        return full_gray[top:top + height, left:left + width]
    
    ## capture full tab screen
    def _capture_full(self) -> np.ndarray:
        """Grab the entire screen covering game + score + finish regions."""
        frame = np.array(self.sct.grab(self.active_box))[:, :, :3]
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    ## capture specific region from screen
    def _capture(self, region: dict) -> np.ndarray:
        """captures specific location in game"""
        frame = np.array(self.sct.grab(region))[:, :, :3]

        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    ## only check when we exceed max steps 
    def _should_check_done(self) -> bool:
        return self.current_step >= self.min_steps_before_done_check
    
    ## load all the config at once
    def load_config(self):
        with open("config.json", "r") as f:
            config = json.load(f)

            print("config.json loaded for game-frame info")
            return config
        
    ## get the screen which lies inside game region
    def get_observation(self, full_gray: np.ndarray= None ): #type: ignore
        gray = self._slice_region(full_gray, self.game_location) 
        # Resize
        resized = cv2.resize(gray, (100, 83))
        self.last_frame = resized

        ## append after resizing img to frames
        self.frames.append(resized)

        if len(self.frames) == 0:
            while len(self.frames) < 4:
                self.frames.append(resized)
        else:
            self.frames.append(resized)

        # Add channel dimension
        #observation = resized[np.newaxis, :, :]

        return np.stack(self.frames, axis=0)
    
    ## model will take a step on an action taken 
    def step(self, action):
        step_start_time = time.time()  ## Start the step timer

        t0 = time.time()
        match action:
            case 1:
                pydirectinput.press('space')

            case 2:
                pydirectinput.keyDown("down")
                time.sleep(0.05)
                pydirectinput.keyUp("down")

        t1 = time.time()

        self.current_step += 1 ## increment everytime a step is taken
        
        full_gray = self._capture_full()
        t2 = time.time()

        obs = self.get_observation(full_gray)
        t3 = time.time()

        terminated = False
        if self._should_check_done():
            terminated = self.is_done(full_gray)
        t4 = time.time()

        reward = -10 if terminated else 1
        truncated = False
        info = {}

        ## DEBUG 
        #if self.current_step % 10 == 0:
        #    print(f"action:{t1-t0:.3f} capture:{t2-t1:.3f} obs:{t3-t2:.3f} done:{t4-t3:.3f}")
        ## if the game ended get score, write in info
        if terminated:
            try:
                score_frame = np.array(self.sct.grab(self.score_location))[:, :, :3]
                gray_score = cv2.cvtColor(score_frame, cv2.COLOR_BGR2GRAY)
                info['score'] = self.get_episode_score(gray_score)
            except Exception:
                info['score'] = 0

        ## Maintain exact loop pacing constraints (FPS Cap)    
        #elapsed_time = time.time() - step_start_time
        #time_left_to_sleep = self.step_duration - elapsed_time
        #if time_left_to_sleep > 0:
        #    time.sleep(time_left_to_sleep)

        return obs, reward, terminated, truncated, info
    
    ## check for game over
    def is_done(self, full_gray: np.ndarray) -> bool:
       """compares the finish region against a saved game over template"""
       screen = self._slice_region(full_gray, self.finish_location)

       template = cv2.resize(self.game_over_template, (screen.shape[1], screen.shape[0])) #type: ignore
       

       diff = cv2.absdiff(screen, template)
       match_ratio = np.mean(diff < 15)
       return True if match_ratio > 0.95 else False
    
    ## restart enviroment from start
    def reset(self, seed=None, options=None): #type: ignore
        super().reset(seed=seed)
        time.sleep(0.5)
        self.current_step = 0
        self.frames.clear()
        ## restart the game
        pydirectinput.click(x=150, y=150)
        pydirectinput.press("space")
        time.sleep(0.3)

        full_gray = self._capture_full()
        obs = self.get_observation(full_gray)
        info = {}
        return obs, info
    
    # visualize the game
    def render(self):
        cv2.imshow("Game Observation Stream", self.last_frame)
        cv2.waitKey(1)

    def close(self):
        cv2.destroyAllWindows()

    ## get each episodes score for logging 
    def get_episode_score(self, pre_captured_gray: np.ndarray):
        """Scrapes the high score digits directly from the canvas area"""

        processed = cv2.threshold(pre_captured_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

        ## Setup Tesseract Digit Config
        custom_config = r'--psm 7 outputbase digits'
        raw_text = pytesseract.image_to_string(processed, config=custom_config).strip()

        ## Extract only numeric digits to filter out random punctuation artifacts
        score_digits = "".join([char for char in raw_text if char.isdigit()])
        
        return int(score_digits) if score_digits else 0
