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
    
    def __init__(self, render_mode = None):
        ## call super to get access to all the available methods of Env Class
        super().__init__()
        print("Initialized Game Enviroment...")
        self.render_mode = render_mode

        ## Enviroment shape
        self.gametime_reward = 0.1
        self.jump_penalty = 0.0
        self.gameover_penalty = -10
        self.duck_penalty = 0.0

        self.prev_frame = None  # For motion detection

        ## create observation_space - game enviroment box
        self.observation_space = Box(low=0, high=255, shape=(83, 100, 4), dtype=np.uint8)
        ## create action space of all actions that can be executed in enviroment
        self.action_space = Discrete(3) ## actions - (jump, duck, do-nothing)

        self.frames = deque(maxlen= 4) ## 4 frames at a time
        self.sct = MSS() ## instanciate MSS - screen capturing

        ## count on current_Step
        self.current_step = 0
        self.min_steps_before_done_check = 10

        ## fps capture
        self.target_fps = 20  
        self.step_duration = 1.0 / self.target_fps 

        ## load all the screen_data points
        config = self.load_config()
        self.active_box = config['active_capture_box']
        self.game_location = config["game_location"]
        self.finish_location = config["finish_location"]
        self.score_location = config['score_location']
        
        self.last_frame = np.zeros((83, 100), dtype=np.uint8) 
        
        ## load the game over template
        self.game_over_template = cv2.imread("assets/game_over_template.png", cv2.IMREAD_GRAYSCALE)
        
        print("Enviroment Created...")

    ## compute each sub-area's slice relative to full_region's top-left
    def _slice_region(self,full_gray: np.ndarray, region) -> np.ndarray:
        """Slices a region safely out of the base screen frame array."""
        top, left = region["top"], region["left"]
        height, width = region["height"], region["width"]

        return full_gray[top:top + height, left:left + width].copy()
    
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

        if full_gray is not None:
            gray = self._slice_region(full_gray, self.game_location) 
            # Resize
            resized = cv2.resize(gray, (100,83))
        else:
            resized = np.zeros((83, 100), dtype = np.uint8)

        # Compute frame difference for motion
        if self.prev_frame is not None:
            diff = cv2.absdiff(resized, self.prev_frame)
        else:
            diff = np.zeros_like(resized)

        self.prev_frame = resized.copy()
        self.last_frame = resized

        ## append after resizing img to frames
        self.frames.append(resized)
        self.frames.append(diff)

        while len(self.frames) < 4:
            self.frames.append(resized if len(self.frames) % 2 == 0 else diff)

        #return np.stack(self.frames, axis=0) ## output shape (4,83,100)
        return np.stack(self.frames, axis=-1) ## OUTPUT SHape (83,100,4)
    
    ## model will take a step on an action taken 
    def step(self, action):
        reward = self.gametime_reward
        step_start_time = time.time()  ## Start the step timer
        t0 = time.time()
        self.last_action = action
        match action:
            case 1:
                pydirectinput.press('space')
                reward += self.jump_penalty

            case 2:
                pydirectinput.keyDown("down")
                reward += self.duck_penalty
                time.sleep(0.08)
                pydirectinput.keyUp("down",_pause =False)

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

        truncated = False
        info = {}

        ## DEBUG 
        #if self.current_step % 10 == 0:
        #    print(f"action:{t1-t0:.3f} capture:{t2-t1:.3f} obs:{t3-t2:.3f} done:{t4-t3:.3f}")
        ## if the game ended get score, write in info
        if terminated:
            try:
                reward = self.gameover_penalty
                score_frame = np.array(self.sct.grab(self.score_location))[:, :, :3]
                gray_score = cv2.cvtColor(score_frame, cv2.COLOR_BGR2GRAY)
                info['score'] = self.get_episode_score(gray_score)
            except Exception:
                info['score'] = 0
        else:
            info['score'] = 0

        ## Maintain exact loop pacing constraints (FPS Cap)    
        elapsed_time = time.time() - step_start_time
        time_left_to_sleep = self.step_duration - elapsed_time
        if time_left_to_sleep > 0:
            time.sleep(time_left_to_sleep)

        return obs, reward, terminated, truncated, info
    
    ## check for game over
    def is_done(self, full_gray: np.ndarray) -> bool:
        """compares the finish region against a saved game over template"""
        screen = self._slice_region(full_gray, self.finish_location)

        template = cv2.resize(self.game_over_template, (screen.shape[1], screen.shape[0])) #type: ignore
       

        diff = cv2.absdiff(screen, template)
        match_ratio = np.mean(diff < 15)

        return True if match_ratio > 0.9 else False
    
    ## restart enviroment from start
    def reset(self, seed=None): #type: ignore
        try:

            super().reset(seed=seed)
            time.sleep(0.5)
            self.current_step = 0
            self.prev_frame = None
            self.frames.clear()
            self.last_score = 0

            ## restart the game
            pydirectinput.press('space') 
            time.sleep(0.1)
            pydirectinput.press("space")
            time.sleep(0.3)

            full_gray = self._capture_full()
            obs = self.get_observation(full_gray)
            info = {}
            return obs, info
        except Exception as e:
            print(f"Error during reset: {e}")
            ## Return blank observation as fallback
            return np.zeros((83, 100, 4), dtype=np.uint8), {'error': str(e)}

    # visualize the game
    def render(self):
        if self.render_mode == "human":
            cv2.imshow("Game Observation Stream", self.last_frame)
            cv2.waitKey(1)
        elif self.render_mode == "rgb_array":
            return self.last_frame

    def close(self):
        cv2.destroyAllWindows()

    def get_episode_info(self):
        """Get diagnostic information about current episode"""
        return {
            'steps': self.current_step,
            'frames_in_buffer': len(self.frames),
            'last_action': self.last_action if hasattr(self, 'last_action') else None,
            'fps': 1.0 / self.step_duration if self.step_duration > 0 else 0
        }
    ## get each episodes score for logging 
    def get_episode_score(self, pre_captured_gray: np.ndarray) -> int:
        """Scrapes the high score digits directly from the canvas area"""

        processed = cv2.threshold(pre_captured_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

        ## remove all noise
        kernel = np.ones((1, 1), np.uint8)
        processed = cv2.morphologyEx(processed, cv2.MORPH_CLOSE, kernel)

        ## Setup Tesseract Digit Config
        custom_config = r'--psm 7 outputbase digits'
        raw_text = pytesseract.image_to_string(processed, config=custom_config).strip()

        ## Extract only numeric digits to filter out random punctuation artifacts
        score_digits = "".join(filter(str.isdigit, raw_text))
        
        try:
            return int(score_digits) if score_digits else 0
        except ValueError:
            return 0
