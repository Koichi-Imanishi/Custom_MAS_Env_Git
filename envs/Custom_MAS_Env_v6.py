# MASのカスタム環境をgymnasiumで作成する
# 壁の配置を学習する
# imageを出力しない

import os
import random
import numpy as np
import matplotlib.pyplot as plt
from copy import copy
from collections import deque, defaultdict
import matplotlib.animation as animation
import gymnasium as gym
from PIL import Image
from IPython.display import Image as IImage
import pygame
import pygame.gfxdraw
import torch
from potentials.pedped_1d import PedPedPotential
from potentials.pedspace import PedSpacePotential
from field_of_view import FieldOfView
import simulator_mod
import stateutils
from pygame.locals import *
from gymnasium.wrappers import RecordVideo, render_collection
from stable_baselines3 import A2C, DQN, PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
#from stable_baselines3.common.env_checker import check_env
from typing import Optional

SCR_RECT = Rect(0, 0, 360, 480)

############ Some color codes  ############
WHITE = (255, 255, 255)
BLUE = (0,   0, 255)
GREEN = (0, 255,   0)
RED = (255,   0,   0)
BLACK = (0, 0, 0)
GREY = (169, 169, 169)
TEXTCOLOR = (0,   0,  0)
###########################################


class CustomMASEnv(gym.Env):

    metadata = {
        'render_modes': ['rgb_array', 'rgb_array_list'],
        "render_fps": 30,
        }

    def __init__(self, render_mode: Optional[str] = "rgb_array"):
        super().__init__()

        print("init")

        self.tau = 0.5 #relaxation time

        # 歩行者スプライトグループ
        self.group = pygame.sprite.RenderUpdates()

        # 強化学習上の最大ステップ数
        self.MAX_STEPS = 32

        # 歩行者の数
        #self.PEOPLE_NUM = 45
        self.PEOPLE_NUM = 90 #test

        # 1ステップで回す時間ステップ数
        self.TIME_STEPS = 32

        self.step_num = 0
        self.episode_num = 0

        self.RECORD_STEP_CYCLE = 8

        # 記録するエピソード周期
        self.RECORD_EPISODE_CYCLE = 100

        self.OUTPUT_NAME = 'test_PPO_lr_00008_n_steps_32_seed_123'

        # 目的地選択肢, 線分の始点と終点，(x1, y1, x2, y2)
        self.DESTINATION_OPT1 = torch.tensor([6.0, 12.0, 12.0, 12.0])
        self.DESTINATION_OPT2 = torch.tensor([6.0, 7.0, 6.0, 11.0]) #左
        self.DESTINATION_OPT3 = torch.tensor([7.0, 6.0, 11.0, 6.0]) #中央
        self.DESTINATION_OPT4 = torch.tensor([12.0, 7.0, 12.0, 11.0]) #右
        self.DESTINATION_OPT5 = torch.tensor([0.0, 8.2, 0.0, 9.8]) #左出口
        self.DESTINATION_OPT6 = torch.tensor([8.2, 0.0, 9.8, 0.0]) #中央出口
        self.DESTINATION_OPT7 = torch.tensor([18.0, 8.2, 18.0, 9.8]) #右出口
        self.ENDPOINT_LEFT = torch.tensor([-10.0, 8.9, -10.0, 9.1])
        self.ENDPOINT_CENTER = torch.tensor([8.9, -10.0, 9.1, -10.0])
        self.ENDPOINT_RIGHT = torch.tensor([28.0, 8.9, 28.0, 9.1])

        # デフォルトの目的地リスト(queue)を作成
        self.DESTINATION_QUEUE1 = deque([self.DESTINATION_OPT1, self.DESTINATION_OPT3, self.DESTINATION_OPT6, self.ENDPOINT_CENTER]) #直進
        self.DESTINATION_QUEUE2 = deque([self.DESTINATION_OPT1, self.DESTINATION_OPT2, self.DESTINATION_OPT5, self.ENDPOINT_LEFT]) #左折
        self.DESTINATION_QUEUE3 = deque([self.DESTINATION_OPT1, self.DESTINATION_OPT4, self.DESTINATION_OPT7, self.ENDPOINT_RIGHT]) #右折
        self.ALL_QUEUE = [self.DESTINATION_QUEUE1, self.DESTINATION_QUEUE2, self.DESTINATION_QUEUE3]

        #self.states = initial_state #1ステップ目の全員の状態

        initial_state = torch.zeros((self.PEOPLE_NUM, 6)) #位置，速度，目的地 y velocity=0

        #destinationsを作成
        self.destinations = torch.empty((self.PEOPLE_NUM, 4))

        self.dest_queues = dict() #全歩行者の目的地キュー
        for i in range(self.PEOPLE_NUM):
            self.dest_queues[i] = self.DESTINATION_QUEUE1.copy()

        initial_state[:, 0:2] = (torch.rand((self.PEOPLE_NUM, 2))) * torch.tensor([5.8, 23.8]) + torch.tensor([6.1, 12.0])

        for i in range(self.PEOPLE_NUM):
            self.destinations[i, :] = self.dest_queues[i].popleft()
            initial_state[i, 3] = torch.normal(torch.full((1, ), -1.0), 0.1)

        #on_calcを作成
        self.on_calc = torch.ones((self.PEOPLE_NUM, 1)) #全員True

        no_accelerations = torch.zeros((initial_state.shape[0], 2), dtype=initial_state.dtype)
        initial_state = torch.cat((initial_state[:, :4], no_accelerations, initial_state[:, 4:]), dim=-1)

        if hasattr(self.tau, 'shape'):
            tau = self.tau
        else:
            tau = self.tau * torch.ones(initial_state.shape[0], dtype=initial_state.dtype)
        initial_state = torch.cat((initial_state, tau.unsqueeze(-1)), dim=-1)

        preferred_speeds = stateutils.speeds(initial_state)
        initial_state = torch.cat((initial_state, preferred_speeds.unsqueeze(-1)), dim=-1)

        self.states = initial_state #1ステップ目の全員の状態

        # 1ステップ目の全員の状態から1人ごとのspriteを作成
        self.pedestrians = []
        for i in range(self.PEOPLE_NUM):
            self.Pedestrian.containers = self.group
            self.pedestrians.append(self.Pedestrian(self.states[i])) #spriteをリストに格納

        self.states = initial_state.expand(self.TIME_STEPS+1, self.PEOPLE_NUM, 10)

        # 描画関連
        self.render_mode = render_mode
        pygame.init()
        self.screen = pygame.display.set_mode(SCR_RECT.size)
        self.clock = pygame.time.Clock()
        self.background = pygame.Surface(SCR_RECT.size)
        self.background.fill((255, 255, 255))
        pygame.draw.lines(self.background, BLACK, False, [(120, 0), (160, 0)], 2) #wall
        pygame.draw.lines(self.background, BLACK, False, [(200, 0), (240, 0)], 2) #wall
        pygame.draw.lines(self.background, RED, False, [(160, 0), (200, 0)], 2)
        pygame.draw.lines(self.background, BLACK, False, [(0, 120), (0, 160)], 2) #wall
        pygame.draw.lines(self.background, BLACK, False, [(0, 200), (0, 240)], 2) #wall
        pygame.draw.lines(self.background, RED, False, [(0, 160), (0, 200)], 2)
        pygame.draw.lines(self.background, BLACK, False, [(358, 120), (358, 160)], 2) #wall
        pygame.draw.lines(self.background, BLACK, False, [(358, 200), (358, 240)], 2) #wall
        pygame.draw.lines(self.background, RED, False, [(358, 160), (358, 200)], 2)
        pygame.draw.rect(self.background, BLACK, (0, 0, 120, 120))
        pygame.draw.rect(self.background, BLACK, (0, 240, 120, 240))
        pygame.draw.rect(self.background, BLACK, (240, 0, 120, 120))
        pygame.draw.rect(self.background, BLACK, (240, 240, 120, 240))
        self.screen.blit(self.background, (0, 0))
        pygame.display.update()

        self.action_space = gym.spaces.Discrete(3) # 0: 直進, 1: 左折, 2: 右折になるようにする

        self.observation_space = gym.spaces.Box(low=0.0, high=60.0, shape=(self.RECORD_STEP_CYCLE, 3), dtype=np.float32)
        self.observation = [[0]*3, [0]*3, [0]*3, [0]*3, [0]*3, [0]*3, [0]*3, [0]*3]

        self.upper_wall_1 = torch.stack([torch.linspace(6, 8, 10), torch.linspace(0, 0, 10)], -1) #固定壁　実際は細かい点の集合
        self.upper_wall_2 = torch.stack([torch.linspace(10, 12, 10), torch.linspace(0, 0, 10)], -1)
        self.upperleft_wall = torch.stack([torch.linspace(6, 6, 30), torch.linspace(0, 6, 30)], -1)
        self.upperright_wall = torch.stack([torch.linspace(12, 12, 30), torch.linspace(0, 6, 30)], -1)
        self.left_wall_1 = torch.stack([torch.linspace(0, 0, 10), torch.linspace(6, 8, 10)], -1)
        self.left_wall_2 = torch.stack([torch.linspace(0, 0, 10), torch.linspace(10, 12, 10)], -1)
        self.leftupper_wall = torch.stack([torch.linspace(0, 6, 30), torch.linspace(6, 6, 30)], -1)
        self.leftlower_wall = torch.stack([torch.linspace(0, 6, 30), torch.linspace(12, 12, 30)], -1)
        self.right_wall_1 = torch.stack([torch.linspace(18, 18, 10), torch.linspace(6, 8, 10)], -1)
        self.right_wall_2 = torch.stack([torch.linspace(18, 18, 10), torch.linspace(10, 12, 10)], -1)
        self.rightupper_wall = torch.stack([torch.linspace(12, 18, 30), torch.linspace(6, 6, 30)], -1)
        self.rightlower_wall = torch.stack([torch.linspace(12, 18, 30), torch.linspace(12, 12, 30)], -1)
        self.lowerleft_wall = torch.stack([torch.linspace(6, 6, 90), torch.linspace(12, 36, 90)], -1)
        self.lowerright_wall = torch.stack([torch.linspace(12, 12, 90), torch.linspace(12, 36, 90)], -1)

        self.ped_ped = PedPedPotential() #歩行者間ポテンシャル
        self.ped_space = PedSpacePotential([self.upper_wall_1,
                                            self.upper_wall_2,
                                            self.upperleft_wall,
                                            self.upperright_wall,
                                            self.left_wall_1,
                                            self.left_wall_2,
                                            self.leftupper_wall,
                                            self.leftlower_wall,
                                            self.right_wall_1,
                                            self.right_wall_2,
                                            self.rightupper_wall,
                                            self.rightlower_wall,
                                            self.lowerleft_wall,
                                            self.lowerright_wall,
                                            ])

        #self.simu = simulator_mod.Simulator(ped_ped=self.ped_ped, ped_space=self.ped_space, oversampling=2, delta_t=0.08, destinations=self.destinations, dest_queues=self.dest_queues, on_calc=self.on_calc)

        #with torch.no_grad():
        #    self.states = self.simu.run(initial_state, 1) #これまでの状態がstackされている, 所定の時間ステップ分の更新
        #initial_stateを作成

    # state作成用内部クラス
    class Pedestrian(pygame.sprite.Sprite):
        # 初期化
        def __init__(self, states): #statesは1人分の状態
            pygame.sprite.Sprite.__init__(self, self.containers)
            self.state = states
            self.position = self.state[0:2] #positionの情報を別に保持
            self.destination = 0 # 目的地初期化
            self.destination_queue = deque() # 目的地キュー

            #歩行者スプライト描画関連 サイズはあとで調整
            self.image = pygame.Surface([20, 20], flags=pygame.SRCALPHA) #透過できるような書き方
            self.image.fill((0, 0, 0, 0)) #透明
            pygame.draw.circle(self.image, BLUE, (10, 10), 10)
            self.x = round((self.position[0].item()) * 20) #positionの情報を使ってpygame上のx座標を計算
            self.y = round((self.position[1].item()) * 20) #positionの情報を使ってpygame上のy座標を計算
            self.rect = self.image.get_rect(center=(self.x, self.y))

        # 歩行者の描写を更新
        def update(self, pedestrian, states): #statesは1人分の全時刻の状態を保持
            self.state = states
            self.position = self.state[0:2]
            pedestrian.x = round((self.position[0].item()) * 20) #positionの情報を使ってpygame上のx座標を計算
            pedestrian.y = round((self.position[1].item()) * 20) #positionの情報を使ってpygame上のy座標を計算
            self.rect = self.image.get_rect(center=(self.x, self.y))

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
        ):
        super().reset(seed=seed)

        print("reset")
        self.episode_num += 1
        self.frames = []
        self.frames1 = []

        self.info = {}
        self.info["left_people_num"] = []
        self.info["center_people_num"] = []
        self.info["right_people_num"] = []
        self.info["action"] = []

        self.before_action = 0

        #self.observation = self.states.to('cpu').detach().numpy().copy() #statesをnumpyに変換

        if self.step_num % self.RECORD_EPISODE_CYCLE == 0:
            self.images = [self.render()]

        return self.observation, self.info

    def render(self, render_mode='rgb_array'):
        self.group.clear(self.screen, self.background)
        dirty_rects = self.group.draw(self.screen)

        pygame.display.update(dirty_rects)

        return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )

    def step(self, action):

        print(f"step EPISODE:{self.episode_num}")

        self.step_num += 1

        if action == self.before_action: #誘導方向が変化しない場合何もしない
            pass

        elif action == 0: #直進
            for i in range(self.PEOPLE_NUM):
                if len(self.dest_queues[i]) == 3:
                    self.dest_queues[i] = self.DESTINATION_QUEUE1.copy()
                    self.dest_queues[i].popleft()

                #elif len(self.dest_queues[i]) == 2:
                #    self.dest_queues[i] = self.DESTINATION_QUEUE1.copy()
                #    self.dest_queues[i].popleft()
                #    self.dest_queues[i].popleft()

        elif action == 1: #左折
            for i in range(self.PEOPLE_NUM):
                if len(self.dest_queues[i]) == 3:
                    self.dest_queues[i] = self.DESTINATION_QUEUE2.copy()
                    self.dest_queues[i].popleft()

                #elif len(self.dest_queues[i]) == 2:
                #    self.dest_queues[i] = self.DESTINATION_QUEUE2.copy()
                #    self.dest_queues[i].popleft()
                #    self.dest_queues[i].popleft()

        else: #右折
            for i in range(self.PEOPLE_NUM):
                if len(self.dest_queues[i]) == 3:
                    self.dest_queues[i] = self.DESTINATION_QUEUE3.copy()
                    self.dest_queues[i].popleft()

                #elif len(self.dest_queues[i]) == 2:
                #    self.dest_queues[i] = self.DESTINATION_QUEUE3.copy()
                #    self.dest_queues[i].popleft()
                #    self.dest_queues[i].popleft()

        self.before_action = action

        self.ped_ped = PedPedPotential() #歩行者間ポテンシャル

        self.ped_space = PedSpacePotential([self.upper_wall_1,
                                            self.upper_wall_2,
                                            self.upperleft_wall,
                                            self.upperright_wall,
                                            self.left_wall_1,
                                            self.left_wall_2,
                                            self.leftupper_wall,
                                            self.leftlower_wall,
                                            self.right_wall_1,
                                            self.right_wall_2,
                                            self.rightupper_wall,
                                            self.rightlower_wall,
                                            self.lowerleft_wall,
                                            self.lowerright_wall,
                                            ]) #壁からのポテンシャル更新

        # あらかじめ定めた時間ステップ分をここで更新(100Δtとか)
        # 粒子の流れはループさせる 周期的境界があるので使用
        # 平均速度で報酬を計算する　100時間ステップ分の平均速度
        # terminatedは0固定，truncatedのみ
        self.simu = simulator_mod.Simulator(ped_ped=self.ped_ped, ped_space=self.ped_space, oversampling=2, delta_t=0.08, destinations=self.destinations, dest_queues=self.dest_queues, on_calc=self.on_calc)
        self.simu.integrator = simulator_mod.PeriodicBoundary_mod(self.simu.integrator, x_boundary=[0.0, 18.0], y_boundary=[0.0, 24.0], destinations=self.destinations, dest_queues=self.dest_queues, action=action, ALL_QUEUE = self.ALL_QUEUE) # 周期的境界
        #self.simu.integrator = simulator_mod.PeriodicBoundary(self.simu.integrator, x_boundary=[-25.0, 25.0]) # 周期的境界

        with torch.no_grad():
            self.states = self.simu.run(self.states[-1], self.TIME_STEPS) #これまでの状態がstackされている, 所定の時間ステップ分の更新

        #for t in range(self.TIME_STEPS):
        #    for i in range(self.PEOPLE_NUM):
        #        self.group.update(self.pedestrians[i], self.states[t][i])
        #    self.render()

        if (self.episode_num-1) % self.RECORD_EPISODE_CYCLE == 0:
            for t in range(self.TIME_STEPS):
                for i in range(self.PEOPLE_NUM):
                    self.group.update(self.pedestrians[i], self.states[t][i])
                self.images.append(self.render())

        self.info["action"].append(action)
        self.info["left_people_num"].append(torch.sum((self.states[-1, :, 0] > 0.0) & (self.states[-1, :, 0] < 6.0) & (self.states[-1, :, 1] > 6.0) & (self.states[-1, :, 1] < 12.0)).item())
        self.info["center_people_num"].append(torch.sum((self.states[-1, :, 0] > 6.0) & (self.states[-1, :, 0] < 12.0) & (self.states[-1, :, 1] > 0.0) & (self.states[-1, :, 1] < 6.0)).item())
        self.info["right_people_num"].append(torch.sum((self.states[-1, :, 0] > 12.0) & (self.states[-1, :, 0] < 18.0) & (self.states[-1, :, 1] > 6.0) & (self.states[-1, :, 1] < 12.0)).item())

        self.observation.pop(0)
        self.observation.append([self.info["left_people_num"][-1], self.info["center_people_num"][-1], self.info["right_people_num"][-1]])

        terminated = False #self.is_terminated() #常にFalse
        #truncated = False
        truncated = self.is_truncated()

        a =max(self.info["left_people_num"][-1], self.info["center_people_num"][-1], self.info["right_people_num"][-1]) - min(self.info["left_people_num"][-1], self.info["center_people_num"][-1], self.info["right_people_num"][-1])
        if a <= 5:
            reward = 55

        elif a <= 10:
            reward = 45

        elif a <= 15:
            reward = 30

        else:
            reward = 0

        reward -= 1 * ((self.info["left_people_num"][-1] <= 3) + (self.info["center_people_num"][-1] <= 3) + (self.info["right_people_num"][-1] <= 3))
        reward -= 1 * ((self.info["left_people_num"][-1] <= 1) + (self.info["center_people_num"][-1] <= 1) + (self.info["right_people_num"][-1] <= 1))
        reward -= 1 * ((self.info["left_people_num"][-1] == 0) + (self.info["center_people_num"][-1] == 0) + (self.info["right_people_num"][-1] == 0))

        reward -= 1 * ((self.info["left_people_num"][-1] >= 20) + (self.info["center_people_num"][-1] >= 20) + (self.info["right_people_num"][-1] >= 20))
        reward -= 1 * ((self.info["left_people_num"][-1] >= 22) + (self.info["center_people_num"][-1] >= 22) + (self.info["right_people_num"][-1] >= 22))
        reward -= 1 * ((self.info["left_people_num"][-1] >= 25) + (self.info["center_people_num"][-1] >= 30) + (self.info["right_people_num"][-1] >= 25))
        reward -= 1 * ((self.info["left_people_num"][-1] >= 30) + (self.info["center_people_num"][-1] >= 40) + (self.info["right_people_num"][-1] >= 30))

        if self.step_num % self.RECORD_STEP_CYCLE == 0:
            if self.info["left_people_num"] == [0] * self.RECORD_STEP_CYCLE:
                reward -= 30

            if self.info["center_people_num"] == [0] * self.RECORD_STEP_CYCLE:
                reward -= 30

            if self.info["right_people_num"] == [0] * self.RECORD_STEP_CYCLE:
                reward -= 30

            if len(set(self.info["action"])) == 1:
                reward -= 30
            elif len(set(self.info["action"])) == 2:
                reward -= 20

            reward = max(reward, 0)

        return self.observation, reward, terminated, truncated, self.info

    def is_terminated(self):
        if self.on_calc.sum().item() == 0:
            print("terminated")
            return True

        else:
            return False

    def is_truncated(self):
        if self.step_num % self.RECORD_STEP_CYCLE == 0:
            print("truncated")
            #print(self.step_num)

            if (self.episode_num-1) % self.RECORD_EPISODE_CYCLE == 0:
                for image in self.images:
                    image = (image).astype(np.uint8)
                    img = Image.fromarray(image)
                    self.frames1.append(img)

                print(len(self.frames1))
                self.frames1[0].save(f"{self.OUTPUT_NAME}_{self.episode_num}.gif", save_all=True, append_images=self.frames1[1:], duration=10, loop=0)
                IImage(filename=f"{self.OUTPUT_NAME}_{self.episode_num}.gif")

            return True

        else:
            return False

# 以下実行部分
# ログフォルダの準備
log_dir = './logs/'
os.makedirs(log_dir, exist_ok=True)
env = CustomMASEnv()
#env = RecordVideo(CustomMASEnv(), video_folder='./', episode_trigger=lambda x: x % 10 == 0)
#env = render_collection.RenderCollection(env, pop_frames=False, reset_clean=True)
#check_env(env)
env = Monitor(env, log_dir, allow_early_resets=True, info_keywords=("action", "left_people_num", "center_people_num", "right_people_num"))
env = DummyVecEnv([lambda: env])
normalized_vec_env = VecNormalize(env)

# モデルの準備
model = PPO('MlpPolicy', normalized_vec_env, learning_rate=0.0008, n_steps=64, batch_size=64, verbose=1, seed=123) #learning_rate，batch_sizen_stepsを調整した seed=123

# 学習の実行
model.learn(total_timesteps=10000)

# 推論の実行
state = normalized_vec_env.reset()

while True:
    # 学習環境の描画
    # env.render() # render後回し

    # モデルの推論
    action, _ = model.predict(state, deterministic=True)

    # 1ステップ実行
    state, rewards, terminated, truncated, info = normalized_vec_env.step(action)

    # エピソード完了
    if terminated or truncated:
        break

# 学習環境の解放
pygame.quit()
normalized_vec_env.close()
