#MASのカスタム環境をgymnasiumで作成する
#壁の配置を学習する

import os
import random
import numpy as np
import matplotlib.pyplot as plt
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
import simulator
import stateutils
from pygame.locals import *
from gymnasium.wrappers import RecordVideo, render_collection
from stable_baselines3 import A2C, DQN, PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
#from stable_baselines3.common.env_checker import check_env
from typing import Optional

SCR_RECT = Rect(0, 0, 1000, 400)

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

    def __init__(self, render_mode: Optional[str] = "rgb_array_list"):
        super().__init__()

        print("init")

        # 強化学習上の最大ステップ数
        self.MAX_STEPS = 100

        # 歩行者の数
        self.PEOPLE_NUM = 45

        # 1ステップで回す時間ステップ数
        self.TIME_STEPS = 15

        self.step_num = 0
        self.episode_num = 0

        self.RECORD_STEP_CYCLE = 32

        # 記録するエピソード周期
        self.RECORD_EPISODE_CYCLE = 10

        self.OUTPUT_NAME = 'test2_of_batch_size32_n_step32'

        self.tau = 0.5 #???

        self.frames = []
        self.frames1 = []

        self.info = {}

        #initial_stateを作成
        initial_state = torch.zeros((self.PEOPLE_NUM, 6)) #位置，速度，目的地 y velocity=0

        for i in range(self.PEOPLE_NUM):
            if i%2 == 0:
                initial_state[i, 0:2] = torch.tensor([-20 + 5*(i%9), -4 + 2*(i//9)]) #整列配置5×9
                initial_state[i, 2] = torch.normal(torch.full((1, ), 1.34), 0.26) #x velocity y velocityは変更していないので0
                initial_state[i, 4] = 100.0 #x destination
            else:
                initial_state[i, 0:2] = torch.tensor([-20 + 5*(i%9), -4 + 2*(i//9)])
                initial_state[i, 2] = torch.normal(torch.full((1, ), -1.34), 0.26) #x velocity y velocityは変更していないので0
                initial_state[i, 4] = -100.0 #x destination

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

        # 描画関連
        self.render_mode = render_mode
        pygame.init()
        self.screen = pygame.display.set_mode(SCR_RECT.size)
        self.clock = pygame.time.Clock()
        self.background = pygame.Surface(SCR_RECT.size)
        self.background.fill((255, 255, 255))
        pygame.draw.lines(self.background, BLACK, False, [(0, 100), (1000, 100)], 2) #wall
        pygame.draw.lines(self.background, BLACK, False, [(0, 300), (1000, 300)], 2) #wall
        pygame.draw.lines(self.background, RED, False, [(0, 100), (0, 300)], 2) #wall
        pygame.draw.lines(self.background, RED, False, [(998, 100), (998, 300)], 2) #wall
        self.screen.blit(self.background, (0, 0))
        pygame.display.update()

        # 歩行者/壁スプライトグループ
        self.wall_group = pygame.sprite.RenderUpdates()
        self.group = pygame.sprite.RenderUpdates()

        self.Wall.containers = self.wall_group
        self.Wall()

        # 1ステップ目の全員の状態から1人ごとのspriteを作成
        self.pedestrians = []
        for i in range(self.PEOPLE_NUM):
            self.Pedestrian.containers = self.group
            self.pedestrians.append(self.Pedestrian(self.states[i])) #spriteをリストに格納



        self.action_space = gym.spaces.Discrete(2) # 障害物の配置場所をN箇所のうち1箇所選択する
        #self.action_space = gym.spaces.Box(low=np.array([-10.0, -4.0, -10.0, -4.0], dtype=np.float32), high=np.array([10.0, 4.0, 10.0, 4.0], dtype=np.float32)) # 壁の両端x, yを連続的に変化させる場合

        self.observation_space = gym.spaces.Box(low=-10000.0, high=10000.0, shape=(self.TIME_STEPS+1, 45, 10), dtype=np.float32)

        self.upper_wall = torch.stack([torch.linspace(-30, 30, 600), torch.linspace(5, 5, 600)], -1) #固定壁　実際は細かい点の集合
        self.lower_wall = torch.stack([torch.linspace(-30, 30, 600), torch.linspace(-5, -5, 600)], -1) #固定壁　実際は細かい点の集合
        self.movable_wall = torch.stack([torch.linspace(-5, 5, 600), torch.linspace(0, 0, 600)], -1) #可動壁 壁のx,yでの位置を指定

        self.ped_ped = PedPedPotential() #歩行者間ポテンシャル
        self.ped_space = PedSpacePotential([self.upper_wall, self.lower_wall, self.movable_wall]) #壁からのポテンシャル更新

        self.simu = simulator.Simulator(ped_ped=self.ped_ped, ped_space=self.ped_space, oversampling=2, delta_t=0.08)
        self.simu.integrator = simulator.PeriodicBoundary(self.simu.integrator, x_boundary=[-25.0, 25.0]) # 周期的境界
        with torch.no_grad():
            self.states = self.simu.run(initial_state, self.TIME_STEPS) #これまでの状態がstackされている, 所定の時間ステップ分の更新

        self.observation = self.states.to('cpu').detach().numpy().copy() #statesをnumpyに変換
        self.step_num = 0
        self.images = [self.render()]


    # state作成用内部クラス
    class Pedestrian(pygame.sprite.Sprite):
        # 初期化
        def __init__(self, states): #statesは1人分の状態
            pygame.sprite.Sprite.__init__(self, self.containers)
            self.state = states
            self.position = self.state[0:2] #positionの情報を別に保持

            #歩行者スプライト描画関連 サイズはあとで調整
            self.image = pygame.Surface([20, 20], flags=pygame.SRCALPHA) #透過できるような書き方
            self.image.fill((0, 0, 0, 0)) #透明
            pygame.draw.circle(self.image, BLUE, (10, 10), 10)
            self.x = round((self.position[0].item()+25) * 20) #positionの情報を使ってpygame上のx座標を計算
            self.y = round((self.position[1].item()+10) * 20) #positionの情報を使ってpygame上のy座標を計算
            self.rect = self.image.get_rect(center=(self.x, self.y))

        # 歩行者の描写を更新
        def update(self, pedestrian, states): #statesは1人分の全時刻の状態を保持
            self.state = states
            self.position = self.state[0:2]
            pedestrian.x = round((self.position[0].item()+25) * 20)
            pedestrian.y = round((self.position[1].item()+10) * 20)
            self.rect = self.image.get_rect(center=(self.x, self.y))

    class Wall(pygame.sprite.Sprite):
        def __init__(self):
            pygame.sprite.Sprite.__init__(self, self.containers)
            self.image = pygame.Surface([1000, 400], flags=pygame.SRCALPHA)
            self.image.fill((0, 0, 0, 0))
            pygame.draw.line(self.image, GREEN, (400, 200), (600, 200), 2)
            self.rect = self.image.get_rect(center=(500, 200))

        def update(self, action):
            self.image.fill((0, 0, 0, 0))
            if action == 0:
                pygame.draw.line(self.image, GREEN, (400, 200), (600, 200), 2)
            elif action == 1:
                pygame.draw.line(self.image, GREEN, (500, 140), (500, 260), 2)
            elif action == 2:
                pygame.draw.line(self.image, GREEN, (0, 0), (200, 40), 2)
            else:
                pygame.draw.line(self.image, GREEN, (0, 40), (200, 0), 2)
            self.rect = self.image.get_rect(center=(500, 200))

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
        ):
        super().reset(seed=seed)
        self.info = {}
        self.info["action"] = []
        self.info["action_ratio(0)"] = 0

        print("reset")
        #self.step_num += 1
        self.episode_num += 1
        self.frames = []
        self.frames1 = []

        if self.step_num % self.RECORD_EPISODE_CYCLE == 0:
            self.images = [self.render()]

        return self.observation, self.info

    def render(self, render_mode='rgb_array_list'):
        self.group.clear(self.screen, self.background)
        self.wall_group.clear(self.screen, self.background)
        dirty_rects = self.group.draw(self.screen)
        dirty_rects_wall = self.wall_group.draw(self.screen)

        pygame.display.update(dirty_rects_wall)
        pygame.display.update(dirty_rects)

        return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )

    def step(self, action):
        # ここに変化させる障害物の情報が必要
        #self.movable_wall = torch.stack([torch.linspace(action[0], action[2], int(max(abs(action[2]-action[0]), abs(action[3]-action[1])))*100),
        #                            torch.linspace(action[1], action[3], int(max(abs(action[2]-action[0]), abs(action[3]-action[1])))*100)], -1) #可動壁 壁のx,yでの位置を指定

        print(f"step EPISODE:{self.episode_num}")

        self.step_num += 1

        if action == 0:
            self.movable_wall = torch.stack([torch.linspace(-5, 5, 100), torch.linspace(0, 0, 100)], -1)
        elif action == 1:
            self.movable_wall = torch.stack([torch.linspace(0, 0, 20), torch.linspace(-3, 3, 20)], -1)
        elif action == 2:
            self.movable_wall = torch.stack([torch.linspace(-5, 5, 100), torch.linspace(-1, 1, 100)], -1)
        else:
            self.movable_wall = torch.stack([torch.linspace(-5, 5, 100), torch.linspace(1, -1, 100)], -1)

        self.ped_ped = PedPedPotential() #歩行者間ポテンシャル
        self.ped_space = PedSpacePotential([self.upper_wall, self.lower_wall, self.movable_wall]) #壁からのポテンシャル更新

        # あらかじめ定めた時間ステップ分をここで更新(100Δtとか)
        # 粒子の流れはループさせる 周期的境界があるので使用
        # 平均速度で報酬を計算する　100時間ステップ分の平均速度
        # terminatedは0固定，truncatedのみ
        self.simu = simulator.Simulator(ped_ped=self.ped_ped, ped_space=self.ped_space, oversampling=2, delta_t=0.08)
        self.simu.integrator = simulator.PeriodicBoundary(self.simu.integrator, x_boundary=[-25.0, 25.0]) # 周期的境界

        with torch.no_grad():
            self.states = self.simu.run(self.states[-1], self.TIME_STEPS) #これまでの状態がstackされている, 所定の時間ステップ分の更新

        self.wall_group.update(action)

        if (self.episode_num-1) % self.RECORD_EPISODE_CYCLE == 0:
            for t in range(self.TIME_STEPS):
                for i in range(self.PEOPLE_NUM):
                    self.group.update(self.pedestrians[i], self.states[t][i])
                self.images.append(self.render())

        self.observation = self.states.to('cpu').detach().numpy().copy() #statesをnumpyに変換
        reward = np.linalg.norm(self.observation[:, :, 2:4], axis=-1).mean()/self.RECORD_STEP_CYCLE
        #print(reward)

        terminated = False #常にFalse
        truncated = self.is_truncated() #10ステップごとにTrue

        #info = {}

        # if terminated or truncated:
        self.info["action"].append(action)
        self.info["action_ratio(0)"] += (action == 0) / self.RECORD_STEP_CYCLE

        return self.observation, reward, terminated, truncated, self.info

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
env = Monitor(env, log_dir, allow_early_resets=True, info_keywords=("action", "action_ratio(0)"))
env = DummyVecEnv([lambda: env])

# モデルの準備
model = PPO('MlpPolicy', env, learning_rate=0.0003, n_steps=32, batch_size=32, verbose=1) #learning_rateはデフォルト，batch_sizeとn_stepsを調整した

# 学習の実行
model.learn(total_timesteps=3000)

# 推論の実行
state = env.reset()

while True:
    # 学習環境の描画
    # env.render() # render後回し

    # モデルの推論
    action, _ = model.predict(state, deterministic=True)

    # 1ステップ実行
    state, rewards, terminated, truncated, info = env.step(action)

    # エピソード完了
    if terminated or truncated:
        break

# 学習環境の解放
pygame.quit()
env.close()
