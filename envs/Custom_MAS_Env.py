#MASのカスタム環境をgymnasiumで作成する
#壁の配置を学習する

import os
import random
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
import pygame
import pygame.gfxdraw
import torch
from .potentials import PedPedPotential, PedSpacePotential
from .field_of_view import FieldOfView
from . import simulator
from . import stateutils
from pygame.locals import *
from gymnasium.wrappers import RecordVideo
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
#from stable_baselines3.common.env_checker import check_env
from typing import Optional

SCR_RECT = Rect(0, 0, 1000, 1000)

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
        'render_modes': ['rgb_array'],
        "render_fps": 30,
        }

    def __init__(self, render_mode: Optional[str] = "rgb_array"):
        super().__init__()

        # 強化学習上の最大ステップ数
        self.MAX_STEPS = 100

        # 1ステップで回す時間ステップ数
        self.TIME_STEPS = 100

        # 描画関連
        self.render_mode = render_mode
        pygame.init()
        self.screen = pygame.display.set_mode(SCR_RECT.size)
        self.clock = pygame.time.Clock()
        self.background = pygame.Surface(SCR_RECT.size)
        self.background.fill((255, 255, 255))
        pygame.draw.lines(self.background, BLACK, False, [(200, 0), (0, 0), (0, 1000), (200, 1000)], 2) #wall
        pygame.draw.lines(self.background, BLACK, False, [(800, 0), (1000, 0), (1000, 1000), (800, 1000)], 2) #wall
        pygame.draw.lines(self.background, RED, False, [(200, 0), (800, 0)], 2) #entrance1
        pygame.draw.lines(self.background, GREEN, False, [(200, 1000), (800, 1000)], 2) #entrance2
        self.screen.blit(self.background, (0, 0))
        pygame.display.update()

        # 歩行者スプライトグループ関連
        self.group = pygame.sprite.RenderUpdates()

        # いったん45人の歩行者を整列させて配置
        for i in range(45):
            self.Pedestrian.containers = self.group
            self.Pedestrian(i) #変数名は必要ならあとでいれる

        self.observation_space = gym.spaces.Box(low=[0], high=[100], dtype=np.int32) #velocityのみ？
        #self.action_space = gym.spaces.Discrete(3) # 障害物の配置場所をN箇所のうち1箇所選択する
        self.action_space = gym.spaces.Box(low=[-10, -10, -4, -4], high=[10, 10, 4, 4], dtype=np.int32) # 壁の両端x, yを連続的に変化させる場合
        #self.reward_range =
        self.reset()


    # スプライト作成用内部クラス
    class Pedestrian(pygame.sprite.Sprite):
        # 初期化
        def __init__(self, i):
            pygame.sprite.Sprite.__init__(self, self.containers)
            self.state = torch.zeros((1, 6)) #位置，速度，目的地 y velocity=0

            randomizer = random.choice([0, 1])
            if randomizer == 0:
                #self.state[0, 0:2] = ((torch.rand((1, 2)) - 0.5) * 2.0) * torch.tensor([25.0, 4.5]) # ランダム配置
                self.state[0, 0:2] = torch.tensor[-20 + 5*(i%9), -4 + 2*(i//9)] #整列配置5×9
                self.state[0, 2] = torch.normal(torch.full((1, ), 1.34), 0.26) #x velocity y velocityは変更していないので0
                self.state[0, 4] = 100.0 #x destination
            else:
                self.state[0, 0:2] = torch.tensor[-20 + 5*(i%9), -4 + 2*(i//9)]
                self.state[0, 2] = torch.normal(torch.full((1, ), -1.34), 0.26) #x velocity y velocityは変更していないので0
                self.state[0, 4] = -100.0 #x destination

            #self.state[0, 6] = True 計算中フラグはいったんいれない
            #歩行者スプライト描画関連 サイズはあとで調整
            self.image = pygame.Surface([2, 2], flags=pygame.SRCALPHA) #透過できるような書き方
            self.image.fill((0, 0, 0, 0)) #透明
            pygame.draw.circle(self.image, BLUE, (1, 1), 1)
            self.rect = self.image.get_rect(center=(100, 100)) #center=(100, 100)のときの左上の座標を取得
            self.position = self.state[0, 0:2]

            #歩行者-歩行者間
            #self.ped_ped = PedPedPotential()

        # 歩行者の描写を更新　いったん放置
        def update(self):
            pass
            #simu = simulator.Simulator(ped_ped=self.ped_ped, ped_space=self.ped_space, oversampling=2, delta_t=0.08)
            #simu.integrator = simulator.PeriodicBoundary(simu.integrator, x_boundary=[-25.0, 25.0])
            #with torch.no_grad():
            #    self.state = simu.run(self.state, 1) #これまでの状態がstackされている

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
        ):
        super().reset(seed=seed)
        self.steps = 0

        # 描写の初期化
        self.group.clear(self.screen, self.background)
        self.group.update(self.pos)
        self.group.draw(self.screen)
        dirty_rects = self.group.draw(self.screen)
        pygame.display.update(dirty_rects)

        # 場所を抽出
        observation = self.state[0, 0:2]
        info = {}

        return observation, info


    # 無理やり時間ステップごとの更新できる？
    def render(self, render_mode='rgb_array'):
        self.group.clear(self.screen, self.background)
        self.group.update(self.pos)
        dirty_rects = self.group.draw(self.screen)
        pygame.display.update(dirty_rects)
        return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )

    def step(self, action):
        upper_wall = torch.stack([torch.linspace(-30, 30, 600), torch.linspace(5, 5, 600)], -1) #固定壁　実際は細かい点の集合
        lower_wall = torch.stack([torch.linspace(-30, 30, 600), torch.linspace(-5, -5, 600)], -1) #固定壁　実際は細かい点の集合

        #if action == 0:
        #    movable_wall = torch.stack([torch.linspace(-15, -5, 100), torch.full((600,), 0)], -1) #可動壁
        #elif action == 1:
        #    movable_wall = torch.stack([torch.linspace(-5, 5, 100), torch.full((600,), 0)], -1) #可動壁
        #else:
        #    movable_wall = torch.stack([torch.linspace(5, 15, 100), torch.full((600,), 0)], -1) #可動壁

        movable_wall = torch.stack([torch.linspace(action[0], action[1], max(abs(action[1]-action[0]), abs(action[3]-action[2]))*100),
                                    torch.linspace(action[2], action[3], max(abs(action[1]-action[0]), abs(action[3]-action[2]))*100)], -1) #可動壁 壁のx,yでの位置を指定

        self.ped_ped = PedPedPotential() #歩行者間ポテンシャル
        self.ped_space = PedSpacePotential([upper_wall, lower_wall, movable_wall]) #壁からのポテンシャル更新
        self.state = self.state[-1, :] #最後の状態のみ引き継ぐ(全部削除するとどうなるかわからないから最後のstateだけ残しておく)

        # あらかじめ定めた時間ステップ分をここで更新(100Δtとか)
        # 粒子の流れはループさせる 周期的境界があるので使用
        # 平均速度で報酬を計算する　100時間ステップ分の平均速度
        # terminatedは0固定，truncatedのみ
        simu = simulator.Simulator(ped_ped=self.ped_ped, ped_space=self.ped_space, oversampling=2, delta_t=0.08)
        simu.integrator = simulator.PeriodicBoundary(simu.integrator, x_boundary=[-25.0, 25.0]) # 周期的境界
        with torch.no_grad():
            self.state = simu.run(self.state, self.TIME_STEPS) #これまでの状態がstackされている, 所定の時間ステップ分の更新

        observation = torch.norm(torch.mean(input=self.state[-100:, 2:4], dim=0)) # 平均velocity　45人の100Δt分の平均速度
        reward = torch.norm(torch.mean(input=self.state[-100:, 2:4], dim=0)) # 平均velocityをそのままrewardにする
        terminated = self.is_terminated()
        truncated = self.is_truncated()
        info = {}

        return observation, reward, terminated, truncated, info # terminatedはFalse固定，truncatedのみ

    def get_reward(self, pos):
        pass # 報酬は平均速度が高いほど高い

    def is_terminated(self):
        return False # terminatedはFalse固定

    def is_truncated(self):
        if self.steps >= self.MAX_STEPS:
            return True
        else:
            return False

# 以下実行部分
# ログフォルダの準備
log_dir = './logs/'
os.makedirs(log_dir, exist_ok=True)
env = CustomMASEnv()
env = RecordVideo(CustomMASEnv(), video_folder='./', episode_trigger=lambda x: x % 1000 == 0)
#check_env(env)
env = Monitor(env, log_dir, allow_early_resets=True)
env = DummyVecEnv([lambda: env])

# モデルの準備
model = PPO('MlpPolicy', env, verbose=1)

# 学習の実行
model.learn(total_timesteps=30000)

# 推論の実行
state = env.reset()

while True:
    # 学習環境の描画
    env.render()

    # モデルの推論
    action, _ = model.predict(state, deterministic=True)

    # 1ステップ実行
    state, rewards, terminated, truncated, info = env.step(action)

    # エピソード完了
    if terminated or truncated:
        env.render()
        break

# 学習環境の解放
pygame.quit()
env.close()
