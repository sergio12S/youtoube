from envirenment import Environment
from agent import DQLAgent
import matplotlib.pyplot as plt
import numpy as np


env = Environment('BTCUSDT', ['close'], 4)

episodes = 10
agent = DQLAgent(gamma=0.5, env=env)
agent.learn(episodes=episodes)

plt.figure(figsize=(10, 6))
x = range(len(agent.averages))
y = np.polyval(np.polyfit(x, agent.averages, deg=3), x)
plt.plot(agent.averages, label='moving average')
plt.plot(x, y, 'r--', label='regression')
plt.xlabel('episodes')
plt.ylabel('total reward')
plt.legend()

agent.test(1)
agent.env.data_['returns'] = env.data_['close'].pct_change()
agent.env.data_['strategy'] = env.data['action'] * env.data_['returns']
agent.env.data_['strategy'].cumsum().plot()
