import sys
import os
import gym
import numpy as np
import matplotlib.pyplot as plt

# Создаем среду Taxi-v3
env = gym.make('Taxi-v3', render_mode='human')
env.reset()
env.render()

# Получаем количество состояний и действий
n_states = env.observation_space.n
n_actions = env.action_space.n
print("n_states=%i, n_actions=%i" % (n_states, n_actions))

def initialize_policy(n_states, n_actions):
    # Создаем массив для хранения вероятности действий
    policy = np.full((n_states, n_actions), 1.0 / n_actions)  # Равномерная политика
    return policy

# Инициализируем политику
policy = initialize_policy(n_states, n_actions)

# Проверка корректности политики
assert type(policy) in (np.ndarray, np.matrix)
assert np.allclose(policy, 1. / n_actions)
assert np.allclose(np.sum(policy, axis=1), 1)

def generate_session(env, policy, t_max=10**4):
    """
    Играть до конца или t_max тиков.
    :param policy: массив вида [n_states,n_actions] с вероятностями действий
    :returns: список состояний, список действий и сумма наград
    """
    states, actions = [], []
    total_reward = 0.
    s, info = env.reset()  # Изменено на возвращение информации
    for t in range(t_max):
        a = np.random.choice(n_actions, p=policy[s])
        result = env.step(a)  # Получаем результат
        
        # Обработка результата в зависимости от количества возвращаемых значений
        if len(result) == 5:
            new_s, r, done, terminated, info = result  # Обработка 5 значений
        elif len(result) == 4:
            new_s, r, done, info = result
        elif len(result) == 3:
            new_s, r, done = result
        else:
            raise ValueError("Неподдерживаемое количество возвращаемых значений из env.step()")
        
        # Логирование состояния на каждом шаге
        print(f"Step: {t}, State: {s}, Action: {a}, Reward: {r}, Done: {done}, Terminated: {terminated}")

        states.append(s)
        actions.append(a)
        total_reward += r
        s = new_s
        
        if done or terminated:  # Проверка на завершение
            break
            
    return states, actions, total_reward

# Генерация сессии
s, a, r = generate_session(env, policy)
assert type(s) == type(a) == list
assert len(s) == len(a)
assert type(r) in [float, np.float64]

# Визуализация начального распределения вознаграждения
sample_rewards = [generate_session(env, policy, t_max=1000)[-1] for _ in range(200)]
plt.hist(sample_rewards, bins=20)
plt.vlines([np.percentile(sample_rewards, 50)], [0], [100], label="50'th percentile", color='green')
plt.vlines([np.percentile(sample_rewards, 90)], [0], [100], label="90'th percentile", color='red')
plt.legend()
plt.xlabel('Rewards')
plt.ylabel('Frequency')
plt.title('Distribution of Rewards from Sessions')
plt.show()