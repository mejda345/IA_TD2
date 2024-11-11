import random
import gym
import numpy as np
from IPython.display import clear_output
import matplotlib.pyplot as plt
import time

def update_q_table(Q, s, a, r, sprime, alpha, gamma):
    Q[s, a] = Q[s, a] + alpha * (r + gamma * np.max(Q[sprime, :]) - Q[s, a])
    return Q

def epsilon_greedy(Q, s, epsilone, decay_rate):
   
    if random.uniform(0, 1) < epsilone * decay_rate:
    
        return random.randint(0, Q.shape[1] - 1)
    else:
        return np.argmax(Q[s, :])

if __name__ == "__main__":
    env = gym.make("Taxi-v3", render_mode="human")

    env.reset()
    env.render()

    # Initialize Q-table
    Q = np.zeros([env.observation_space.n, env.action_space.n])

    # Hyperparameters
    alpha = 0.01  # Learning rate
    gamma = 0.8   # Discount factor
    epsilon = 0.2 # Exploration probability
    decay_rate = 0.99  # Decay rate for the exploration probability
    n_epochs = 100 # Number of episodes to train
    max_itr_per_epoch = 100 # Max iterations per episode

    rewards = []

    for e in range(n_epochs):
        r = 0

        S, _ = env.reset()  # Reset the environment for a new episode

        for _ in range(max_itr_per_epoch):
            # Select action using epsilon-greedy policy with decay rate
            A = epsilon_greedy(Q=Q, s=S, epsilone=epsilon, decay_rate=decay_rate)

            # Perform action and observe new state and reward
            Sprime, R, done, _, info = env.step(A)

            # Accumulate reward
            r += R

            # Update Q-table
            Q = update_q_table(
                Q=Q, s=S, a=A, r=R, sprime=Sprime, alpha=alpha, gamma=gamma
            )

            # Update current state
            S = Sprime

            # Stop if the episode is complete
            if done:
                break

        print("Episode #", e, ": Reward =", r)
        rewards.append(r)

        # Print the learning progress
        if e % 10 == 0:
            print("Learning progress:")
            print("  Episode:", e)
            print("  Average reward:", np.mean(rewards[-10:]))
            print("  Max reward:", np.max(rewards[-10:]))
            print("  Min reward:", np.min(rewards[-10:]))
            print()

    print("Average reward =", np.mean(rewards))

    # Plot the rewards as a function of episodes
    plt.plot(rewards)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Training Rewards")
    plt.show()

    print("Training finished.\n")

    """
    Evaluate the Q-learning algorithm
    """
    env.close()