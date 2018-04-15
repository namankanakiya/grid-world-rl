from gridworld import GridWorldMDP
from qlearn import QLearner
import random

import numpy as np
import matplotlib.pyplot as plt
import time


def plot_convergence(utility_grids, policy_grids):
    fig, ax1 = plt.subplots()
    ax1.set_xlabel("# Iterations")
    ax2 = ax1.twinx()
    utility_ssd = np.sum(np.square(np.diff(utility_grids)), axis=(0, 1))
    ax1.plot(utility_ssd, 'b.-')
    ax1.set_ylabel('Change in Utility', color='b')

    policy_changes = np.count_nonzero(np.diff(policy_grids), axis=(0, 1))
    ax2.plot(policy_changes, 'r.-')
    ax2.set_ylabel('Change in Best Policy', color='r')


def create_big_maze():
    shape = (25,25)
    goal = (0, -1)
    numTraps = 20
    numObstacles = 15
    spacesTaken = []
    spacesTaken.append(goal)
    obstacles = []
    traps = []
    default_reward = -0.1
    goal_reward = 1
    trap_reward = -1
    reward_grid = np.zeros(shape) + default_reward
    terminal_mask = np.zeros_like(reward_grid, dtype=np.bool)
    goal_mask = np.zeros_like(reward_grid, dtype=np.bool)
    avoid_mask = np.zeros_like(reward_grid, dtype=np.bool)
    obstacle_mask = np.zeros_like(reward_grid, dtype=np.bool)

    reward_grid[goal] = goal_reward
    goal_mask[goal] = True
    terminal_mask[goal] = True
    numHigh = 24
    random.seed(45)
    for i in range(numTraps):
        while True:
            newTrap = (random.randint(0, numHigh), random.randint(0, numHigh))
            if newTrap not in spacesTaken:
                spacesTaken.append(newTrap)
                traps.append(newTrap)
                reward_grid[newTrap] = trap_reward
                terminal_mask[newTrap] = True
                avoid_mask[newTrap] = True
                break

    for i in range(numObstacles):
        while True:
            newObstacle = (random.randint(0, numHigh), random.randint(0, numHigh))
            if newObstacle not in spacesTaken:
                spacesTaken.append(newObstacle)
                obstacles.append(newObstacle)
                reward_grid[newObstacle] = 0
                obstacle_mask[newObstacle] = True
                break

    return reward_grid, terminal_mask, goal_mask, avoid_mask, obstacle_mask


def create_small_maze():
    shape = (6, 8)
    goal = (0, -1)
    trap = (1, -1)
    obstacles = []
    obstacles.append((1, 1))
    obstacles.append((2, 2))
    obstacles.append((3, 5))
    obstacles.append((2, 6))
    default_reward = -0.1
    goal_reward = 1
    trap_reward = -1

    reward_grid = np.zeros(shape) + default_reward
    reward_grid[goal] = goal_reward
    reward_grid[trap] = trap_reward
    for obstacle in obstacles:
        reward_grid[obstacle] = 0

    terminal_mask = np.zeros_like(reward_grid, dtype=np.bool)
    terminal_mask[goal] = True
    terminal_mask[trap] = True

    goal_mask = np.zeros_like(reward_grid, dtype=np.bool)
    goal_mask[goal] = True

    avoid_mask = np.zeros_like(reward_grid, dtype=np.bool)
    avoid_mask[trap] = True

    obstacle_mask = np.zeros_like(reward_grid, dtype=np.bool)
    for obstacle in obstacles:
        obstacle_mask[obstacle] = True
    return reward_grid, terminal_mask, goal_mask, avoid_mask, obstacle_mask


def generate_maze(newInt):
    if newInt == 0:
        return create_small_maze()
    elif newInt == 1:
        return create_big_maze()


def print_spaces():
    print("_____________________________________________")
    print("__________________")
    print("_____________________________________________")

if __name__ == '__main__':

    for i in range(2):
        print_spaces()
        print("Generating Maze")
        reward_grid, terminal_mask, goal_mask, avoid_mask, obstacle_mask = generate_maze(i)

        if i == 0:
            prefix = "small_"
        else:
            prefix = "large_"
        # Default run of small, 0.8 stochasticity, discount = 0.9
        gw = GridWorldMDP(reward_grid=reward_grid,
                          obstacle_mask=obstacle_mask,
                          terminal_mask=terminal_mask,
                          action_probabilities=[
                              (-1, 0.1),
                              (0, 0.8),
                              (1, 0.1),
                          ],
                          no_action_probability=0.0,
                          goal_mask=goal_mask,
                          avoid_mask=avoid_mask)

        mdp_solvers = {'Value Iteration' : gw.run_value_find,
                       'Policy Iteration' : gw.run_policy_find}

        for solver_name, solver_fn in mdp_solvers.items():
            print_spaces()
            print('Final result of {}:'.format(solver_name))
            policy_grids, utility_grids, i = solver_fn(iterations=25, discount=0.9)
            print(i)
            plt.figure()
            gw.plot_policy(utility_grids[:, :, -1])
            plot_convergence(utility_grids, policy_grids)
            plt.show()

        for solver_name, solver_fn in mdp_solvers.items():
            print(solver_name)
            if solver_name == 'Value Iteration':
                middle = 'value_'
            else:
                middle = 'policy_'
            times = []
            convergedAt = []
            stochast = []
            for i in range(11):
                stochas = round((1 - i*.1), 1)
                otherActions = (1-stochas) / 2
                leftAction = (-1, otherActions)
                normalAction = (0, stochas)
                rightAction = (1, otherActions)
                gw = GridWorldMDP(reward_grid=reward_grid,
                          obstacle_mask=obstacle_mask,
                          terminal_mask=terminal_mask,
                          action_probabilities=[
                              leftAction,
                              normalAction,
                              rightAction,
                          ],
                          no_action_probability=0.0,
                          goal_mask=goal_mask,
                          avoid_mask=avoid_mask)
                now = time.time()
                print(stochas)
                policy_grids, utility_grids, con = solver_fn(iterations=25, discount=0.9)
                ms_elapsed = (time.time() - now) * 1000
                times.append(ms_elapsed)
                convergedAt.append(con)
                print('Number of ms allowed: {}'.format(ms_elapsed))
                print(con)
                print(policy_grids[:, :, -1])
                print(utility_grids[:, :, -1])
                fig = plt.figure()
                gw.plot_policy(utility_grids[:, :, -1])
                plt.title("{} Best Policy at Stochasticity of {}".format(solver_name, round((1-stochas), 1)))
                stochast.append(round((1-stochas), 1))
                plt.savefig("../figures/{}{}{}.png".format(prefix, middle, i))
                plt.close(fig)

            fig, ax1 = plt.subplots()
            ax1.set_title('{} Stochasticity vs. time vs. convergence steps'.format(solver_name))
            ax1.set_xlabel("Stochasticity")
            ax2 = ax1.twinx()
            ax1.plot(stochast, times, 'b.-')
            ax1.set_ylabel('Time (ms)', color='b')
            ax2.plot(stochast, convergedAt, 'r.-')
            ax2.set_ylabel('Convergence Steps', color='r')
            plt.savefig("../figures/{}{}{}.png".format(prefix, middle, "stochas_time_converge"))
            plt.close(fig)




        # ql = QLearner(num_states=(shape[0] * shape[1]),
        #               num_actions=4,
        #               learning_rate=0.8,
        #               discount_rate=0.9,
        #               random_action_prob=0.5,
        #               random_action_decay_rate=0.99,
        #               dyna_iterations=0)
        #
        # start_state = gw.grid_coordinates_to_indices(start)
        #
        # iterations = 1000
        # flat_policies, flat_utilities = ql.learn(start_state,
        #                                          gw.generate_experience,
        #                                          iterations=iterations)
        # start = (-1, 0)
        # new_shape = (gw.shape[0], gw.shape[1], iterations)
        # ql_utility_grids = flat_utilities.reshape(new_shape)
        # ql_policy_grids = flat_policies.reshape(new_shape)
        # print('Final result of QLearning:')
        # print(ql_policy_grids[:, :, -1])
        # print(ql_utility_grids[:, :, -1])
        #
        # plt.figure()
        # gw.plot_policy(ql_utility_grids[:, :, -1], ql_policy_grids[:, :, -1])
        # plot_convergence(ql_utility_grids, ql_policy_grids)
        # plt.show()
