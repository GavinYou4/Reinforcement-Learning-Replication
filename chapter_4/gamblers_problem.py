import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use('Agg')

# end goal - stop when this is reached
GOAL = 100

# all possible states of the problem - 0 to 100
STATES = np.arange(GOAL + 1)

# probability of flipping heads
P_HEADS = 0.4

def value_iteration():
    # give all states a value of 0, give the goal state a value of 1
    state_values = np.zeros(GOAL + 1)
    state_values[GOAL] = 1.0

    sweeps_history = []

    #value iteration loop
    while True:

        old_state_values = state_values.copy()
        sweeps_history.append(old_state_values)

        for state in STATES[1:GOAL]:
            # possible actions are the minimum of the current state and the difference between the goal and the current state
            actions = np.arange(1, min(state, GOAL - state) + 1)

            action_returns = []
            for action in actions:
                # calculate the expected return for each action
                expected_return = P_HEADS * old_state_values[state + action] + (1 - P_HEADS) * old_state_values[state - action]
                action_returns.append(expected_return)
            # choose action with highest expected return, that value becomes the new value for the state
            new_state_value = max(action_returns)
            state_values[state] = new_state_value

        # Calculate Delta, the maximum change in state values across all states
        change = abs(state_values - old_state_values).max()

        # if the change is less than a small threshold, we can stop iterating
        if change < 1e-9:
            sweeps_history.append(state_values.copy())
            break
    
    # derive the optimal policy for each state in S
    policy = np.zeros(GOAL + 1)
    for state in STATES[1:GOAL]:
        # possible actions are the minimum of the current state and the difference between the goal and the current state
        actions = np.arange(min(state, GOAL - state) + 1)
        action_returns = []
        for action in actions:
            # For each action, calculate the expected return based on the current state values
            expected_return = P_HEADS * state_values[state + action] + (1 - P_HEADS) * state_values[state - action]
            action_returns.append(expected_return)
        # choose the one with the highest return
        best_action = actions[np.argmax(np.round(action_returns[1:], 5)) + 1]  # +1 to account for the fact that we started actions from 1
        policy[state] = best_action

    # plot the value function and policy
    plt.figure(figsize=(10, 20))
    plt.subplot(2, 1, 1)
    for sweep, state_values in enumerate(sweeps_history):
        # plot the value function for each sweep
        plt.plot(STATES, state_values, label='Sweep {}'.format(sweep))
    plt.xlabel('Capital')
    plt.ylabel('Value Estimates')
    plt.legend(loc = 'best')

    plt.subplot(2, 1, 2)
    plt.scatter(STATES, policy)
    plt.xlabel('Capital')
    plt.ylabel('Final policy (stake)')

    plt.savefig("gamblers_problem.png")
    plt.close()

if __name__ == "__main__":
    value_iteration()