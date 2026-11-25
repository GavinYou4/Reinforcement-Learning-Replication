import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.table import Table



#4x4 grid world
WORLD_SIZE = 4

ACTIONS = [np.array([0, -1]), np.array([-1, 0]), np.array([0, 1]), np.array([1, 0])]  # left, up, right, down

# 25% chance of moving in each direction
ACTION_PROB = 0.25


def check_terminal_state(state):
    """ Check if the state is a terminal state (top-left or bottom-right corner) """
    x, y = state
    return (x == 0 and y == 0) or (x == WORLD_SIZE - 1 and y == WORLD_SIZE - 1)

def step(state, action):
    """Take a step in the environment given a state and an action."""
    if check_terminal_state(state):
        return state, 0

    # Calculate the new position by applying the action
    new_state = (np.array(state) + action).tolist()
    x, y = new_state

    # Check if the new position is out of bounds, if so, stay in the same state
    if x < 0 or x >= WORLD_SIZE or y < 0 or y >= WORLD_SIZE:
        new_state = state

    return new_state, -1 #reward of -1 for each step taken

def draw_image(image):
    fig, ax = plt.subplots()
    # hide the axes
    ax.set_axis_off()
    # create a table to display the values
    tb = Table(ax, bbox=[0, 0, 1, 1])

    # calculate dimensions of each cell
    nrows, ncols = image.shape
    width, height = 1.0 / ncols, 1.0 / nrows

    # populate table with corresponding state values from image
    for (i, j), val in np.ndenumerate(image):
        tb.add_cell(i, j, width, height, text=val, loc='center', facecolor='white')

    # add row and column labels to identify the states
    for i in range(len(image)):
        tb.add_cell(i, -1, width, height, text=i, loc='right', edgecolor='none', facecolor='none')
        tb.add_cell(-1, i, width, height / 2, text=i, loc='center', edgecolor='none', facecolor='none')
    # add completed table to the figure
    ax.add_table(tb)

def calculate_value(in_place = True, discount = 1.0):
    # initialize the state values to zero
    new_state_values = np.zeros((WORLD_SIZE, WORLD_SIZE))
    # Track amount of sweeps through the grid
    iterations = 0
    while True:
        iterations += 1
        if in_place:
            # Reference same array for updates, so that the new values are used immediately in the next calculations
            state_values = new_state_values
        else:
            # Create a copy of the state values, so that the new values are not used until the next sweep
            state_values = new_state_values.copy()
        # Store the old state values for convergence checking
        old_state_values = state_values.copy()

        # Check every state in the grid world
        for i in range(WORLD_SIZE):
            for j in range(WORLD_SIZE):
                # Initialize the new value for the state to zero
                new_value = 0
                for action in ACTIONS:
                    # Consider every possible action from the state
                    # Obtain the resulting state and immediate reward
                    (next_i, next_j), reward = step((i, j), action)
                    # Use Bellman equation each action contributes to the expected value of the state, weighted by the probability of taking that action
                    new_value += ACTION_PROB * (reward + discount * state_values[next_i, next_j])
                # Store the new value for the state in the new_state_values array
                new_state_values[i, j] = new_value
        
        # Check for convergence
        max_change = np.abs(new_state_values - old_state_values).max()
        # Stop when the largest change is less than theta = 0.0001
        if max_change < 1e-4:
            break
    return new_state_values, iterations

def draw_value_function():
    _, sync_iterations = calculate_value(in_place=False)
    values, async_iterations = calculate_value(in_place=True)

    print("Synchronous iterations:", sync_iterations)
    print("Asynchronous iterations:", async_iterations)
    draw_image(np.round(values, decimals=2))
    plt.savefig('grid_world_value_function.png')
    plt.close()

if __name__ == "__main__":
    draw_value_function()
