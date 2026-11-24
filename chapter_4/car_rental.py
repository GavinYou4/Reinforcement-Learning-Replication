import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import poisson

matplotlib.use('Agg')

# maximum number of cars at each location
MAX_CARS = 20

# Maximum number of cars that can be moved overnight
MAX_MOVE_OF_CARS = 5

# expected amount of requests and returns at first location
EXPECTED_REQUESTS_FIRST_LOC = 3
EXPECTED_RETURNS_FIRST_LOC = 3

# expected amount of requests and returns at second location
EXPECTED_REQUESTS_SECOND_LOC = 4
EXPECTED_RETURNS_SECOND_LOC = 2

# Discount rate (gamma)
DISCOUNT = 0.9

# Reward for each car rented
RENTAL_REWARD = 10

#Cost of moving a car
MOVE_CAR_COST = 2

# possible actions set - moving cars from one location to another
# negative values mean moving cars from location 2 to location 1, positive values mean moving cars from location 1 to location 2
ACTIONS = np.arange(-MAX_MOVE_OF_CARS, MAX_MOVE_OF_CARS + 1)

# Poisson upper bound for requests and returns - we will ignore the probability mass beyond this point
POISSON_UPPER_BOUND = 11

# poissson probability distribution cache - to avoid recalculating the same values over and over again
poisson_cache = dict()

def poisson_probability(n, lam):
    """ Returns the probability of n events occurring in a Poisson distribution with mean lam """
    global poisson_cache
    key = 10*n + lam
    if key not in poisson_cache:
        poisson_cache[key] = poisson.pmf(n, lam)
    return poisson_cache[key]

def expected_return(state, action, state_values, returned_cars):
    # initialize the total return
    returns = 0.0

    # calculate cost of moving cars, abs(action) is the number of cars moved, multiplied by the cost per car
    returns -= MOVE_CAR_COST * abs(action)

    # change the state based on the action taken, ensuring we don't exceed the maximum number of cars at each location
    NUM_CARS_FIRST_LOC = min(state[0] - action, MAX_CARS)
    NUM_CARS_SECOND_LOC = min(state[1] + action, MAX_CARS)

    # loop over all possible rental requests at the first location
    for rental_requests_first_loc in range(POISSON_UPPER_BOUND):
        # loop over all possible rental requests at the second location
        for rental_requests_second_loc in range(POISSON_UPPER_BOUND):
            # calculate the probability of this combination of rental requests occurring
            prob = poisson_probability(rental_requests_first_loc, EXPECTED_REQUESTS_FIRST_LOC) * poisson_probability(rental_requests_second_loc, EXPECTED_REQUESTS_SECOND_LOC)
            
            # Store the number of cars at both locations before rentals
            number_of_cars_first_loc = NUM_CARS_FIRST_LOC
            number_of_cars_second_loc = NUM_CARS_SECOND_LOC

            # calculate the number of cars rented at each location, which is the minimum of the number of rental requests and the number of cars available
            cars_rented_first_loc = min(number_of_cars_first_loc, rental_requests_first_loc)
            cars_rented_second_loc = min(number_of_cars_second_loc, rental_requests_second_loc)

            # calculate the reward for renting cars, which is the number of cars rented multiplied by the rental reward
            reward = (cars_rented_first_loc + cars_rented_second_loc) * RENTAL_REWARD

            # update the number of cars at each location after rentals
            number_of_cars_first_loc -= cars_rented_first_loc
            number_of_cars_second_loc -= cars_rented_second_loc

            if returned_cars:
                next_first = min(number_of_cars_first_loc + EXPECTED_RETURNS_FIRST_LOC, MAX_CARS)
                next_second = min(number_of_cars_second_loc + EXPECTED_RETURNS_SECOND_LOC, MAX_CARS)
                returns += prob * (reward + DISCOUNT * state_values[next_first, next_second])
            else:
                # loop over all possible returns at the first location
                for returned_cars_first_loc in range(POISSON_UPPER_BOUND):
                    # loop over all possible returns at the second location
                    for returned_cars_second_loc in range(POISSON_UPPER_BOUND):
                        # calculate the probability of this combination of returns occurring
                        prob_return = poisson_probability(returned_cars_first_loc, EXPECTED_RETURNS_FIRST_LOC) * poisson_probability(returned_cars_second_loc, EXPECTED_RETURNS_SECOND_LOC)
                        # update the number of cars at each location after returns, ensuring we don't exceed the maximum number of cars at each location
                        number_of_cars_first_loc = min(number_of_cars_first_loc + returned_cars_first_loc, MAX_CARS)
                        number_of_cars_second_loc = min(number_of_cars_second_loc + returned_cars_second_loc, MAX_CARS)
                        # calculate the total return for this combination of rental requests and returns, which is the reward plus the discounted value of the next state
                        returns += prob * prob_return * (reward + DISCOUNT * state_values[number_of_cars_first_loc, number_of_cars_second_loc])
    return returns

def value_iteration(constant_returned_cars=True):
    # initialize the state values to zero
    state_values = np.zeros((MAX_CARS + 1, MAX_CARS + 1))
    # initialize the policy to zero
    policy = np.zeros(state_values.shape, dtype=int)
    # initialize the history of state values for plotting
    iterations = 0
    _, axes = plt.subplots(2, 3, figsize=(40, 20))
    plt.subplots_adjust(wspace=0.2, hspace=0.2)
    axes = axes.flatten()

    while True:
        fig = sns.heatmap(np.flipud(policy), cmap="YlGnBu", ax=axes[iterations])
        fig.set_ylabel('Number of cars at first location')
        fig.set_xlabel('Number of cars at second location')
        fig.set_title(f'Policy after {iterations} iterations')
        fig.set_yticks(list(reversed(range(MAX_CARS + 1))))

        iterations += 1

        while True:
            # store the old state values for convergence checking
            old_state_values = state_values.copy()

            for i in range(MAX_CARS + 1):
                for j in range(MAX_CARS + 1):
                    # calculate the expected return for the current state and action, using the old state values
                    new_state_value = expected_return((i, j), policy[i, j], old_state_values, constant_returned_cars)
                    state_values[i, j] = new_state_value
            max_change = np.abs(state_values - old_state_values).max()
            if max_change < 1e-4:
                break
        # policy improvement step
        policy_stable = True
        for i in range(MAX_CARS + 1):
            for j in range(MAX_CARS + 1):
                old_action = policy[i, j]
                action_returns = []
                for action in ACTIONS:
                    # check if the action is valid (i.e., does not result in negative cars at either location)
                    if (0 <= i - action <= MAX_CARS) and (0 <= j + action <= MAX_CARS):
                        # calculate the expected return for the current state and action, using the current state values
                        expected_return_value = expected_return((i, j), action, state_values, constant_returned_cars)
                        action_returns.append(expected_return_value)
                    else:
                        # if the action is invalid, append a very low value to ensure it is not chosen
                        action_returns.append(-np.inf)
                # choose the action with the highest expected return
                best_action = ACTIONS[np.argmax(action_returns)]
                policy[i, j] = best_action
                if old_action != best_action:
                    policy_stable = False
        if policy_stable:
            # plot the value function
            fig = sns.heatmap(np.flipud(state_values), cmap="YlGnBu", ax=axes[-1])
            fig.set_ylabel('Number of cars at first location')
            fig.set_xlabel('Number of cars at second location')
            fig.set_title(f'Value function after {iterations} iterations')
            fig.set_yticks(list(reversed(range(MAX_CARS + 1))))
            break
    plt.savefig("car_rental.png")
    plt.close()
    
if __name__ == "__main__":
    value_iteration(constant_returned_cars=True)

            