def custom_reward_fn(state, target_position):
    # Extract the current position of the ant (assuming the ant's position is the first 3 values)
    ant_position = state[:3]  # Ant's position in x, y, z coordinates

    # Calculate the Euclidean distance to the target
    distance_to_target = np.linalg.norm(ant_position - target_position)

    # You can shape the reward based on the distance, the closer, the higher the reward
    reward = -distance_to_target  # Negative reward for distance (smaller is better)

    # Optionally, you can give a bonus for reaching the target
    if distance_to_target < 0.5:  # A threshold for being "close enough"
        reward += 10  # Large reward for reaching the target

    return reward
