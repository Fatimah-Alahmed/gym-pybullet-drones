import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Parameters for analysis
TARGET_POS = np.array([1.0, 1.0, 1.0])  # Adjust to your actual target position
SUCCESS_THRESHOLD = 0.1  # Define a threshold for success (e.g., 0.1 meters)
filename = "step_data_save-11.05.2024_13.00.38.csv"  # Adjust to the generated CSV file

# Load the step-level data from the CSV file
data_pd = pd.read_csv(filename)

# Initialize counters for success rate calculation
successful_trials = 0
total_trials = data_pd['trial_number'].nunique()

# Set up a 3D plot for the trajectories
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Analyze each trial separately
for trial_number in data_pd['trial_number'].unique():
    # Extract data for the current trial
    trial_data = data_pd[data_pd['trial_number'] == trial_number]
    
    # Extract the position from 'obs' assuming the first three elements are x, y, z
    positions = np.stack(trial_data['obs'].apply(lambda x: np.fromstring(x.strip('[]'), sep=' ')).values)[:, :3]
    
    # Plot the trajectory in 3D
    ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], label=f'Trial {trial_number}')
    
    # Calculate distance to the target position for the last step
    last_position = positions[-1]
    distance_to_target = np.linalg.norm(last_position - TARGET_POS)
    
    # Determine if the trial is successful
    if distance_to_target < SUCCESS_THRESHOLD:
        successful_trials += 1

# Calculate and print the success rate
success_rate = successful_trials / total_trials
print(f"Success rate: {success_rate * 100:.2f}%")

# Configure the 3D plot
ax.set_xlabel('X Position')
ax.set_ylabel('Y Position')
ax.set_zlabel('Z Position')
ax.set_title('Drone Trajectories in 3D')
plt.legend()
plt.show()
