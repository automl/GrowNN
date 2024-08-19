import gymnasium as gym
import matplotlib.pyplot as plt

# Create the MuJoCo environment
env = gym.make("Humanoid-v4", render_mode="rgb_array", width=1280, height=720)  # You can replace 'Humanoid-v4' with any MuJoCo environment

# Reset the environment to the initial state
observation, info = env.reset()

# Render the environment and capture the image
image = env.render()

# Plot the image using matplotlib
plt.imshow(image)
plt.axis("off")  # Turn off axis labels
plt.show()

# Close the environment
env.close()
