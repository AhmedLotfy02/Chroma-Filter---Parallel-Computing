import matplotlib.pyplot as plt

# Data
resolutions = ['640x480', '700x900', '1920x1080']
gpu_times = [0.00022975, 0.00028273, 0.00044120]
pytorch_times = [0.000225, 0.000239, 0.000316]
speedup = [1.02, 1.18, 1.39]

# Plotting the execution times
plt.figure(figsize=(10, 6))
plt.plot(resolutions, gpu_times, marker='o', label='GPU')
plt.plot(resolutions, pytorch_times, marker='o', label='PyTorch')
plt.title('Execution Time Comparison')
plt.xlabel('Image Resolution')
plt.ylabel('Execution Time (seconds)')
plt.legend()
plt.grid(True)
plt.show()

# Plotting the speedup ratios
plt.figure(figsize=(10, 6))
plt.plot(resolutions, speedup, marker='o', color='green', label='Speedup (GPU/PyTorch)')
plt.title('Speedup Ratio (GPU vs PyTorch)')
plt.xlabel('Image Resolution')
plt.ylabel('Speedup Ratio')
plt.legend()
plt.grid(True)
plt.show()

# Data
resolutions = ['480x240', '640x480', '700x900', '1920x1080']
gpu_times = [0.00031553, 0.00022975, 0.00028273, 0.00044120]
cpu_times = [0.00137312, 0.00205611, 0.00420714, 0.0140391]
speedup = [cpu / gpu for cpu, gpu in zip(cpu_times, gpu_times)]

# Plotting the execution times
plt.figure(figsize=(10, 6))
plt.plot(resolutions, gpu_times, marker='o', label='GPU')
plt.plot(resolutions, cpu_times, marker='o', label='CPU')
plt.title('Execution Time Comparison')
plt.xlabel('Image Resolution')
plt.ylabel('Execution Time (seconds)')
plt.legend()
plt.grid(True)
plt.show()

# Plotting the speedup ratios
plt.figure(figsize=(10, 6))
plt.plot(resolutions, speedup, marker='o', color='green', label='Speedup (CPU/GPU)')
plt.title('Speedup Ratio (CPU vs GPU)')
plt.xlabel('Image Resolution')
plt.ylabel('Speedup Ratio')
plt.legend()
plt.grid(True)
plt.show()
