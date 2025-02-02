import yaml
import numpy as np
import matplotlib.pyplot as plt

# data_local = {}
# data_bf = {}
# data_dstar = {}
# data_dstar_lite = {}
all_data_time = []
all_data_cost = []
for N in [10, 20, 50, 100]:
    for algo in ["brute-force", "dstar", "local"]:
        file = algo + "_" + str(N) + "_prefix.yaml"
        try:
            with open(file, 'r') as file:
                try:
                    data = yaml.safe_load(file)
                except yaml.YAMLError as e:
                    print(f"Error reading YAML file: {e}")
                  
        except IOError as e:
            pass
        
        data = np.array(data)
        time = data[:, 0]
        cost = data[:, 1]
        # time = np.log10(time)
        # max_value = np.max(time)
        # min_value = np.min(time)
        # mean_value = np.mean(time)
        # std_deviation = np.std(time)
        # sorted = np.sort(time)
        # middle_index = len(sorted) // 2
        # middle_value = sorted[middle_index] 
        # percentile_25 = np.percentile(time, 25)
        # percentile_75 = np.percentile(time, 75)
        all_data_time.append(time)
        all_data_cost.append(cost)
 
# # Creating plot
# plt.boxplot(data)


labels = ['','10','','','20', '','','50','','','100','']
bplot1 = plt.boxplot(all_data_time,
                     vert=True,  # vertical box alignment
                     patch_artist=True,
                     labels=labels,
                     showfliers=False,
                     widths=(0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75))  # will be used to label x-ticks
plt.yscale('symlog', linthreshy=1) #, subsy=[1,2,3,4,5,6,7,8,9])
plt.axvline(0.5, color = 'r', linestyle='--') 
colors = ['magenta', 'maroon', 'navy', 'olive', 'orange','magenta', 'maroon', 'navy', 'olive', 'orange','magenta', 'maroon', 'navy', 'olive', 'orange','magenta', 'maroon', 'navy', 'olive', 'orange']
plt.tick_params(bottom = False)
for bplot in (bplot1,bplot1,bplot1,bplot1):
    for patch, color in zip(bplot1['boxes'], colors):
        patch.set_facecolor(color)
plt.gca().set_ylim(bottom=-0.1)
plt.show()

