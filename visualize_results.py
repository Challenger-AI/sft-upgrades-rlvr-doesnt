# get src/results.txt and visualize its results

import os
import matplotlib.pyplot as plt

path = "./src/qwq/results.txt"
assert os.path.exists(path), "You first need to run run.sh to get results"

with open(path) as f:
    result_lines = f.readlines()

accs = []
tasks = []
for line in result_lines:
    task, acc = line.split(" accuracy: ")
    accs.append(int(acc))
    tasks.append(task)

plt.plot(tasks, accs)
plt.savefig("figure1.png")

path = "./src/base/results.txt"
assert os.path.exists(path), "You first need to run run.sh to get results"

with open(path) as f:
    result_lines = f.readlines()

accs = []
tasks = []
for line in result_lines:
    task, acc = line.split(" accuracy: ")
    accs.append(int(acc))
    tasks.append(task)

plt.plot(tasks, accs)
plt.savefig("figure2.png")

path = "./src/deepseek/results.txt"
assert os.path.exists(path), "You first need to run run.sh to get results"

with open(path) as f:
    result_lines = f.readlines()

accs = []
tasks = []
for line in result_lines:
    task, acc = line.split(" accuracy: ")
    accs.append(int(acc))
    tasks.append(task)

plt.plot(tasks, accs)
plt.savefig("figure3.png")