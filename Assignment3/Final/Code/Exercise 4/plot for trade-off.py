import json
import matplotlib.pyplot as plt
import os

# The path to Json files -> change to your path
file_path = "C:/Users/25738/Desktop/Exercise 4/GSEMO_json"  

# The names of Json files
json_files = [
    "IOHprofiler_f2101_MaxCoverage2101.json",
    "IOHprofiler_f2102_MaxCoverage2102.json",
    "IOHprofiler_f2103_MaxCoverage2103.json",
    "IOHprofiler_f2104_MaxCoverage2104.json",
    "IOHprofiler_f2201_MaxInfluence2201.json",
    "IOHprofiler_f2202_MaxInfluence2202.json",
    "IOHprofiler_f2203_MaxInfluence2203.json",
    "IOHprofiler_f2204_MaxInfluence2204.json",
]

plt.figure(figsize=(7,6))

labels = [
    "MC2100","MC2101","MC2102","MC2103",
    "MI2200","MI2201","MI2202","MI2203"
]

for fname, label in zip(json_files, labels):
    path = os.path.join(file_path, fname)
    with open(path, "r") as f:
        data = json.load(f)
    
    sol = data["scenarios"][0]["runs"][0]["best"]["x"] # best solution
    fitness = data["scenarios"][0]["runs"][0]["best"]["y"] # best fitness

    cost = sum(sol) # cost

    # Plot
    plt.scatter(cost, fitness, s=120, marker="o", label=label)

plt.title("GSEMO Best Solutions across 8 Instances (Trade-off Overview)")
plt.xlabel("Cost")
plt.ylabel("Fitness (Coverage / Influence)")
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend()
plt.tight_layout()
plt.show()
