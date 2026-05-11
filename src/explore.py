import os
from collections import Counter

names = ['Dust Mask', 'Eye Wear', 'Shield', 'boots', 'glove', 'helmet', 'person', 'vest']

class_counts = Counter()
labels_path = r"C:\Users\moham\PycharmProjects\ppeDetection\data\ppe-pk.v3i.yolov8\test\labels"

for file in os.listdir(labels_path):
    if file.endswith(".txt"):
        with open(os.path.join(labels_path, file)) as f:
            for line in f:
                cls = int(line.split()[0])
                class_counts[cls] += 1


print(class_counts)

for k, v in class_counts.items():
    print(names[k], v)