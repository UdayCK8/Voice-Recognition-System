import os

dataset = "dataset"

print("\nDataset Summary")
print("-" * 30)

for label in os.listdir(dataset):

    folder = os.path.join(dataset, label)

    if os.path.isdir(folder):

        count = len([
            f for f in os.listdir(folder)
            if f.lower().endswith(".wav")
        ])

        print(label, ":", count)