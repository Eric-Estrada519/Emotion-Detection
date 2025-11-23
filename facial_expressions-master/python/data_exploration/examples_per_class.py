import matplotlib
matplotlib.use("TkAgg")

import matplotlib.pyplot as plt
from model_code.data_loader import get_loader


loader = get_loader(batch_size=1, shuffle=True)
dataset = loader.dataset

examples = {}
classes = sorted(set(dataset.data["emotion"]))

for img, label in loader:
    label = label[0].lower()
    if label not in examples:
        examples[label] = img[0]
    if len(examples) == len(classes):
        break

plt.figure(figsize=(4 * len(classes), 4))
for i, cls in enumerate(classes):
    img = examples[cls]
    plt.subplot(1, len(classes), i + 1)
    plt.imshow(img.permute(1, 2, 0))
    plt.title(cls)
    plt.axis("off")

plt.show()


"""Shows the data that is not processed by the model_code"""
