# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.tri as tri
#
# # First set of proportions
# proportions1 = [0.6, 0.75, 0.8, 0.9, 0.7, 0.8]
# labels1 = ['Sadness', 'Joy', 'Love', 'Anger', 'Fear', 'Surprise']
# N = len(proportions1)
# proportions1 = np.append(proportions1, 1)
# theta = np.linspace(0, 2 * np.pi, N, endpoint=False)
# x1 = np.append(np.sin(theta), 0)
# y1 = np.append(np.cos(theta), 0)
# triangles = [[N, i, (i + 1) % N] for i in range(N)]
# triang_backgr = tri.Triangulation(x1, y1, triangles)
# triang_foregr = tri.Triangulation(x1 * proportions1, y1 * proportions1, triangles)
#
# # Second set of proportions
# proportions2 = [0.7, 0.6, 0.9, 0.8, 0.75, 0.8]
# labels2 = labels1
# proportions2 = np.append(proportions2, 1)
# x2 = np.append(np.sin(theta), 0)
# y2 = np.append(np.cos(theta), 0)
# triang_foregr2 = tri.Triangulation(x2 * proportions2, y2 * proportions2, triangles)
#
# cmap1 = cmap1 = plt.cm.Reds
# cmap2 = plt.cm.Greens
# cmap = plt.cm.flag
#
# colors1 = np.linspace(0, 1, N + 1)
# colors2 = np.linspace(0, 1, N + 1)
#
# plt.triplot(triang_backgr, colors1, cmap=cmap, shading='gouraud', alpha=.2)
# plt.triplot(triang_foregr, colors1, cmap=cmap1, shading='gouraud', alpha=.1)
# plt.triplot(triang_foregr2, colors2, cmap=cmap2, shading='gouraud', alpha=.1)
#
# plt.triplot(triang_backgr, color='white', lw=2)
#
# for label, color, xi, yi in zip(labels1, colors1, x1, y1):
#     plt.text(xi * 1.05, yi * 1.05, label,
#              ha='left' if xi > 0.1 else 'right' if xi < -0.1 else 'center',
#              va='bottom' if yi > 0.1 else 'top' if yi < -0.1 else 'center')
#
# for label, color, xi, yi in zip(labels2, colors2, x2, y2):
#     plt.text(xi * 1.05, yi * 1.05, label,
#              ha='left' if xi > 0.1 else 'right' if xi < -0.1 else 'center',
#              va='bottom' if yi > 0.1 else 'top' if yi < -0.1 else 'center')
#
# plt.axis('off')
# plt.gca().set_aspect('equal')
# plt.show()

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import json
# Load the dataset
from datasets import load_dataset

dataset = load_dataset('go_emotions')

dataset = load_dataset('dair-ai/emotion')
# Split the dataset into train and validation sets
train_dataset = dataset['train']
valid_dataset = dataset['test']

device='cuda'

do_sample=False
generate_new_data=False
illus=True
model_name='gpt2-medium'
#checkpoint_name = 'gpt2-large_context_ctx_1_lr_5e-05-v3.ckpt'
checkpoint_name = '{}_context_ctx_1_lr_5e-05-v4'.format(model_name)
use_visual_data=True

saving_dir='/graphics/scratch2/staff/Hassan/checkpoints/lyrics_to_prompts/vis_emotion/emotions/visual_{}/{}/'.format(use_visual_data,checkpoint_name)

# with open(saving_dir +'pred_test_normal_best_model_pred_{}'.format(checkpoint_name)) as file:
#     normal_pred = json.load(file)

with open(saving_dir +'pred_test_visual_best_model_pred_{}'.format(checkpoint_name)) as file:
    visual_pred = json.load(file)

true_labels=valid_dataset['label']


import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Define class labels
labels = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']

# Example predicted labels
predicted = visual_pred
# Example true labels
true = true_labels

from sklearn.metrics import classification_report

# print(classification_report(true_labels, predicted))




# Compute confusion matrices

predicted2=[i-1 if 1 ==4 else i for i in predicted ]


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap, to_rgba_array

# Compute confusion matrices
cm1 = confusion_matrix(true, predicted)
cm2 = confusion_matrix(true, predicted2)

# Calculate percentages
cm_percentage1 = cm1.astype('float') / cm1.sum(axis=1)[:, np.newaxis]
cm_percentage2 = cm2.astype('float') / cm2.sum(axis=1)[:, np.newaxis]

# Set up the figure and axes
fig, ax = plt.subplots(figsize=(10, 6))

# Define custom color maps for each matrix
cmap1 = sns.color_palette("Blues")
cmap2 = sns.color_palette("Greens")

# Convert color maps to RGBA arrays
cmap1_rgba = to_rgba_array(cmap1)
cmap2_rgba = to_rgba_array(cmap2)

# Combine the RGBA arrays into a single color map
combined_cmap_rgba = np.concatenate((cmap1_rgba, cmap2_rgba))

# Create a custom color map using the combined RGBA array
combined_cmap = ListedColormap(combined_cmap_rgba)

# Create the combined confusion matrix by concatenating diagonally divided matrices
combined_cm = np.concatenate((cm_percentage1, cm_percentage2), axis=1)

# Plot the combined confusion matrix as a heatmap with diagonal division
sns.heatmap(combined_cm, annot=True, fmt=".1%", cmap=combined_cmap, cbar=True, square=True,
            annot_kws={"fontsize": 12}, cbar_kws={"shrink": 0.7}, mask=np.triu(np.ones_like(combined_cm)))

# Set x and y axis labels
tick_marks = np.arange(len(labels))
combined_tick_marks = np.concatenate((tick_marks, tick_marks + len(labels)))
plt.xticks(combined_tick_marks + 0.5, labels + labels, rotation=45, ha="right")
plt.yticks(tick_marks + 0.5, labels, rotation=0)

# Set axis labels and title
plt.xlabel('Predicted', fontsize=14)
plt.ylabel('True', fontsize=14)

# Increase font size of tick labels on both axes
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

# Save the plot with tight layout
plt.savefig('combined_CM.png', bbox_inches='tight')

# Display the plot
plt.show()
