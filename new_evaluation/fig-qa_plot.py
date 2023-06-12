import matplotlib.pyplot as plt
import numpy as np

# Model 1 - Fine-tuned version
model1_fine_tune = [0.6893939393939394, 0.75, 0.6019417475728155, 0.6502732240437158]
# Model 1 - Zero-shot version
model1_zero_shot = [0.5808080808080808, 0.5955882352941176, 0.5533980582524272, 0.6338797814207651]
# Model 2 - Fine-tuned version
model2_fine_tune = [0.6515151515151515, 0.6691176470588235, 0.5728155339805825, 0.6338797814207651]
# Model 2 - Zero-shot version
model2_zero_shot = [0.5416666666666666, 0.5514705882352942, 0.5145631067961165, 0.6284153005464481]

# Labels for x-axis
labels = ['objects', 'visual', 'social', 'cultural']

# Sort the results and reorder labels accordingly
sorted_indices = np.argsort(model1_fine_tune)
sorted_labels = [labels[i] for i in sorted_indices]
model1_fine_tune = [model1_fine_tune[i] for i in sorted_indices]
model1_zero_shot = [model1_zero_shot[i] for i in sorted_indices]
model2_fine_tune = [model2_fine_tune[i] for i in sorted_indices]
model2_zero_shot = [model2_zero_shot[i] for i in sorted_indices]

# Set the figure size to square
plt.figure(figsize=(8, 5))
# Plotting the data
plt.plot(sorted_labels, model1_fine_tune, 'b-', label='ViPE-S (Fine-tuned)')
plt.plot(sorted_labels, model1_zero_shot, 'b--', label='ViPE-S (Zero-shot)')
plt.plot(sorted_labels, model2_fine_tune, 'r-', label='GPT2 (Fine-tuned)')
plt.plot(sorted_labels, model2_zero_shot, 'r--', label='GPT2 (Zero-shot)')

# Adding labels and title
plt.xlabel('Categories', fontsize=14)
plt.ylabel('Accuracies', fontsize=14)


# Adding legend with increased font size
plt.legend(fontsize=12)

# Increase font size of tick labels on both axes
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)



# Save the plot with tight layout
plt.savefig('fig-qa_graph.png', bbox_inches='tight')
# Display the plot
plt.show()
