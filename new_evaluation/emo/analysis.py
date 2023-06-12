import json
import json
# Load the dataset
from datasets import load_dataset



dataset=load_dataset('emo')
# Remove the last class
num_classes = 3  # Number of classes in the dataset
def convert_label(example):
    example['label'] -= 1
    return example

dataset = dataset.filter(lambda example: example['label'] > 0)
dataset = dataset.map(convert_label)


# Split the dataset into train and validation sets
train_dataset = dataset['train']
valid_dataset = dataset['test']
saving_dir='/graphics/scratch2/staff/Hassan/checkpoints/lyrics_to_prompts/vis_emotion/emo/'
device='cuda'

do_sample=False
generate_new_data=False
illus=True
model_name='gpt2-medium'
checkpoint_name = '{}_context_ctx_7_lr_5e-05-v4'.format(model_name)
#checkpoint_name = 'gpt2-large_context_ctx_1_lr_5e-05-v3.ckpt'

with open(saving_dir +'test_normal_best_model_pred_{}'.format(checkpoint_name)) as file:
    normal_pred = json.load(file)

with open(saving_dir +'test_visual_best_model_pred_{}'.format(checkpoint_name)) as file:
    visual_pred = json.load(file)

true_labels=valid_dataset['label']


import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# Define class labels
labels = ['happy', 'sad', 'angry']

# Example predicted labels
predicted = visual_pred
# Example true labels
true = true_labels



from sklearn.metrics import classification_report

print(classification_report(true_labels, predicted))
# Compute confusion matrix
cm = confusion_matrix(true, predicted)

# Calculate percentages
cm_percentage = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

# Plot confusion matrix
plt.imshow(cm_percentage, interpolation='nearest', cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.colorbar()
tick_marks = np.arange(len(labels))
plt.xticks(tick_marks, labels, rotation=45)
plt.yticks(tick_marks, labels)
plt.xlabel("Predicted")
plt.ylabel("True")

# Add percentage values to the plot
thresh = cm_percentage.max() / 2.
for i, j in np.ndindex(cm.shape):
    plt.text(j, i, f"{cm_percentage[i, j]*100:.1f}%",
             horizontalalignment="center",
             color="white" if cm_percentage[i, j] > thresh else "black")

plt.show()


print('print hard samples')

with open(saving_dir + 'vis_emotion_test_sample_{}_{}'.format(do_sample,checkpoint_name)) as file:
    text_valid = json.load(file)


for num , (t_label, p_label) in enumerate(zip(true_labels,visual_pred)):

    if t_label != p_label and labels[t_label]=='angry':
        print(valid_dataset['text'][num])
        print(text_valid[num])
        print('confused with ', labels[p_label])
        print(' ')


# "i am addicted im addicted too and i havent even downloaded it yet facewithtearsofjoy : sadness
#  A person sitting in front of a computer screen, eyes glued to the screen, fingers tapping on the keyboard, addicted to the video game
# confused with  happy"

# "what is gvsu you do know what the definition of is isright no i'm afraid not : sad
#  A confused student scratching his head, trying to understand the definition of right and wrong
# confused with  angry"