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
import os
do_sample=False
generate_new_data=False
illus=False

#checkpoint_name = 'gpt2-large_context_ctx_1_lr_5e-05-v3.ckpt'
use_visual_data=True
use_chatgpt=True
chat_gpt_random=False
shuffle=False
os.environ['CUDA_VISIBLE_DEVICES']='1'

path_to_jsons = '/home/shahmoha/PycharmProjects/chatgpt/visual_emotions'
if chat_gpt_random:
    path_to_jsons='/home/shahmoha/PycharmProjects/chatgpt/visual_emotions_random'

model_name='gpt2-medium'
device='cuda'
checkpoint_name = '{}_context_ctx_3_lr_5e-05-v4'.format(model_name)
if use_chatgpt:
    checkpoint_name='chat_gpt'

saving_dir = '/graphics/scratch2/staff/Hassan/checkpoints/lyrics_to_prompts/vis_emotion/Vis_emotions_chatgpt/vis_{}_shuffle_{}_chatgpt_{}_random_{}/{}/'.format(
    use_visual_data, shuffle, use_chatgpt, chat_gpt_random, checkpoint_name)



# with open(saving_dir +'pred_test_normal_best_model_pred_{}'.format(checkpoint_name)) as file:
#     normal_pred = json.load(file)

with open(saving_dir +'pred_test_visual_best_model_pred_{}'.format(checkpoint_name)) as file:
    visual_pred = json.load(file)

true_labels=valid_dataset['label']


import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# Define class labels
labels = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']

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

# Adding labels and title
plt.xlabel('Categories', fontsize=14)
plt.ylabel('Accuracies', fontsize=14)


# Adding legend with increased font size
plt.legend(fontsize=12)

# Increase font size of tick labels on both axes
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)



# Save the plot with tight layout
plt.savefig('CM.png', bbox_inches='tight')
# Display the plot
plt.show()



print('print hard samples')

with open(saving_dir + 'vis_emotion_test_sample_{}_{}'.format(do_sample,checkpoint_name)) as file:
    text_valid = json.load(file)

# Define class labels
labels = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']
test_dataset = dataset['test']

for num , (t_label, p_label) in enumerate(zip(true_labels,visual_pred)):

    if t_label != p_label and labels[t_label]=='surprise':
        print(test_dataset['text'][num])
        print(text_valid[num])
        print('confused with ', labels[p_label])
        print(' ')

# 'i feel that the packaging is really lovely and the product itself just does everything you ask: love
#  A woman holding a beautifully wrapped gift box, smiling with excitement
# confused with  joy

# 'i feel it would be foolish and perhaps a little disrespectful to consider doing the long hilly race : sadness
#  A hiker standing at the base of a steep mountain, contemplating whether to climb it or not
# confused with  fear'

# 'i do feel discouraged by what my supervisor said: sadness
#  A frustrated employee sitting at a desk, staring at a computer screen with a frown
# confused with  anger

# 'i was studying i always had the feeling that the process was unpleasant but it was absolutely necessary: saddness
#  A student sitting at a desk, surrounded by books and notes, looking tired and frustrated'
#anger

# 'i feel this strange sort of liberation: surprise
#  A woman standing on top of a mountain, arms stretched out, feeling the wind in her hair
# confused with  joy


"""
visual
         precision    recall  f1-score   support

           0       0.57      0.74      0.64       581
           1       0.67      0.73      0.70       695
           2       0.56      0.16      0.25       159
           3       0.64      0.46      0.54       275
           4       0.54      0.53      0.53       224
           5       0.70      0.32      0.44        66

    accuracy                           0.61      2000
   macro avg       0.61      0.49      0.52      2000
weighted avg       0.62      0.61      0.60      2000

"""