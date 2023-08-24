# import seaborn as sns
# import matplotlib.pyplot as plt
# import numpy as np
#
# # Example data (replace with your actual data)
# list1_scores = np.random.normal(0, 1, 1000000)
# list2_scores = np.random.normal(1, 1, 1000000)
# list3_scores = np.random.normal(2, 1, 1000000)
# list4_scores = np.random.normal(3, 1, 1000000)
#
# data = np.vstack([list1_scores, list2_scores, list3_scores, list4_scores]).T
#
# sns.violinplot(data=data)
# plt.xticks([0, 1, 2, 3], ['List 1', 'List 2', 'List 3', 'List 4'])
# plt.ylabel('Scores')
# plt.title('Violin Plot of List Scores')
# plt.show()
#
#
# import matplotlib.pyplot as plt
# import numpy as np
#
# # Example data (replace with your actual data)
# list1_scores = np.random.normal(0, 1, 1000000)
# list2_scores = np.random.normal(1, 1, 1000000)
# list3_scores = np.random.normal(2, 1, 1000000)
# list4_scores = np.random.normal(3, 1, 1000000)
#
# data = [list1_scores, list2_scores, list3_scores, list4_scores]
#
# plt.boxplot(data)
# plt.xticks([1, 2, 3, 4], ['List 1', 'List 2', 'List 3', 'List 4'])
# plt.ylabel('Scores')
# plt.title('Box Plot of List Scores')
# plt.show()
#
# import matplotlib.pyplot as plt
# import numpy as np
#
# # Simulated data (replace with your actual data)
# vipe_scores = np.random.random(1000000)
# lyric_scores = np.random.random(1000000)
# chatgpt_scores = np.random.random(1000000)
# gpt2_scores = np.random.random(1000000)
#
# vipe_scores.sort()
# lyric_scores.sort()
# chatgpt_scores.sort()
# gpt2_scores.sort()
#
# # Create x values (e.g., indices or any other meaningful values)
# x_values = np.arange(1000000)
#
# # Plot the lines for each list
# plt.figure(figsize=(10, 6))
# plt.plot(x_values, vipe_scores, label='VIPE')
# plt.plot(x_values, lyric_scores, label='Lyrics')
# plt.plot(x_values, chatgpt_scores, label='ChatGPT')
# plt.plot(x_values, gpt2_scores, label='GPT-2')
#
# plt.xlabel('Sample Index')
# plt.ylabel('Score')
# plt.title('Line Plots of List Scores')
# plt.legend()
# plt.show()

import pickle
with open('/graphics/scratch2/students/ghoshadh/SongAnimator/datasets/retrieval/metaphor_id.pickle', 'rb') as handle:
    metaphor_id = pickle.load(handle)

with open('/graphics/scratch2/students/ghoshadh/SongAnimator/datasets/retrieval/prompt_dict_vipe.pickle', 'rb') as handle:
    vipe = pickle.load(handle)

d=2