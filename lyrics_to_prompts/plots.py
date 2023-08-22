LM_results={'gpt2':{'ss_all':[65.60, 65.87,65.79,65.91,66.12],'ss_nli':[66.89,67.16,67.07,67.19,67.37],'cider':[26.35,27.40,26.90,27.17,28.00],
            'rouge':[22.72,22.94,22.92,23.00,23.12]},
            'gpt2-medium':{'ss_all':[66.72,0,67.24,67.79],'ss_nli':[68.00,0,68.48,69.00,0], 'cider':[31.89,0,33.89,36.70],'rouge':[23.53,0,23.93,24.40]}}

#latest results
LM_results={'ViPE-S':{'cider':[26.36,27.43,26.92,27.23,28.06], },  'ViPE-M':{ 'cider':[31.95,31.57,34.00,36.70,39.51]}}

import matplotlib.pyplot as plt
import numpy as np
# Set the figure size to square
plt.figure(figsize=(8, 5))

context_values=[0,1,3,5,7]
context_values= [str(i) for i in context_values]
style={'ViPE-S':'r-', 'ViPE-M':'b-'}

plt.figure(figsize=(8, 2))

for model, results, in LM_results.items():

    print(model)
    for num, (metric, results_metric) in enumerate(results.items()):

        # for ctx, score in enumerate(results_metric):
        #     print('metric: {}, context_size: {} :, score: {}'.format(metric, context_values[ctx], score))
        #     print('_________________')

            # Plotting the data
        plt.plot(context_values, results_metric, style[model], label=model)


# Adding labels and title
plt.xlabel('Context Sizes', fontsize=14)
plt.ylabel('CIDEr Score', fontsize=14)

# Adding legend with increased font size
plt.legend(fontsize=12)

# Increase font size of tick labels on both axes
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

# Save the plot with tight layout
plt.savefig('context_effect.png', bbox_inches='tight')
# Display the plot
plt.show()


