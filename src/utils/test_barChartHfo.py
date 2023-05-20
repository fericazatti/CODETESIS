import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


x = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
y = [10, 20, 30, 40, 50, 60, 70, 80]
status_description = ['nan', 'soz', 'resect,soz', 'soz', 'soz', 'resect,soz', 'nan', 'resect,soz'] 


data = {
    'values': x,
    'counts': y,
    'status_description': status_description 
}

df = pd.DataFrame(data)

# threshold = 30

# colors = ['red' if val < threshold else ''  for val in y]

status = df['status_description'].unique()

colors = []
for i in range(df.shape[0]): colors.append('')
df = df.assign(colors= colors)

color = 0x4e4e4e
for state in status:    
    for i in range(df.shape[0]):
        if (df.at[i, 'status_description'] == state):
            df.at[i, 'colors'] ='#' + hex(color)[2:]
    color = color + 0x006456


colors = []
for state in df['status_description']:
    if (state == 'nan'):
        # comment:
        colors.append('grey')
    elif (state == 'soz'):
        # comment: 
        colors.append('blue')
    else:
        # comment: 
        colors.append('red')


fig, ax = plt.subplots()

for state in df['status_description'].unique():    
    ax.bar(
        df.loc[df['status_description'] == state]['values'],
        df.loc[df['status_description'] == state]['counts']/df['counts'].max(),
        color=colors, label=state
    )    


# ax.bar(df['values'], df['counts']/df['counts'].max(), color=colors, label=df['status_description'].unique())
ax.axhline(y=0.75, color='r', linestyle='--')
ax.legend()

plt.show()
