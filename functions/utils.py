import pickle
from tkinter import filedialog
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

def open_file_as_dataframe(default_path):

    # Load csv file from file dialog and generate pandas dataframe
    source = input("Select source. Load from file (F/f) or default location of webscraped file (D/d).")
    
    if source.lower() == "d":
        df = pd.read_csv(default_path, sep = ',')
    elif source.lower() -- "f":
        df = pd.read_csv(filedialog.askopenfilename(), sep = ',')
    else:
        print("Wrong input.")

    return df

def dataframe_to_csv(list :list, file_path :str):

    dataframe_scrape = pd.DataFrame(list)
    dataframe_scrape.to_csv(file_path, index=False)
    print(dataframe_scrape.info())
    print(dataframe_scrape.describe())

    return

def save_dataframe_to_pkl(df, file_path):

    with open(file_path, 'wb') as f:
        pickle.dump(df, f)

    return

def open_pkl_as_dataframe(default_path):
    import pickle
    # Load csv file from file dialog and generate pandas dataframe
    source = input("Select source of pickle file. Load from file (F/f) or default file path (D/d).")
    
    if source.lower() == "d":
        df = pd.read_pickle(default_path)
        
    elif source.lower() == "f":
        df = pd.read_pickle(filedialog.askopenfilename())
    else:
        print("Wrong input.")

    return df

def save_csv_pickle(df):
    from tkinter import filedialog
    import pickle

    csv_or_pickle = input("Output as csv or pickle? 1 for csv file, 2 for pickle")

    if csv_or_pickle == 1:
        # Save data to new csv file
        #output_csv = r'..\data\clean\after_step_1_cleaning.csv'  # Fill your path to file
        #file_path = filedialog.asksaveasfilename()                  # Let user select filename
        df.to_csv(filedialog.asksaveasfilename(), index=False)       # Save data to new csv file, let user select filename
    
    elif csv_or_pickle == 2:
        with open(filedialog.asksaveasfilename(), 'wb') as f:
            pickle.dump(df, f)
    
    else:
        csv_or_pickle = input("Try again. Output as csv or pickle? 1 for csv file, 2 for pickle")
    
    return

def barplot(df,feature):
    import seaborn as sns
    import matplotlib.pyplot as plt
    import pandas as pd

    # Checking the amount of properties per province
    frequency = df[feature].value_counts()

    # Plotting the frequency
    plt.figure(figsize=(15, 4))
    sns.barplot(x=frequency.index, y=frequency.values)
    plt.title(f'Properties per {feature}')
    plt.xlabel(feature)
    plt.ylabel('Frequency')
    plt.show()

    return

def figure_boxplots(steps):
    import seaborn as sns
    import matplotlib.pyplot as plt

    # Function to create plots
    sns.color_palette("colorblind")

    # Create a figure with the number of required subplots and grid distribution
    fig, ax = plt.subplots(nrows=len(steps) // 4 if len(steps) % 4 == 0 else len(steps) // 4 + 1, ncols=len(steps), figsize=(15, 5))
    
    # Flatten the ax array for easier indexing
    ax = ax.ravel()

    # Loop over the columns and create a scatter plot for each
    for i, step in enumerate(steps):
        sns.boxplot(x=step, orient='h', color='blue', ax=ax[i])
        ax[i].set_title(f'{step.name} - box plot for outlier detection') # Set title for each plot


    #plt.legend(loc='upper center')     # Move the legend to the right side
    plt.tight_layout()
    plt.show()
    
    return

def figure_barplots(steps,col):
    import seaborn as sns
    import matplotlib.pyplot as plt

    # Function to create plots
    sns.color_palette("colorblind")

    # Create a figure with the number of required subplots and grid distribution
    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(15, 5))
    
    # Flatten the ax array for easier indexing
    ax = ax.ravel()

    # Loop over the columns and create a scatter plot for each
    for index, (key, value) in enumerate(steps.items()):
        sns.barplot(x=value.index, y=value.values, ax=ax[index])
        ax[index].set_title(f"Bar plot consolidation categorical data {col} - {key}") # Set title for each plot

    #plt.legend(loc='upper center')     # Move the legend to the right side
    plt.tight_layout()
    plt.show()

    return


