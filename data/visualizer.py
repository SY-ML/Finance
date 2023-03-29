import matplotlib.pyplot as plt
from config_dbnomics import ConfigDBnomics
from dfs_dbnomics import DataFrameDBnomics

cfg = ConfigDBnomics()
dict_code = cfg.dict_code

# 로그스케일 차트
# Create the gold-to-silver ratio data frame
df_ratio = pd.DataFrame({'period': df_gold['period'], 'ratio': df_gold['value'] / df_silver['value']})

# Create a figure and axis for the line plot
fig, ax1 = plt.subplots()

# Plot the gold and silver data frames on the same line plot
ax1.plot(df_gold['period'], df_gold['value'], label='Gold', color='gold')
ax1.plot(df_silver['period'], df_silver['value'], label='Silver', color='silver')
ax1.set_xlabel('Period')
ax1.set_ylabel('Value')
ax1.legend(loc='upper left')

# Set primary y-axis to log scale
ax1.set_yscale('log')

# Create a twin x-axis for the bar plot
ax2 = ax1.twinx()

# Plot the gold-to-silver ratio as a bar plot
ax2.bar(x=df_ratio['period'], height=df_ratio['ratio'], label='Gold-to-Silver Ratio', alpha=0.3)
ax2.set_ylabel('Gold-to-Silver Ratio')
ax2.legend(loc='upper right')

# Set secondary y-axis to log scale (if needed)
# ax2.set_yscale('log')

# Show the combined plot
plt.show()

# 일반 차트
def plot_multiple_series_by_keword(ls_dfs, x='period', y='value'):

    # Create a figure and axis for the line plot
    f, ax1 = plt.subplot()
    for df in ls_dfs:
        label = df['dataset_code'].unique()
        # Plot the data frames on the same line plot
        ax1.plot(df[x], df[y], label=label)
    ax1.set_xlabel('Period')
    ax1.set_ylabel('Value')
    ax1.legend(loc='upper left')
    # Show the combined plot
    plt.show()


    #
    # # Create a twin x-axis for the bar plot
    # ax2 = ax1.twinx()
    #
    # # Plot the gold-to-silver ratio as a bar plot
    # ax2.bar(x=df_ratio['period'], height=df_ratio['ratio'], label='Gold-to-Silver Ratio', alpha=0.3)
    # ax2.set_ylabel('Gold-to-Silver Ratio')
    # ax2.legend(loc='upper right')
    #
