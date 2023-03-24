from settings import Settings
import matplotlib.pyplot as plt
sy = Settings()

if __name__ == '__main__':

    start_date = '2000-01-01'
    end_date = '2022-12-31'

    import dbnomics
    import pandas as pd


    # Define the data_goldset and series codes
    code_gold = "LBMA/gold/gold_D_USD_PM"
    code_silver = "LBMA/silver/silver_D_USD"
    
    series_code = "USD"

    # Define the date range of interest
    # start_date = "2010-01-01"
    # end_date = "2023-03-21"

    # Fetch the data_gold from DBnomics

    def get_data_and_make_dataframe(code):
        data = dbnomics.fetch_series(code)
        df = pd.DataFrame(data)

        return df

    df_gold, df_silver = [get_data_and_make_dataframe(code) for code in [code_gold, code_silver]]

    print(df_gold, df_silver)
    #

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
    df_ratio = pd.DataFrame({'period': df_gold['period'], 'ratio': df_gold['value'] / df_silver['value']})

    # Create a figure and axis for the line plot
    fig, ax1 = plt.subplots()

    # Plot the gold and silver data frames on the same line plot
    ax1.plot(df_gold['period'], df_gold['value'], label='Gold', color='gold')
    ax1.plot(df_silver['period'], df_silver['value'], label='Silver', color='silver')
    ax1.set_xlabel('Period')
    ax1.set_ylabel('Value')
    ax1.legend(loc='upper left')

    # Create a twin x-axis for the bar plot
    ax2 = ax1.twinx()

    # Plot the gold-to-silver ratio as a bar plot
    ax2.bar(x=df_ratio['period'], height=df_ratio['ratio'], label='Gold-to-Silver Ratio', alpha=0.3)
    ax2.set_ylabel('Gold-to-Silver Ratio')
    ax2.legend(loc='upper right')

    # Show the combined plot
    plt.show()


    def plot_with_moving_average(df, x, y):

        import matplotlib.pyplot as plt

        # Define the window sizes for the moving averages
        windows = [5, 10, 20, 80, 120, 240]

        # Define the data_gold and x-axis values
        x = df['period']
        y = df['value']

        # Create the figure and axis objects
        fig, ax = plt.subplots(figsize=(10, 5))

        # Plot the data_gold
        ax.plot(x, y, label='Value')

        # Plot the moving averages
        for window in windows:
            ma = df['value'].rolling(window=window, min_periods=1).mean()
            ax.plot(x, ma, label=f'MA ({window})')

        # Set the axis labels and legend
        ax.set_xlabel('Period')
        ax.set_ylabel('Value')
        ax.legend()

        # Show the plot
        plt.show()

    plot_with_moving_average(df = df_gold, x='period', y='value')