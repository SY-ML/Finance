from settings import Settings

sy = Settings()

if __name__ == '__main__':

    start_date = '2000-01-01'
    end_date = '2022-12-31'

    import dbnomics
    import pandas as pd


    # Define the dataset and series codes
    dataset_code = "LBMA/gold/gold_D_USD_PM"
    # dataset_code = "LBMA/silver/silver_D_USD"
    series_code = "USD"

    # Define the date range of interest
    # start_date = "2010-01-01"
    # end_date = "2023-03-21"

    # Fetch the data from DBnomics
    data = dbnomics.fetch_series(dataset_code)

    # Convert the data to a pandas dataframe
    df = pd.DataFrame(data)
    print(df.columns)

    # Print the dataframe
    print(df)

    import matplotlib.pyplot as plt

    # Define the window sizes for the moving averages
    windows = [5, 10, 20, 80, 120, 240]

    # Define the data and x-axis values
    x = df['period']
    y = df['value']

    # Create the figure and axis objects
    fig, ax = plt.subplots(figsize=(10, 5))

    # Plot the data
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