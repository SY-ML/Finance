from data.dbnomics_data import DBnomicsData
import data.visualizer as viz
#061924 :(
if __name__ == '__main__':
    # dbnomics config and data fetching
    dbn_data = DBnomicsData()
    # Update data
    # dbn_data.fetch_and_save_data()
    dfs_dbn = dbn_data.load_dataframes()
    # unpack dataframes into individual variables (same as your original script)
    df_gold, df_silver, df_USprate, df_usdidx_broad, df_usdidx_afe, df_usidx_eme, df_usdidx_mjcrn, df_usdidx_itp  = dfs_dbn
    print(df_gold, df_silver, df_USprate)
    print(df_usdidx_broad.head())
    viz.plot_multiple_series_by_keyword(ls_dfs=[df_usdidx_broad, df_usdidx_afe, df_usidx_eme, df_usdidx_mjcrn, df_usdidx_itp])
