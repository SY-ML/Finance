from data.config_dbnomics import ConfigDBnomics
from data.datagen_dbnomics import DataGenerator_DBnomics
from data.dfs_dbnomics import DataFrameDBnomics
import data.visualizer as viz

from settings import Settings
sy = Settings()

if __name__ == '__main__':
    # dbnomics config
    cfg_dbn = ConfigDBnomics()
    codes_dbn = cfg_dbn.dict_code
    ls_keys_dbn = codes_dbn.keys()

    # Update data
    # dg_dbn = DataGenerator_DBnomics()

    dfs_dbn = [DataFrameDBnomics(code = code) for code in codes_dbn.values()]
    df_gold, df_silver, df_USprate, df_usdidx_broad, df_usdidx_afe, df_usidx_eme, df_usdidx_mjcrn, df_usdidx_itp  = [ cl.df for cl in dfs_dbn]

    print(df_gold, df_silver, df_USprate)
    print(df_usdidx_broad.head())
    exit()
    viz.plot_multiple_series_by_keyword(ls_dfs=[df_usdidx_broad, df_usdidx_afe, df_usidx_eme, df_usdidx_mjcrn, df_usdidx_itp])

