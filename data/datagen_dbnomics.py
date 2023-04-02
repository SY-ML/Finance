import dbnomics
import pandas as pd
from .config_dbnomics import ConfigDBnomics

'''
DBnomics ULR is as follows:

    https://db.nomics.world/
'''
cfg = ConfigDBnomics()
archive_path = cfg.archive_path
ls_data_code = cfg.dict_code.values()

class DataGenerator_DBnomics():
    def __init__(self):
        # DBnomics Code - Used to retrieve data

        for code in ls_data_code:
            file_name = cfg.converter_code_to_local_file_name(code) # Replace characters that file name cannot contain with _
            path_data = f'{archive_path}/{file_name}.csv' # path of dataframe to save to and load from
            data = dbnomics.fetch_series(code) # Fetch data by code
            df = pd.DataFrame(data) # Make it DataFrame
            df.to_csv(path_data, index=False) # Export as csv

        self.df = df