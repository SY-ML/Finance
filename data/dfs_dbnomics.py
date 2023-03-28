import pandas as pd
from config_dbnomics import ConfigDBnomics

cfg = ConfigDBnomics()

class DataFrameDBnomics():
    def __init__(self, code):
        archive_path = cfg.archive_path
        file_name = cfg.converter_code_to_local_file_name(code)
        read_from = f'{archive_path}/{file_name}.csv'

        self.df = pd.read_csv(read_from)


