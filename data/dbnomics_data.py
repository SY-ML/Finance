import dbnomics
import pandas as pd
import os

class DBnomicsData:
    def __init__(self):
        self.archive_path = './DBnomics'
        self.dict_code = {
            'gold': "LBMA/gold/gold_D_USD_PM",
            'silver': "LBMA/silver/silver_D_USD",
            'prate_us': 'FED/PRATES_PRATES_POLICY_RATES/RESBME_N.D',
            'usdidx_broad' : 'FED/H10/JRXWTFB_N.B',
            'usdidx_afe': 'FED/H10/JRXWTFN_N.B',
            'usdidx_eme': 'FED/H10/JRXWTFO_N.B',
            'usdidx_mjcrn': 'FED/H10/V0.JRXWTFN_N.B',
            'usdidx_itp': 'FED/H10/V0.JRXWTFO_N.B',
        }

    def converter_code_to_local_file_name(self, code):
        file_name = code
        for chr in ['/', '.']:  # replace characters that file names cannot contain with _
            file_name = file_name.replace(chr, '_')

        return file_name

    def fetch_and_save_data(self):
        for code in self.dict_code.values():
            file_name = self.converter_code_to_local_file_name(code)
            path_data = os.path.join(self.archive_path, f'{file_name}.csv')
            data = dbnomics.fetch_series(code)
            df = pd.DataFrame(data)
            df.to_csv(path_data, index=False)

    def load_dataframes(self):
        dataframes = []
        for code in self.dict_code.values():
            file_name = self.converter_code_to_local_file_name(code)
            read_from = os.path.join(self.archive_path, f'{file_name}.csv')
            df = pd.read_csv(read_from)
            dataframes.append(df)

        return dataframes
