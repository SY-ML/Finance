import dbnomics
import pandas as pd

'''
DBnomics ULR is as follows:

    https://db.nomics.world/
'''

# archive path - csv file will be saved to the path below
archive_path = '../DBnomics'



class Data_DBnomics():
    def __init__(self, code, keep_updated = True):
        # DBnomics Code - Used to retrieve data
        self.code = code

        # Settings for save and load
        file_name = code.replace('/', '_') # file names cannot contain /; hence, replace / with _
        self.path_data = f'{archive_path}/{file_name}.csv' # path of dataframe to save to and load from

        # Fetch data and save as csv
        if keep_updated: # if you want it not updated, give False
            self.get_data_and_save_as_csv()

    def get_data_and_save_as_csv(self):
        data = dbnomics.fetch_series(self.code)
        df = pd.DataFrame(data)
        df.to_csv(self.path_data, index=False)
        return df


dict_data = {'gold': "LBMA/gold/gold_D_USD_PM",
           'silver': "LBMA/silver/silver_D_USD",
             'int_us': 'FRED/DFEDTAR'}


gd = Data_DBnomics(dict_data['gold'], keep_updated=False)
sv = Data_DBnomics(dict_data['silver'], keep_updated=False)
int_us = Data_DBnomics(dict_data['int_us'])
