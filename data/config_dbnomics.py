
class ConfigDBnomics():
    def __init__(self):
        self.archive_path = './DBnomics'
        self.dict_code = {'gold': "LBMA/gold/gold_D_USD_PM",
                 'silver': "LBMA/silver/silver_D_USD",
                 'prate_us': 'FED/PRATES_PRATES_POLICY_RATES/RESBME_N.D'}
        # self.dict_file_name = [self.converter_code_to_local_file_name(code) for code in self.dict_code.values()]

    def converter_code_to_local_file_name(self, code):
        """

        :param code:
        :return:
        """
        file_name = code
        for chr in ['/', '.']:  # replace characters that file names cannot contain with _
            file_name = file_name.replace(chr, '_')

        return file_name

