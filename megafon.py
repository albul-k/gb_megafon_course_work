import os
import dill
import pandas as pd


class Megafon():
    """Predict target"""

    def __init__(self, path_pipeline: str, path_result: str) -> None:
        """Initialization

        Parameters
        ----------
        path_pipeline : str
            path to dill file with pipeline
        path_result : str
            path to result file
        """
        self.path_pipeline = path_pipeline
        self.file_out_result = path_result

        # Load pipeline
        with open(self.path_pipeline, 'rb') as file:
            self.model = dill.load(file)

    def run(self, path_data: str) -> None:
        """Run predictions

        Parameters
        ----------
        path_data : str
            path to file with init data
        """
        df_init = pd.read_csv(path_data)

        df = pd.DataFrame()
        df[['id', 'vas_id', 'buy_time']] = df_init[['id', 'vas_id', 'buy_time']]
        df['target'] = self.model.predict_proba(df_init)[:,1]
        df.to_csv(self.file_out_result)


if __name__ == '__main__':

    megafon = Megafon(
        path_pipeline='pipeline.dill',
        path_result='answers_test.csv'
    )
    megafon.run(path_data='data/data_test.csv')
