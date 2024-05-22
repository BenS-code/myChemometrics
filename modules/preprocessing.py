import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist


# from matplotlib.patches import Circle

pd.options.mode.chained_assignment = None


class DataInspection:

    def __init__(self, df_x, df_y):
        self.df_x = df_x
        self.df_y = df_y

    def anomaly_detection(self):
        dissimilarities = cdist(self.df_x, self.df_x, metric="correlation")
        absolute_normalized_dissimilarities_vector = (
            np.abs((dissimilarities[1] - np.mean(dissimilarities[1])) /
                   np.std(dissimilarities[1])))

        return absolute_normalized_dissimilarities_vector


class DataFiltering:

    def __init__(self, df_x, df_y):
        self.df_x = df_x
        self.df_y = df_y

    def filter_data(self, x_threshold, y_threshold):
        selected_x_columns = self.df_x.columns
        selected_y_columns = self.df_y.columns
        df_temp = pd.concat([self.df_y, self.df_x], axis=1)

        dissimilarities = DataInspection(df_temp[selected_x_columns],
                                         df_temp[selected_y_columns]).anomaly_detection()

        rows_to_remove = np.where(dissimilarities > x_threshold)

        df_temp = df_temp.drop(index=rows_to_remove[0].tolist())

        for col in df_temp[selected_y_columns].columns:
            mean = df_temp[col].mean()
            std = df_temp[col].std()
            df_temp = df_temp[(df_temp[col] - mean).abs() <= (y_threshold * std)]

        df_temp = df_temp.dropna()
        df_temp = df_temp.reset_index()

        self.df_x = df_temp[selected_x_columns]
        self.df_y = df_temp[selected_y_columns]

        return self.df_x, self.df_y


class DataStandardization:

    def __init__(self, df_x, df_y):
        self.df_x = df_x
        self.df_y = df_y

    def apply_labels_normalization(self):

        """
        Apply Normalization
        """

        self.df_y = (self.df_y - np.mean(self.df_y, axis=0)) / np.std(self.df_y, axis=0)

        return self.df_y

    def apply_msc(self):
        """
        Apply Multiplicative Scatter Correction (MSC) to NIR spectra.
        """
        temp_input = self.df_x
        for i in range(temp_input.shape[0]):
            temp_input.iloc[i] -= temp_input.iloc[i].mean()

        ref = np.mean(temp_input, axis=0)

        # Define a new array and populate it with the corrected data
        data_msc = temp_input.copy()
        for i in range(temp_input.shape[0]):
            # Run regression
            fit = np.polyfit(ref, temp_input.iloc[i, :], 1, full=True)
            # Apply correction
            data_msc.iloc[i, :] = (temp_input.iloc[i, :] - fit[0][1]) / fit[0][0]

        self.df_x = data_msc.copy()

        return self.df_x

    def apply_snv(self):
        """
        Apply Standard Normal Variate (SNV) correction to NIR spectra.
        """

        temp_input = self.df_x

        data_snv = temp_input.copy()
        for i in range(temp_input.shape[0]):
            # Apply correction
            data_snv.iloc[i, :] = (temp_input.iloc[i, :] -
                                   np.mean(temp_input.iloc[i, :])) / np.std(temp_input.iloc[i, :])

        self.df_x = data_snv.copy()

        return self.df_x
