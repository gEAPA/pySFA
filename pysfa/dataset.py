import pandas as pd
import numpy as np
import os

file_path = os.path.dirname(__file__)


class production_data:
    """Example datasets provided by the pySFA
    """

    def __init__(self, dmu, x, y, b=None, z=None):
        """General data structure

        Args:
            dmu (String): decision making unit.
            x (Numbers): input variables.
            y (Numbers): output variables.
            b (Numbers, optional): bad output variables. Defaults to None.
            z (Numbers, optional): contextual variables. Defaults to None.
        """
        self.decision_making_unit = dmu
        self.x, self.y, self.b, self.z = x, y, b, z


def load_Finnish_electricity_firm(x_select=['Energy', 'Length', 'Customers'], y_select=['OPEX', 'CAPEX', 'TOTEX'], z_select=['PerUndGr']):
    """Loading Finnish electricity firm data

    Args:
        x_select (list, optional): input variables. Defaults to ['Energy', 'Length', 'Customers'].
        y_select (list, optional): output variable. Defaults to ['OPEX', 'CAPEX', 'TOTEX'].
        z_select (list, optional): contextual variable. Defaults to ['PerUndGr'].

    Returns:
        Numbers: selected input-output
    """
    dataframe = pd.read_csv(
        file_path+"/data/electricityFirms.csv")
    dmu = np.asanyarray(dataframe.index.tolist()).T
    x = np.column_stack(
        [np.asanyarray(dataframe[selected]).T for selected in x_select])
    y = np.column_stack(
        [np.asanyarray(dataframe[selected]).T for selected in y_select])
    if z_select != None:
        z = np.column_stack(
            [np.asanyarray(dataframe[selected]).T for selected in z_select])
    return production_data(dmu, x, y, z=z)


def load_Tim_Coelli_frontier(x_select=['capital', 'labour'], y_select=['output']):
    """Loading Tim Coelli 4.1 data

    Args:
        x_select (list, optional): input variables. Defaults to ['capital', 'labour'].
        y_select (list, optional): output variable. Defaults to ['output'].

    Returns:
        Numbers: selected input-output
    """
    dataframe = pd.read_csv(
        file_path+"/data/41Firm.csv")
    dmu = np.asanyarray(dataframe['firm']).T
    x = np.column_stack(
        [np.asanyarray(dataframe[selected]).T for selected in x_select])
    y = np.column_stack(
        [np.asanyarray(dataframe[selected]).T for selected in y_select])
    return production_data(dmu, x, y)
