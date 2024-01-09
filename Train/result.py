import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from keras.metrics import mean_squared_error
from math import sqrt
import pandas as pd
import os
import xlsxwriter
import openpyxl
from openpyxl import Workbook
from openpyxl.styles import Font


def result(rmse_values):
    excel_filename = "rmse_values.xlsx"
    if not os.path.exists(excel_filename):
        wb = Workbook()
        wb.save(excel_filename)
    wb = openpyxl.load_workbook(excel_filename)
    sheet = wb.active
    sheet["A1"] = "5 min"
    sheet["B1"] = "10 min"
    sheet["C1"] = "15 min"
    sheet["D1"] = "20 min"
    sheet["E1"] = "25 min"
    sheet["F1"] = "30 min"
    sheet["G1"].font = Font(bold=True)

    sheet.append(rmse_values)
    wb.save(excel_filename)

    return 


