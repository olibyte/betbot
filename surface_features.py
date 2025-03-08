#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pandas as pd
from config import *
df = pd.read_csv(ATP_DATA_PATH)

unique_surfaces = df['Surface'].unique()
# print(unique_surfaces)

df = pd.get_dummies(df, columns=['Surface'], prefix='Surface')
