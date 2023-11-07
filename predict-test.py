#!/usr/bin/env python
# coding: utf-8

import requests


url = "http://localhost:9696/predict"

customer = {
    "var38": 117310.979016494,
    "var15": 35,
    "saldo_medio_var5_hace3": 0.0,
    "saldo_medio_var5_ult3": 22.74,
    "saldo_medio_var5_hace2": 25.02,
    "num_var45_ult3": 9,
    "num_var45_hace3": 3,
    "num_var45_hace2": 0,
    "saldo_medio_var5_ult1": 20.43,
    "saldo_var30": 9.66,
    "saldo_var5": 9.66,
    "num_var45_ult1": 6,
    "num_var22_ult3": 0,
    "imp_trans_var37_ult1": 0.0,
    "saldo_var42": 9.66,
    "num_var22_hace3": 0,
    "num_var22_hace2": 0,
    "num_med_var45_ult3": 3,
    "imp_op_var39_efect_ult3": 0.0,
    "imp_ent_var16_ult1": 0.0,
    "num_meses_var39_vig_ult3": 2,
    "saldo_var37": 0.0,
    "imp_op_var39_ult1": 20.34,
    "var36": 99,
    "imp_op_var39_comer_ult3": 20.34,
    "num_var22_ult1": 0,
    "imp_var43_emit_ult1": 0.0,
    "imp_op_var41_comer_ult3": 20.34,
    "imp_op_var39_efect_ult1": 0.0,
    "num_op_var39_ult3": 9,
    "imp_op_var41_comer_ult1": 20.34,
    "num_med_var22_ult3": 0,
    "imp_op_var41_ult1": 20.34,
    "imp_op_var39_comer_ult1": 20.34,
    "var3": 2,
    "num_var35": 6,
    "num_op_var41_ult3": 9,
    "num_op_var39_hace2": 0,
    "num_op_var39_ult1": 9,
    "saldo_var26": 0.0,
    "num_meses_var5_ult3": 2,
    "num_var43_recib_ult1": 0,
    "ind_var39_0": 1,
    "num_op_var39_comer_ult3": 9,
    "num_op_var39_efect_ult3": 0,
    "num_var37_med_ult2": 0,
    "num_var4": 2,
    "saldo_medio_var13_corto_hace2": 0.0,
    "num_op_var41_comer_ult3": 9,
    "num_op_var39_comer_ult1": 9,
    "saldo_var25": 0.0,
    "imp_op_var41_efect_ult3": 0.0,
}


response = requests.post(url, json=customer).json()
print(response)
