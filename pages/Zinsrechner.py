import os
import streamlit as st
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt

# Kredit in EUR
# Zins in % [0-100]
# Anfänglicher Tilgungssatz in % [0-100]
# Laufzeit in Jahren
def zinsrechner(kredit, zins, tilgung_begin, laufzeit, tilgung):

    monatsrate = (kredit*(tilgung_begin+zins)/100)/12
    st.write('monatsrate', monatsrate)
    laufzeit_in_monaten = laufzeit*12
    
    betrag = np.zeros(laufzeit_in_monaten)
    betrag[0] = kredit - tilgung

    for t in range(1,laufzeit_in_monaten):
        df = (betrag[t-1] * zins/100 )/12 - monatsrate
        betrag[t] = max(0, betrag[t-1] + df)
    
    return betrag

def zinseszins(anfangsbetrag, zins, laufzeit):
    betrag = np.zeros(laufzeit)
    betrag[0] = anfangsbetrag

    for j in range(1, laufzeit):
        df = betrag[j-1] * zins/100
        betrag[j] = betrag[j-1] + df

    return betrag


kredit_KFW      = 120000
kredit_staat    =  25000
kredit_spar     =  72000
kredit_bayern   = 115000

betrag_spar = zinsrechner(kredit_spar, 1.5, 2.00, 15, 0)
betrag_spar_tilg = zinsrechner(kredit_spar, 1.5, 2.00, 15, 1000)

betrag_KFW = zinsrechner(kredit_KFW, 0.95, 3.21, 10, 0)
betrag_KFW_tilg = zinsrechner(kredit_KFW-1000, 0.95, 3.21, 10, 18000)
st.write('KFW Ende', betrag_KFW_tilg[-1])
st.write('KFW differenz', betrag_KFW[-1] - betrag_KFW_tilg[-1] - 18000)

st.write(betrag_spar[-1])
st.write(betrag_spar_tilg[-1])

st.write('differenz', betrag_spar[-1] - betrag_spar_tilg[-1] - 1000)

anlage = zinseszins(1000, 2, 10)
# st.write(anlage)
st.write(anlage[-1] - 1000)