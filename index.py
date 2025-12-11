import streamlit as st

pages = {
    "Grundlagen": [
        st.Page("pages/0_UU_Diskrete_Fouriertransformation.py", title="Diskrete Fouriertransformation"),
        st.Page("pages/0_UU_Lineare_Gleichungssysteme.py", title="Lineare Gleichungssysteme")
    ],
    "Einführung in Data Science": [
        st.Page("pages/64511_FUH_LE4.0_Statistische_Methoden.py", title="4.0 Statistische Methoden"),
        st.Page("pages/64511_FUH_LE4.1_Merkmalrepräsentation.py", title="4.1 Merkmalextraktion"),
        st.Page("pages/64511_FUH_LE4.2_Ähnlichkeitsmodellierung.py", title="4.2 Ähnlichkeitsmodellierung"),
        st.Page("pages/64511_FUH_LE4.3_Assoziationsregeln.py", title="4.3 Assoziationsregeln (Apriori)"),
        st.Page("pages/64511_FUH_LE4.4_Clustering.py", title="4.4 Clustering (k-Means)"),
        st.Page("pages/64511_FUH_LE4.5_Lineare_Diskriminanzfunktion.py", title="4.5 Lineare Diskriminanzfunktion"),
        st.Page("pages/64511_FUH_LE4.6_Entscheidungsbaum.py", title="4.6 Information Gain")
    ],
    "Maschinelles Lernen": [
        st.Page("pages/64401_FUH_3.5_Hauptkomponentenanalyse.py", title="3.5 Hauptkomponentenanalyse"),
        st.Page("pages/0_ETH_Spektrales_Clustering.py", title="Spektrales Clustering"),
    ],
    "Mathematische Grundlagen von Data Science": [
        st.Page("pages/61811_FUH_A1.2_Eigenwerte, Eigenvektoren und Diagonalisierbarkeit.py", title="1.2 Eigenwerte, Eigenvektoren und Diagonalisierbarkeit"),
        st.Page("pages/61811_FUH_A1.3.0_Singulärwertzerlegung.py", title="1.3 Singulärwertzerlegung"),
        st.Page("pages/61811_FUH_A1.3.1_Experiment.py", title="1.3.1 Experiment Zerlegung von Bildern"),
        st.Page("pages/61811_FUH_A2.1.0_Wahrscheinlichkeit.py", title="2.1 Wahrscheinlichkeiten"),
        st.Page("pages/61811_FUH_A2.1.1_Experiment_Korrelation.py", title="2.1.1 Herleitung Korrelationskoeffizient"),
        st.Page("pages/61811_FUH_A2.1.2_Experiment_Monte_Carlo.py", title="2.1.2 Experiment Monte Carlo"),
        st.Page("pages/61811_FUH_A3.1.1_Experiment_Lagrange.py", title="3.1.1 Experiment Lagrange"),
    ],
}

pg = st.navigation(pages)
pg.run()

# st.title('Index')

# st.write('''
#     Code base on [github](https://github.com/ruediluethi/numerical-mathematics)
# ''')