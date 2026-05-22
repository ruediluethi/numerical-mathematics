import streamlit as st
import numpy as np
import math 
from matplotlib import pyplot as plt

st.write(r'''
    Mittels Splines wird der Werkzeugpfad aus Messdaten approximiert
    und so als mathematische Funktion beschrieben.

    Idee ist die Messdaten durch Bernsteinpolynome zu approximieren.
    Wenn irgendwie möglich sollten jeweils mehrere Punkte eines
    Zeitfensters zu einem Stützpunkt aggregiert/approximiert werden.
    
    Ähnlich wie ein rolling mean, aber eben mittels Bernstein-Approximation.


    ==> Spline kann durch äquivalente Stützpunkte beschrieben werden,
    welche wiederum durch interpolation zu exakt definierten Spline aus
    der Approximation führen.
    
''')

st.write(r'''
    Abtragsimulation mit Spline als Input
    auf dynamischen Octo-Trees
    - möglichst mit Krümmung als Split-Kriterium
''')

st.write(r'''
    Feinmaschige gleichmäßige Raumaufteilung
    von jedem Segment wird die nächste Distanz zum Spline bestimmt
    ist d < r ==> Segment wird abgetragen
    ==> erstmal check    
''')

st.write(r'''
    nehmen wir mal quadratische splines an,
    dann werden drei Punkte benötigt um die Koeffizienten zu bestimmen
    plus extrema, dann kann in einem rechteck abgeschätzt werden wo der spline durchgeht
        
''')