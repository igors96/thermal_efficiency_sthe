import streamlit as st
import pandas as pd
import xgboost as xg
from htbuilder import div, big, h2, styles
from htbuilder.units import rem

# Creating an instance of XGBoost Regressor and fitting the model to the training data

xgb_r = xg.XGBRegressor()
xgb_r.load_model("model_xgb_r.model")

# Getting heat exchanger data

def get_heat_exchanger_data():

    st.sidebar.markdown("# Insert heat exchanger parameters:")
    flow_hot = st.sidebar.slider('Flow of hot fluid (kg/s)', 0.5, 3.0, 2.0, 0.1)
    flow_cold = st.sidebar.slider('Flow of cold fluid (kg/s)', 0.5, 3.0, 2.0, 0.1)
    temp_hot = st.sidebar.slider('Temperature of hot fluid (ºC)', 40, 80, 40, 1)
    temp_cold = st.sidebar.slider('Temperature of cold fluid (ºC)', 5, 30, 5, 1)
    shell_passes = st.sidebar.slider('Shell passes', 1, 3, 1, 1)

    heat_exchanger_data = {'flow_hot (kg/s)' : flow_hot,
                            'flow_cold (kg/s)' : flow_cold,
                            'temp_hot (ºC)' : temp_hot,
                            'temp_cold (ºC)' : temp_cold,
                            'shell_passes' : shell_passes}
                        
    features = pd.DataFrame(heat_exchanger_data, index = [0])

    return features

heat_exchanger_input = get_heat_exchanger_data()

# Making predictions
prediction = xgb_r.predict(heat_exchanger_input)
rounded = list(map('{:.2f}%'.format, prediction))

# Creating parameters to use after
color = '#FF4D4C'
title = 'HEAT EXCHANGER THERMAL EFFICIENCY'
value = rounded

st.markdown('This application is very simple. The intention here is to provide the calculus of thermal efficiency of the shell and tube heat exchanger. Just input five parameters in the left side: flows and temperatures of the hot and cold fluid, and the shell passes. Automatically will appear, below, the value of the thermal efficiency.')

st.markdown(
        div(
            style=styles(
                text_align="left",
                color = color,
                padding=(rem(1), 0, rem(1), 0),
            )
        )(
            h2(style=styles(font_size=rem(2), font_weight=800, padding=0))(title),
            big(style=styles(font_size=rem(4), font_weight=800, line_height=1.5))(
                value
            ),
        ),
        unsafe_allow_html=True,
    )