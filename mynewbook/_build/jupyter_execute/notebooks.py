#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import datetime as dt
import json
from datetime import datetime

import plotly.io as pio
import plotly.graph_objects as go
import plotly.express as px
import warnings
warnings.filterwarnings('ignore')

pio.renderers.default = "notebook"
pio.templates.default = "plotly_white"


# this enables relative path imports
import os
from dotenv import load_dotenv
load_dotenv()
_PROJECT_PATH: str = os.environ["_project_path"]
_PICKLED_DATA_FILENAME: str = os.environ["_pickled_data_filename"]

import sys
from pathlib import Path
project_path = Path(_PROJECT_PATH)
sys.path.append(str(project_path))

import config_v2 as cfg

from library_report_v2 import Cleaning as cln
from library_report_v2 import Graphing as grp
from library_report_v2 import Processing as pro
from library_report_v2 import Configuration as repcfg


# In[2]:


periodo_historico = cfg.BASELINE
periodo_de_estudio = cfg.STUDY


# In[3]:


def show_response_contents(df):
    print("The response contains:")
    print(json.dumps(list(df['variable'].unique()), sort_keys=True, indent=4))
    print(json.dumps(list(df['device'].unique()), sort_keys=True, indent=4))


# In[4]:


DEVICE_NAME = 'Clinica Cardio Infantil'


# In[5]:


df = pd.read_pickle(project_path / 'data' / _PICKLED_DATA_FILENAME)
#df = df.query("device_name == @DEVICE_NAME")
show_response_contents(df)


# In[6]:


conteo_variable_df = df['variable'].value_counts().to_frame().reset_index()
conteo_variable_df.columns = ['variable', 'conteo']


# In[7]:


df = df.sort_values(by=['variable','datetime'])
df = pro.datetime_attributes(df)


# In[8]:


ea_dia = df.query("variable == 'consumo-electricidad-dia'").copy()
ea_mes = df.query("variable == 'consumo-electricidad-mes'").copy()
cs_agua_dia = df.query("variable == 'consumo-agua-dia'").copy()
cs_agua_mes = df.query("variable == 'consumo-agua-mes'").copy()
ea_termica_hora = df.query("variable == 'energia-termica-hora-hora'").copy()
ea_termica_mes = df.query("variable == 'energia-termica-mes'").copy()
t_out = df.query("variable == 't-out'").copy()
t_input = df.query("variable == 't-input'").copy()
cop_dia = df.query("variable == 'cop_dia'").copy()
df_pa = ea_dia


# In[9]:


ea_dia = cln.remove_outliers_by_zscore(ea_dia, zscore=4)
ea_mes = cln.remove_outliers_by_zscore(ea_mes, zscore=4)
cs_agua_dia = cln.remove_outliers_by_zscore(cs_agua_dia, zscore=4)
cs_agua_mes = cln.remove_outliers_by_zscore(cs_agua_mes, zscore=4)
ea_termica_hora = cln.remove_outliers_by_zscore(ea_termica_hora, zscore=4)
ea_termica_mes = cln.remove_outliers_by_zscore(ea_termica_mes, zscore=4)
t_out = cln.remove_outliers_by_zscore(t_out, zscore=4)
t_input = cln.remove_outliers_by_zscore(t_input, zscore=4)
cop_dia = cln.remove_outliers_by_zscore(cop_dia, zscore=4)
df_pa = cln.remove_outliers_by_zscore(df_pa, zscore=4)


# In[10]:


ea_total_month = ea_dia.groupby(by=["variable"]).resample('M').sum().reset_index().set_index('datetime')
ea_total_month = pro.datetime_attributes(ea_total_month)

cs_agua_month = cs_agua_dia.groupby(by=["variable"]).resample('M').sum().reset_index().set_index('datetime')
cs_agua_month = pro.datetime_attributes(cs_agua_month)

ea_termica_month = ea_termica_hora.groupby(by=["variable"]).resample('M').sum().reset_index().set_index('datetime')
ea_termica_month = pro.datetime_attributes(ea_termica_month) 

t_out_month = t_out.groupby(by=["variable"]).resample('M').sum().reset_index().set_index('datetime')
t_out_month = pro.datetime_attributes(t_out_month)

t_input_month = t_input.groupby(by=["variable"]).resample('M').sum().reset_index().set_index('datetime')
t_input_month = pro.datetime_attributes(t_input_month)

cop_dia_month = cop_dia.groupby(by=["variable"]).resample('M').sum().reset_index().set_index('datetime')
cop_dia_month = pro.datetime_attributes(cop_dia_month)

t_out_day = t_out.groupby(by=["variable"]).resample('D').mean().reset_index().set_index('datetime')
t_out_day = pro.datetime_attributes(t_out_day)


# In[11]:


xm = pd.concat([ea_total_month, cs_agua_month])
xm = xm.sort_values(by=['year', 'month'])


# In[12]:


# combinar las columnas "year" y "month" en una sola columna "fecha"
xm["fecha"] = pd.to_datetime(xm[["year", "month"]].assign(day=1))


# 
# <div style="page-break-before: always;"></div>

# ## Informe Clinica Cardio

# ### Globales de electricidad y agua
# El consumo consolidad en el último trimestre en agua y electricidad en el sistema de las bombas de calor es el siguiente:

# In[13]:


# PLOTS BASE LINE VS XM  . data xm
from IPython import display
import plotly.express as px
fig = px.bar(
    xm,
    x= "fecha",
    y="value",
    barmode='group',
    color='variable',
    color_discrete_sequence=repcfg.FULL_PALETTE,
    labels={'month':'Mes', 'value':'Consumo [kWh/mes]'},
    title=f"{DEVICE_NAME}: Consumo Mensual de energía activa [kWh/mes]",
)

fig.add_hline(y=ea_total_month['value'].mean(), line_dash="dash", line_color=repcfg.FULL_PALETTE[1], annotation_text=f"Línea base: {ea_total_month['value'].mean():.2f} kWh/mes", annotation_position="top left")

"""
fig.update_layout(
    font_family=repcfg.CELSIA_FONT,
    font_size=repcfg.PLOTLY_TITLE_FONT_SIZE,
    font_color=repcfg.FULL_PALETTE[1],
    title_x=repcfg.PLOTLY_TITLE_X,
    width=repcfg.JBOOK_PLOTLY_WIDTH,
    height=repcfg.JBOOK_PLOTLY_HEIGHT
)

fig.show()

"""
fig.update_layout(
    font_family=repcfg.CELSIA_FONT,
    font_size=repcfg.PLOTLY_TITLE_FONT_SIZE,
    font_color=repcfg.FULL_PALETTE[1],
    title_x=repcfg.PLOTLY_TITLE_X,
    width=repcfg.JBOOK_PLOTLY_WIDTH + 100,  # Aumentar el ancho de la gráfica en 100 unidades
    height=repcfg.JBOOK_PLOTLY_HEIGHT,
    xaxis_title="Mes",
    xaxis_tickangle=45,  # Rotar las etiquetas de los meses
    xaxis_tickfont_size=10,  # Reducir el tamaño de las etiquetas
)

fig.show()



# ### Consumo Electricidad

# In[14]:


from IPython import display

# Agrupar por "variable" y "month" y "year", y sumar los valores de "value"
tabla1 = xm[xm["variable"] == "consumo-electricidad-dia"].groupby(["year", "month", "variable"])["value"].sum().reset_index()
tabla2 = xm[xm["variable"] == "consumo-agua-dia"].groupby(["year", "month", "variable"])["value"].sum().reset_index()

# Renombrar la columna "variable"
tabla1["variable"] = "consumo electricidad"
tabla2["variable"] = "consumo agua"




# Calcular el consumo linea base (promedio de los meses 1, 2 y 3)
consumo_linea_base = tabla1[tabla1['month'].isin([1, 2, 3])]['value'].mean()

# Calcular el % de variación respecto al consumo linea base
tabla1['consumo linea base'] = consumo_linea_base.round(2)
tabla1['%variacion'] = ((tabla1['value'] - consumo_linea_base) / consumo_linea_base) * 100

# Redondear la columna %variacion a dos decimales
tabla1['%variacion'] = tabla1['%variacion'].round(2)

# Mostrar las tablas en Jupyter Notebook
display.display(tabla1)
#display.display(tabla2)


# In[15]:


# Convertir la columna "year" a tipo categórico para mantener la paleta de colores discreta
tabla1['year'] = tabla1['year'].astype('category')

# Separar los datos de línea base (meses 1 al 3) y periodo de estudio (meses 4 al 6)
linea_base = tabla1[tabla1['month'].between(1, 3)]
periodo_estudio_energia = tabla1[tabla1['month'].between(4, 6)]

# Calcular el valor promedio de la línea base
valor_promedio_linea_base = linea_base['value'].mean()

# Crear el gráfico
fig = px.bar(
    pd.concat([linea_base, periodo_estudio_energia]),
    x='month',
    y='value',
    color='year',  # El color ahora representará las distintas categorías de la columna "year"
    facet_col='variable',
    labels={'month': 'Mes', 'value': 'Consumo de electricidad [kWh/mes]', 'year': 'Año'},
    title='Comportamiento del Consumo de Electricidad en Periodo de Estudio vs. Consumo Línea Base',
    category_orders={'month': [1, 2, 3, 4, 5, 6]},
    color_discrete_map={2023: '#d5752d', 2022: '#59595b'}  # Asignar los colores de la paleta a cada año
)

# Agregar la línea horizontal del valor promedio de la línea base
fig.add_hline(y=valor_promedio_linea_base, line_dash="dash", line_color='#ca0045', annotation_text=f"Promedio Línea Base: {valor_promedio_linea_base:.2f} kWh/mes", annotation_position="top left")

# Aplicar los mismos ajustes al diseño de la figura que en la primera gráfica
fig.update_layout(
    font_family=repcfg.CELSIA_FONT,
    font_size=repcfg.PLOTLY_TITLE_FONT_SIZE,
    font_color=repcfg.FULL_PALETTE[1],
    title_x=repcfg.PLOTLY_TITLE_X,
    width=repcfg.JBOOK_PLOTLY_WIDTH + 100,  # Aumentar el ancho de la gráfica en 100 unidades
    height=repcfg.JBOOK_PLOTLY_HEIGHT,
    xaxis_title="Mes",
    xaxis_tickangle=45,  # Rotar las etiquetas de los meses
    xaxis_tickfont_size=10,  # Reducir el tamaño de las etiquetas
)


# Mostrar el gráfico
fig.show()


# In[16]:


# Etiqueta para salto de página
print("<div style=\"page-break-before: always;\"></div>")

# Celda de código
print("Este es el contenido de la segunda página.")


# <div style="page-break-before: always;"></div>

# ### Consumo Agua

# In[17]:


# Calcular el consumo linea base (promedio de los meses 1, 2 y 3)
consumo_linea_base_2 = tabla2[tabla2['month'].isin([1, 2, 3])]['value'].mean()

# Calcular el % de variación respecto al consumo linea base
tabla2['consumo linea base'] = consumo_linea_base_2.round(2)
tabla2['%variacion'] = ((tabla2['value'] - consumo_linea_base_2) / consumo_linea_base_2) * 100

# Redondear la columna %variacion a dos decimales
tabla2['%variacion'] = tabla2['%variacion'].round(2)

# Mostrar el DataFrame con las nuevas columnas
#print(tabla2)
display.display(tabla2)


# In[18]:


# Convertir la columna "year" a tipo categórico para mantener la paleta de colores discreta
tabla2['year'] = tabla2['year'].astype('category')

# Separar los datos de línea base (meses 1 al 3) y periodo de estudio (meses 4 al 6)
linea_base = tabla2[tabla2['month'].between(1, 3)]
periodo_estudio_agua = tabla2[tabla2['month'].between(4, 6)]

# Calcular el valor promedio de la línea base
valor_promedio_linea_base = linea_base['value'].mean()

# Crear el gráfico
fig = px.bar(
    pd.concat([linea_base, periodo_estudio_agua]),
    x='month',
    y='value',
    color='year',  # El color ahora representará las distintas categorías de la columna "year"
    facet_col='variable',
    labels={'month': 'Mes', 'value': 'Consumo de agua [m3/mes]', 'year': 'Año'},
    title='Comportamiento del Consumo de Agua en Periodo de Estudio vs. Consumo Línea Base',
    category_orders={'month': [1, 2, 3, 4, 5, 6]},
    color_discrete_map={2023: '#13a2e1', 2022: '#59595b'}  # Asignar los colores de la paleta a cada año
)

# Agregar la línea horizontal del valor promedio de la línea base
fig.add_hline(y=valor_promedio_linea_base, line_dash="dash", line_color='#ca0045', annotation_text=f"Promedio Línea Base: {valor_promedio_linea_base:.2f} m3/mes", annotation_position="top left")

# Aplicar los mismos ajustes al diseño de la figura que en la primera gráfica
fig.update_layout(
    font_family=repcfg.CELSIA_FONT,
    font_size=repcfg.PLOTLY_TITLE_FONT_SIZE,
    font_color=repcfg.FULL_PALETTE[1],
    title_x=repcfg.PLOTLY_TITLE_X,
    width=repcfg.JBOOK_PLOTLY_WIDTH + 100,  # Aumentar el ancho de la gráfica en 100 unidades
    height=repcfg.JBOOK_PLOTLY_HEIGHT,
    xaxis_title="Mes",
    xaxis_tickangle=45,  # Rotar las etiquetas de los meses
    xaxis_tickfont_size=10,  # Reducir el tamaño de las etiquetas
)


# Mostrar el gráfico
fig.show()


# ### Para tener en cuenta

# In[19]:


"""
# Generar el texto con las conclusiones
texto_conclusiones = f"Variación en el consumo de energía durante el periodo de estudio:\n\n"
texto_conclusiones += f"Durante el periodo de estudio, el consumo de energía presenta una disminución promedio del {periodo_estudio_energia['%variacion'].mean():.2f}% en comparación con la línea base.\n"

mes_max_aumento = tabla1.loc[tabla1['%variacion'].idxmax(), 'month']
valor_max_aumento = tabla1['%variacion'].max()
texto_conclusiones += f"En el mes {mes_max_aumento}, se observa el mayor aumento en el consumo de energía en relación con la línea base, con un incremento del {valor_max_aumento:.2f}%.\n"

mes_max_disminucion = tabla1.loc[tabla1['%variacion'].idxmin(), 'month']
valor_max_disminucion = tabla1['%variacion'].min()
texto_conclusiones += f"En el mes {mes_max_disminucion}, se registra la mayor disminución en el consumo de energía en relación con la línea base, con una reducción del {valor_max_disminucion:.2f}%."

print(texto_conclusiones)
"""


# In[20]:


from IPython.display import Markdown

texto_conclusiones = f"Variación en el consumo de energía durante el periodo de estudio:\n\n"
texto_conclusiones += f"- Durante el periodo de estudio, el consumo de energía presenta una disminución promedio del {periodo_estudio_energia['%variacion'].mean():.2f}% en comparación con la línea base.\n"

mes_max_aumento = tabla1.loc[tabla1['%variacion'].idxmax(), 'month']
valor_max_aumento = tabla1['%variacion'].max()
texto_conclusiones += f"- En el mes {mes_max_aumento}, se observa el mayor aumento en el consumo de energía en relación con la línea base, con un incremento del {valor_max_aumento:.2f}%.\n"

mes_max_disminucion = tabla1.loc[tabla1['%variacion'].idxmin(), 'month']
valor_max_disminucion = tabla1['%variacion'].min()
texto_conclusiones += f"- En el mes {mes_max_disminucion}, se registra la mayor disminución en el consumo de energía en relación con la línea base, con una reducción del {valor_max_disminucion:.2f}%."

# Mostrar el texto con formato justificado y viñetas
Markdown(texto_conclusiones)


# In[21]:


from IPython.display import Markdown

texto_conclusiones2 = f"Variación en el consumo de agua durante el periodo de estudio:\n\n"
texto_conclusiones2 += f"- Durante el periodo de estudio, el consumo de agua presenta una disminución promedio del {periodo_estudio_agua['%variacion'].mean():.2f}% en comparación con la línea base.\n"

mes_max_aumento = tabla2.loc[tabla2['%variacion'].idxmax(), 'month']
valor_max_aumento = tabla2['%variacion'].max()
texto_conclusiones2 += f"- En el mes {mes_max_aumento}, se observa el mayor aumento en el consumo de agua en relación con la línea base, con un incremento del {valor_max_aumento:.2f}%.\n"

mes_max_disminucion = tabla2.loc[tabla2['%variacion'].idxmin(), 'month']
valor_max_disminucion = tabla2['%variacion'].min()
texto_conclusiones2 += f"- En el mes {mes_max_disminucion}, se registra la mayor disminución en el consumo de agua en relación con la línea base, con una reducción del {valor_max_disminucion:.2f}%."

# Mostrar el texto con formato justificado y viñetas
Markdown(texto_conclusiones2)


# <div style="page-break-before: always;"></div>

# ### Desagregado del consumo

# In[22]:


#df_plot = pd.concat([ea_dia, cs_agua_dia])

list_vars = [
    'consumo-electricidad-dia',
    'consumo-agua-dia'
]

alpha = 0.75
fig = go.Figure()
hex_color_primary = repcfg.FULL_PALETTE[0]
hex_color_secondary = repcfg.FULL_PALETTE[1]

idx = 0
for variable in list_vars:
    df_var = ea_dia.query("variable == @variable")
    hex_color = repcfg.FULL_PALETTE[idx % len(repcfg.FULL_PALETTE)]
    rgba_color = grp.hex_to_rgb(hex_color, alpha)
    idx += 1

    if (len(df_var) > 0):
        fig.add_trace(go.Scatter(
            x=df_var.index,
            y=df_var.value,
            line_color=rgba_color,
            name=variable,
            showlegend=True,
        ))



fig.update_layout(
    title=f"{DEVICE_NAME}: Consumo de energía activa [kWh/día]",
    font_family=repcfg.CELSIA_FONT,
    font_size=repcfg.PLOTLY_TITLE_FONT_SIZE,
    font_color=repcfg.FULL_PALETTE[1],
    title_x=repcfg.PLOTLY_TITLE_X,
    width=repcfg.JBOOK_PLOTLY_WIDTH,
    height=repcfg.JBOOK_PLOTLY_HEIGHT,
    yaxis=dict(title_text="Consumo Activa [kWh/día]")
)

fig.update_traces(mode='lines')
# fig.update_xaxes(rangemode="tozero")
fig.update_yaxes(rangemode="tozero")
fig.show()


# El consumo de agua diario medio oscila entre 10m³ y 12m³

# In[23]:


#df_plot = pd.concat([ea_dia, cs_agua_dia])

list_vars = [
    'consumo-electricidad-dia',
    'consumo-agua-dia'
]

alpha = 0.75
fig = go.Figure()
hex_color_primary = repcfg.FULL_PALETTE[0]
hex_color_secondary = repcfg.FULL_PALETTE[1]

idx = 0
for variable in list_vars:
    df_var = cs_agua_dia.query("variable == @variable")
    hex_color = repcfg.FULL_PALETTE[idx % len(repcfg.FULL_PALETTE)]
    rgba_color = grp.hex_to_rgb(hex_color, alpha)
    idx += 1

    if (len(df_var) > 0):
        fig.add_trace(go.Scatter(
            x=df_var.index,
            y=df_var.value,
            line_color=rgba_color,
            name=variable,
            showlegend=True,
        ))



fig.update_layout(
    title=f"{DEVICE_NAME}: Consumo de agua [m3/día]",
    font_family=repcfg.CELSIA_FONT,
    font_size=repcfg.PLOTLY_TITLE_FONT_SIZE,
    font_color=repcfg.FULL_PALETTE[1],
    title_x=repcfg.PLOTLY_TITLE_X,
    width=repcfg.JBOOK_PLOTLY_WIDTH,
    height=repcfg.JBOOK_PLOTLY_HEIGHT,
    yaxis=dict(title_text="Consumo agua [m3/día]")
)

fig.update_traces(mode='lines')
# fig.update_xaxes(rangemode="tozero")
fig.update_yaxes(rangemode="tozero")
fig.show()


# In[24]:


df_temp = pd.concat([t_out_day])

list_vars = [
    't-out',
    't-input'
]

alpha = 0.75
fig = go.Figure()
hex_color_primary = repcfg.FULL_PALETTE[0]
hex_color_secondary = repcfg.FULL_PALETTE[1]

idx = 0
for variable in list_vars:
    df_var = df_temp.query("variable == @variable")
    hex_color = repcfg.FULL_PALETTE[idx % len(repcfg.FULL_PALETTE)]
    rgba_color = grp.hex_to_rgb(hex_color, alpha)
    idx += 1

    if (len(df_var) > 0):
        fig.add_trace(go.Scatter(
            x=df_var.index,
            y=df_var.value,
            line_color=rgba_color,
            name=variable,
            showlegend=True,
        ))



fig.update_layout(
    title=f"{DEVICE_NAME}: Temperatura Salida [°C]",
    font_family=repcfg.CELSIA_FONT,
    font_size=repcfg.PLOTLY_TITLE_FONT_SIZE,
    font_color=repcfg.FULL_PALETTE[1],
    title_x=repcfg.PLOTLY_TITLE_X,
    width=repcfg.JBOOK_PLOTLY_WIDTH,
    height=repcfg.JBOOK_PLOTLY_HEIGHT,
    yaxis=dict(title_text="Temperatura [°C]")
)

fig.update_traces(mode='lines')
# fig.update_xaxes(rangemode="tozero")
fig.update_yaxes(rangemode="tozero")
fig.show()


# In[25]:


#df_plot = pd.concat([ea_dia, cs_agua_dia])

list_vars = [
    'cop_dia',

]

alpha = 0.75
fig = go.Figure()
hex_color_primary = repcfg.FULL_PALETTE[0]
hex_color_secondary = repcfg.FULL_PALETTE[1]

idx = 0
for variable in list_vars:
    df_var = cop_dia.query("variable == @variable")
    hex_color = repcfg.FULL_PALETTE[idx % len(repcfg.FULL_PALETTE)]
    rgba_color = grp.hex_to_rgb(hex_color, alpha)
    idx += 1

    if (len(df_var) > 0):
        fig.add_trace(go.Scatter(
            x=df_var.index,
            y=df_var.value,
            line_color=rgba_color,
            name=variable,
            showlegend=True,
        ))



fig.update_layout(
    title=f"{DEVICE_NAME}: Coeficiente de Rendimiento",
    font_family=repcfg.CELSIA_FONT,
    font_size=repcfg.PLOTLY_TITLE_FONT_SIZE,
    font_color=repcfg.FULL_PALETTE[1],
    title_x=repcfg.PLOTLY_TITLE_X,
    width=repcfg.JBOOK_PLOTLY_WIDTH,
    height=repcfg.JBOOK_PLOTLY_HEIGHT,
    yaxis=dict(title_text="Coeficiente de Rendimiento")
)

fig.update_traces(mode='lines')
# fig.update_xaxes(rangemode="tozero")
fig.update_yaxes(rangemode="tozero")
fig.show()


# In[26]:


df_pa_bl, df_pa_st = pro.split_into_baseline_and_study(df_pa, baseline=cfg.BASELINE, study=cfg.STUDY, inclusive='both')

if (len(df_pa_bl) > 0) & (len(df_pa_st) > 0):
    df_pa_bl_day = (
        df_pa_bl
        .reset_index()
        .groupby(['device_name','variable','hour'])['value']
        .agg(['median','mean','std','min',pro.q_low,pro.q_high,'max','count'])
        .reset_index()
    )

    df_pa_st_day = (
        df_pa_st
        .reset_index()
        .groupby(['device_name','variable','hour'])['value']
        .agg(['median','mean','std','min',pro.q_low,pro.q_high,'max','count'])
        .reset_index()
    )
    """
    grp.compare_baseline_day_by_hour(
        df_pa_bl_day,
        df_pa_st_day,
        title=f"{DEVICE_NAME}: Día típico",
        bl_label="Promedio línea base",
        st_label="Promedio semanal",
        bl_ci_label="Intervalo línea base",
        include_ci=True,
        fill_ci=True
    )

    """

    df_pa_bl_week = (
        df_pa_bl
        .reset_index()
        .groupby(['device_name','variable','cont_dow'])['value']
        .agg(['median','mean','std','min',pro.q_low,pro.q_high,'max','count'])
        .reset_index()
    )

    df_pa_st_week = (
        df_pa_st
        .reset_index()
        .groupby(['device_name','variable','cont_dow'])['value']
        .agg(['median','mean','std','min',pro.q_low,pro.q_high,'max','count'])
        .reset_index()
    )

    grp.compare_baseline_week_by_day(
        df_pa_bl_week,
        df_pa_st_week,
        title=f"{DEVICE_NAME}: Semana típica",
        bl_label="Promedio línea base",
        st_label="Promedio semanal",
        bl_ci_label="Intervalo línea base",
        include_ci=True,
        fill_ci=True

    )


# ## Consideraciones importantes
# 
#  El COP (Coeficiente de Rendimiento) es un indicador de eficiencia en la calentamiento del agua. A mayor valor, mayor eficiencia energética del sistema.
# 
#  Durante los meses de enero y febrero, el rendimiento se vio afectado debido al polvo generado por las actividades de construcción.
#  
#  A partir de marzo, el rendimiento se estabilizó en 3.2.
