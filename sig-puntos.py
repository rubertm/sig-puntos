import dash

from dash import dcc
from dash import html
from dash.dependencies import Input, Output, State
from dash import no_update
#import dash_table
from dash import dash_table


# from dash import dash_table

#from sklearn.preprocessing import OrdinalEncoder
import pandas as pd
# import shapely.geometry
import numpy as np
#import json
import plotly.express as px
import plotly.graph_objects as go
#from plotly.subplots import make_subplots

import base64
import datetime
import re
import io
#import os
#import requests

# --------------------------
token = 'pk.eyJ1Ijoicm1vcHl0aG9uIiwiYSI6ImNrbmZ6MGZyMDF3Yncyd2s4ODVoMmR1Z3EifQ.FCGeYHLeHwjRkksgEyIrSw'
# with open('js_areasxxxxxxxxxx.json') as cs:
#    js_area = geojson.load(cs)

# ------------------------------------------------------------------
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets,
                suppress_callback_exceptions=True
                )
server = app.server
# ------------------------------------------------------------------

# ------------------------------------------------------------------

app.layout = html.Div([

    dcc.Upload(
        id='upload-data',
        children=html.Div([
            'App para la visualización geográfica de puntos  ',
            html.A('seleccione el archivo')
        ]),
        style={
            'width': '98%',
            'height': '30px',
            'lineHeight': '30px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px',
            'color': '#1F618D',
            'fontSize': 18
        },
        # Multiples archivos se pueden cargar
        multiple=True
    ),
    # html.Div(id='output-div'),
    html.Div(id='output-datatable'),
])


def parse_contents(contents, filename, date):
    content_type, content_string = contents.split(',')

    decoded = base64.b64decode(content_string)
    try:
        if 'csv' in filename:
            # Assume that the user uploaded a CSV file
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
        elif 'xls' in filename:
            # Assume that the user uploaded an excel file
            df = pd.read_excel(io.BytesIO(decoded), sheet_name='bdgrafico', dtype={"ejex": str})

            dp = pd.read_excel(io.BytesIO(decoded), sheet_name='puntos', dtype={"CAPA_GEOGRAFICA": str})
            radioitemsp = list(dp.columns)
            radioitemsp = radioitemsp[5:(len(radioitemsp) - 1)]

    except Exception as e:
        print(e)
        return html.Div([
            'error al procesar este archivo'
        ])

    return html.Div([  # inicio

        html.Div(
            [

                html.Label("Clasificación Geográfica",
                           style={'fontSize': 13, 'text-align': 'left', 'padding': '0px 30px 0px 15px',
                                  'color': '#1A5276', "font-weight": "bold"}),
                dcc.Dropdown(id='geografica',
                             options=[{'label': c, 'value': c} for c in sorted(dp.CAPA_GEOGRAFICA.unique())],
                             multi=True,
                             value=[dp.CAPA_GEOGRAFICA[0]],
                             clearable=False,
                             style={'fontSize': 12, 'color': '#1B2631'}
                             ),

                # html.Br(),
                html.Label("Clasificación del mapa",
                           style={'fontSize': 13, 'text-align': 'left', 'padding': '0px 30px 0px 15px',
                                  'color': '#1A5276', "font-weight": "bold"}),
                dcc.RadioItems(id='clasificacionpunto',
                               options=[{'label': i, 'value': i} for i in radioitemsp], value=radioitemsp[1],
                               style={'fontSize': 12, 'color': '#1B2631'},
                               labelStyle={'display': 'block'}  # 'inline-block'
                               ),

                html.Label("Nivel de Transparencia del usuario",
                           style={'fontSize': 13, 'text-align': 'left', 'padding': '0px 30px 0px 15px',
                                  'color': '#1A5276', "font-weight": "bold"}),
                dcc.Slider(id='transparenciap',
                           min=0, max=1, step=0.1, value=0.4,
                           marks={0: {'label': '0', 'style': {'color': '#77b0b1', 'fontSize': 9}},
                                  0.2: {'label': '0.2', 'style': {'fontSize': 9}},
                                  0.4: {'label': '0.4', 'style': {'fontSize': 9}},
                                  0.6: {'label': '0.6', 'style': {'fontSize': 9}},
                                  0.8: {'label': '0.8', 'style': {'fontSize': 9}},
                                  1: {'label': 'Sin Transparencia', 'style': {'color': '#f50', 'fontSize': 9}}},
                           ),

                # html.Br(),

                html.Label("Mapas de fondo",
                           style={'fontSize': 13, 'text-align': 'left', 'padding': '0px 30px 0px 15px',
                                  'color': '#1A5276', "font-weight": "bold"}),
                dcc.RadioItems(id='estilomapa',
                               options=[{'label': i, 'value': i} for i in
                                        ['carto-positron', 'streets', 'satellite', 'satellite-streets']],
                               value='carto-positron',
                               style={'fontSize': 12, 'color': '#1B2631'},
                               labelStyle={'display': 'block'}  # 'inline-block'
                               ),

                # html.Button(id="submit-button", children="Crear Mapa",  style={'fontSize': 10, 'text-align': 'center',
                #                                                        'color': '#154360','padding': '0px 30px 0px 15px'}),

                # html.Br(),

                html.Label("Gráfico de: ",
                           style={'fontSize': 13, 'text-align': 'left', 'padding': '0px 30px 0px 15px',
                                  'color': '#1A5276', "font-weight": "bold"}),
                dcc.Dropdown(id='graficodatos',
                             options=[{'label': c, 'value': c} for c in sorted(df.escenario.unique())],
                             multi=False,
                             value=(df.escenario.unique())[0],
                             searchable=False,
                             style={'fontSize': 12, 'color': '#1B2631'},
                             # clearable=False, multi=False, value=(df.escenario.unique())[0]
                             ),
                # html.Hr(),
                dcc.Graph(id='grafica1',
                          figure={},
                          style={'float': 'left', 'display': 'block',
                                 'borderBottom': 'thin lightgrey solid',
                                 # 'backgroundColor': '#F8F9F9',
                                 'padding': '0px 0px 0px 0px'
                                 }),

                html.Label("Contacto:rmopython@gmail.com, WhatsApp (+57) 301 7565982, fecha:2022.",
                           style={'fontSize': 10, 'text-align': 'left', 'padding': '0px 1px 0px 5px',
                                  'color': '#1A5276',
                                  "font-weight": "bold", 'display': 'block', }),
            ],
            style={'width': '30%', 'display': 'inline-block'}),

        html.Div([

            dcc.Graph(id='grafica0',
                      figure={},
                      clickData=None,
                      hoverData=None,
                      config={
                          'staticPlot': False,  # True, False
                          'scrollZoom': True,  # True, False
                          'doubleClick': 'reset',  # 'reset', 'autosize' or 'reset+autosize', False
                          'showTips': False,  # True, False
                          'displayModeBar': True,  # True, False, 'hover'
                          'watermark': True,
                          # 'modeBarButtonsToRemove': ['pan2d','select2d'],
                      },

                      ),
        ], style={'width': '67%', 'float': 'right', 'display': 'inline-block',
                  'borderBottom': 'thin lightgrey solid',
                  # 'backgroundColor': '#F8F9F9',
                  'padding': '5px 5px 5px 5px'

                  }),

        html.Hr(),

        html.H6(filename),
        html.H6(datetime.datetime.fromtimestamp(date)),

        dash_table.DataTable(
            data=dp.to_dict('records'),
            columns=[{'name': i, 'id': i} for i in dp.columns],
            page_size=3
        ),

        dcc.Store(id='stored-data-p', data=dp.to_dict('records')),
        dcc.Store(id='stored-data-df', data=df.to_dict('records')),

        html.Hr(),  # horizontal line

        # For debugging, display the raw contents provided by the web browser
        html.Div('Raw Content'),
        html.Pre(contents[0:200] + '...', style={
            'whiteSpace': 'pre-wrap',
            'wordBreak': 'break-all'
        })
    ])  # Final


@app.callback(Output('output-datatable', 'children'),
              Input('upload-data', 'contents'),
              State('upload-data', 'filename'),
              State('upload-data', 'last_modified'))
def update_output(list_of_contents, list_of_names, list_of_dates):
    if list_of_contents is not None:
        children = [
            parse_contents(c, n, d) for c, n, d in
            zip(list_of_contents, list_of_names, list_of_dates)]
        return children


@app.callback(Output('grafica0', 'figure'),
              # Input('submit-button','n_clicks'),
              Input('geografica', 'value'),
              Input('clasificacionpunto', 'value'),
              Input('transparenciap', 'value'),
              Input('estilomapa', 'value'),
              State('stored-data-p', 'data')
              )
def make_graphs(opc_geografica, opc_clasificacionp, opc_transparenciap, opc_estilomapa, datap):
    if opc_geografica is None:
        return no_update
    else:
        dp0 = pd.DataFrame(datap)
        dp1 = dp0[(dp0.CAPA_GEOGRAFICA.isin(opc_geografica))]
        # colorp = dp1[opc_clasificacionp]
        # textp0 = dp1['text']
        # textp = textp0.values.tolist()
        # Nodos del proyecto !!!!! Los siguientes codigos son para otro tipo de grafico

        # dp2 = pd.unique(dp1[opc_clasificacionp])
        # a = 8 + 1 * len(dp2)
        # marker_size = np.arange(8, a, 1)
        # marker_color = ['#2E86C1', '#CB4335', '#28B463', '#D68910', '#8E44AD', '#1ABC9C', '#F1C40F', '#00FFFF',
        #                '#800080', '#FF00FF']
        # fig = go.Figure()
        # for i in range(0, len(dp2)):
        # dp3 = dp1[dp1[opc_clasificacionp] == dp2[i]]
        # Nodos del proyecto
        # lons3 = []
        # lats3 = []

        # lons3 = np.empty(len(dp3))
        # lons3 = dp3['X_INICIO']
        # lats3 = np.empty(len(dp3))
        # lats3 = dp3['Y_INICIO']

        # text2 = []
        # text2 = np.empty(len(dp3), dtype=object)
        # text2 = dp3['text']

        # fig.add_trace(go.Scattermapbox(mode='markers', lon=lons3, lat=lats3, name=dp2[i],
        #                           hovertext=text2, opacity=opc_transparenciap,
        #                           marker=dict(size=marker_size[i],
        #                                       color=marker_color[i],
        #                                       )))
        #fig = go.Figure()

        # object, int64, 'float64'
        type_dato = (dp1[opc_clasificacionp].dtype)
        #dpsize = dp1
        #print(dp1[opc_clasificacionp])
        #dpsize[opc_clasificacionp] = dpsize[opc_clasificacionp].astype('category').cat.codes

        if type_dato == 'object':
            fig = px.scatter_mapbox(dp1, lat=dp1['Y_INICIO'], lon=dp1['X_INICIO'], color=dp1[opc_clasificacionp],
                                    # size=sizep,
                                    size_max=10,
                                    opacity=opc_transparenciap,
                                    hover_name=dp1['text'],
                                    # text=dp1['CODIGO'],
                                    custom_data=['CODIGO']
                                    )
            fig.update_layout(margin={"r": 150, "t": 35, "l": 10, "b": 10}, showlegend=True,
                              mapbox_style=opc_estilomapa,
                              mapbox_accesstoken='pk.eyJ1Ijoicm1vcHl0aG9uIiwiYSI6ImNrbmZ6MGZyMDF3Yncyd2s4ODVoMmR1Z3EifQ.FCGeYHLeHwjRkksgEyIrSw',
                              mapbox_zoom=15,
                              mapbox_center={'lat': dp1['Y_INICIO'].mean(), 'lon': dp1['X_INICIO'].mean()},
                              width=1000, height=650,
                              )
        else:
            fig = px.scatter_mapbox(dp1, lat=dp1['Y_INICIO'], lon=dp1['X_INICIO'], color=dp1[opc_clasificacionp],
                                size=dp1[opc_clasificacionp],
                                size_max=20,
                                opacity=opc_transparenciap,
                                #color_continuous_scale=px.colors.cyclical.IceFire,
                                hover_name=dp1['text'],
                                # text=dp1['CODIGO'],
                                custom_data=['CODIGO']
                                )
            fig.update_layout(margin={"r": 150, "t": 35, "l": 10, "b": 10}, showlegend=True,
                          mapbox_style=opc_estilomapa,
                          mapbox_accesstoken='pk.eyJ1Ijoicm1vcHl0aG9uIiwiYSI6ImNrbmZ6MGZyMDF3Yncyd2s4ODVoMmR1Z3EifQ.FCGeYHLeHwjRkksgEyIrSw',
                          mapbox_zoom=15, mapbox_center={'lat': dp1['Y_INICIO'].mean(), 'lon': dp1['X_INICIO'].mean()},
                          width=1000, height=650,
                          coloraxis_colorbar=dict(title=opc_clasificacionp, titleside='top',
                                                 titlefont=dict(size=11, family='Arial, sans-serif'),
                                                 thicknessmode='fraction',
                                                 len=0.5, lenmode='fraction', outlinewidth=1, y=0.3,
                                                 )
                          )
        return fig


@app.callback(
    Output('grafica1', 'figure'),
    Input(component_id='grafica0', component_property='hoverData'),
    Input(component_id='grafica0', component_property='clickData'),
    Input(component_id='grafica0', component_property='selectedData'),
    Input('graficodatos', 'value'),
    State('stored-data-df', 'data'),
)
def make_graph2(hov_data, clk_data, slct_data, slct_grafico, datadf):
    if slct_grafico is None:
        return no_update
    else:
        if clk_data is None:
            df0 = pd.DataFrame(datadf)
            # df1 = df0[(df0.escenario == slct_grafico) & (df0.ejex >= 2035)]
            df11 = df0[(df0.escenario == slct_grafico)]
            df10 = df11.reset_index()
            codigo_n = df10['grafico'][0]

            df1 = df10[(df10.grafico == codigo_n)]

            titulog = df1['Identificacion'][0]

            ejex0 = df1.ejex.unique()
            grafico0 = df1.grafico.unique()
            nejex0 = len(df1.ejex.unique())
            ngrafico0 = len(df1.grafico.unique())
            nxxng = nejex0 * ngrafico0
            ejex1 = []
            ejex1 = np.empty(nxxng, dtype=object)
            grafico1 = []
            grafico1 = np.empty(nxxng, dtype=object)
            ejey1 = []
            ejey1 = np.empty(nxxng, dtype=object)

            n = -1
            for i in range(0, ngrafico0):
                for j in range(0, nejex0):
                    n = n + 1
                    grafico1[n] = grafico0[i]
                    ejex1[n] = ejex0[j]
                    ejey2 = df1[(df1.ejex == ejex0[j]) & (df1.grafico == grafico0[i])]
                    ejey1[n] = round(ejey2['ejey'].sum(), 2)

            df3 = pd.DataFrame({'ejex': ejex1, 'grafico': grafico1, 'ejey': ejey1})
            for i, row in enumerate(df3['ejex']):
                p = re.compile(" 00:00:00")
                datetime = p.split(df3['ejex'][i])[0]
                df3.iloc[i, 0] = datetime

            df3['text'] = 'grafico: ' + df3['grafico'].astype(str) + '<br>' + \
                          'ejey: ' + df3['ejey'].astype(str)

            df4 = pd.unique(df3['grafico'])
            grafico_color = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A', '#19D3F3', '#FF6692', '#B6E880',
                             '#FF97FF', '#FECB52']

            fig2 = go.Figure()
            for i in range(0, len(df4)):
                df5 = df3[df3['grafico'] == df4[i]]
                y = df5['ejey']
                fig2.add_trace(go.Bar(x=df3['ejex'], y=y, text=df5, textposition='auto', marker_color=grafico_color[i],
                                      name=df4[i].astype(str),

                                      )
                               )

            fig2.update_layout(
                title=f'Identificacion: {titulog}',
                titlefont=dict(size=10, family='Arial, sans-serif'),
                template='plotly_white',
                width=500, height=250, margin={"r": 50, "t": 75, "l": 10, "b": 10},
                barmode='stack',
                yaxis_title="eje y"
                # xaxis={'categoryorder': 'category ascending'},
            )
            fig2.update_xaxes(tickangle=90, tickfont=dict(family='Rockwell', color='blue', size=10))

            return fig2
        else:
            # print(f'hover data: {hov_data}')
            # print(hov_data['points'][0]['customdata'][0])
            # print(f'click data: {clk_data}')
            # print(f'selected data: {slct_data}')

            df00 = pd.DataFrame(datadf)
            # df1 = df0[(df0.escenario == slct_grafico) & (df0.ejex >= 2035)]
            df10 = df00[(df00.escenario == slct_grafico)]
            hov_codigo1 = clk_data['points'][0]['customdata']
            hov_codigo = hov_codigo1[0]
            # print(hov_codigo1, type(hov_codigo1), hov_codigo, type(hov_codigo))
            df12 = df10[(df10.grafico == hov_codigo)]
            df1 = df12.reset_index()
            titulog = df1['Identificacion'][0]
            # print(titulog, type(titulog))
            ejex0 = df1.ejex.unique()
            grafico0 = df1.grafico.unique()
            nejex0 = len(df1.ejex.unique())
            ngrafico0 = len(df1.grafico.unique())
            nxxng = nejex0 * ngrafico0
            ejex1 = []
            ejex1 = np.empty(nxxng, dtype=object)
            grafico1 = []
            grafico1 = np.empty(nxxng, dtype=object)
            ejey1 = []
            ejey1 = np.empty(nxxng, dtype=object)

            n = -1
            for i in range(0, ngrafico0):
                for j in range(0, nejex0):
                    n = n + 1
                    grafico1[n] = grafico0[i]
                    ejex1[n] = ejex0[j]
                    ejey2 = df1[(df1.ejex == ejex0[j]) & (df1.grafico == grafico0[i])]
                    ejey1[n] = round(ejey2['ejey'].sum(), 2)

            df3 = pd.DataFrame({'ejex': ejex1, 'grafico': grafico1, 'ejey': ejey1})
            for i, row in enumerate(df3['ejex']):
                p = re.compile(" 00:00:00")
                datetime = p.split(df3['ejex'][i])[0]
                df3.iloc[i, 0] = datetime

            df3['text'] = 'grafico: ' + df3['grafico'].astype(str) + '<br>' + \
                          'ejey: ' + df3['ejey'].astype(str)

            df4 = pd.unique(df3['grafico'])
            grafico_color = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A', '#19D3F3', '#FF6692', '#B6E880',
                             '#FF97FF', '#FECB52']

            fig2 = go.Figure()
            for i in range(0, len(df4)):
                df5 = df3[df3['grafico'] == df4[i]]
                y = df5['ejey']
                fig2.add_trace(go.Bar(x=df3['ejex'], y=y, text=df5, textposition='auto', marker_color=grafico_color[i],
                                      name=df4[i].astype(str),

                                      )
                               )

            fig2.update_layout(title=f'Identificacion: {titulog}',
                               titlefont=dict(size=10, family='Arial, sans-serif'),
                               template='plotly_white',
                               width=500, height=250, margin={"r": 50, "t": 75, "l": 10, "b": 10},
                               barmode='stack',
                               yaxis_title="eje y"
                               # xaxis={'categoryorder': 'category ascending'},
                               )
            fig2.update_xaxes(tickangle=90, tickfont=dict(family='Rockwell', color='blue', size=10))

            return fig2
# ------------------------------------------------------------------
if __name__ == '__main__':
    #app.run_server(debug=False, host="127.0.0.1", port=8085)
    app.run_server(debug=True)
