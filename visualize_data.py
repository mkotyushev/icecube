import sqlite3
from dash import Dash, dcc, html, Input, Output, State
import plotly.express as px
import plotly.graph_objects as go

import pandas as pd


# Load metadata
database_path = '/workspace/icecube/data/batch_656.db'
with sqlite3.connect(database_path) as conn:
    meta_query = f'SELECT * FROM meta_table'
    df = pd.read_sql(meta_query, conn)
df = df.sample(100)

# Load sensors geometry
df_sensor_geometry = pd.read_csv('data/dataset/sensor_geometry.csv')

with sqlite3.connect(database_path) as conn:
    pulse_query = f'SELECT * FROM pulse_table WHERE event_id = {df.iloc[0]["event_id"]}'
    df_pulse = pd.read_sql(pulse_query, conn)

# Create static figure of sensors geometry
# with blank trace for data
original_data = [
    go.Scatter3d(
        visible = True,
        x = df_sensor_geometry.x,
        y = df_sensor_geometry.y,
        z = df_sensor_geometry.z,
        mode='markers',
        marker=dict(
            size=1,
        ),
        uirevision='constant',
    ),
    go.Scatter3d(
        visible = True,
        mode='markers',
        uirevision='constant',
    )
]

fig = go.Figure(data=original_data)
fig.layout.scene.camera.projection.type = "orthographic"
fig.layout.uirevision = 'constant'

app = Dash(__name__)
app.layout = html.Div([
    dcc.Graph(id='graph-with-slider', figure=fig),
    dcc.Slider(
        df['event_id'].min(),
        df['event_id'].max(),
        step=None,
        value=df['event_id'].min(),
        marks={str(event_id): str(event_id) for event_id in df['event_id'].unique()},
        id='event_id-slider'
    ),
    dcc.Slider(
        id='time-slider'
    ),
    dcc.Interval(id="animate", disabled=True),
    html.Button("Play/Stop", id="play"),
])


@app.callback(
    Output('time-slider', 'value'),
    Output('time-slider', 'min'),
    Output('time-slider', 'max'),
    Output('graph-with-slider', 'figure'),
    Input('event_id-slider', 'value'))
def update_figure_on_event_id_change(event_id):
    global database_path, df_pulse
    with sqlite3.connect(database_path) as conn:
        pulse_query = f'SELECT * FROM pulse_table WHERE event_id = {event_id}'
        df_pulse = pd.read_sql(pulse_query, conn)

    data = [
        original_data[0],
        go.Scatter3d(
            visible = True,
            x = df_pulse.x,
            y = df_pulse.y,
            z = df_pulse.z,
            mode='markers',
            marker={
                'size': df_pulse.charge * 10,
                'opacity': 0.5,
            },
            uirevision='constant',
        ),
    ]

    return df_pulse.time.max(), df_pulse.time.min(), df_pulse.time.max(), {
        'data': data,
        'layout': {
            'uirevision': 'constant',
        },
    }


@app.callback(
    Output('graph-with-slider', 'figure', allow_duplicate=True),
    Input('time-slider', 'value'), prevent_initial_call=True)
def update_figure_on_time_change(time):
    global df_pulse

    mask = df_pulse.time <= time
    data = [
        original_data[0],
        go.Scatter3d(
            visible = True,
            x = df_pulse.x[mask],
            y = df_pulse.y[mask],
            z = df_pulse.z[mask],
            mode='markers',
            marker={
                'size': df_pulse.charge[mask] * 10,
                'opacity': 0.5,
            },
            uirevision='constant',
        ),
    ]

    return {
        'data': data,
        'layout': {
            'uirevision': 'constant',
        },
    }


@app.callback(
    Output("time-slider", "value", allow_duplicate=True),
    Input('animate', 'n_intervals'),
    prevent_initial_call=True,
)
def update_output(n_intervals):
    global df_pulse
    selected_value = df_pulse.iloc[(n_intervals % len(df_pulse))]['time']
    return selected_value


@app.callback(
    Output("animate", "disabled"),
    Input("play", "n_clicks"),
    Input("animate", "disabled"),
)
def toggle(n_clicks, disabled):
    if n_clicks:
        return not disabled
    return disabled


if __name__ == '__main__':
    app.run_server(host='127.0.0.1', port='8050', debug=True)
