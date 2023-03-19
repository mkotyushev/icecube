import sqlite3
from dash import Dash, dcc, html, Input, Output
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
df_pulse = None

# Create static figure of sensors geometry
# with blank trace for data
data = [
    go.Scatter3d(
        visible = True,
        x = df_sensor_geometry.x,
        y = df_sensor_geometry.y,
        z = df_sensor_geometry.z,
        mode='markers',
        marker=dict(
            size=1,
        )
    ),
    go.Scatter3d(
        visible = True,
        mode='markers',
    )
]

fig = go.Figure(data=data)
fig.layout.scene.camera.projection.type = "orthographic"

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
    )
])


@app.callback(
    Output('time-slider', 'min'),
    Output('time-slider', 'max'),
    Output('graph-with-slider', 'figure'),
    Input('event_id-slider', 'value'))
def update_figure(event_id):
    global database_path, fig, df_pulse
    with sqlite3.connect(database_path) as conn:
        pulse_query = f'SELECT * FROM pulse_table WHERE event_id = {event_id}'
        df_pulse = pd.read_sql(pulse_query, conn)

    fig.data[1]['x'] = df_pulse.x
    fig.data[1]['y'] = df_pulse.y
    fig.data[1]['z'] = df_pulse.z
    fig.data[1]['marker']['size'] = df_pulse.charge * 10

    return df_pulse.time.min(), df_pulse.time.max(), fig


@app.callback(
    Output('graph-with-slider', 'figure', allow_duplicate=True),
    Input('time-slider', 'value'), prevent_initial_call=True)
def update_figure(time):
    global database_path, fig, df_pulse

    if df_pulse is not None:
        mask = df_pulse.time <= time
        fig.data[1]['x'] = df_pulse.x[mask]
        fig.data[1]['y'] = df_pulse.y[mask]
        fig.data[1]['z'] = df_pulse.z[mask]
        fig.data[1]['marker']['size'] = df_pulse.charge[mask] * 10

    return fig


if __name__ == '__main__':
    app.run_server(host='127.0.0.1', port='8050', debug=True)
