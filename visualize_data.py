import sqlite3
from dash import Dash, dcc, html, Input, Output
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

from icecube_utils import angles_to_xyz
from train_large import seed_everything

seed_everything(0)

# Load metadata
database_path = '/workspace/icecube/data/batch_656.db'
with sqlite3.connect(database_path) as conn:
    meta_query = f'SELECT * FROM meta_table'
    df_meta = pd.read_sql(meta_query, conn)
df_meta = df_meta.sample(100)
event_ids_str = ", ".join(df_meta["event_id"].astype(int).astype(str).to_list())

# Load sensors geometry
df_sensor_geometry = pd.read_csv('data/dataset/sensor_geometry.csv')

# Load pulse data
with sqlite3.connect(database_path) as conn:
    pulse_query = f'SELECT * FROM pulse_table WHERE event_id in ({event_ids_str})'
    df_pulses = pd.read_sql(pulse_query, conn)
df_pulse = df_pulses[df_pulses['event_id'] == df_meta.iloc[0]['event_id']]

# Load prediction results
df_prediction = pd.read_hdf('results/angles_merged.h5', key='df')

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
        ),
        uirevision='constant',
        name='Sensors'
    ),
    go.Scatter3d(
        visible = True,
        mode='markers',
        uirevision='constant',
    ),
    go.Scatter3d(
        visible = True,
        uirevision='constant',
    ),
    go.Scatter3d(
        visible = True,
        uirevision='constant',
    )
]

fig = go.Figure(data=data)
fig.layout.scene.camera.projection.type = "orthographic"
fig.layout.uirevision = 'constant'

app = Dash(__name__)
app.layout = html.Div([
    dcc.Graph(id='graph-with-slider', figure=fig),
    dcc.Slider(
        0,
        len(df_meta),
        step=None,
        value=0,
        marks={str(event_index): str(event_index) for event_index in range(len(df_meta))},
        id='event_id-slider'
    ),
    dcc.Slider(
        id='time-slider'
    ),
    dcc.Interval(id="animate", disabled=True, interval=500),
    html.Button("Play/Stop", id="play"),
])


@app.callback(
    Output('time-slider', 'value'),
    Output('time-slider', 'min'),
    Output('time-slider', 'max'),
    Output('graph-with-slider', 'figure'),
    Input('event_id-slider', 'value'))
def update_figure_on_event_id_change(event_index):
    global database_path, df_pulses, df_pulse, df_meta, df_prediction, data

    df_pulse = df_pulses[df_pulses['event_id'] == df_meta.iloc[event_index]['event_id']]

    zenith, azimuth = df_meta.iloc[event_index]['zenith'], df_meta.iloc[event_index]['azimuth']
    true_x, true_y, true_z = angles_to_xyz(azimuth, zenith)
    true_x, true_y, true_z = true_x * 500, true_y * 500, true_z * 500

    zenith_pred, azimuth_pred = df_prediction.iloc[event_index]['zenith'], df_prediction.iloc[event_index]['azimuth']
    pred_x, pred_y, pred_z = angles_to_xyz(azimuth_pred, zenith_pred)
    pred_x, pred_y, pred_z = pred_x * 500, pred_y * 500, pred_z * 500

    print(
        f'Event {event_index} with '
        f'true direction ({true_x}, {true_y}, {true_z}), '
        f'predicted direction ({pred_x}, {pred_y}, {pred_z})'
    )

    data = [
        data[0],
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
            name='Pulses'
        ),
        go.Scatter3d(
            visible = True,
            x=[0, true_x],
            y=[0, true_y],
            z=[0, true_z],
            marker={
                'size': 0,
                'sizemin': 0,
            },
            line={
                'color': 'green'
            },
            name='True direction'
        ),
        go.Scatter3d(
            visible = True,
            x=[0, pred_x],
            y=[0, pred_y],
            z=[0, pred_z],
            marker={
                'size': 0,
                'sizemin': 0,
            },
            line={
                'color': 'yellow'
            },
            name='Predicted direction'
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
    global df_pulse, data

    mask = df_pulse.time <= time
    data = [
        data[0],
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
            name='Pulses'
        ),
        data[2],
        data[3],
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
