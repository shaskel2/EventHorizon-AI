import dash
from dash import dcc, html, Input, Output
import plotly.graph_objects as go
import torch
import numpy as np
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from models.pinn import OpticalEventHorizonPINN
from physics.wave_eq import get_gradients

# 1. Initialize the Dash Web Server
app = dash.Dash(__name__)

# 2. Load the Model & Generate Data
model = OpticalEventHorizonPINN(num_layers=6, hidden_dim=100)
model.load_state_dict(torch.load("data/processed/event_horizon_pinn.pth"))
model.eval()

nx, nt = 100, 100
L, T = 10.0, 5.0
x_flat = np.linspace(0, L, nx)
t_flat = np.linspace(0, T, nt)
X, Time = np.meshgrid(x_flat, t_flat)

x_tensor = torch.tensor(X.flatten()[:, None], dtype=torch.float32, requires_grad=True)
t_tensor = torch.tensor(Time.flatten()[:, None], dtype=torch.float32, requires_grad=True)

u_pred, v_pred = model(x_tensor, t_tensor)
u_t, v_t = get_gradients(u_pred, t_tensor), get_gradients(v_pred, t_tensor)
u_x, v_x = get_gradients(u_pred, x_tensor), get_gradients(v_pred, x_tensor)
u_xx, v_xx = get_gradients(u_x, x_tensor), get_gradients(v_x, x_tensor)

intensity_tensor = u_pred**2 + v_pred**2
f_u = -v_t + 0.5 * u_xx + intensity_tensor * u_pred
f_v =  u_t + 0.5 * v_xx + intensity_tensor * v_pred

# Detach and reshape for Plotly
error_surface = (f_u**2 + f_v**2).detach().numpy().reshape(nt, nx)
intensity = intensity_tensor.detach().numpy().reshape(nt, nx)
u_surface = u_pred.detach().numpy().reshape(nt, nx)

# 3. Pre-generate the Static 3D Figures
fig_intensity_3d = go.Figure(data=[go.Surface(z=intensity, x=x_flat, y=t_flat, colorscale='Inferno')])
fig_intensity_3d.update_layout(title="Intensity |ψ|²", scene=dict(xaxis_title='Space', yaxis_title='Time', zaxis_title='Intensity'), template='plotly_dark', margin=dict(l=0, r=0, b=0, t=40))

fig_real_3d = go.Figure(data=[go.Surface(z=u_surface, x=x_flat, y=t_flat, colorscale='Viridis')])
fig_real_3d.update_layout(title="Real Component (u)", scene=dict(xaxis_title='Space', yaxis_title='Time', zaxis_title='Amplitude'), template='plotly_dark', margin=dict(l=0, r=0, b=0, t=40))

fig_error_3d = go.Figure(data=[go.Surface(z=error_surface, x=x_flat, y=t_flat, colorscale='Reds')])
fig_error_3d.update_layout(title="Physics Error Map", scene=dict(xaxis_title='Space', yaxis_title='Time', zaxis_title='Error'), template='plotly_dark', margin=dict(l=0, r=0, b=0, t=40))

# 4. Construct the UI Layout
# Notice we removed the global 'color': 'white' so the tooltip background renders correctly
app.layout = html.Div(style={'backgroundColor': '#111111', 'display': 'flex', 'minHeight': '100vh', 'fontFamily': 'sans-serif'}, children=[
    
    # --- SIDEBAR ---
    html.Div(style={'width': '25%', 'padding': '20px', 'backgroundColor': '#1E1E1E', 'borderRight': '1px solid #333'}, children=[
        html.H2("Forensics Panel", style={'borderBottom': '1px solid #444', 'paddingBottom': '10px', 'color': 'white'}),
        html.P("Use the controls below to slice the spacetime data and isolate specific variables.", style={'color': '#AAA', 'fontSize': '14px'}),
        
        html.Label("Time Step (t) Slice:", style={'fontWeight': 'bold', 'marginTop': '20px', 'display': 'block', 'color': 'white'}),
        dcc.Slider(
            id='time-slider',
            min=0, max=T, step=T/nt, value=0,
            # Explicitly styling the labels so they show up bright white against the dark background
            marks={
                0: {'label': '0.0s', 'style': {'color': '#ffffff'}},
                2.5: {'label': '2.5s', 'style': {'color': '#ffffff'}},
                5.0: {'label': '5.0s', 'style': {'color': '#ffffff'}}
            },
            tooltip={"placement": "bottom", "always_visible": True} 
        ),
        
        html.Div(id='error-metric-display', style={'marginTop': '40px', 'padding': '15px', 'backgroundColor': '#2A2A2A', 'borderRadius': '5px'})
    ]),
    
    # --- MAIN CONTENT AREA ---
    html.Div(style={'width': '75%', 'padding': '20px', 'overflowY': 'scroll'}, children=[
        html.H1("EventHorizon-AI Analytics", style={'color': 'white'}),
        
        # Row 1: The Interactive 2D Cross Section
        html.Div([
            dcc.Graph(id='2d-slice-graph', style={'height': '40vh'})
        ], style={'marginBottom': '20px'}),
        
        # Row 2: All three 3D Overviews restored
        html.Div(style={'display': 'flex', 'justifyContent': 'space-between'}, children=[
            html.Div(dcc.Graph(figure=fig_intensity_3d, style={'height': '45vh'}), style={'width': '32%'}),
            html.Div(dcc.Graph(figure=fig_real_3d, style={'height': '45vh'}), style={'width': '32%'}),
            html.Div(dcc.Graph(figure=fig_error_3d, style={'height': '45vh'}), style={'width': '32%'}),
        ])
    ])
])

# 5. The Callback
@app.callback(
    [Output('2d-slice-graph', 'figure'),
     Output('error-metric-display', 'children')],
    [Input('time-slider', 'value')]
)
def update_dashboard(selected_time):
    t_idx = (np.abs(t_flat - selected_time)).argmin()
    
    slice_intensity = intensity[t_idx, :]
    slice_error = error_surface[t_idx, :]
    
    fig_2d = go.Figure()
    fig_2d.add_trace(go.Scatter(x=x_flat, y=slice_intensity, mode='lines', name='Wave Intensity', line=dict(color='#00ffcc', width=3)))
    fig_2d.add_trace(go.Scatter(x=x_flat, y=slice_error, mode='lines', name='Physics Error', line=dict(color='#ff4444', width=2, dash='dash')))
    
    fig_2d.update_layout(title=f"Cross-Section at t = {selected_time:.2f}s", xaxis_title="Space (x)", yaxis_title="Amplitude / Error", template='plotly_dark', margin=dict(l=0, r=0, b=0, t=40))
    
    max_err = np.max(slice_error)
    metric_text = html.Div([
        html.H4("Instantaneous Diagnostics", style={'color': 'white', 'marginTop': '0px'}),
        html.P(f"Max Error at t={selected_time:.2f}s:", style={'color': '#AAAAAA', 'marginBottom': '5px'}),
        html.H3(f"{max_err:.6f}", style={'color': '#ff4444' if max_err > 0.01 else '#00ffcc', 'marginTop': '0px'})
    ])
    
    return fig_2d, metric_text

if __name__ == '__main__':
    app.run(debug=True)
