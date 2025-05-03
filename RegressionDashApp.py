import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import numpy as np
import pickle

imgsrc= "https://info.topmba.com/hubfs/Full-QS-logo_Social_Media-Mar-07-2023-02-49-12-1555-PM.png"

featurelabels=['Academic Reputation Score','Employer Reputation Score',
               'Intl. Research Network Score','Employment Outcomes Score','Sustainability Score',
               'No. of Citations','Total Faculty','International Faculty','Total Students',
               'International Students']

# Load the pre-trained model (assuming 'model.pkl' is the saved model)
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# Initialize Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Define app layout
app.layout = dbc.Container([
    ##html.Img(src=imgsrc, style={"width": "30px", "margin": "10px 0"}),
    ##html.H1("QS Score Predictor"),

    dbc.Row([
        dbc.Col(html.Img(src=imgsrc , style={"width": "300px", "height": "80px"}), width="auto"),
        dbc.Col(html.H1("QS Score Predictor"), width="auto")  # Auto width to fit content
    ], align="center", className="g-2"),  # Center align and add gap
    html.Hr(),

    # Input fields for the 10 independent variables
    dbc.Row([
        dbc.Col([html.Label(featurelabels[i]), dcc.Input(id=f"input-{i}", type="number", placeholder=f"", className="form-control")], md=6)
        for i in range(10)
    ]),

    html.Br(),

    # Prediction Button
    dbc.Button("Predict Overall Score", id="predict-btn", color="primary", className="mt-2"),
    html.Br(),
    html.Br(),

    # Output area
    html.H4("Predicted Overall Score:"),
    html.Div(id="prediction-output", style={"fontSize": "24px", "fontWeight": "bold"}),
], style={
    "backgroundColor": "#FFD580",  # Dark background
    "height": "100vh",             # Full page height
    "width":"200vh",
    "padding": "20px"
})

# Define callback for prediction
@app.callback(
    Output("prediction-output", "children"),
    Input("predict-btn", "n_clicks"),
    [State(f"input-{i}", "value") for i in range(10)]
)
def predict_score(n_clicks, *inputs):
    if n_clicks is None:
        return "Enter values and click Predict."
    try:
        # Prepare input data for the model
        input_data = np.array(inputs).reshape(1, -1)
        prediction = model.predict(input_data)[0]
        return f"{prediction[0]:.2f}"
    except Exception as e:
        return f"Error: {e}"

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
