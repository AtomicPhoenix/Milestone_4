import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.express as px
import pandas as pd
import base64
import io

from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import r2_score

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# at end of code modify and run with 
# app.run_server(debug=True) instead of False
# if run locally but we needed False for streamlit 

app.layout = html.Div(
    className='main-container',
    children=[

        dcc.Store(id='stored-data', storage_type='memory'),
        dcc.Store(id='trained-pipeline', storage_type='memory'),
        dcc.Store(id='feature-list', storage_type='memory'),

        # Upload File Container
        html.Div(
            className='upload-container',
            children=[
                dcc.Upload(
                    id='upload-data',
                    children=html.Div("Upload File", className='upload-text'),
                    multiple=False
                )
            ]
        ),

        # Select Target Container
        html.Div(
            className='select-target-container',
            children=[
                html.Label("Select Target:", className='target-label'),
                dcc.Dropdown(
                    id='target-dropdown',
                    options=[],
                    placeholder="Choose target var",
                    clearable=True,
                    className='target-dropdown'
                ),
            ]
        ),

        # Bar Charts Container (side by side)
        html.Div(
            className='charts-container',
            children=[
                html.Div(
                    className='chart-section',
                    children=[
                        dcc.RadioItems(
                            id='categorical-radio',
                            options=[],
                            labelStyle={'display': 'inline-block', 'margin-right': '10px'},
                            className='radio-items'
                        ),
                        dcc.Graph(
                            id='bar-chart-categorical',
                            style={'width': '100%', 'height': '100%'}
                        )
                    ]
                ),
                html.Div(
                    className='chart-section',
                    children=[
                        dcc.Graph(
                            id='bar-chart-correlation',
                            style={'width': '100%', 'height': '100%'}
                        )
                    ]
                )
            ]
        ),

        # Train Container
        html.Div(
            className='train-container',
            children=[
                html.Div(id='feature-checkboxes', className='checkbox-container'),
                html.Button("Train", id='train-button', className='train-button'),
            ]
        ),

        # Predict Container
        html.Div(
            className='predict-container',
            children=[
                html.Div("The R2 score is: N/A", id='r2-score-text', className='r2-score-display'),
                html.Div(
                    className='predict-row',
                    children=[
                        dcc.Input(
                            id='predict-input',
                            type='text',
                            placeholder="Features comma-separated",
                            className='predict-input'
                        ),
                        html.Button("Predict", id='predict-button', className='predict-button'),
                        html.Span("Predicted tip is:", className='predict-label'),
                        html.Span("", id='predict-output-value', className='prediction-result')
                    ]
                ),
            ]
        )
    ]
)

# A. Parse uploaded data and store
@app.callback(
    Output('stored-data', 'data'),
    Output('feature-list', 'data'),
    Input('upload-data', 'contents'),
    State('upload-data', 'filename')
)
def store_uploaded_data(contents, filename):
    if contents is None:
        return dash.no_update, dash.no_update

    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    # UTF-8 fallback Latin-1
    try:
        decoded_string = decoded.decode('utf-8')
    except UnicodeDecodeError:
        decoded_string = decoded.decode('latin-1')

    df = pd.read_csv(io.StringIO(decoded_string))

    numeric_cols = df.select_dtypes(include=['float64', 'int64', 'float32', 'int32']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

    data_store = {
        'data': df.to_dict('records'),
        'numeric_cols': numeric_cols,
        'categorical_cols': categorical_cols
    }
    feature_list = df.columns.tolist()
    return data_store, feature_list

#Populate target dropdown
@app.callback(
    Output('target-dropdown', 'options'),
    Input('stored-data', 'data')
)
def populate_target_options(stored_data):
    if not stored_data:
        return []
    numeric_cols = stored_data['numeric_cols']
    return [{'label': col, 'value': col} for col in numeric_cols]

#Populate / default radio for categorical
@app.callback(
    Output('categorical-radio', 'options'),
    Output('categorical-radio', 'value'),
    Input('stored-data', 'data')
)
def populate_categorical_radio(stored_data):
    if not stored_data:
        return [], None
    cat_cols = stored_data['categorical_cols']
    options = [{'label': c, 'value': c} for c in cat_cols]
    default_val = cat_cols[0] if cat_cols else None
    return options, default_val

#Bar chart - average target by categorical
@app.callback(
    Output('bar-chart-categorical', 'figure'),
    Input('categorical-radio', 'value'),
    Input('target-dropdown', 'value'),
    Input('stored-data', 'data')
)
def update_bar_chart_categorical(cat_col, target_col, stored_data):
    if not stored_data:
        return px.bar(title="No data loaded")
    if cat_col is None or target_col is None:
        return px.bar(title="Awaiting selection")
    df = pd.DataFrame(stored_data['data'])
    if cat_col not in df.columns or target_col not in df.columns:
        return px.bar(title="Invalid columns")

    avg_df = df.groupby(cat_col)[target_col].mean().reset_index()
    fig = px.bar(avg_df, x=cat_col, y=target_col, title=f"Average {target_col} by {cat_col}")

    # Example styling
    fig.update_traces(
        texttemplate='%{y:.2f}',
        textposition='inside',
        textfont=dict(color='grey'),
        marker=dict(
            color='rgba(203,223,235,255)',
            line=dict(color='rgb(188,188,191)', width=2)
        )
    )
    fig.update_layout(
        autosize=True,
        margin=dict(t=50)
    )
    return fig

# Correlation bar chart now depends on the selected features. ***
@app.callback(
    Output('bar-chart-correlation', 'figure'),
    Input('target-dropdown', 'value'),
    Input({'type': 'feature-checklist', 'index': dash.ALL}, 'value'),  # Listen to selected features
    State('stored-data', 'data')
)
def update_bar_chart_correlation(target_col, all_feature_values, stored_data):
    """Plot correlation only for the numeric features the user selected (excluding target)."""
    if not stored_data:
        return px.bar(title="No data loaded")
    if not target_col:
        return px.bar(title="Awaiting target")

    # Flatten the selected features
    selected_features = []
    for fv in all_feature_values:
        if fv:
            selected_features.extend(fv)
    selected_features = list(set(selected_features))

    df = pd.DataFrame(stored_data['data'])
    if target_col not in df.columns:
        return px.bar(title="Invalid target")

    # Filter only numeric columns that are selected
    numeric_cols = df.select_dtypes(include=['int64','float64','int32','float32']).columns
    numeric_selected = [col for col in selected_features if col in numeric_cols]

    if len(numeric_selected) == 0:
        return px.bar(title="No numeric features selected")

    corr_vals = []
    for col in numeric_selected:
        # skip target if included in numeric_selected
        if col == target_col:
            continue
        corr = df[col].corr(df[target_col])
        corr_vals.append({'column': col, 'corr_value': abs(corr)})

    if len(corr_vals) == 0:
        return px.bar(title="No numeric features remain after excluding target")

    corr_df = pd.DataFrame(corr_vals)
    
    # Dynamically set the title to reflect the selected target
    fig = px.bar(
        corr_df, 
        x='column', 
        y='corr_value', 
        title=f"Correlation of Selected Numeric Features with {target_col}"
    )

    fig.update_traces(
        texttemplate='%{y:.2f}',
        textposition='inside',
        textfont=dict(color='white'),
        marker=dict(color='rgba(101,110,242,255)')
    )
    fig.update_layout(
        autosize=True,
        margin=dict(t=50),
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=True, range=[0, corr_df['corr_value'].max() * 1.2])
    )
    return fig


#Generate feature checkboxes (inline, smaller font)
@app.callback(
    Output('feature-checkboxes', 'children'),
    Input('feature-list', 'data'),
    Input('target-dropdown', 'value')
)
def generate_feature_checkboxes(feature_list, target_col):
    if not feature_list:
        return []

    checkboxes = []
    for feat in feature_list:
        if feat == target_col:
            continue
        checkboxes.append(
            dbc.Checklist(
                options=[{'label': feat, 'value': feat}],
                value=[feat],  # default selected
                id={'type': 'feature-checklist', 'index': feat},
                inline=True,
                style={'fontSize': '12px', 'margin-right': '10px'}
            )
        )
    return checkboxes

# G. Train callback
@app.callback(
    Output('trained-pipeline', 'data'),
    Output('r2-score-text', 'children'),
    Input('train-button', 'n_clicks'),
    State('stored-data', 'data'),
    State({'type': 'feature-checklist', 'index': dash.ALL}, 'value'),
    State('target-dropdown', 'value')
)
def train_model(n_clicks, stored_data, all_feature_values, target_col):
    if not n_clicks or not stored_data or not target_col:
        return dash.no_update, "The R2 score is: N/A"
    
    df = pd.DataFrame(stored_data['data'])
    
    # Flatten the selected features from the dynamic checklists
    selected_features = []
    for fv in all_feature_values:
        if fv:
            selected_features.extend(fv)
    selected_features = list(set(selected_features))
    
    if len(selected_features) == 0:
        return dash.no_update, "The R2 score is: N/A (No features selected)"
    
    # Drop rows where target is missing
    df = df.dropna(subset=[target_col])  # IMPORTANT: remove rows w/ NaN in target
    
    X = df[selected_features].copy()
    y = df[target_col].copy()
    
    # If the target col is still empty after dropping NaN, return
    if y.empty:
        return dash.no_update, "The R2 score is: N/A (No valid rows after dropping NaN target)"
    
    numeric_feats = X.select_dtypes(include=['int64','float64','int32','float32']).columns
    cat_feats    = X.select_dtypes(include=['object','category']).columns
    
    transformers = []
    if len(numeric_feats) > 0:
        transformers.append(('num', SimpleImputer(strategy='mean'), numeric_feats))
    if len(cat_feats) > 0:
        transformers.append(
            ('cat',
             Pipeline([
                 ('imputer', SimpleImputer(strategy='most_frequent')),
                 ('ohe', OneHotEncoder(handle_unknown='ignore'))
             ]),
             cat_feats)
        )
    preprocessor = ColumnTransformer(transformers=transformers, remainder='drop')
    
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', RandomForestRegressor(n_estimators=50, random_state=42))
    ])
    
    pipeline.fit(X, y)
    preds = pipeline.predict(X)
    r2 = r2_score(y, preds)

    pipeline_data = {'trained': True, 'r2': r2}
    return pipeline_data, f"The R2 score is: {r2:.2f}"


# H. Predict callback
@app.callback(
    Output('predict-output-value', 'children'),
    Input('predict-button', 'n_clicks'),
    State('predict-input', 'value'),
    State('trained-pipeline', 'data'),
    State({'type': 'feature-checklist', 'index': dash.ALL}, 'value'),
    State('target-dropdown', 'value'),
    State('stored-data', 'data')
)
def predict_value(n_clicks, input_string, pipeline_data, all_feature_values, target_col, stored_data):
    if not n_clicks or not pipeline_data or not input_string:
        return ""

    # Recollect selected features
    selected_features = []
    for fv in all_feature_values:
        if fv:
            selected_features.extend(fv)
    selected_features = list(set(selected_features))

    df = pd.DataFrame(stored_data['data'])

    # Drop rows with missing target to avoid ValueError
    df = df.dropna(subset=[target_col])

    X = df[selected_features].copy()
    y = df[target_col].copy()

    numeric_feats = X.select_dtypes(include=['int64','float64','int32','float32']).columns
    cat_feats = X.select_dtypes(include=['object','category']).columns

    preprocessor = ColumnTransformer(transformers=[
        ('num', SimpleImputer(strategy='mean'), numeric_feats),
        ('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('ohe', OneHotEncoder(handle_unknown='ignore'))
        ]), cat_feats)
    ], remainder='drop')

    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', RandomForestRegressor(n_estimators=50, random_state=42))
    ])
    # Fit the pipeline on the filtered data
    pipeline.fit(X, y)

    user_values = [v.strip() for v in input_string.split(',')]
    if len(user_values) != len(selected_features):
        return "Error: Mismatch in # input values vs. selected features."

    input_dict = {}
    for feat, val in zip(selected_features, user_values):
        try:
            val_converted = float(val)
            input_dict[feat] = [val_converted]
        except ValueError:
            input_dict[feat] = [val]

    input_df = pd.DataFrame(input_dict)
    pred = pipeline.predict(input_df)[0]
    return f"{pred:.2f}"


if __name__ == '__main__':
    app.run_server(debug=False)

# change to true if run locally
