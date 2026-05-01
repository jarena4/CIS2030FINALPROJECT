from dash import Dash, html, dcc, Input, Output
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from scipy import stats

app = Dash(__name__, suppress_callback_exceptions=True)

# Data
df = pd.read_csv("dashboard_eda_data.csv")

GENRE_COL = "genre" if "genre" in df.columns else "track_genre"

FEATURES = [
    "danceability", "energy", "valence", "acousticness",
    "tempo", "loudness", "speechiness", "instrumentalness"
]
FEATURES = [c for c in FEATURES if c in df.columns]

# ensure numeric
for col in FEATURES:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# clean
eda_df = df.dropna(subset=[GENRE_COL] + FEATURES).copy()

# try t-test file 
try:
    ttest_result = pd.read_csv("dashboard_ttest_result.csv")
except FileNotFoundError:
    classical = eda_df[eda_df[GENRE_COL] == "Classical"]["acousticness"]
    electronic = eda_df[eda_df[GENRE_COL] == "Electronic"]["acousticness"]
    t_stat, p_value = stats.ttest_ind(classical, electronic, nan_policy="omit")
    ttest_result = pd.DataFrame([{
        "classical_avg_acousticness": classical.mean(),
        "electronic_avg_acousticness": electronic.mean(),
        "t_stat": t_stat,
        "p_value": p_value
    }])

# from proj
cm_data = [
    [181,87,57,3,115,81,87,56,82,150],
    [75,413,43,118,73,75,4,48,6,45],
    [57,44,325,11,121,54,23,155,12,92],
    [19,27,15,776,1,11,2,39,0,10],
    [40,31,82,4,503,11,41,49,36,100],
    [42,39,43,7,22,533,37,105,41,24],
    [42,7,12,1,21,29,346,9,424,13],
    [18,32,123,82,33,112,44,428,13,19],
    [76,22,10,0,32,28,506,15,188,24],
    [189,80,142,8,224,45,29,51,37,107]
]

genres = ['Alternative','Anime','Blues','Classical','Country','Electronic','Hip-Hop','Jazz','Rap','Rock']

model_results = pd.DataFrame([
    {'model': 'Logistic Regression', 'accuracy': 0.3831, 'f1': 0.3718},
    {'model': 'Random Forest',       'accuracy': 0.4160, 'f1': 0.4082},
    {'model': 'KMeans + RF',         'accuracy': 0.4174, 'f1': 0.4091},
    {'model': 'Tuned RF',            'accuracy': 0.4220, 'f1': 0.4116},
])

# style
ACCENT = '#4f63c6'
PAGE_BG = '#f5f5f5'
BORDER = '#e0e0e0'
TEXT = '#333'
MUTED = '#777'

HEADER_STYLE = {
    'background': ACCENT,
    'color': 'white',
    'padding': '18px 32px',
    'fontSize': '1.4rem',
    'fontWeight': 'bold',
    'margin': '0',
}

SUBHEADER_STYLE = {
    'background': 'white',
    'color': MUTED,
    'padding': '10px 32px',
    'fontSize': '0.9rem',
    'borderBottom': f'1px solid {BORDER}',
}

CARD_STYLE = {
    'background': 'white',
    'border': f'1px solid {BORDER}',
    'borderRadius': '6px',
    'padding': '20px',
    'marginBottom': '20px',
}

TAB_STYLE = {
    'padding': '12px 20px',
    'fontWeight': '600',
    'fontSize': '0.88rem',
    'background': 'white',
    'border': f'1px solid {BORDER}',
    'color': TEXT,
}

SELECTED_TAB_STYLE = {
    'padding': '12px 20px',
    'fontWeight': '600',
    'fontSize': '0.88rem',
    'background': '#f0f2ff',
    'borderTop': f'3px solid {ACCENT}',
    'color': ACCENT,
}

INSIGHT_STYLE = {
    'background': '#f8f8f8',
    'borderLeft': f'4px solid {ACCENT}',
    'padding': '11px 15px',
    'marginTop': '14px',
    'fontSize': '0.82rem',
    'lineHeight': '1.6',
    'color': '#444',
    'borderRadius': '0 4px 4px 0',
}

# figure helpers
def style_fig(fig):
    fig.update_layout(
        paper_bgcolor='white',
        plot_bgcolor='white',
        font=dict(family='Arial, sans-serif', color=TEXT),
        margin=dict(t=45, b=60, l=60, r=30),
    )
    return fig


def make_genre_distribution():
    counts = eda_df[GENRE_COL].value_counts().reset_index()
    counts.columns = ['genre', 'count']
    fig = px.bar(counts, x='genre', y='count', title='Distribution of Music Genres')
    fig.update_traces(marker_color=ACCENT)
    fig.update_layout(xaxis_title='Genre', yaxis_title='Number of Tracks', xaxis_tickangle=-45)
    return style_fig(fig)


def make_correlation_heatmap():
    temp = eda_df[FEATURES + [GENRE_COL]].copy()
    temp['genre_encoded'] = temp[GENRE_COL].astype('category').cat.codes
    corr = temp[FEATURES + ['genre_encoded']].corr()
    fig = px.imshow(
        corr,
        text_auto='.2f',
        color_continuous_scale='RdBu_r',
        zmin=-1,
        zmax=1,
        title='Correlation Between Audio Features and Genre'
    )
    fig.update_layout(xaxis_tickangle=-45)
    return style_fig(fig)


def make_outlier_boxplot(feature):
    q1 = eda_df[feature].quantile(0.25)
    q3 = eda_df[feature].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    outlier_pct = (((eda_df[feature] < lower) | (eda_df[feature] > upper)).mean()) * 100

    fig = px.box(eda_df, y=feature, points='outliers', title=f'Outlier Detection: {feature.capitalize()}')
    fig.update_traces(marker_color=ACCENT, line_color=ACCENT)
    fig.update_layout(yaxis_title=feature.capitalize(), xaxis_title='')
    return style_fig(fig), f'{outlier_pct:.1f}% of tracks are outliers for {feature} using the 1.5×IQR rule.'


def make_sql_result_table(sort_col):
    sql_result = (
        eda_df.groupby(GENRE_COL)[['acousticness', 'energy', 'danceability', 'speechiness', 'tempo']]
        .mean()
        .reset_index()
        .rename(columns={GENRE_COL: 'genre'})
    )

    for c in ['acousticness', 'energy', 'danceability', 'speechiness']:
        sql_result[c] = sql_result[c].round(3)
    sql_result['tempo'] = sql_result['tempo'].round(1)

    sql_result = sql_result.sort_values(sort_col, ascending=False)

    rows = []
    for _, row in sql_result.iterrows():
        rows.append(html.Tr([
            html.Td(html.Strong(row['genre']), style={'padding':'9px 12px','borderBottom':'1px solid #eee'}),
            html.Td(f"{row['acousticness']:.3f}", style={'padding':'9px 12px','borderBottom':'1px solid #eee'}),
            html.Td(f"{row['energy']:.3f}", style={'padding':'9px 12px','borderBottom':'1px solid #eee'}),
            html.Td(f"{row['danceability']:.3f}", style={'padding':'9px 12px','borderBottom':'1px solid #eee'}),
            html.Td(f"{row['speechiness']:.3f}", style={'padding':'9px 12px','borderBottom':'1px solid #eee'}),
            html.Td(f"{row['tempo']:.1f}", style={'padding':'9px 12px','borderBottom':'1px solid #eee'}),
        ]))

    return html.Table([
        html.Thead(html.Tr([
            html.Th(c, style={'padding':'9px 12px','background':'#f5f5f5',
                    'borderBottom':'2px solid #e0e0e0','color':'#555',
                    'fontSize':'0.75rem','textTransform':'uppercase'})
            for c in ['Genre','Acousticness','Energy','Danceability','Speechiness','Tempo']
        ])),
        html.Tbody(rows),
    ], style={'width':'100%','borderCollapse':'collapse','fontSize':'0.82rem'})


def make_feature_by_genre(feature):
    sample_df = eda_df.sample(min(2000, len(eda_df)), random_state=42)
    order = eda_df[GENRE_COL].value_counts().index.tolist()
    fig = px.strip(
        sample_df,
        x=GENRE_COL,
        y=feature,
        category_orders={GENRE_COL: order},
        title=f'{feature.capitalize()} by Genre'
    )
    fig.update_traces(marker=dict(size=4, opacity=0.45, color=ACCENT))
    fig.update_layout(xaxis_title='Genre', yaxis_title=feature.capitalize(), xaxis_tickangle=-45)
    return style_fig(fig)


def make_ttest_card():
    row = ttest_result.iloc[0]
    p = row['p_value']
    conclusion = 'statistically significant' if p < 0.05 else 'not statistically significant'

    return html.Div([
        html.Div([
            html.Div([
                html.Div(f"{row['classical_avg_acousticness']:.3f}", style={'fontSize':'1.6rem','fontWeight':'bold','color':ACCENT}),
                html.Div('Classical avg acousticness', style={'fontSize':'0.78rem','color':MUTED}),
            ], style=CARD_STYLE),
            html.Div([
                html.Div(f"{row['electronic_avg_acousticness']:.3f}", style={'fontSize':'1.6rem','fontWeight':'bold','color':ACCENT}),
                html.Div('Electronic avg acousticness', style={'fontSize':'0.78rem','color':MUTED}),
            ], style=CARD_STYLE),
            html.Div([
                html.Div(f"{row['t_stat']:.3f}", style={'fontSize':'1.6rem','fontWeight':'bold','color':ACCENT}),
                html.Div('T-statistic', style={'fontSize':'0.78rem','color':MUTED}),
            ], style=CARD_STYLE),
            html.Div([
                html.Div(f"{row['p_value']:.6f}", style={'fontSize':'1.6rem','fontWeight':'bold','color':ACCENT}),
                html.Div('P-value', style={'fontSize':'0.78rem','color':MUTED}),
            ], style=CARD_STYLE),
        ], style={'display':'grid','gridTemplateColumns':'repeat(4, 1fr)','gap':'12px'}),
        html.Div([
            'The t-test compares acousticness for Classical and Electronic tracks. The result is ',
            conclusion,
            ', supporting the EDA finding that Classical tracks are much more acoustic than Electronic tracks.'
        ], style=INSIGHT_STYLE)
    ])


def make_model_chart():
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=model_results['model'], y=model_results['accuracy'],
        mode='lines+markers', name='Accuracy',
        line=dict(color=ACCENT, width=2), marker=dict(size=9),
    ))
    fig.add_trace(go.Scatter(
        x=model_results['model'], y=model_results['f1'],
        mode='lines+markers', name='Weighted F1',
        line=dict(color='#d65f3a', width=2, dash='dot'), marker=dict(size=7),
    ))
    fig.update_layout(
        paper_bgcolor='white', plot_bgcolor='white',
        margin=dict(t=10,b=60,l=60,r=20),
        yaxis=dict(range=[0.35,0.45], title='Score', gridcolor='#eee'),
        xaxis=dict(gridcolor='#eee'),
        legend=dict(orientation='h', y=1.1),
    )
    return fig

# layout
app.layout = html.Div([
    html.H1('CIS 2450: Big Data Analytics', style=HEADER_STYLE),
    html.Div('Music Genre Classification Dashboard', style=SUBHEADER_STYLE),

    html.Div([
        html.Div([html.Div('50,000', style={'fontSize':'1.8rem','fontWeight':'bold','color': ACCENT}), html.Div('Total Tracks', style={'fontSize':'0.75rem','color':'#888'})], style={**CARD_STYLE, 'textAlign':'center','flex':'1','minWidth':'120px'}),
        html.Div([html.Div(str(eda_df[GENRE_COL].nunique()), style={'fontSize':'1.8rem','fontWeight':'bold','color': ACCENT}), html.Div('Genres', style={'fontSize':'0.75rem','color':'#888'})], style={**CARD_STYLE, 'textAlign':'center','flex':'1','minWidth':'120px'}),
        html.Div([html.Div('2', style={'fontSize':'1.8rem','fontWeight':'bold','color': ACCENT}), html.Div('Data Sources', style={'fontSize':'0.75rem','color':'#888'})], style={**CARD_STYLE, 'textAlign':'center','flex':'1','minWidth':'120px'}),
        html.Div([html.Div(str(len(FEATURES)), style={'fontSize':'1.8rem','fontWeight':'bold','color': ACCENT}), html.Div('Audio Features',style={'fontSize':'0.75rem','color':'#888'})], style={**CARD_STYLE, 'textAlign':'center','flex':'1','minWidth':'120px'}),
        html.Div([html.Div('42.2%', style={'fontSize':'1.8rem','fontWeight':'bold','color': ACCENT}), html.Div('Best Accuracy',style={'fontSize':'0.75rem','color':'#888'})], style={**CARD_STYLE, 'textAlign':'center','flex':'1','minWidth':'120px'}),
        html.Div([html.Div('4', style={'fontSize':'1.8rem','fontWeight':'bold','color': ACCENT}), html.Div('Models Trained',style={'fontSize':'0.75rem','color':'#888'})], style={**CARD_STYLE, 'textAlign':'center','flex':'1','minWidth':'120px'}),
    ], style={'display':'flex','gap':'16px','padding':'20px 32px','flexWrap':'wrap'}),

    html.Div([
        dcc.Tabs(id='tabs', value='eda', children=[
            dcc.Tab(label='EDAs', value='eda', style=TAB_STYLE, selected_style=SELECTED_TAB_STYLE),
            dcc.Tab(label='Model Comparison', value='models', style=TAB_STYLE, selected_style=SELECTED_TAB_STYLE),
            dcc.Tab(label='Confusion Matrix', value='confusion', style=TAB_STYLE, selected_style=SELECTED_TAB_STYLE),
        ]),
        html.Div(id='tab-content', style={'padding': '24px 32px'}),
    ]),

    html.Footer('CIS 2450: Music Genre Classification Final Project',
                style={'textAlign':'center','padding':'20px','fontSize':'0.73rem','color':'#aaa','borderTop':'1px solid #e0e0e0'}),
], style={'fontFamily': 'Arial, sans-serif', 'background': PAGE_BG, 'minHeight': '100vh'})

# tab content
@app.callback(Output('tab-content', 'children'), Input('tabs', 'value'))
def render_tab(tab):
    if tab == 'eda':
        return html.Div([
            html.Div([
                html.H3('EDA 1: Distribution of Music Genres'),
                dcc.Graph(figure=make_genre_distribution()),
                html.Div('The dataset is evenly balanced across the selected music genres, which helps prevent the classifier from being dominated by one genre.', style=INSIGHT_STYLE),
            ], style=CARD_STYLE),

            html.Div([
                html.H3('EDA 2: Correlation Between Audio Features and Genre'),
                dcc.Graph(figure=make_correlation_heatmap()),
                html.Div('The correlation heatmap shows strong relationships between several audio features, especially energy/loudness and acousticness/energy.', style=INSIGHT_STYLE),
            ], style=CARD_STYLE),

            html.Div([
                html.H3('EDA 3: Audio Features vs Genre'),
                html.Label('Select feature:', style={'fontWeight':'600','fontSize':'0.85rem','marginRight':'10px'}),
                dcc.Dropdown(
                    id='genre-feature',
                    options=[{'label': f.capitalize(), 'value': f} for f in FEATURES],
                    value=FEATURES[0],
                    clearable=False,
                    style={'width':'260px','marginBottom':'14px'},
                ),
                dcc.Graph(id='genre-feature-chart'),
                html.Div('This plot shows how each audio feature varies across genres using a sampled strip plot', style=INSIGHT_STYLE),
            ], style=CARD_STYLE),

            html.Div([
                html.H3('EDA 4: Outlier Detection Across Audio Features'),
                html.Label('Select feature:', style={'fontWeight':'600','fontSize':'0.85rem','marginRight':'10px'}),
                dcc.Dropdown(
                    id='outlier-feature',
                    options=[{'label': f.capitalize(), 'value': f} for f in FEATURES],
                    value=FEATURES[0],
                    clearable=False,
                    style={'width':'260px','marginBottom':'14px'},
                ),
                dcc.Graph(id='outlier-chart'),
                html.Div(id='outlier-insight', style=INSIGHT_STYLE),
            ], style=CARD_STYLE),

            html.Div([
                html.H3('EDA 5: SQL Audio Feature Averages by Genre'),
                html.Pre(
                    'SELECT genre, AVG(acousticness), AVG(energy), AVG(danceability),\n'
                    '       AVG(speechiness), AVG(tempo)\n'
                    'FROM tracks GROUP BY genre ORDER BY avg_acousticness DESC',
                    style={'background':'#f8f8f8','padding':'10px 14px','borderRadius':'6px',
                           'fontSize':'0.75rem','color':'#555','marginBottom':'12px'}
                ),
                html.Div([
                html.Label('Sort by:', style={'fontWeight':'600','fontSize':'0.85rem','marginRight':'10px'}),
                dcc.Dropdown(
                    id='sql-sort',
                    options=[{'label': c.capitalize(), 'value': c} for c in ['acousticness','energy','danceability','speechiness','tempo']],
                    value='acousticness',
                    clearable=False,
                    style={'width': '200px'},
                ),
            ], style={'display':'flex','alignItems':'center','marginBottom':'16px'}),
            html.Div(id='sql-table'),
            ], style=CARD_STYLE),

            html.Div([
                html.H3('EDA 6: Classical vs Electronic Acousticness T-Test'),
                make_ttest_card(),
            ], style=CARD_STYLE),
        ])

    if tab == 'models':
        return html.Div([
            dcc.Graph(id='model-chart', figure=make_model_chart()),
            html.Div([
                'Each model improved on the previous. The ~42% ceiling suggests audio features alone are insufficient '
                'to perfectly separate genres like Alternative, Rock, Rap, and Hip-Hop.'
            ], style=INSIGHT_STYLE),
        ], style=CARD_STYLE)

    if tab == 'confusion':
        return html.Div([
            html.Div([
                html.Label('View:', style={'fontWeight':'600','fontSize':'0.85rem','marginRight':'10px'}),
                dcc.Dropdown(
                    id='cm-mode',
                    options=[
                        {'label': 'Normalized (% of true class)', 'value': 'normalized'},
                        {'label': 'Raw counts', 'value': 'raw'},
                    ],
                    value='normalized',
                    clearable=False,
                    style={'width': '260px'},
                ),
            ], style={'display':'flex','alignItems':'center','marginBottom':'16px'}),
            dcc.Graph(id='cm-chart'),
            html.Div([
                'Classical', ' is the easiest genre to predict. ',
                'Hip-Hop & Rap', ' is the hardest pair because their audio profiles overlap strongly.'
            ], style=INSIGHT_STYLE),
        ], style=CARD_STYLE)
    
# call backs
@app.callback(
    Output('outlier-chart', 'figure'),
    Output('outlier-insight', 'children'),
    Input('outlier-feature', 'value'),
)
def update_outlier(feature):
    return make_outlier_boxplot(feature)


@app.callback(Output('genre-feature-chart', 'figure'), Input('genre-feature', 'value'))
def update_genre_feature(feature):
    return make_feature_by_genre(feature)


@app.callback(Output('cm-chart', 'figure'), Input('cm-mode', 'value'))
def update_cm(mode):
    if mode == 'normalized':
        z = [[round(v / sum(row), 3) for v in row] for row in cm_data]
    else:
        z = cm_data
    fig = go.Figure(go.Heatmap(
        z=z, x=genres, y=genres,
        colorscale='Blues', showscale=True,
        text=cm_data,
        hovertemplate='True: <b>%{y}</b><br>Predicted: <b>%{x}</b><br>Count: %{text}<extra></extra>',
    ))
    fig.update_layout(paper_bgcolor='white', plot_bgcolor='white',
                      margin=dict(t=10,b=100,l=100,r=60),
                      xaxis=dict(tickangle=-40, title='Predicted'),
                      yaxis=dict(title='True Label', autorange='reversed'))
    return fig


@app.callback(Output('sql-table', 'children'), Input('sql-sort', 'value'))
def update_sql_table(sort_col):
    return make_sql_result_table(sort_col)


if __name__ == '__main__':
    app.run(debug=True, port=8050)