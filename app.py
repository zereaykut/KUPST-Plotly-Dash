import dash
from dash import dcc, html, callback, Output, Input, State
import dash_bootstrap_components as dbc

import os
import requests
import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta

import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
pio.templates.default = "plotly_white"

wdir = os.getcwd()

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.SPACELAB])
app.title = "KUPST Dashboard"

main_url = "https://seffaflik.epias.com.tr/electricity-service"


def control_date(start_date: date, end_date: date):
    delta = end_date - start_date
    if start_date > end_date:
        return "Start Date can't be greater than end date"
    elif delta.days >= 30:
        return f"Date range can't be greater than 30 days. Used date range is {delta.days} days."
    else:
        return 200


def get_org_info(start_date: date, end_date: date):
    response = requests.post(
        f"{main_url}/v1/generation/data/organization-list",
        json={
            "startDate": f"{start_date}T00:00:00+03:00",
            "endDate": f"{end_date}T23:00:00+03:00",
            "region": "TR1",
        },
    )
    if response.status_code == 200:
        data = response.json()
        data = data.get("items", None)
        if data is None:
            data = "response is None or Empty"
        else:
            data = pd.DataFrame(data)
    else:
        data = "response error"
    return data


def get_uevcb_info(start_date: date, org_id: int):
    response = requests.post(
        f"{main_url}/v1/generation/data/uevcb-list",
        json={
            "startDate": f"{start_date}T00:00:00+03:00",
            "organizationId": org_id,
            "region": "TR1",
        },
    )
    if response.status_code == 200:
        data = response.json()
        data = data.get("items", None)
        if data is None:
            data = "response is None or Empty"
        else:
            data = pd.DataFrame(data)
    else:
        data = "response error"
    return data


def get_grt_info():
    response = requests.get(f"{main_url}/v1/generation/data/powerplant-list")
    if response.status_code == 200:
        data = response.json()
        data = data.get("items", None)
        if data is None:
            data = "response is None or Empty"
        else:
            data = pd.DataFrame(data)
    else:
        data = "response error"
    return data


def get_kudup(uevcb_id: int, org_id: int, start_date: date, end_date: date):
    response = requests.post(
        f"{main_url}/v1/generation/data/sbfgp",
        json={
            "startDate": f"{start_date}T00:00:00+03:00",
            "endDate": f"{end_date}T23:00:00+03:00",
            "organizationId": org_id,
            "uevcbId": uevcb_id,
            "region": "TR1",
        },
    )
    if response.status_code == 200:
        data = response.json()
        data = data.get("items", None)
        if data is None:
            data = "response is None or Empty"
        else:
            data = pd.DataFrame(data)
            data["date"] = pd.to_datetime(data["date"].str.split("+", expand=True)[0])
            data = data[["date", "toplam"]].rename(columns={"toplam": "kudup"})
    else:
        data = "response error"
    return data


def get_grt(grt_id: int, start_date: date, end_date: date):
    response = requests.post(
        f"{main_url}/v1/generation/data/realtime-generation",
        json={
            "startDate": f"{start_date}T00:00:00+03:00",
            "endDate": f"{end_date}T23:00:00+03:00",
            "powerPlantId": grt_id,
            "region": "TR1",
        },
    )
    if response.status_code == 200:
        data = response.json()
        data = data.get("items", None)
        if data is None:
            data = f"get_grt() response is None or Empty {datetime.now()}\n"
            with open(f"{wdir}/data/get_grt_error.txt", "w") as f:
                f.write(data)
        else:
            data = pd.DataFrame(data)
            data["date"] = pd.to_datetime(data["date"].str.split("+", expand=True)[0])
            data = data[["date", "total"]].rename(columns={"total": "grt"})
    else:
        data = f"get_grt() response error {datetime.now()}\n"
        with open(f"{wdir}/data/error.txt", "a") as f:
            f.write(data)
    return data

def get_mcp(start_date: date, end_date: date):
    response = requests.post(
        f"{main_url}/v1/markets/dam/data/mcp",
        json={
            "startDate": f"{start_date}T00:00:00+03:00",
            "endDate": f"{end_date}T23:00:00+03:00",
            "region": "TR1",
        },
    )
    if response.status_code == 200:
        data = response.json()
        data = data.get("items", None)
        if data is None:
            data = "response is None or Empty"
        else:
            data = pd.DataFrame(data)
            data["date"] = pd.to_datetime(data["date"].str.split("+", expand=True)[0])
            data = data[["date", "price"]].rename(columns={"price": "mcp"})
    else:
        data = "response error"
    return data


def get_smp(start_date: date, end_date: date):
    response = requests.post(
        f"{main_url}/v1/markets/bpm/data/system-marginal-price",
        json={
            "startDate": f"{start_date}T00:00:00+03:00",
            "endDate": f"{end_date}T23:00:00+03:00",
            "region": "TR1",
        },
    )
    if response.status_code == 200:
        data = response.json()
        data = data.get("items", None)
        if data is None:
            data = "response is None or Empty"
        else:
            data = pd.DataFrame(data)
            data["date"] = pd.to_datetime(data["date"].str.split("+", expand=True)[0])
            data = data[["date", "systemMarginalPrice"]].rename(
                columns={"systemMarginalPrice": "smp"}
            )
    else:
        data = "response error"
    return data


def kupst(df, tolerance_coefficient: float):
    df["tolerance_coefficient"] = tolerance_coefficient

    df["imbalance_amount"] = df["grt"] - df["kudup"]
    # Tolerance Amount
    df["tolerance_amount"] = df["kudup"] * df["tolerance_coefficient"]
    # Tolerance Exceed Imbalance Amount
    df["tolerance_exceed_imbalance_amount"] = df[
        (abs(df["imbalance_amount"]) > df["tolerance_amount"])
    ]["imbalance_amount"]

    # Positive Imbalance Amount
    df["positive_imbalance_amount"] = df[df["imbalance_amount"] >= 0][
        "imbalance_amount"
    ]
    df["negative_imbalance_amount"] = df[df["imbalance_amount"] < 0]["imbalance_amount"]

    # Positive Imbalance Payment
    df["positive_imbalance_payment"] = (
        df["positive_imbalance_amount"] * df[["mcp", "smp"]].min(axis=1) * 0.97
    )
    df["negative_imbalance_payment"] = (
        df["negative_imbalance_amount"] * df[["mcp", "smp"]].max(axis=1) * 1.03
    )
    # Imbalance Payment
    df["imbalance_payment"] = df[
        ["positive_imbalance_payment", "negative_imbalance_payment"]
    ].sum(axis=1)

    # Imbalance Cost
    df["imbalance_cost"] = df["imbalance_amount"] * df["mcp"] - df["imbalance_payment"]
    # .unit Imbalance Cost
    df["unit_imbalance_cost"] = df["imbalance_cost"] / df["grt"].replace(0, np.nan)

    ## KUPST
    df["mcp_smp_3_percent"] = df[["mcp", "smp"]].max(axis=1) * 0.03

    df["kupsm"] = abs(df["tolerance_exceed_imbalance_amount"]) - df["tolerance_amount"]
    df["kupst"] = df["kupsm"] * df["mcp_smp_3_percent"]

    df["unit_kupst"] = df["kupst"] / df["grt"].replace(0, np.nan)
    return df


def kupst_report(df):
    tolerance_coefficient = df["tolerance_coefficient"].mean()
    total_generation = df["grt"].sum()
    total_imbalance_payment = df["imbalance_payment"].sum()
    total_positive_eia = df["positive_imbalance_amount"].sum()
    total_negative_eia = df["negative_imbalance_amount"].sum()
    total_imbalance_cost = df["imbalance_cost"].sum()
    unit_imbalance_cost = total_imbalance_cost / total_generation
    unit_kupst_cost = df["kupst"].sum() / total_generation
    total_kupst_cost = df["kupst"].sum()
    imbalance_plus_kupst_unit_cost = unit_imbalance_cost + unit_kupst_cost

    dict_report = {
        "Total Generation": [total_generation],
        "Total Electricity Imbalance Payment": [total_imbalance_payment],
        "Total Imbalance Cost": [total_imbalance_cost],
        "Unit Imbalance Cost": [unit_imbalance_cost],
        "Unit KUPST Cost": [unit_kupst_cost],
        "Total KUPST Cost": [total_kupst_cost],
        "Imbalance + Unit KUPST Cost": [imbalance_plus_kupst_unit_cost],
        "Total Positive Electricity Imbalance Amount": [total_positive_eia],
        "Total Negative Electricity Imbalance Amount": [total_negative_eia],
    }
    df_report = pd.DataFrame(dict_report)
    return df_report

#%% Info 
df_grt_info = get_grt_info()
df_grt_info.to_csv(f"{wdir}/data/df_grt_info.csv", index=False)

grt_info_list = sorted(df_grt_info["name"].to_list())
grt_info_list = [{"label":i, "value":i} for i in grt_info_list]


def get_plot_height(df):
    lenght = len(df)
    if lenght <= 10:
        return 500
    return (lenght // 10 + 1) * 400

#%% Pages
# Page layout fo Unit Imbalance Cost for Powerplants
kupst_layout = html.Div([
    html.H1('KUPST Dashboard'),
    html.P(),
    dbc.Row([html.P("Start & End Days"),
        dbc.Col([dcc.DatePickerRange(id='date-picker-range', minimum_nights=5, clearable=False, with_portal=True, start_date=date.today() - timedelta(days=7), end_date=date.today()),]),
    ]),
    html.P(),
    dbc.Row([
        dbc.Col([dbc.Row([html.P("Company for KUDUP Data"), dcc.Dropdown(id="org-kudup", value="SİBELRES ELEKTRİK ÜRETİM A.Ş.", multi=False)]),]),
        dbc.Col([dbc.Row([html.P("Powerplant for KUDUP Data"),dcc.Dropdown(id="pp-kudup", multi=False),])]),
        dbc.Col([dbc.Row([html.P("Powerplant for Real Time Generation Data"),dcc.Dropdown(id="pp-grt", options=grt_info_list, value="SİBEL RES-40W0000000156631", multi=False)]), ]),
    ]),
    html.P(),
    dbc.Row([html.P("Tolerance Coefficient"),
        dbc.Col([dcc.Input(id='tol-coef', type='number', value=0.1, min=0.05, max=1, step=0.01)]),
    ]),
    html.P(),
    dcc.Graph(id='plot-grt-kudup', figure={}),
    dcc.Graph(id='plot-imbalance-amount', figure={}),
    dcc.Graph(id='plot-kupst', figure={}),
])

about_dashboard_layout = html.Div([
    html.H1('About Dashboard'),
    html.P(),
    ])

# Page layout fo Unit Imbalance Cost for Organizations
contact_layout = html.Div([
    html.H1('Contact'),
    html.P(),
    dbc.Row([dcc.Link('Github', href='https://github.com/zereaykut'),]),
    dbc.Row([dcc.Link('LinkedIn', href='https://www.linkedin.com/in/halil-aykut-zere-90694520b/'),])
    ])

#%% Layout
# Register the page layouts with their corresponding URLs
sidebar = html.Div(
    [
        # html.H2("Menü", className="display-4"),
        html.Hr(),
        dbc.Button("☰", id="open-offcanvas", n_clicks=0),
        dbc.Offcanvas(
        dbc.Nav(
            [
                dbc.NavLink("KUPST Dashboard", href="/", active="exact"),
                dbc.NavLink("About Dashboard", href="/about-dashboard", active="exact"),
                dbc.NavLink("Contact", href="/contact", active="exact"),
            ],
            vertical=True,
            pills=True,
        )
            ,id="offcanvas", is_open=False,
        ),
    ],
)

content = html.Div(id="page-content")

app.layout = html.Div([dcc.Location(id="url"),  
                       dbc.Row([
                           dbc.Col([sidebar], xs=4, sm=4, md=2, lg=2, xl=2, xxl=1), 
                           dbc.Col([content], xs=8, sm=8, md=10, lg=10, xl=10, xxl=10)
                           ])
                        ])

# Callback to toggle navigation bar visibility
@app.callback(
    Output("offcanvas", "is_open"),
    Input("open-offcanvas", "n_clicks"),
    [State("offcanvas", "is_open")],
)
def toggle_offcanvas(n1, is_open):
    if n1:
        return not is_open
    return is_open

# Callback to update the page content based on the URL
@app.callback(Output('page-content', 'children'),
              [Input('url', 'pathname')])
def display_page(pathname):
    if pathname == "/":
        return kupst_layout
    elif pathname == "/about-dashboard":
        return about_dashboard_layout
    elif pathname == "/contact":
        return contact_layout
    else:
        return html.Div(
        [
            html.H1("404: Not found", className="text-danger"),
            html.Hr(),
            html.P(f"The pathname {pathname} was not recognised..."),
        ],
        className="p-3 bg-light rounded-3",
    )

#%% Parameters
@callback(Output('org-kudup', 'options'),
          Input('date-picker-range', 'start_date'), Input('date-picker-range', 'end_date'),)
def select_org_kudup(start_date, end_date):
    df_org_info = get_org_info(start_date, end_date)
    df_org_info.to_csv(f"{wdir}/data/df_org_info.csv", index=False)

    org_info_list = sorted(df_org_info["organizationName"].to_list())
    org_info_list = [{"label":i, "value":i} for i in org_info_list]

    return org_info_list

@callback(Output('pp-kudup', 'options'), Output('pp-kudup', 'value'),
          Input('date-picker-range', 'start_date'), Input('date-picker-range', 'end_date'), Input('org-kudup', 'value'),)
def select_pp_kudup(start_date, end_date, org_name):
    df_org_info = pd.read_csv(f"{wdir}/data/df_org_info.csv")
    org_info = df_org_info[df_org_info["organizationName"] == org_name]
    org_id = int(org_info["organizationId"])
    with open(f"{wdir}/data/org_id.txt", "w") as f:
        f.write(str(org_id))

    df_uevcb_info = get_uevcb_info(start_date, org_id)
    df_uevcb_info.to_csv(f"{wdir}/data/df_uevcb_info.csv", index=False)

    pp_info_list = sorted(df_uevcb_info["name"].to_list())
    pp_selected = pp_info_list[0]
    pp_info_list = [{"label":i, "value":i} for i in pp_info_list]

    return pp_info_list, pp_selected

@callback(Output('plot-grt-kudup', 'figure'), Output('plot-imbalance-amount', 'figure'), Output('plot-kupst', 'figure'),
          Input('date-picker-range', 'start_date'), Input('date-picker-range', 'end_date'), Input('pp-kudup', 'value'), Input('pp-grt', 'value'), Input('tol-coef', 'value'),)
def plot(start_date, end_date, uevcb_name, grt_name, tolerance_coefficient):
    with open(f"{wdir}/data/org_id.txt", "r") as f:
        org_id = int(f.readline())

    df_grt_info = pd.read_csv(f"{wdir}/data/df_grt_info.csv")
    grt_info = df_grt_info[df_grt_info["name"] == grt_name]
    grt_id = int(grt_info["id"])

    df_uevcb_info = pd.read_csv(f"{wdir}/data/df_uevcb_info.csv")
    uevcb_info = df_uevcb_info[df_uevcb_info["name"] == uevcb_name]
    uevcb_id = int(uevcb_info["id"])

    df_grt = get_grt(grt_id, start_date, end_date)
    df_kudup = get_kudup(uevcb_id, org_id, start_date, end_date)
    df_mcp = get_mcp(start_date, end_date)
    df_smp = get_smp(start_date, end_date)

    df = pd.concat(
        [
            # df_grt.set_index("date"),
            df_kudup.set_index("date"),
            df_mcp.set_index("date"),
            df_smp.set_index("date"),
        ],
        axis=1,
    )

    df = kupst(df, tolerance_coefficient)
    df_report = kupst_report(df)

    # Real Time Generation & KUDUP Plot
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df.index, y=df["grt"], mode="lines", name="Real Time Generation"
        )
    )
    fig.add_trace(
        go.Scatter(x=df.index, y=df["kudup"], mode="lines", name="KUDÜP")
    )
    fig.update_layout(title_text="Real Time Generation & KUDÜP", hovermode="x unified")

    # Imbalance Amount
    fig_imbalance_amount = go.Figure()
    fig_imbalance_amount.add_trace(
        go.Scatter(
            x=df.index,
            y=df["positive_imbalance_amount"].fillna(0),
            mode="lines",
            name="Positive Imbalance",
            marker={"color": "red"},
        )
    )
    fig_imbalance_amount.add_trace(
        go.Scatter(
            x=df.index,
            y=df["negative_imbalance_amount"].fillna(0),
            mode="lines",
            name="Negative Imbalance",
            marker={"color": "blue"},
        )
    )
    fig_imbalance_amount.update_layout(title_text="Imbalance Amount", hovermode="x unified")
    st.plotly_chart(
        fig_imbalance_amount, theme="streamlit", use_container_width=True
    )

    # Imbalance Cost
    fig_imbalance_cost = make_subplots(specs=[[{"secondary_y": True}]])
    fig_imbalance_cost.add_trace(
        go.Scatter(
            x=df.index,
            y=df["imbalance_cost"].fillna(0),
            mode="lines",
            name="Imbalance Cost",
            marker={"color": "red"},
        ),
        secondary_y=True,
    )
    fig_imbalance_cost.add_trace(
        go.Scatter(
            x=df.index,
            y=df["kupst"].fillna(0),
            mode="lines",
            name="Kupst",
            marker={"color": "blue"},
        ),
        secondary_y=False,
    )
    fig_imbalance_cost.update_layout(title_text="Kupst", hovermode="x unified")
    fig_imbalance_cost.update_yaxes(
        title_text="Imbalance Cost", secondary_y=True
    )
    fig_imbalance_cost.update_yaxes(title_text="Kupst", secondary_y=False)

    return fig, fig_imbalance_amount, fig_imbalance_cost


if __name__ == "__main__":
    app.run(debug=True) # local run
    # app.run_server(host="0.0.0.0", port=8002, debug=False) # network run
