import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.figure_factory as ff
import plotly.offline

pd.set_option ('display.expand_frame_repr', False)
pd.set_option ("display.max_rows", 300)
pd.set_option ("display.max_columns", None)
pd.set_option ("display.width", 10000)
pd.set_option ('display.float_format', lambda x: '%.2f' % x)


def input_func(data, skill1="dribbling", skill2="potential", skill3="overall", skill4="passing",
               skill5="defending"):
    st.sidebar.markdown ('**_Player Name_**')
    player = st.sidebar.selectbox ("", sorted (df["short_name"].values.tolist ()))
    face_url = data.loc[data["short_name"] == player, "player_face_url"].values[0]
    club_url = data.loc[data["short_name"] == player, "club_logo_url"].values[0]
    nation_url = data.loc[data["short_name"] == player, "nation_flag_url"].values[0]
    position = data.loc[data["short_name"] == player, "first_position"].values[0]
    market_value = data.loc[data["short_name"] == player, "value_eur"].values[0].astype (int)

    col1, col2 = st.sidebar.columns([1, 1])
    with col1:
        st.image (face_url)
        st.markdown (position)

    with col2:
        st.subheader ("_**MARKET VALUE**_")
        st.markdown (f'**{market_value}**')
        st.markdown ('**_Club Logo_**')
        st.image (club_url)

    df_radar = pd.DataFrame (dict (
        r=[data.loc[data["short_name"] == player, skill1].values[0],
           data.loc[data["short_name"] == player, skill2].values[0],
           data.loc[data["short_name"] == player, skill3].values[0],
           data.loc[data["short_name"] == player, skill4].values[0],
           data.loc[data["short_name"] == player, skill5].values[0]],
        theta=[skill1, skill2, skill3, skill4, skill5]))
    fig2 = px.line_polar (df_radar, r='r', theta='theta', line_close=True, height=250, range_r=(0, 100))
    st.sidebar.plotly_chart (fig2, use_container_width=True)

    return player, position


def recommendation(data, player):
    segment = data.loc[data["short_name"] == player, "Segment"].values[0]
    skill_cols = [col for col in data.columns if (col.endswith ("_url") == False) & (col != "sofifa_id")]
    new_data = data.loc[~(data["short_name"] == player)]

    st.subheader ("**TOP 5 RECOMMENDED PLAYERS**")
    sort_cols = [col for col in skill_cols if col != "Segment"]

    col1, col2 = st.columns([1,1])
    with col1:
        sortBy = st.selectbox ("Sort By", sorted (sort_cols))
    with col2:
        asc = st.selectbox ("Sort Type", ["Descending", "Ascending"])
        if asc == "Descending":
            recommendation = new_data.loc[new_data["Segment"] == segment, skill_cols].sort_values (sortBy,
                                                                                                   ascending=False).reset_index (
                drop=True)
        else:
            recommendation = new_data.loc[new_data["Segment"] == segment, skill_cols].sort_values (sortBy,
                                                                                                   ascending=True).reset_index (
                drop=True)

    st.subheader("Distribution of Recommended Players' Segment")
    fig = px.scatter_3d (recommendation, x='defending', y='overall', z='value_eur',
                         color="Segment", log_x=True, hover_name="short_name", hover_data=["overall"], height=450,
                         width=800, labels={"defending":"Defending", "overall":"Overall", "value_eur":"Market Value"})
    st.plotly_chart (fig, use_container_width=True)

    top_5 = recommendation.iloc[0:5].reset_index (drop=True)
    for i in range (len (top_5)):
        col1, col2, col3 = st.columns([3, 2, 4])
        with col1:
            rec_face_url = data.loc[data["short_name"] == top_5["short_name"].values[i], "player_face_url"].values[0]
            st.image (rec_face_url)
            rec_short_name = data.loc[data["short_name"] == top_5["short_name"].values[i], "short_name"].values[0]
            st.text (rec_short_name)
            rec_pos = data.loc[data["short_name"] == top_5["short_name"].values[i], "first_position"].values[0]
            st.markdown (rec_pos)

        with col2:
            rec_value = data.loc[data["short_name"] == top_5["short_name"].values[i], "value_eur"].values[0].astype (
                int)
            st.subheader ("_**MARKET VALUE**_")
            st.markdown (f'**{rec_value}**')
            rec_club_url = data.loc[data["short_name"] == top_5["short_name"].values[i], "club_logo_url"].values[0]
            st.image (rec_club_url)
        with col3:
            radar_plot (top_5, rec_short_name)

    return recommendation


def radar_plot(data, player, skill1="dribbling", skill2="potential", skill3="overall", skill4="passing",
               skill5="defending"):
    df_radar = pd.DataFrame (dict (
        r=[data.loc[data["short_name"] == player, skill1].values[0],
           data.loc[data["short_name"] == player, skill2].values[0],
           data.loc[data["short_name"] == player, skill3].values[0],
           data.loc[data["short_name"] == player, skill4].values[0],
           data.loc[data["short_name"] == player, skill5].values[0]],
        theta=[skill1, skill2, skill3, skill4, skill5]))
    fig2 = px.line_polar (df_radar, r='r', theta='theta', line_close=True, height=300, range_r=(0, 100))
    st.plotly_chart (fig2, use_container_width=True)


df_ = pd.read_csv ("Final_Table.csv", low_memory=False)

df_.drop ("Unnamed: 0", axis=1, inplace=True)
df = df_.copy ()

st.title ("FIFA 22 PLAYER SEGMENTATION")
st.write ("""
 * The app analyze and segment FIFA players by using PCA and K-Means Unsupervised Learning Algorithm.""")

if st.button("""Click Here To See The Descriptive Statistics of All Segments"""):
    fig3 = px.box (df, x="Segment", y="overall", color="Segment", width=700, height=500, labels={"overall":"Overall"})
    st.plotly_chart (fig3, use_container_width=True)

player, position = input_func (df)

rec_df = recommendation (df, player)



