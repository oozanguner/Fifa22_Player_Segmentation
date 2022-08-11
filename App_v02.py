
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from functools import reduce
import warnings
import datetime as dt
import missingno as msno
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer
from yellowbrick.cluster.elbow import kelbow_visualizer
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from scipy import stats
import plotly.express as px

pd.set_option ("display.max_rows", 300)
pd.set_option ("display.max_columns", None)
pd.set_option ("display.width", 10000)
pd.set_option ('display.float_format', lambda x: '%.2f' % x)

warnings.filterwarnings ("ignore")
def grab_cols(dataframe):
    num_cols = []
    id_jersey_cols = []
    cat_cols = []
    for col in dataframe.columns:
        if ("jersey" not in col) & ("_id" not in col):
            try:
                dataframe[col].astype (float)
                num_cols.append (col)
            except:
                cat_cols.append (col)
        else:
            id_jersey_cols.append (col)

    return num_cols, cat_cols, id_jersey_cols


def create_data():
    for i in range (15, 23):
        globals ()[f"df_{i}"] = pd.read_csv (f"/Users/ozanguner/PycharmProjects/Miuul_Fifa_Clustering/players_{i}.csv",
                                             low_memory=False)
        print (f"Dataframe that containing 20{i} historical statistics were read")
        nums, cats, id_jerseys = grab_cols (globals ()[f"df_{i}"])

    intersect_cols = (list (
        set.intersection (set (df_15.columns), set (df_16.columns), set (df_17.columns), set (df_18.columns),
                          set (df_19.columns),
                          set (df_20.columns), set (df_21.columns), set (df_22.columns))))

    url_cols = [col for col in intersect_cols if col.endswith ("_url")]
    dropped_cols = [col for col in intersect_cols if (len (col) <= 3) | (col.endswith ("_url"))]

    for i in range (15, 23):
        globals ()[f"df_{i}"].drop (dropped_cols, axis=1, inplace=True)
        print (f"Unuseful features were dropped from dataframe that containing 20{i} historical statistics")

    for i in range (15, 23):
        nums, cats, id_jerseys = grab_cols (globals ()[f"df_{i}"])
        globals ()[f"new_df_{i}"] = globals ()[f"df_{i}"][nums]
        col_names = [col + f"_{i}" for col in globals ()[f"df_{i}"].columns if col in nums]
        globals ()[f"new_df_{i}"].columns = col_names
        globals ()[f"new_df_{i}"].insert (0, column="sofifa_id", value=globals ()[f"df_{i}"]["sofifa_id"])

    new_dfs = [new_df_22, new_df_21, new_df_20, new_df_19, new_df_18, new_df_17, new_df_16, new_df_15]

    for i in range (22, 15, -1):
        globals ()[f"new_df_{i - 1}"] = pd.merge (globals ()[f"new_df_{i}"], globals ()[f"new_df_{i - 1}"], how="left",
                                                  on="sofifa_id")
    print ("All historical datas were merged")
    merged_new_df = new_df_15.copy ()

    return merged_new_df, df_22, url_cols, dropped_cols, intersect_cols


merged_new_df, df_22, url_cols, dropped_cols, intersect_cols = create_data ()


# FEATURE ENGINEERING
def create_trend_feats(merged_new_df, df_22):
    trend_cols = [col for col in merged_new_df.columns if col != "sofifa_id"]


    overall_cols = [col for col in merged_new_df.columns if col.startswith ("overall")]
    merged_new_df["NEW_years_played"] = merged_new_df[overall_cols].notna ().sum (axis=1)
    print ("Players' active year features were created.")

    unique_cols = list (set ([
        col.split ("_")[0] if len (col.split ("_")) == 2 else col.split ("_")[0] + "_" + col.split ("_")[1] if len (
            col.split ("_")) == 3 else col.split ("_")[0] + "_" + col.split ("_")[1] + "_" + col.split ("_")[2] for col
        in
        trend_cols if "club_contract" not in col]))


    for col in unique_cols:
        merged_new_df[[f"{col}_22", f"{col}_21", f"{col}_20", f"{col}_19", f"{col}_18", f"{col}_17", f"{col}_16",
                       f"{col}_15"]] = merged_new_df.loc[:,
                                       [f"{col}_22", f"{col}_21", f"{col}_20", f"{col}_19", f"{col}_18", f"{col}_17",
                                        f"{col}_16", f"{col}_15"]].fillna (method="ffill", axis=1)
    print ("Missing data from previous years were filled.")


    for col in unique_cols:
        merged_new_df[f"NEW_{col}_total_trend"] = ((merged_new_df[f"{col}_22"] - merged_new_df[f"{col}_21"]) /
                                                   merged_new_df[
                                                       f"{col}_21"]) + (
                                                          (merged_new_df[f"{col}_21"] - merged_new_df[f"{col}_20"]) /
                                                          merged_new_df[f"{col}_20"]) + ((
                                                                                                 merged_new_df[
                                                                                                     f"{col}_20"] -
                                                                                                 merged_new_df[
                                                                                                     f"{col}_19"]) / \
                                                                                         merged_new_df[f"{col}_19"]) + (
                                                          (
                                                                  merged_new_df[f"{col}_19"] - merged_new_df[
                                                              f"{col}_18"]) / \
                                                          merged_new_df[f"{col}_18"]) + ((
                                                                                                 merged_new_df[
                                                                                                     f"{col}_18"] -
                                                                                                 merged_new_df[
                                                                                                     f"{col}_17"]) / \
                                                                                         merged_new_df[f"{col}_17"]) + (
                                                          (
                                                                  merged_new_df[f"{col}_17"] - merged_new_df[
                                                              f"{col}_16"]) / \
                                                          merged_new_df[f"{col}_16"]) + ((
                                                                                                 merged_new_df[
                                                                                                     f"{col}_16"] -
                                                                                                 merged_new_df[
                                                                                                     f"{col}_15"]) / \
                                                                                         merged_new_df[f"{col}_15"])
    print ("Trend features were created.")

    last_trend_cols = [col for col in merged_new_df.columns if col.endswith ("trend")]


    for col in unique_cols:
        merged_new_df[f"NEW_{col}_avg_trend"] = merged_new_df[f"NEW_{col}_total_trend"] / (
                merged_new_df["NEW_years_played"] - 1)
        merged_new_df[f"NEW_{col}_avg_trend"] = merged_new_df[f"NEW_{col}_avg_trend"].fillna (0)
    print ("Average trend statistics were created")

    avg_trend_cols = [col for col in merged_new_df.columns if col.endswith ("avg_trend")]
    avg_trend_cols.insert (0, "sofifa_id")
    avg_trend_cols.insert (1, "NEW_years_played")

    merged_final_df = merged_new_df[avg_trend_cols]


    final_df = pd.merge (df_22, merged_final_df, how="inner", on="sofifa_id")

    return final_df, unique_cols


final_df, unique_cols = create_trend_feats (merged_new_df, df_22)


def feature_eng(final_df, unique_cols):
    final_df["NEW_first_position"] = final_df.apply (lambda x: x["player_positions"].split (",")[0], axis=1)
    final_df["NEW_total_position_count"] = final_df.apply (lambda x: len (x["player_positions"].split (",")), axis=1)


    final_df[["physic", "passing", "pace", "defending", "dribbling", "shooting"]] = final_df[
        ["physic", "passing", "pace", "defending", "dribbling", "shooting"]].fillna (0)


    final_df["goalkeeping_speed"] = final_df["goalkeeping_speed"].fillna (0)

    foot_mapping = {"Left": 0,
                    "Right": 1}

    final_df["NEW_foot_isRight"] = final_df["preferred_foot"].map (foot_mapping)

    final_df.loc[final_df["club_team_id"].isnull (), "NEW_isFreeAgent"] = 1
    final_df["NEW_isFreeAgent"].fillna (0, inplace=True)

    final_df.loc[final_df["club_loaned_from"].isnull (), "NEW_isLoan"] = 0
    final_df["NEW_isLoan"].fillna (1, inplace=True)

    today_date = dt.datetime (2022, 1, 1)

    filled_value = (today_date - dt.timedelta (days=365)).strftime ("%Y-%m-%d")
    final_df["club_joined"] = final_df["club_joined"].fillna (filled_value)
    final_df["club_joined"] = pd.to_datetime (final_df["club_joined"])
    final_df["NEW_club_joined"] = final_df["club_joined"].apply (lambda x: today_date.year - x.year)


    final_df["NEW_league_level"] = final_df["league_level"].apply (lambda x: 1 if x == 1 else 2)


    final_df["club_contract_valid_until"] = final_df["club_contract_valid_until"].fillna (
        int (today_date.year))

    final_df["NEW_contract_valid_left"] = final_df["club_contract_valid_until"] - final_df["club_joined"].dt.year

    final_df.loc[final_df["nation_team_id"].isnull (), "NEW_isNationalTeam"] = 0
    final_df["NEW_isNationalTeam"].fillna (1, inplace=True)


    final_df["wage_eur"] = final_df["wage_eur"].fillna (0)

    final_df.loc[final_df["goalkeeping_speed"].notna (), "NEW_isGoalkeeper"] = 1
    final_df["NEW_isGoalkeeper"].fillna (0, inplace=True)

    final_df["value_eur"] = final_df["value_eur"].fillna (0)

    final_df["NEW_player_traits_count"] = final_df["player_traits"].apply (
        lambda x: 0 if pd.isnull (x) else len (x.split (",")))

    # PLAYER TRAITS FEATURE ENGINEERING
    final_df["player_traits"].replace (np.nan, "", inplace=True)

    traits = set ()
    for i in final_df["player_traits"].values:
        range_ = len (i.split (","))
        for k in range (range_):
            traits.add (i.split (",")[k].strip ())

    traits_list = list (traits)
    traits_list.remove ('')

    traits_df = pd.DataFrame (index=range (len (final_df)), columns=traits_list)
    print ("Characteristic features are being created.")
    for ind, i in enumerate (final_df["player_traits"].str.split (", ")):
        for k in traits_list:
            if k in i:
                traits_df[k][ind] = 1
            else:
                traits_df[k][ind] = 0
    print ("Characteristic features were created.")
    final_df = pd.concat ([final_df, traits_df], axis=1)

    new_traits_list = ["NEW_" + "_".join (col.split (" ")) for col in final_df.columns if col in traits_list]
    traits_mapping = dict (zip (traits_list, new_traits_list))

    final_df.rename (columns=traits_mapping, inplace=True)

    final_df["NEW_work_rate1"] = final_df["work_rate"].apply (lambda x: x.split ("/")[0])
    final_df["NEW_work_rate2"] = final_df["work_rate"].apply (lambda x: x.split ("/")[1])

    work_rate_mapping = {"Low": 1,
                         "Medium": 2,
                         "High": 3}

    final_df["NEW_work_rate1"] = final_df["NEW_work_rate1"].map (work_rate_mapping)
    final_df["NEW_work_rate2"] = final_df["NEW_work_rate2"].map (work_rate_mapping)

    final_df["NEW_body_type1"] = final_df["body_type"].apply (lambda x: x.split (" ")[0])
    final_df["NEW_body_type2"] = final_df["body_type"].str.extract ('([(0-9].+)')

    # https://fifaforums.easports.com/en/discussion/598277/body-types-best-to-worst-and-why-that-90-rated-player-turns-like-a-boat  sitesindeki bilgilerden yola çıkarak mapping yapılmıştır.
    body_type1_mapping = {"Unique": 4,
                          "Lean": 3,
                          "Normal": 2,
                          "Stocky": 1}
    final_df["NEW_body_type1"] = final_df["NEW_body_type1"].map (body_type1_mapping)

    body_type2_mapping = {"(170-)": 1,
                          "(170-185)": 2,
                          "(185+)": 3}

    final_df["NEW_body_type2"] = final_df["NEW_body_type2"].map (body_type2_mapping)
    final_df["NEW_body_type2"].fillna (4, inplace=True)
    final_df["NEW_body_type2"] = final_df["NEW_body_type2"].astype (int)

    final_df["release_clause_eur"] = final_df["release_clause_eur"].fillna (
        0)

    effort_cols = unique_cols.copy ()
    effort_cols.remove ("league_level")
    new_cols = [col for col in final_df.columns if col.startswith ("NEW_") and col != "NEW_league_level_avg_trend"]

    final_feats = effort_cols + new_cols

    final_df2 = final_df[final_feats].copy ()

    obj_cols = [col for col in final_df2.columns if final_df2[col].dtypes == "O" and col != "NEW_first_position"]

    final_df2 = pd.get_dummies (data=final_df2, columns=["NEW_first_position"], drop_first=True)

    final_df2[obj_cols] = final_df2[obj_cols].astype (int)

    value_cols = ["release_clause_eur", "value_eur", "wage_eur", "NEW_release_clause_eur_avg_trend",
                  "NEW_value_eur_avg_trend", "NEW_wage_eur_avg_trend"]

    final_df3 = final_df2.drop (value_cols, axis=1)

    return final_df3


final_df3 = feature_eng (final_df, unique_cols)
final_df3.head()
var_df = final_df3.var()
dropped_feat = var_df[var_df.values==0].index[0]
final_df3.drop(dropped_feat, axis=1, inplace=True)
def check_dist(df_dist, col):
    plt.title('Distribution of ' + col)
    sns.distplot(df_dist[col],color = "b")
    plt.show()
for col in final_df3.columns:
    check_dist(final_df3, col)
def scale_data(final_df, final_df3):
    # SCALING
    sc = StandardScaler ()
    sc_df = sc.fit_transform (final_df3)
    scale_df = pd.DataFrame (sc_df, index=final_df["sofifa_id"], columns=final_df3.columns)

    return scale_df


scale_df = scale_data (final_df, final_df3)


def deter_comp_pca(scale_df, threshold=0.90):
    # Decreasing Dimension with PCA
    col_counts = len (scale_df.columns)
    pca = PCA (n_components=col_counts)
    scaled_pca = pca.fit_transform (scale_df)
    explained_var = np.cumsum (pca.explained_variance_ratio_)
    n_comp = np.where (explained_var < threshold)[0][
                 -1] + 1  # Açıklanabilirlik üzerinden n_component sayısını belirleme

    return n_comp




n_comp = deter_comp_pca (scale_df)


def dec_dim_pca(scale_df, n_comp):
    fin_pca = PCA (n_components=n_comp)
    fin_df = fin_pca.fit_transform (scale_df)
    df_last = pd.DataFrame (fin_df,index=scale_df.index)

    return df_last




df_last = dec_dim_pca (scale_df, n_comp)


def optimal_cluster_count(df_last, k_range=50, metric_="distortion"):
    # Determine the optimal number of clusters
    print ("Determining optimal number of clusters.")
    kmeans = KMeans (random_state=99)
    elbow = KElbowVisualizer (kmeans, k=k_range, metric=metric_)
    elbow.fit (df_last)
    elbow.show()
    k_ = elbow.elbow_value_
    print ("Optimal number of clusters was determined.")
    return k_


k_ = optimal_cluster_count (df_last)
# FOR ALL PLAYERS
def modelling(df_last, k_):
    # Modelleme
    model = KMeans (random_state=99, n_clusters=k_).fit (df_last)
    segments = model.labels_

    return segments

segments = modelling(df_last, k_)
ident_cols = ["sofifa_id","short_name","NEW_first_position"]+unique_cols
rec_df = final_df.loc[final_df["sofifa_id"].isin(df_last.index), ident_cols]
rec_df["Segment"] = segments
rec_df.head()
rec_df.to_csv("Recommendation_Table.csv")

#rec_df = pd.read_csv("/Users/ozanguner/PycharmProjects/Miuul_Fifa_Clustering/Recommendation_Table.csv")
#rec_df.drop("Unnamed: 0", axis=1, inplace=True)
url_df = pd.read_csv("/Users/ozanguner/PycharmProjects/Miuul_Fifa_Clustering/players_22.csv", low_memory=False)
url_cols = [col for col in url_df if (col.endswith("_url")) | (col == "sofifa_id")]
url_df2 = url_df[url_cols]

df_fin = rec_df.merge(url_df2, on="sofifa_id", how = "left")
df_fin.rename(columns={"NEW_first_position":"first_position"}, inplace=True)
df_fin.to_csv("Final_Table.csv")



