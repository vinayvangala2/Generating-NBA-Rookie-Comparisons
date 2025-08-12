import streamlit as st
import pandas as pd
import numpy as np
import torch
from torch import nn
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

# -----------------------
# Load and train model
# -----------------------
@st.cache_resource
def load_model():
    df = pd.read_csv("rookie_train_pre2023.csv")
    player_names = df["Player"].values
    features_df = df.drop(columns=["Player", "Season", "SeasonStart"], errors="ignore")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features_df.astype(np.float32))
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

    class Autoencoder(nn.Module):
        def __init__(self, input_dim, encoding_dim=8):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, 128),
                nn.Tanh(),
                nn.Linear(128, 64),
                nn.Tanh(),
                nn.Linear(64, encoding_dim)
            )
            self.decoder = nn.Sequential(
                nn.Linear(encoding_dim, 64),
                nn.Tanh(),
                nn.Linear(64, 128),
                nn.Tanh(),
                nn.Linear(128, input_dim)
            )

        def forward(self, x):
            encoded = self.encoder(x)
            decoded = self.decoder(encoded)
            return decoded

    input_dim = X_tensor.shape[1]
    model = Autoencoder(input_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.SmoothL1Loss()

    for epoch in range(200):
        model.train()
        optimizer.zero_grad()
        output = model(X_tensor)
        loss = loss_fn(output, X_tensor)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        encoded_data = model.encoder(X_tensor).numpy()

    knn = NearestNeighbors(n_neighbors=6, metric="euclidean")
    knn.fit(encoded_data)

    return model, scaler, knn, player_names, encoded_data

# -----------------------
# Load post-2023 rookies
# -----------------------
@st.cache_data
def load_test_data():
    test_df = pd.read_csv("rookie_test_2023on.csv")
    test_df.dropna(inplace=True)
    return test_df

# Load resources
model, scaler, knn, player_names, encoded_data = load_model()
test_df = load_test_data()

# -----------------------
# Streamlit UI
# -----------------------
st.title("2024 and 2025 NBA Rookies: Comparison Generator")
st.markdown("Uses a combination of 30 advanced stats (TS%, ORB%, etc,; scaled to league average where appropriate), NBA Draft Combine measurements (height, wingspan, etc.), and player attributes (age, games played, etc.) to find the most similar rookie seasons from 2000 - 2023.")

# Dropdown to select rookie
rookie_list = sorted(test_df["Player"].unique())
selected_rookie = st.selectbox("Select a recent rookie:", rookie_list)

top_n = st.slider("Number of similar rookies:", 1, 10, 5)

# Button to compute comps
if st.button("Find Similar Rookies"):
    row = test_df[test_df["Player"] == selected_rookie]
    row_features = row.drop(columns=["Player", "Season", "SeasonStart"], errors="ignore")

    row_scaled = scaler.transform(row_features.astype(np.float32))
    row_tensor = torch.tensor(row_scaled, dtype=torch.float32)

    with torch.no_grad():
        row_encoded = model.encoder(row_tensor).numpy()

    distances, indices = knn.kneighbors(row_encoded, n_neighbors=top_n)

    # Show results
    st.subheader(f"Top {top_n} Comps for {selected_rookie}")
    results = []
    for rank, (i, dist) in enumerate(zip(indices[0], distances[0]), 1):
        results.append({"Rank": rank, "Similar Rookie": player_names[i], "Similarity Score": round(dist, 4)})

    results_df = pd.DataFrame(results)
    st.dataframe(
        results_df[['Similar Rookie', "Similarity Score"]],
        hide_index=True,
        use_container_width=True
    )

    st.markdown("Similarity scores are determined through Euclidian distance, lower scores (~0 is lowest) mean a stronger similarity.")

with st.expander("ℹ️ About this model"):

    st.markdown("Originally, I wanted to build out this tool for draft prospects, to make it easier to develop 'comps' for players, seeing as 'comps' are the backbone of most NBA pre-draft content these days. With a lot of the combine data and college advanced stat data being inaccessible or sparse, I pivoted to using rookie season advanced stats and a subset of available NBA Draft Combine stats to create a proof-of-concept model. Hence, this model here. The app allows you to select from a list of 2023 and 2024 draft class NBA rookies with available data, and find out their 10 most similar rookie seasons from 2000 to 2023. Since a model like this has no ground truth to judge accuracy, no one can say for sure that this model is generating 'correct' responses. Rather, the goal is to develop a more pointed set of questions. For example, Rob Dillingham's closest comparison is former 53rd overall pick Kenny Satterfield. This should sound some alarms, but also force a deeper look into Satterfield's career trajectory and how Dillingham may follow or deviate from that. The 30 features are as follows, with some advanced stats being scaled to that season's league average: Age, G, GS, MP, PER, TS%, 3PAr, FTr, ORB%, DRB%, TRB%, AST%, STL%, BLK%, TOV%, USG%, OWS, DWS, WS, WS/48, OBPM, DBPM, BPM, VORP, HGT, WGT, BMI, WNGSPN, STNDRCH, BAR. The advanced stats data was scaped from Basketball Reference, and the NBA Draft Combine data was pulled from Kaggle. The process of building out the final dataset can be seen in the Jupyter Notebook files included in this repository. This process did come with caveats. Since not all players took part in the NBA Draft Combine and/or did not take part in every drill, not all rookies to ever touch an NBA floor from 2000 to 2025 are included, and only a subset of these overall NBA Draft Combine measurements could be included. Therefore, comparisons are being made to a somewhat limited set of players. The model itself begins with an autoencoder with a 3-layer feedforward neural net. I chose an autoencoder to handle dimensionality reduction, lowering the data to 8 dimensions from 30, while discovering any non-linear patterns that may be present in the data. This felt crucial to gaining good insights into highly interrelated variables. The output was fed into a K-Nearest Neighbors model, which was used to determine similarity score (using Euclidian distance). Future work would include testing out a few other possibilities for the autoencoder, and using PCA as a benchmark to see if the orthogonality constraint doesn't impact the comparisons. I would also like to pull together enough data to create a proper draft prospect model, as I see the value of a tool like this in making better draft selections.")

