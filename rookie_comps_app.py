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
    st.markdown("MODEL: An autoencoder, encoded into 8 dimensions, and fed into a K-Nearest Neighbors model. Similarity scores are determined through Euclidian distance. An autoencoder was chosen in order to discover any non-linear relationships between statistics, and KNN is used to determine comparisons by distance from other rookie seasons. Alternatives could include PCA to reduce the dimensionality of the data rather than autoencoders, however this would only find linear relationships amongst variables.")
    st.markdown("GOAL: This model is not a means to an end. Even with further iteration, there is no ground truth for comparisons that could determine the 'accuracy' of the model. Rather, the comparisons generated can be used to ask pointed questions about players, and give a more specific direction to look in for how a current rookie's development curve may mimic or deviate from their generated comparisons.")
    st.markdown("CAVEATS: Unfortunately, not all players have taken part in the combine, much less every combine drill. Hence, not all rookies are included in the dataset, and not all possible combine measurements are included (for example, no lane agility measurement is used due to missing data). More data availability would alter the predictions created.")
