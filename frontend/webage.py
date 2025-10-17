import streamlit as st
from backend.data.parsedata import dataset, features_df  # dataset for UI, features_df for model

st.title("SteamGameSuggester - Dataset from 26/04/2025")

# Safety: handle empty dataset
if not dataset:
    st.warning("No games loaded.")
    st.stop()

# 1) Build option list
app_ids = list(dataset.keys())

def label_for(app_id: str) -> str:
    g = dataset.get(app_id, {})
    name = g.get("name") or "Unknown"
    price = g.get("price", 0.0) or 0.0
    mc = g.get("metacritic_score", 0) or 0
    price_str = "Free" if float(price) == 0 else f"${float(price):.2f}"
    mc_str = f" • MC {int(mc)}" if mc else ""
    return f"{name} (ID: {app_id}) — {price_str}{mc_str}"

# 2) Let user pick up to 10
selected_ids = st.multiselect(
    "Pick up to 10 games:",
    options=app_ids,
    max_selections=10,  # requires Streamlit >= 1.25
    format_func=label_for,
    placeholder="Type to search by name, ID…",
)

st.caption(f"{len(selected_ids)}/10 selected")

# 3) Preview chosen games
if selected_ids:
    rows = []
    for app_id in selected_ids:
        g = dataset.get(app_id, {})
        rows.append({
            "AppID": app_id,
            "Name": g.get("name", ""),
            "Price": g.get("price", 0.0),
            "User Score": g.get("user_score", 0),
            "Metacritic": g.get("metacritic_score", 0),
            "Positive": g.get("positive", 0),
            "Negative": g.get("negative", 0),
            "Genres": ", ".join(g.get("genres", [])) if isinstance(g.get("genres"), list) else g.get("genres", ""),
        })
    st.dataframe(rows, use_container_width=True)

# 4) On confirm, persist selection and build the feature matrix for your model
if st.button("Use these"):
    st.session_state["selected_app_ids"] = selected_ids
    if not selected_ids:
        st.warning("Select at least one game.")
    else:
        # features_df has exactly the whitelisted features you chose in parsedata.py
        ids_in_index = [sid for sid in selected_ids if sid in features_df.index]
        X = features_df.loc[ids_in_index]

        st.success(f"Saved {len(ids_in_index)} selections.")
        st.write("Feature matrix shape:", X.shape)
        st.dataframe(X, use_container_width=True)
        # e.g. preds = model.predict(X)

chosen = st.session_state.get("selected_app_ids", [])