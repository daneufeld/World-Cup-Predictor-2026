import streamlit as st
import os
import json
import random
import pandas as pd

# Import your functions
from simulator import (
    parse_groups_from_file,
    fetch_live_news_headlines,
    analyze_team_sentiment,
    generate_ratings_with_gemini,
    calculate_probabilities,
    simulate_group_stage_opta,
    run_genai_tournament_director,
)

CACHE_FILE = "team_ratings_cache.json"

st.set_page_config(page_title="2026 World Cup Predictor", page_icon="📊", layout="wide")

# --- INITIALIZATION ---
if "team_ratings" not in st.session_state: st.session_state.team_ratings = None
if "opta_results" not in st.session_state: st.session_state.opta_results = None
if "round_name" not in st.session_state: st.session_state.round_name = "Sechzehntelfinale"
if "current_round_teams" not in st.session_state: st.session_state.current_round_teams = None
if "current_matches" not in st.session_state: st.session_state.current_matches = None
if "current_winners" not in st.session_state: st.session_state.current_winners = None

st.title("📊 2026 World Cup Forecasting Dashboard")
st.markdown("Predictive index modeling, tournament simulations, and bracket execution.")

# Try loading cached ratings immediately if available
if st.session_state.team_ratings is None and os.path.exists(CACHE_FILE):
    with open(CACHE_FILE, "r", encoding="utf-8") as f:
        st.session_state.team_ratings = json.load(f)

# --- STEP 1: RATINGS ---
if st.session_state.team_ratings is None:
    FIFA_FILE = "fifa_rankings.json"
    
    if os.path.exists(FIFA_FILE):
        with open(FIFA_FILE, "r", encoding="utf-8") as f:
            raw_fifa_data = json.load(f)
        
        # Transform raw FIFA points (1200-1900) onto our clean 50-99 math index
        final_ratings = {}
        for team, points in raw_fifa_data.items():
            # Linear scaling normalization formula
            min_f, max_f = 1200.0, 1900.0
            norm = (points - min_f) / (max_f - min_f)
            norm = max(0.0, min(1.0, norm)) # Clamp bounds securely
            
            final_ratings[team] = round(50 + (norm * 49), 1)
            
        st.session_state.team_ratings = final_ratings
    else:
        st.error(f"🚨 '{FIFA_FILE}' missing! Please place it in the project root folder.")
        st.stop()

# --- STEP 2: GROUP STAGE PREDICTIONS ---
st.markdown("---")
st.header("🎲 2. Group Stage Probabilities")

if st.session_state.opta_results is None:
    st.warning("Group stage simulations have not been calculated yet.")
    if st.button("📊 Berechne Aufstiegschancen"):
        with st.spinner("Simuliere Gruppenphase (1,000 Iterations)..."):
            st.session_state.opta_results = simulate_group_stage_opta(parse_groups_from_file(), st.session_state.team_ratings)
        st.rerun()
    st.stop()
else:
    # Format and present the Opta simulation results
    with st.expander("📈 View Advancement Probability Rankings", expanded=True):
        raw_data = []
        for team, metrics in st.session_state.opta_results.items():
            raw_data.append({
                "Team": team,
                "SPI Strength": st.session_state.team_ratings.get(team, 72.0),
                "Avg Points Expected": metrics.get("avg_points", 0.0),
                "Advance to Knockout %": f"{round(metrics.get('advance', 0.0) * 100, 1)}%"
            })
        df_opta = pd.DataFrame(raw_data).sort_values(by="Avg Points Expected", ascending=False).reset_index(drop=True)
        st.dataframe(df_opta, use_container_width=True)

# --- STEP 3: PLAY THROUGH THE KNOCKOUT BRACKET ---
st.markdown("---")
st.header(f"🏆 3. Live Knockout Stage: {st.session_state.round_name}")

def run_match_sim():
    # Package data in the format expected by the tournament director fallback logic
    round_data = run_genai_tournament_director(st.session_state.current_round_teams, {})
    matches = round_data.get("matches", [])
    winners = []
    for match in matches:
        t_a, t_b = match["team_A"], match["team_B"]
        probs = calculate_probabilities(
            st.session_state.team_ratings.get(t_a, 72.0),
            st.session_state.team_ratings.get(t_b, 72.0),
        )
        p_a_ko = probs["prob_A"] + probs["prob_draw"] / 2
        winners.append(t_a if random.random() < p_a_ko else t_b)
    st.session_state.current_matches = matches
    st.session_state.current_winners = winners

def advance_round():
    round_map = {
        "Sechzehntelfinale": "Achtelfinale", 
        "Achtelfinale": "Viertelfinale", 
        "Viertelfinale": "Halbfinale", 
        "Halbfinale": "Finale",
        "Finale": "Sieger"
    }
    st.session_state.current_round_teams = {"teams_in_remaining_round": st.session_state.current_winners}
    st.session_state.current_matches = None
    st.session_state.current_winners = None
    st.session_state.round_name = round_map.get(st.session_state.round_name, "Sieger")

# Seed the knockout stage using the top 32 qualifying simulation results
if st.session_state.current_round_teams is None and st.session_state.opta_results is not None:
    top_32 = sorted(st.session_state.opta_results.items(), key=lambda x: x[1]["advance"], reverse=True)[:32]
    st.session_state.current_round_teams = {"teams_in_remaining_round": [t[0] for t in top_32]}

if st.session_state.round_name == "Sieger":
    st.balloons()
    st.success("🏁 The Tournament simulation is complete!")
    if st.button("🔄 Reset Tournament Simulation"):
        st.session_state.opta_results = None
        st.session_state.round_name = "Sechzehntelfinale"
        st.session_state.current_round_teams = None
        st.session_state.current_matches = None
        st.session_state.current_winners = None
        st.rerun()
else:
    if st.session_state.current_matches is None:
        st.info(f"Ready to simulate the matches for the {st.session_state.round_name}.")
        st.button(f"🔮 {st.session_state.round_name} ausspielen", on_click=run_match_sim)
    else:
        # Layout matches dynamically into columns
        cols = st.columns(2)
        for idx, (match, winner) in enumerate(zip(st.session_state.current_matches, st.session_state.current_winners)):
            with cols[idx % 2].container(border=True):
                st.write(f"**⚽ Match {idx+1}: {match['team_A']} vs. {match['team_B']}**")
                st.markdown(f"**Predicted Winner:** `{winner}`")
        
        st.markdown("---")
        if st.session_state.round_name == "Finale" and st.session_state.current_winners:
            st.balloons()
            st.success(f"🏆 DER WELTMEISTER IST: {st.session_state.current_winners[0].upper()}")
            st.button("🏁 Complete Simulation", on_click=advance_round)
        else:
            st.button("➡️ Proceed to Next Round", on_click=advance_round)