import os
import random
import json
import time
import re
import math
from openai import OpenAI
from dotenv import load_dotenv
from transformers import pipeline
from duckduckgo_search import DDGS

# 1. OpenRouter Setup
load_dotenv()
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY")
)

# Globale Konfigurationen
CACHE_FILE = "team_ratings_cache.json"
sentiment_pipeline = None

# Realistische FIFA-Basisstärken als math-Fallback, falls die API gestresst ist
BASE_TEAM_RATINGS = {
    "Frankreich": 87, "Brasilien": 86, "England": 86, "Argentinien": 85, "Spanien": 85,
    "Portugal": 84, "Niederlande": 83, "Deutschland": 83, "Belgien": 82, "Kroatien": 81,
    "Uruguay": 81, "Schweiz": 79, "Marokko": 79, "Kolumbien": 79, "Japan": 78,
    "Senegal": 77, "USA": 77, "Schweden": 77, "Österreich": 76, "Ukraine": 76,
    "Türkei": 76, "Südkorea": 76, "Australien": 75, "Ecuador": 75, "Tschechien": 75,
    "Algerien": 75, "Ägypten": 75, "Tunesien": 74, "Mexiko": 74, "Paraguay": 74,
    "Bosnien-Herzegowina": 73, "Ghana": 73, "Saudi-Arabien": 72, "Irak": 71, "Norwegen": 78, 
    "Südafrika": 71, "Katar": 70, "DR Kongo": 70, "Usbekistan": 69, "Panama": 69,
    "Jordanien": 68, "Kap Verde": 68, "Haiti": 66, "Kanada": 74, "Neuseeland": 66,
    "Curacao": 64
}

def load_sentiment_model():
    """Lädt das Hugging Face Modell nur bei Bedarf (Lazy Loading)."""
    global sentiment_pipeline
    if sentiment_pipeline is None:
        print("🤗 Loading English Hugging Face Sentiment Model (DistilBERT)...")
        sentiment_pipeline = pipeline(
            "text-classification", 
            model="distilbert-base-uncased-finetuned-sst-2-english"
        )
    return sentiment_pipeline

def parse_groups_from_file(filename="wm_gruppen.md"):
    groups = {}
    if not os.path.exists(filename):
        print(f"❌ Datei {filename} nicht gefunden! Bitte erstelle sie im Projektordner.")
        return groups
    with open(filename, "r", encoding="utf-8") as f:
        content = f.read()
    matches = re.findall(r"(Gruppe\s+[A-L]):\s*(.+)", content)
    for group_name, teams_str in matches:
        groups[group_name] = [t.strip() for t in teams_str.split(",")]
    return groups

def fetch_live_news_headlines(team_name, max_results=4):
    print(f"🔍 Searching live news for {team_name}...")
    query = f"{team_name} national football team news"
    headlines = []
    try:
        with DDGS() as ddgs:
            results = ddgs.news(query, max_results=max_results)
            for r in results:
                headlines.append(r["title"])
        if not headlines:
            headlines = [f"{team_name} football team prepares for international tournament."]
        return headlines
    except Exception:
        return [f"{team_name} football team updates and squad analysis."]

def analyze_team_sentiment(headlines):
    try:
        model = load_sentiment_model()
        results = model(headlines)
        score_mapping = {"POSITIVE": 1.0, "NEGATIVE": -1.0}
        total_score = 0.0
        for res in results:
            total_score += score_mapping[res["label"]] * res["score"]
        return round(total_score / len(headlines), 2)
    except Exception:
        return 0.0

def generate_ratings_with_gemini(all_team_sentiments):
    """Schickt ALLE 48 Teams in einem einzigen, großen Request zu Gemini."""
    print(f"🧬 Gemini berechnet die Ratings für alle {len(all_team_sentiments)} Teams im Single-Task-Verfahren...")
    
    sentiment_context = "".join([f"- {team}: Sentiment Score {score}\n" for team, score in all_team_sentiments.items()])

    # Hier zwingen wir das Modell, exakt deine übergebenen deutschen Schreibweisen beizubehalten!
    system_instruction = (
        "You are a master sports data analyst. Assign current football team ratings from 1 to 100.\n"
        "Dynamically adjust their metrics based on the provided live media sentiment.\n\n"
        "CRITICAL: You MUST use the exact team names provided in the list as the JSON keys. "
        "Do not translate them to English (e.g., if provided 'Deutschland', use 'Deutschland', not 'Germany').\n\n"
        "You MUST return a JSON object matching this schema exactly for ALL input teams:\n"
        "{\n"
        '  "Deutschland": {"squad_quality": 83, "recent_form": 82, "tactics": 84}\n'
        "}\n"
        "Do not output any plain text before or after the JSON!"
    )
    
    try:
        response = client.chat.completions.create(
            model="google/gemini-2.5-flash",
            messages=[
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": f"Here are all tournament teams and their sentiments:\n{sentiment_context}"}
            ],
            response_format={"type": "json_object"},
            timeout=30
        )       
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"⚠️ Gemini-Fehler oder Limit erreicht ({e}). Nutze mathematisches Lokal-Fallback basierend auf Sentiment.")
        fallback_output = {}
        for team in all_team_sentiments:
            base = BASE_TEAM_RATINGS.get(team, 72)
            sentiment_modifier = int(all_team_sentiments[team] * 5)
            final_rating = max(50, min(99, base + sentiment_modifier))
            fallback_output[team] = {"squad_quality": final_rating, "recent_form": final_rating, "tactics": final_rating}
        return fallback_output

def get_all_team_ratings(all_teams):
    if os.path.exists(CACHE_FILE):
        print(f"📂 Gefundene Ratings im lokalen Cache ({CACHE_FILE}) geladen!")
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            cache_data = json.load(f)
            simulated_sentiments = {team: 0.1 for team in all_teams}
            return cache_data, simulated_sentiments
            
    print("🌐 Kein Cache gefunden. Starte globale Live-Recherche...")
    sentiments = {}
    for team in all_teams:
        headlines = fetch_live_news_headlines(team)
        sentiments[team] = analyze_team_sentiment(headlines)
        time.sleep(0.1) # Leichtes Delay für die News-Suche
        
    # Hier der Game Changer: Nur noch EIN einziger API-Call für alle Teams!
    gemini_output = generate_ratings_with_gemini(sentiments)
    
    final_ratings = {}
    for team, metrics in gemini_output.items():
        avg_rating = (metrics["squad_quality"] + metrics["recent_form"] + metrics["tactics"]) / 3
        final_ratings[team] = round(avg_rating, 1)
        
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(final_ratings, f, indent=2, ensure_ascii=False)
        
    print(f"💾 Alle 48 Team-Ratings erfolgreich in '{CACHE_FILE}' gespeichert!")
    return final_ratings, sentiments

def calculate_probabilities(rating_a, rating_b):
    diff = (rating_a - rating_b) / 10.0
    exp_diff = math.exp(diff)
    prob_a = exp_diff / (1 + exp_diff)
    prob_draw = 0.24
    return {
        "prob_A": round(prob_a * (1.0 - prob_draw), 3),
        "prob_draw": prob_draw,
        "prob_B": round((1.0 - prob_a) * (1.0 - prob_draw), 3)
    }

def simulate_single_group(teams, ratings, num_simulations=1000):
    total_points = {team: 0 for team in teams}
    pairings = [(teams[i], teams[j]) for i in range(len(teams)) for j in range(i + 1, len(teams))]
    
    match_probs = {(t_a, t_b): calculate_probabilities(ratings.get(t_a, 72.0), ratings.get(t_b, 72.0)) for t_a, t_b in pairings}
    
    for _ in range(num_simulations):
        for team_a, team_b in pairings:
            probs = match_probs[(team_a, team_b)]
            rand = random.random()
            if rand < probs["prob_A"]:
                total_points[team_a] += 3
            elif rand < (probs["prob_A"] + probs["prob_draw"]):
                total_points[team_a] += 1
                total_points[team_b] += 1
            else:
                total_points[team_b] += 3
                
    return {team: round(pts / num_simulations, 2) for team, pts in total_points.items()}

def simulate_group_stage_opta(tournament_groups, ratings, num_simulations=1000):
    """
    Simuliert die komplette Gruppenphase N-mal und gibt pro Team
    Aufstiegswahrscheinlichkeit + Schnitt-Punkte zurück.
    """
    all_teams = [team for teams in tournament_groups.values() for team in teams]
    advancement_counts = {team: 0 for team in all_teams}
    total_points = {team: 0.0 for team in all_teams}

    for _ in range(num_simulations):
        third_place_pool = [] 

        for group_name, teams in tournament_groups.items():
            pairings = [(teams[i], teams[j]) for i in range(len(teams)) for j in range(i + 1, len(teams))]
            points = {team: 0 for team in teams}

            for team_a, team_b in pairings:
                probs = calculate_probabilities(ratings.get(team_a, 72.0), ratings.get(team_b, 72.0))
                rand = random.random()
                if rand < probs["prob_A"]:
                    points[team_a] += 3
                elif rand < probs["prob_A"] + probs["prob_draw"]:
                    points[team_a] += 1
                    points[team_b] += 1
                else:
                    points[team_b] += 3

            for team, pts in points.items():
                total_points[team] += pts

            sorted_group = sorted(points.items(), key=lambda x: x[1], reverse=True)

            # Top 2 advance
            advancement_counts[sorted_group[0][0]] += 1
            advancement_counts[sorted_group[1][0]] += 1
            third_place_pool.append((sorted_group[2][0], sorted_group[2][1]))

        # Handle 8 best 3rd-placed teams
        sorted_thirds = sorted(third_place_pool, key=lambda x: x[1], reverse=True)[:8]
        for team, _ in sorted_thirds:
            advancement_counts[team] += 1
            
    # CRITICAL FIX: Return the aggregated calculations!
    return {
        team: {
            "advance": round(advancement_counts[team] / num_simulations, 3),
            "avg_points": round(total_points[team] / num_simulations, 2)
        }
        for team in all_teams
    }

def load_context_file(filename):
    if os.path.exists(filename):
        with open(filename, "r", encoding="utf-8") as f:
            return f.read()
    return "Keine spezifischen Turnierregeln gefunden."

def run_genai_tournament_director(qualified_teams, team_sentiments):
    print("\n🧠 Der GenAI Tournament Director analysiert das Reglement und berechnet die K.-o.-Paarungen...")
    
    reglement_context = load_context_file("turnierbaum.md")
    teams_context = json.dumps(qualified_teams, indent=2, ensure_ascii=False)
    sentiment_context = json.dumps(team_sentiments, indent=2, ensure_ascii=False)
    
    system_instruction = (
        "You are the official FIFA Tournament Director and a precise data model.\n"
        "Your task is to create the upcoming knockout matches strictly following the tournament tree structure in turnierbaum.md.\n"
        "You MUST return a JSON object matching this schema exactly:\n"
        "{\n"
        '  "matches": [\n'
        '    {"match_id": 1, "team_A": "Germany", "team_B": "Scotland", "sentiment_advantage": "Germany", "prediction_reasoning": "Short sentence."}\n'
        '  ]\n'
        "}\n"
        "Do not output any plain text before or after the JSON!"
    )
    
    user_prompt = f"Tree:\n{reglement_context}\nTeams:\n{teams_context}\nSentiment:\n{sentiment_context}"
    
    try:
        response = client.chat.completions.create(
            model="google/gemini-2.5-flash",
            messages=[
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": user_prompt}
            ],
            response_format={"type": "json_object"},
            timeout=15
        )
        return json.loads(response.choices[0].message.content)
    except Exception:
        print("⚠️ API-Limit beim K.-o.-Baum erreicht! Algorithmic Fallback übernimmt regelkonform...")
        # Falls die API abkackt, paart Python die Teams algorithmisch anstatt abzubrechen!
        matches = []
        if "group_winners" in qualified_teams:
            gw = list(qualified_teams["group_winners"].values())
            gr = list(qualified_teams["group_runners_up"].values())
            bt = list(qualified_teams["best_eight_third_placed_teams"])
            for i in range(8):
                matches.append({"match_id": i+1, "team_A": gw[i], "team_B": bt[i], "prediction_reasoning": "Group winner vs Best Third fallback."})
            for i in range(4):
                matches.append({"match_id": i+9, "team_A": gw[8+i], "team_B": gr[i], "prediction_reasoning": "Group winner vs Runner up."})
            for i in range(0, 8, 2):
                matches.append({"match_id": len(matches)+1, "team_A": gr[4+i], "team_B": gr[4+i+1], "prediction_reasoning": "Runner up battle."})
        else:
            flat_teams = qualified_teams.get("teams_in_remaining_round", [])
            for i in range(0, len(flat_teams), 2):
                if i+1 < len(flat_teams):
                    matches.append({"match_id": (i//2)+1, "team_A": flat_teams[i], "team_B": flat_teams[i+1], "prediction_reasoning": "Knockout progression."})
        return {"matches": matches}

if __name__ == "__main__":
    print("=" * 60)
    print("🏆 FULL 48-TEAM WORLD CUP LIVE PIPELINE (STABLE) 🏆")
    print("=" * 60)
    
    tournament_groups = parse_groups_from_file()
    if not tournament_groups:
        exit()
        
    all_flat_teams = [team for teams in tournament_groups.values() for team in teams]
    team_ratings, active_sentiments = get_all_team_ratings(all_flat_teams)
    
    print("\n" + "="*50)
    print("📊 GRUPPENPHASEN-SIMULATION (Schnitt aus 1.000 Runs):")
    print("="*50)
    
    group_tables = {}
    for group_name, teams in tournament_groups.items():
        exp_points = simulate_single_group(teams, team_ratings)
        sorted_group = sorted(exp_points.items(), key=lambda x: x[1], reverse=True)
        group_tables[group_name] = sorted_group
        
        print(f"\n🔹 {group_name}:")
        for rank, (team, pts) in enumerate(sorted_group, 1):
            print(f"  {rank}. {team:<25} -> {pts} Pkt. (Rating: {team_ratings.get(team, 72)})")

    winners, runners_up, all_thirds = {}, {}, []
    for group_name, table in group_tables.items():
        winners[group_name] = table[0][0]
        runners_up[group_name] = table[1][0]
        all_thirds.append({"team": table[2][0], "points": table[2][1]})
        
    best_thirds = sorted(all_thirds, key=lambda x: x["points"], reverse=True)[:8]
    best_thirds_teams = [t["team"] for t in best_thirds]
    
    current_round_teams = {
        "group_winners": winners,
        "group_runners_up": runners_up,
        "best_eight_third_placed_teams": best_thirds_teams
    }
    round_name = "Sechzehntelfinale"
    
    while True:
        round_data = run_genai_tournament_director(current_round_teams, active_sentiments)
        matches = round_data.get("matches", [])
        
        if not matches:
            print("⚠️ Struktur unterbrochen. Abbruch.")
            break
            
        print("\n" + "═"*70)
        print(f"🏆 OFFIZIELLES {round_name.upper()} 🏆")
        print("═"*70)
        
        next_round_input_teams = []
        for match in matches:
            t_a, t_b = match["team_A"], match["team_B"]
            
            probs = calculate_probabilities(team_ratings.get(t_a, 72.0), team_ratings.get(t_b, 72.0))
            p_a_ko = probs["prob_A"] + (probs["prob_draw"] / 2)
            
            winner = t_a if random.random() < p_a_ko else t_b
            print(f"⚽ Spiel {match['match_id']}: {t_a:<18} vs. {t_b:<18} -> ✨ SIEGER: {winner}")
            next_round_input_teams.append(winner)
            
        if len(next_round_input_teams) == 16:
            round_name = "Achtelfinale"
            current_round_teams = {"teams_in_remaining_round": next_round_input_teams}
        elif len(next_round_input_teams) == 8:
            round_name = "Viertelfinale"
            current_round_teams = {"teams_in_remaining_round": next_round_input_teams}
        elif len(next_round_input_teams) == 4:
            round_name = "Halbfinale"
            current_round_teams = {"teams_in_remaining_round": next_round_input_teams}
        elif len(next_round_input_teams) == 2:
            round_name = "Finale"
            current_round_teams = {"teams_in_remaining_round": next_round_input_teams}
        elif len(next_round_input_teams) == 1:
            print("\n" + "👑"*35)
            print(f"🎉 DER WELTMEISTER IST: {next_round_input_teams[0].upper()} !!! 🎉")
            print("👑"*35 + "\n")
            break
        else:
            break
            
        time.sleep(1)