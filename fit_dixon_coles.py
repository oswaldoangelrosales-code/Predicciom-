import sys
import pandas as pd
from penaltyblog.models import DixonColesGoalModel

url = "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"
df = pd.read_csv(url)
df["date"] = pd.to_datetime(df["date"])

official_tournaments = [
    "FIFA World Cup",
    "FIFA World Cup qualification",
    "UEFA Euro",
    "UEFA Euro qualification",
    "UEFA Nations League",
    "Copa América",
    "Gold Cup",
    "African Cup of Nations",
    "African Cup of Nations qualification",
    "AFC Asian Cup",
    "AFC Asian Cup qualification",
    "CONCACAF Nations League",
    "CONMEBOL–UEFA Cup of Champions"
]

df = df[df["date"] >= "2024-01-01"]
df = df[df["tournament"].isin(official_tournaments)]
print("Partidos usados:", len(df))
print(df["tournament"].value_counts())

# select and rename columns
data = df[["home_team", "away_team", "home_score", "away_score"]].copy()
data.columns = [
    "home_team",
    "away_team",
    "home_goals",
    "away_goals",
]

data = data.dropna()
data["home_goals"] = data["home_goals"].astype(int)
data["away_goals"] = data["away_goals"].astype(int)
data = data[(data["home_goals"] >= 0) & (data["away_goals"] >= 0)]

import numpy as np

# convert to writable numpy arrays to avoid read-only buffer issues in C extensions
home_goals = np.asarray(data["home_goals"].to_numpy()).astype(int).copy()
away_goals = np.asarray(data["away_goals"].to_numpy()).astype(int).copy()
home_team = np.asarray(data["home_team"].to_numpy(dtype=object)).copy()
away_team = np.asarray(data["away_team"].to_numpy(dtype=object)).copy()

model = DixonColesGoalModel(
    home_goals,
    away_goals,
    home_team,
    away_team,
)

print("Fitting Dixon–Coles model (this may take a moment)...")
model.fit()
print("Fit completed.")

# Print useful attributes if available
for attr in ("params", "theta", "attack", "defence", "home_adv", "rho"):
    if hasattr(model, attr):
        print(f"{attr}:", getattr(model, attr))

# set home advantage to 0 explicitly
# Try setting in both public and internal storage (depending on implementation)
try:
    if hasattr(model, "params") and isinstance(model.params, dict):
        model.params["home_advantage"] = 0.0
except Exception:
    pass
try:
    if hasattr(model, "_params") and isinstance(model._params, dict):
        model._params["home_advantage"] = 0.0
except Exception:
    pass

print("home_advantage (model.params):", model.params.get("home_advantage") if hasattr(model, "params") and isinstance(model.params, dict) else None)
print("home_advantage (model._params):", model._params.get("home_advantage") if hasattr(model, "_params") and isinstance(model._params, dict) else None)

# fallback: show keys in model's __dict__
print("model.__dict__ keys:", sorted(k for k in model.__dict__.keys()))

# Try using the model's predict method for a sample match; fallback if teams missing
home = sys.argv[1]
away = sys.argv[2]
print(f"\nPredicting match: {home} vs {away}")
pred = None
try:
    # Some implementations expect team names present in the fitted teams
    pred = model.predict(home_team=home, away_team=away)
    print("model.predict output:", pred)
except Exception as e:
    print("model.predict failed:", repr(e))
    # Fallback: compute expected goals using mean attack/defence when teams are unknown
    params = getattr(model, "params", {}) or getattr(model, "_params", {})
    import numpy as _np

    # helpers to get team params or use mean
    attack_vals = [v for k, v in params.items() if k.startswith("attack_")]
    defence_vals = [v for k, v in params.items() if k.startswith("defence_")]
    mean_attack = float(_np.mean(attack_vals)) if attack_vals else 1.0
    mean_defence = float(_np.mean(defence_vals)) if defence_vals else 0.0

    def get_param(prefix, team_name, mean_val):
        return float(params.get(f"{prefix}_{team_name}", mean_val))

    a_home = get_param("attack", home, mean_attack)
    d_home = get_param("defence", home, mean_defence)
    a_away = get_param("attack", away, mean_attack)
    d_away = get_param("defence", away, mean_defence)

    home_adv = float(params.get("home_advantage", 0.0))

    lambda_home = _np.exp(home_adv + a_home + d_away)
    lambda_away = _np.exp(a_away + d_home)

    pred = {"exp_goals_home": float(lambda_home), "exp_goals_away": float(lambda_away)}
    print("Fallback predicted expected goals:", pred)

# Compute score probability grid (use model's method if available, otherwise Poisson product)
from scipy.stats import poisson
import numpy as np

if hasattr(pred, "home_goal_expectation"):
    lambda_h = pred.home_goal_expectation
    lambda_a = pred.away_goal_expectation
else:
    lambda_h = pred["exp_goals_home"]
    lambda_a = pred["exp_goals_away"]
# Ajuste manual de fuerza por selección
team_strength = {
    "Japan": 1.25,
    "Tunisia": 0.90,
    "Netherlands": 1.20,
    "Sweden": 1.05,
    "Ecuador": 1.05,
    "Curacao": 0.75,
    "Mexico": 1.10,
    "United States": 1.15,
    "Argentina": 1.30,
    "Brazil": 1.30,
    "France": 1.30,
    "Spain": 1.25,
    "England": 1.25,
    "Portugal": 1.25,
    "Germany": 1.20,
}

home_strength = team_strength.get(home, 1.0)
away_strength = team_strength.get(away, 1.0)

lambda_h = lambda_h * home_strength
lambda_a = lambda_a * away_strength
max_goals = 8
score_probs = np.zeros((max_goals + 1, max_goals + 1))

for i in range(max_goals + 1):
    for j in range(max_goals + 1):
        score_probs[i, j] = poisson.pmf(i, lambda_h) * poisson.pmf(j, lambda_a)

home_win = score_probs[np.tril_indices(score_probs.shape[0], -1)].sum()
draw = np.trace(score_probs)
away_win = score_probs[np.triu_indices(score_probs.shape[0], 1)].sum()

print("\nProbabilidades:")
print(f"{home} gana: {home_win:.1%}")
print(f"Empate: {draw:.1%}")
print(f"{away} gana: {away_win:.1%}")
# Ambos anotan (BTTS)
btts_yes = score_probs[1:, 1:].sum()
btts_no = 1 - btts_yes

print("\nAmbos anotan:")
print(f"Sí: {btts_yes*100:.1f}%")
print(f"No: {btts_no*100:.1f}%")

# Over / Under 2.5 goles
over25 = 0
for i in range(score_probs.shape[0]):
    for j in range(score_probs.shape[1]):
        if i + j >= 3:
            over25 += score_probs[i, j]

under25 = 1 - over25

print("\nMás/Menos 2.5 goles:")
print(f"Over 2.5: {over25*100:.1f}%")
print(f"Under 2.5: {under25*100:.1f}%")
# Doble oportunidad
one_x = home_win + draw
x_two = draw + away_win
one_two = home_win + away_win

print("\nDoble oportunidad:")
print(f"1X ({home} o Empate): {one_x*100:.1f}%")
print(f"X2 (Empate o {away}): {x_two*100:.1f}%")
print(f"12 ({home} o {away}): {one_two*100:.1f}%")
# Over / Under 1.5 y 3.5 goles
over15 = 0
over35 = 0

for i in range(score_probs.shape[0]):
    for j in range(score_probs.shape[1]):
        total_goals = i + j

        if total_goals >= 2:
            over15 += score_probs[i, j]

        if total_goals >= 4:
            over35 += score_probs[i, j]

under15 = 1 - over15
under35 = 1 - over35

print("\nMás/Menos 1.5 goles:")
print(f"Over 1.5: {over15*100:.1f}%")
print(f"Under 1.5: {under15*100:.1f}%")

print("\nMás/Menos 3.5 goles:")
print(f"Over 3.5: {over35*100:.1f}%")
print(f"Under 3.5: {under35*100:.1f}%")
flat = []
for i in range(score_probs.shape[0]):
    for j in range(score_probs.shape[1]):
        flat.append((score_probs[i, j], i, j))

flat.sort(reverse=True)

print("\nMarcadores más probables:")
for prob, home_goals, away_goals in flat[:5]:
    print(f"{home_goals}-{away_goals}: {prob:.1%}")
    import matplotlib.pyplot as plt

labels = [f"{home} gana", "Empate", f"{away} gana"]
values = [home_win * 100, draw * 100, away_win * 100]

plt.figure()
plt.bar(labels, values)
plt.title(f"{home} vs {away}")
plt.ylabel("Probabilidad %")
plt.savefig("prediccion.png", bbox_inches="tight")
print("Grafica guardada en prediccion.png")
plt.figure(figsize=(7, 5))
plt.imshow(score_probs * 100)
plt.colorbar(label="Probabilidad %")
plt.title(f"Mapa de calor: {home} vs {away}")
plt.xlabel(f"Goles {away}")
plt.ylabel(f"Goles {home}")

for i in range(score_probs.shape[0]):
    for j in range(score_probs.shape[1]):
        plt.text(j, i, f"{score_probs[i,j]*100:.1f}%", ha="center", va="center")

plt.savefig("heatmap_marcadores.png", bbox_inches="tight")
print("Heatmap guardado en heatmap_marcadores.png")
# print("\nScore probability grid (rows=home goals, cols=away goals):\n", score_probs)
