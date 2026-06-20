import pandas as pd
from penaltyblog.models import DixonColesGoalModel

url = "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"
df = pd.read_csv(url)


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
home = "Netherlands"
away = "Sweden"
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
score_probs = None
if hasattr(pred, "correct_score_grid"):
    try:
        score_probs = pred.correct_score_grid()
    except Exception as _e:
        score_probs = None

if score_probs is None:
    from scipy.stats import poisson
    import numpy as _np

    lambda_h = pred.home_goal_expectation
    lambda_a = pred.away_goal_expectation
    if lambda_h is None or lambda_a is None:
        raise ValueError("No expected goals available to compute score probabilities")

    max_goals = 6
    grid = _np.zeros((max_goals + 1, max_goals + 1))
    for i in range(max_goals + 1):
        for j in range(max_goals + 1):
            grid[i, j] = poisson.pmf(i, lambda_h) * poisson.pmf(j, lambda_a)

    score_probs = grid
home_win = score_probs[np.tril_indices(score_probs.shape[0], -1)].sum()
draw = np.trace(score_probs)
away_win = score_probs[np.triu_indices(score_probs.shape[0], 1)].sum()

print("\nProbabilidades:")
print(f"{home} gana: {home_win:.1%}")
print(f"Empate: {draw:.1%}")
print(f"{away} gana: {away_win:.1%}")
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
print("\nScore probability grid (rows=home goals, cols=away goals):\n", score_probs)
