# 📚 Code Reference — Infra & Baselines

Quick reference for all functions and classes in the infra layer.

---

## `src/env_utils.py`

### `make_env()`

```python
make_env(
    seed: int = 0,
    render_mode: str | None = None,
    continuous: bool = False,
    record_video: bool = False,
    video_folder: str | None = None,
) -> gym.Env
```

Crée et configure un `LunarLander-v3`. Point d'entrée unique pour l'environnement — personne ne touche à gymnasium directement.

| Param | Default | Description |
|---|---|---|
| `seed` | `0` | Graine pour la reproductibilité |
| `render_mode` | `None` | `"rgb_array"` pour vidéo, `"human"` pour affichage, `None` pour headless |
| `continuous` | `False` | Espace d'action continu si `True` |
| `record_video` | `False` | Enregistre les épisodes en `.mp4` |
| `video_folder` | `None` | Dossier de sortie (requis si `record_video=True`) |

```python
# Exemples
env = make_env(seed=0)
env = make_env(seed=0, record_video=True, video_folder="videos/test")
```

---

### `get_termination_reason()`

```python
get_termination_reason(
    obs: np.ndarray,
    terminated: bool,
    truncated: bool,
    info: dict,
) -> str
```

Détecte pourquoi un épisode s'est terminé. À appeler après chaque `env.step()` qui retourne `terminated=True` ou `truncated=True`.

| Return | Condition |
|---|---|
| `"landing"` | Les deux jambes touchent, vitesse faible, angle droit |
| `"crash"` | La coque a touché le sol |
| `"out_of_view"` | Le module est sorti du cadre visible |
| `"sleep"` | Limite de steps atteinte (`truncated=True`) |
| `"running"` | Épisode encore en cours |

```python
reason = get_termination_reason(obs, terminated, truncated, info)
```

---

## `src/logger.py`

### `EpisodeLogger`

```python
EpisodeLogger(
    csv_path: str,
    run_name: str | None = None,
    verbose: bool = True,
)
```

Enregistre les métriques par épisode dans un CSV et affiche un résumé en temps réel dans le terminal.

**Colonnes CSV :** `episode`, `score`, `length`, `terminated`, `truncated`, `reason`, `timestamp` + colonnes custom via `extra={}`

---

#### `.log_episode()`

```python
logger.log_episode(
    score: float,
    length: int,
    terminated: bool,
    truncated: bool,
    reason: str,
    extra: dict | None = None,
)
```

À appeler à chaque fin d'épisode. `extra` permet d'ajouter des colonnes custom (ex: `{"epsilon": 0.3, "loss": 0.02}`).

---

#### `.print_summary()`

```python
logger.print_summary(last_n: int = 100)
```

Affiche mean, std, min, max des scores + répartition des causes de fin sur les `last_n` derniers épisodes.

---

#### `.get_recent_mean()`

```python
logger.get_recent_mean(n: int = 100) -> float
```

Retourne la moyenne des `n` derniers scores. Utile pour vérifier la progression pendant l'entraînement.

---

#### `.is_solved()`

```python
logger.is_solved(threshold: float = 200.0, window: int = 100) -> bool
```

Retourne `True` si la moyenne sur les `window` derniers épisodes dépasse `threshold`. C'est le critère de succès du projet.

---

## `src/policies.py`

### `RandomPolicy`

```python
RandomPolicy(action_space: gym.spaces.Discrete, seed: int = 0)
```

Borne basse. Sélectionne une action au hasard à chaque step. Sert uniquement de point de comparaison.

```python
policy = RandomPolicy(env.action_space, seed=0)
action = policy.select_action(obs)  # → int entre 0 et 3
```

---

### `HeuristicPolicy`

```python
HeuristicPolicy()
```

Contrôleur à règles manuelles, sans apprentissage. Doit battre la politique aléatoire.

**Priorités dans l'ordre :**
1. Corriger l'angle si `|theta| > 0.2`
2. Freiner la chute si `vy < -0.3`
3. Corriger le drift horizontal si `|vx| > 0.3`
4. Ne rien faire

```python
policy = HeuristicPolicy()
action = policy.select_action(obs)
```

---

## Interface commune des policies

Toutes les policies exposent la même interface — le DQN aussi devra la respecter.

```python
policy.select_action(obs: np.ndarray) -> int
policy.reset()   # appelé au début de chaque épisode
```