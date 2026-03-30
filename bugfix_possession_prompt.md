# Bug Fix Prompt — Possession % y Ball Assignment

Hay dos bugs concretos a corregir. Nada más. No refactorices nada que no esté mencionado.

---

## Bug 1 — ByteTrack procesado dos veces: causa el % de posesión incorrecto

### Qué pasa

En `main.py`, la calibración de colores de equipo hace esto:

```python
# Línea 30
first_tracks = tracker.get_object_tracks(first_frames, read_from_stub=False)
```

Internamente, `get_object_tracks` llama a `self.tracker.update_with_detections(...)` para cada uno de los ~500 frames. Eso avanza el estado interno de ByteTrack (confirmed tracks, lost tracks, track IDs).

Después, el loop principal abre `frame_stream()` desde el frame 0 y vuelve a procesar los mismos 500 frames:

```python
# Línea 57 — dentro del loop
window_tracks = tracker.get_object_tracks(window_frames, read_from_stub=False)
```

ByteTrack ya tiene estado de esos 500 frames. Al procesarlos de nuevo, asigna IDs distintos a los mismos jugadores. El cache de `player_team_dict` en `TeamAssigner` tiene entradas de la calibración con los IDs "viejos" — los nuevos IDs no matchean → muchos jugadores no tienen equipo asignado → la posesión defaultea a equipo 1 en cada frame → % incorrecto.

### Fix

En `main.py`, agregar UNA línea después de `assign_team_color` para resetear ByteTrack:

```python
team_assigner.assign_team_color(first_frames[0], first_tracks['players'][0])
tracker.tracker = sv.ByteTrack()   # ← AGREGAR ESTA LÍNEA
print("Team colors calibrated.")
```

Esto resetea el estado interno del tracker para que el loop principal empiece limpio desde frame 0.

También hay que limpiar el cache de `player_team_dict` del `TeamAssigner`, porque tiene entradas de los IDs "sucios" de la calibración:

```python
team_assigner.assign_team_color(first_frames[0], first_tracks['players'][0])
tracker.tracker = sv.ByteTrack()          # reset ByteTrack
team_assigner.player_team_dict = {}       # limpiar cache de IDs viejos
print("Team colors calibrated.")
```

Esas dos líneas van juntas, justo después de `assign_team_color`. No tocar nada más en esa sección.

---

## Bug 2 — Sin histéresis en la asignación de pelota: causa el flickering de posesión

### Qué pasa

`PlayerBallAssigner.assign_ball_to_player` en cada frame asigna la pelota al jugador más cercano sin memoria del frame anterior.

Ejemplo concreto del problema:
- Frame 100: jugador A está a 62px, jugador B a 68px → A tiene la pelota ✓
- Frame 101: jugador A está a 64px, jugador B a 63px (se cruzaron por 1px) → cambia a B ✗
- Frame 102: vuelve a A → flickering visible en el video

Además, si la pelota está interpolada (posición estimada, no detectada), el error de posición puede ser de varios píxeles, generando cambios espurios.

### Fix

Reemplazar la clase `PlayerBallAssigner` en `player_ball_assigner/player_ball_assigner.py` con esta versión que agrega histéresis:

```python
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))
from utils import get_center_of_bbox, measure_distance


class PlayerBallAssigner:
    def __init__(self):
        self.max_player_ball_distance = 70
        self._last_assigned = -1  # estado para histéresis

    def assign_ball_to_player(self, players, ball_bbox):
        ball_position = get_center_of_bbox(ball_bbox)

        minimum_distance = float('inf')
        assigned_player = -1

        for player_id, player in players.items():
            player_bbox = player['bbox']
            distance_left = measure_distance((player_bbox[0], player_bbox[-1]), ball_position)
            distance_right = measure_distance((player_bbox[2], player_bbox[-1]), ball_position)
            distance = min(distance_left, distance_right)

            if distance < self.max_player_ball_distance:
                if distance < minimum_distance:
                    minimum_distance = distance
                    assigned_player = player_id

        # Histéresis: el poseedor actual mantiene la pelota a menos que
        # otro jugador esté al menos un 20% más cerca.
        # Esto elimina el flickering cuando dos jugadores están a distancias similares.
        if self._last_assigned != -1 and self._last_assigned in players:
            last_bbox = players[self._last_assigned]['bbox']
            last_dist = min(
                measure_distance((last_bbox[0], last_bbox[-1]), ball_position),
                measure_distance((last_bbox[2], last_bbox[-1]), ball_position)
            )
            if last_dist < self.max_player_ball_distance:
                # Solo cambiar de poseedor si el nuevo está >20% más cerca
                if assigned_player != self._last_assigned:
                    if minimum_distance > last_dist * 0.80:
                        assigned_player = self._last_assigned

        if assigned_player != -1:
            self._last_assigned = assigned_player

        return assigned_player
```

El único cambio de lógica es el bloque de histéresis al final. El resto del método es idéntico al original.

### Nota sobre el estado entre ventanas

`player_assigner` se instancia UNA sola vez fuera del loop (línea 43 de `main.py`), así que `_last_assigned` persiste correctamente entre ventanas. No hay que cambiar nada en `main.py` para esto.

---

## Verificación después de aplicar los fixes

1. Correr `main.py` sobre el clip de prueba corto.
2. Verificar que el overlay "Team 1 Ball Control" y "Team 2 Ball Control" muestren porcentajes que sumen ~100% y que no sean 90%/10% o similares extremos desde el primer segundo.
3. Verificar que el triángulo rojo sobre el jugador con pelota no cambie de jugador frame a frame cuando la pelota está quieta o en movimiento lento.
4. Confirmar que no hay nuevos errores de importación o KeyError.

## Lo que NO hay que tocar

- `tracker.py` — no modificar
- `team_assigner.py` — no modificar
- `camara_movement_estimator.py` — no modificar
- `speed_and_distance_estimator.py` — no modificar
- `view_transformer.py` — no modificar
- Cualquier otra parte de `main.py` fuera de las dos líneas indicadas
