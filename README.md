# parcial_desde_cero - navegacion autonoma DARwIn-OP (Webots R2023b)

## Estado por modulos
| Modulo | Estado |
|---|---|
| 1. Locomocion C++ (`controllers/locomotion`) | Escrito. Sintaxis comprobada con cabeceras FALSAS. **No compilado con Webots, no probado.** |
| 2. Entorno + recompensas + mundo (`race_supervisor`, `tools/generate_world.py`) | Escrito. Logica Python probada con un simulador cinematico falso (`tests/test_env_mock.py`). **Nada probado en Webots.** |
| 3. Agentes (Taller 2, PPO, DQN, A2C) | Pendiente. Interfaz en `agents/base.py`; solo hay baselines `random` y `heuristic`. |
| 4. Automatizacion de experimentos | Pendiente. |
| 5. Resultados / informe IEEE | Pendiente. |

## Arquitectura
Supervisor Python (`race_supervisor`) <-> campo `customData` de cada robot (`"seq,forward,turn"`) <-> controlador C++ `locomotion` (maquina de estados + GaitManager/MotionManager oficiales).
4 carriles identicos (ROBOT_0..3) separados por paredes: un agente por carril, los robots no chocan entre si.
Observacion (8): dist. a meta, sin/cos del rumbo, 5 rangos de "lidar virtual" calculados por el Supervisor desde la geometria (no son sensores del robot).
Acciones (5): forward, fwd_left, fwd_right, turn_left, turn_right. Escudo de seguridad (safe learning): con obstaculo de frente solo se permite girar; se cuenta como "intervencion".

## Uso (SOLO cuando lo autorices/ejecutes)
1. `python tools/generate_world.py`  -> escribe `worlds/race.wbt` (ya incluido generado).
2. Compilar `controllers/locomotion` (Webots: Build o `make` en esa carpeta).
3. Abrir `worlds/race.wbt`. Resultados en `results/<run>/episodes.csv`.
Prueba sin Webots: `python tests/test_env_mock.py`.

## Incertidumbres que SOLO se resuelven probando
- Namespace `managers::` y firmas de los gestores (ver respuesta anterior).
- `config.ini` escrito de memoria: comparar con `controllers/walk/config.ini` oficial.
- `WorldInfo` sin `contactProperties`: si el robot resbala, copiar el de `darwin-op/worlds/darwin-op.wbt`.
- Que `Darwin-op` exponga `customData`, y que su eje de avance sea +x en ENU (yaw = atan2(R10, R00)).
- Reinicio de episodio con `resetPhysics()` + `restartController()` (articulaciones tras una caida). Alternativa: `simulationReset()`.
- Altura inicial 0.33 m, umbral de caida (`up_z < 0.7`), paso de decision 512 ms: valores iniciales sin calibrar.
- En el doble cinematico el heuristico llego a la meta en 6/8 carriles; los otros 2 expiraron (no investigado). Con marcha real puede variar.
