MATRIZ DE 20 EXPERIMENTOS
========================

No se ejecutaron simulaciones al preparar estos archivos.
Cada experimento tiene un JSON y un mundo independientes.

Para ejecutar uno desde PowerShell:
$env:SCENARIO = "RUTA_COMPLETA_AL_JSON"
& "C:\Program Files\Webots\webots.exe" "RUTA_COMPLETA_AL_WBT"

Experimentos:
01 | exp01_start_x_04 | eval | 30 episodios
02 | exp02_start_y_narrow | eval | 30 episodios
03 | exp03_start_y_medium | eval | 30 episodios
04 | exp04_start_y_wide | eval | 30 episodios
05 | exp05_start_x_08 | eval | 30 episodios
06 | exp06_goal_x_60 | eval | 30 episodios
07 | exp07_goal_x_65 | eval | 30 episodios
08 | exp08_goal_x_70 | eval | 30 episodios
09 | exp09_goal_x_73 | eval | 30 episodios
10 | exp10_goal_x_range | eval | 30 episodios
11 | exp11_lane_length_85 | eval | 30 episodios
12 | exp12_lane_width_16 | eval | 30 episodios
13 | exp13_lane_width_24 | eval | 30 episodios
14 | exp14_lane_length_90 | eval | 30 episodios
15 | exp15_lane_spacing_35 | eval | 30 episodios
16 | exp16_lr_reference | train | 100 episodios
17 | exp17_lr_low_a2c_ppo | train | 100 episodios
18 | exp18_lr_high_a2c_ppo | train | 100 episodios
19 | exp19_ppo_clip_01 | train | 100 episodios
20 | exp20_ppo_clip_03 | train | 100 episodios
