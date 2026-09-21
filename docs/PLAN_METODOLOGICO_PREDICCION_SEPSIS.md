# Plan metodológico de predicción de sepsis con innovaciones Koopman

Estado: especificación computacional del experimento, no resultados finales.
Fecha de verificación bibliográfica: 2026-09-17. Revisión metodológica recibida:
2026-09-21.

> **No es el manuscrito.** Este documento fija el estimando, las comparaciones,
> los gates y la ejecución reproducible. No redacta Abstract, Introduction,
> Discussion ni Conclusion.

## 1. Pregunta y decisión metodológica

La pregunta primaria es si la desviación causal de la fisiología observada
respecto de una dinámica multiorgánica no inminente aporta señal predictiva
verificable sobre `baseline + CV` para onset verdadero en las próximas 1–6 h.
La representación propuesta es una aproximación mínima EDMD/Koopman con Ridge,
NumPy y scikit-learn. No se introduce una red profunda ni una dependencia nueva
antes de demostrar que esta señal existe.

La hipótesis no es que “Koopman” mejore por ser más complejo. Es que una
innovación observada

\[
r_{j,i,t} = \frac{x_{j,i,t}-\hat x_{j,i,t\mid t^-}}{s_j}
\]

puede separar deterioro multiorgánico inesperado de nivel, tendencia simple y
proceso de medición. La predicción `\hat x` se aprende sólo con controles y con
observaciones sépticas a más de 12 h del onset. El residuo sólo existe cuando
la variable objetivo se midió realmente; no se calcula contra forward-fill.

## 2. Evidencia reconstruida y su límite

El run histórico 26224 produjo un paquete para la etiqueta persistente del
Challenge, pero quedó `PENDING_FINAL_VALIDATION` y no fue promovido. Sus números
no seleccionan modelos, comparadores, tolerancias ni expected de tests en este
experimento. `baseline + CV` es C0 porque así lo fija el diseño actual; todas
sus métricas se recalcularán para el nuevo estimando.

El job 26224 reservó 8 CPU y 32 GB; duró 26:40:55, alcanzó 17.38 GB y consumió
aproximadamente 1.58 cores promedio. El nuevo run debe perfilar antes de reservar
recursos definitivos.

## 3. Consulta reproducible y búsqueda iterativa

Se aplicó una **consulta reproducible** por título/DOI en fuentes primarias de
editoriales, proceedings oficiales, PubMed/PMC y repositorios de autores. Rango
principal: **2021–2026**. Consultas registradas:

1. `sepsis early prediction ICU distribution shift 2021..2026`;
2. `physiological dynamics sepsis prediction latent dynamics`;
3. `Koopman nonstationary multivariate time series anomaly detection`;
4. `Koopman physiological time series domain adaptation`;
5. `continuous discrete state space irregularly sampled time series`;
6. búsqueda exacta de cada DOI incluido en Referencias.

La búsqueda se iteró desde modelos generales de series temporales hacia dinámica
anómala, muestreo irregular, benchmarking ICU y transporte entre dominios. En
esta búsqueda acotada no se encontró una publicación que combine exactamente
innovaciones EDMD fold-local, onset verdadero 1–6 h, censoring explícito,
validación nested y A↔B en los datos PhysioNet 2019. Esto motiva el experimento,
pero no constituye una afirmación universal de novedad.

Se excluyen explícitamente Zahibi y Zabihi: no se usan como comparador, fuente
metodológica ni evidencia. Los estándares oficiales antiguos del Challenge y su
scorer se conservan sólo como excepción necesaria para definir el dataset y la
Utility secundaria, no como literatura de innovación reciente.

## 4. Estimando primario y cohortes

Para paciente `i` y hora `t`:

\[
Y_{i,t}=1 \Longleftrightarrow 1\leq t_{\mathrm{onset},i}-t\leq6.
\]

Reglas fail-closed:

- onset y post-onset no son horas elegibles;
- pacientes con onset izquierdo-censurado no entran al estimando primario;
- en controles se excluyen las últimas seis horas por seguimiento incompleto;
- `Challenge shifted persistent label`, onset reconstruido y `Y_{i,t}` se
  almacenan por separado;
- el análisis secundario de Utility oficial nunca se denomina “sepsis en las
  próximas seis horas”.

La solicitud inicial decía cinco horas terminales. Se corrigió a seis porque el
estimando incluye `t+1,...,t+6`; conservar `t=T-5` afirmaría ausencia de onset en
`T+1`, un instante no observado.

## 5. Representaciones preespecificadas

- **C0:** baseline heredado corregido + las seis familias CV de 8 h.
- **C1:** estado causal de los 40 predictores oficiales.
- **C2:** C1 + deltas y pendientes observadas.
- **C3:** C1 + innovaciones Koopman y energía multiorgánica.

C0 es el comparador primario; C2 controla la posibilidad de que C3 sólo replique
una pendiente. `SourceSet` nunca es predictor.

Dentro de cada training fold se eligen como máximo 20 señales entre las 34
dinámicas: cobertura observada ≥5 %, al menos 1,000 pacientes con dos mediciones,
orden por cobertura y desempate alfabético. El esquema de salida siempre conserva
las 34 columnas; señales no soportadas y transiciones no estimables son `NaN`.

## 6. Definición EDMD/Koopman

Por señal seleccionada, el estado causal anterior concatena último valor
observado y edad de observación, ambos conocidos antes de `t`. Mediana e IQR se
ajustan sólo en training. Sólo el estado predictor puede imputarse con medianas
de training.

Se comparan dos lifts preespecificados:

\[
\psi_{1}(z)=z, \qquad
\psi_{2}(z)=[z,\{z_a z_b:a\leq b\}].
\]

Para cada variable actualmente observada:

\[
\hat\beta_j=\arg\min_\beta
\sum_{(i,t)\in\mathcal N_j}
(\tilde x_{j,i,t}-\psi(z_{i,t^-})^\top\beta)^2
+\alpha\lVert\beta\rVert_2^2,
\quad \alpha=1.
\]

`\mathcal N_j` contiene controles y puntos a más de 12 h del onset. Se emiten:
innovación estandarizada por señal, energía instantánea media, media y máximo
causales de energía en 8 h, y número de innovaciones observadas.

Para acotar el costo sin seleccionar por outcome se usa una muestra determinista
máxima de 20,000 transiciones no inminentes por señal y fold, identificada por el
hash de pacientes y el nombre de señal. Ridge usa `lsqr`; la transformación se
predice en bloques de 50,000 filas. Cada lift se ajusta una sola vez por inner
fold y se reutiliza para los candidatos XGBoost del mismo fold. Los operadores
por señal se ajustan en paralelo con, como máximo, los CPU asignados y un límite
absoluto de 32 threads; BLAS interno queda en un thread durante esa región para
que procesos × threads nunca exceda la asignación.

## 7. Validación completamente nested

Cada outer fold agrupa pacientes. En cada inner fold se reajustan selección de
señales, normalización, operador, modelo XGBoost y early stopping. La selección
usa Average Precision ponderada para que cada paciente aporte el mismo peso
total. El `logloss` de XGBoost gobierna sólo el early stopping interno y no se
renombra ni se reporta como AP.

Con inner OOF se elige entre identidad y recalibración logística sobre
`logit(p)` por Brier ponderado. También se selecciona el threshold sin acceder al
outer fold. Ningún outer outcome selecciona lift, hiperparámetro, árboles,
calibrador o threshold.

Las predicciones held-out producidas por el early stopping de cada inner fold se
reutilizan para formar el inner OOF del candidato ganador. No se reentrena una
segunda copia sobre los mismos inner folds: eso conserva la independencia y
elimina fits redundantes. El número de árboles del outer fit sigue siendo la
mediana preespecificada de las iteraciones inner.

## 8. Política de alarma

- ventana útil: onset−6 h a onset−1 h;
- período refractario: 6 h;
- presupuesto: ≤0.25 falsas alarmas por paciente-día;
- threshold: máxima sensibilidad útil dentro del presupuesto en inner OOF;
- un aviso remoto, post-onset o repetido no cuenta como TP útil.

El presupuesto es un límite experimental para comparación, no una recomendación
clínica.

## 9. Métricas, inferencia y gates

Primaria: diferencia C3−C0 en AP ponderada por paciente. Se usa bootstrap
pareado por paciente, 300 repeticiones e IC 95 %. Gate: límite inferior >0.

Gates adicionales:

- sensibilidad útil C3 > C0;
- falsas alarmas C3 ≤0.25/paciente-día;
- sin deterioro de Brier respaldado por el IC pareado;
- mediana de lead time C3 no inferior a C0;
- al menos un threshold de DCA donde el límite inferior pareado C3−C0 sea
  positivo y el límite inferior de C3 no sea menor que treat-all ni treat-none.

Se reportan separadamente AP, AUROC, Brier, intercept/slope, ECE de 10 bins
iguales, Utility oficial secundaria, episodios de alarma, lead time y net benefit.
La unidad inferencial primaria es el paciente, no la fila horaria.

### 9.1 Ablaciones, semillas y balance

C0–C3 constituyen la ablación de representación y se publican juntas sin elegir
post hoc la más favorable. Además, C0 y C3 se someten a dos sensibilidades
preespecificadas que no modifican el gate primario:

- bases de semilla `20260906`, `20261007` y `20261108`, manteniendo el esquema
  outer y la configuración elegida exclusivamente en outer-train;
- pesos `equal_patient`, `equal_row` y
  `equal_patient_then_row_class`, manteniendo siempre evaluación AP/AUROC/Brier
  con igual peso total por paciente.

Para limitar compute y evitar un nuevo selection effect, estas sensibilidades
reutilizan en cada outer fold el lift, hiperparámetro y número de árboles ya
seleccionados nested en outer-train. Sólo reentrenan el modelo con la semilla o
los pesos predeclarados. Se reportan todas las configuraciones en probabilidades
raw como estabilidad de ranking; ninguna se calibra, selecciona ni reemplaza al
resultado primario. El OOF de sensibilidad almacena una sola identidad horaria y
diez columnas de probabilidad (`C0/C3 × cinco configuraciones`), evitando repetir
identificadores y targets diez veces sin perder trazabilidad.

## 10. Transporte y escalamiento condicionado

Se ejecutan **A → B** y **B → A**. Selección, dinámica, calibración y threshold se
ajustan sólo en la fuente; ninguna etiqueta del destino participa en fitting.
Esto se llama transporte entre SourceSets públicos, no validación externa.

Estados posibles:

- `INTERNAL_IMPROVEMENT_CONFIRMED`;
- `TRANSPORT_ROBUSTNESS_CONFIRMED`;
- `PROMISING_BUT_NOT_TRANSPORTABLE`;
- `NO_VERIFIED_IMPROVEMENT`.

MIMIC-IV/eICU completos permanecen `BLOCKED_EXTERNAL_DATA` mientras no existan
credenciales/datasets completos. Si C3 pasa gates, se detiene. Si mejora sólo
internamente, la siguiente hipótesis será operador por SourceSet con embedding y
alineamiento espectral [9]. Si la irregularidad deja pocos residuos, se evaluará
un modelo continuous-discrete [4]. Si C3 no mejora con soporte suficiente, se
registra el resultado negativo y no se añade una red mayor.

## 11. Alternativas examinadas y descartadas ahora

- Transformer/foundation model: demasiados grados de libertad antes de probar la
  señal dinámica y mayor riesgo de selección/compute.
- Red Koopman profunda: no necesaria para el primer test de innovación.
- Imputar la observación objetivo y calcular residuo: crea evidencia artificial.
- Fine-tuning en el destino: usa datos de destino y responde otra pregunta; la
  evidencia reciente muestra que no es universalmente robusto [7], [8].
- Entrenar operadores en A+B para transporte: contamina el destino.
- Mixed A/B CV como “external validation”: denominación científicamente falsa.
- Comparar sólo AUROC: insuficiente con prevalencia baja y carga de alarmas.

## 12. Ejecución, recursos y reproducibilidad

`bash run.sh` es el único entrypoint. Encadena con `afterok`:

1. suite completa;
2. `prepare` CPU;
3. benchmark estratificado fijo de 4,000 pacientes, dos lifts Koopman y diez
   fits XGBoost de 200 árboles, CPU 8/16/32 y GPU 8/16/32 cores; CEDIA exige al menos
   8 CPU para cualquier job de la partición GPU;
4. `model` con el perfil seleccionado;
5. `finalize` CPU;
6. promoción sólo después del manifiesto de recursos y validación final.

Todo cálculo científico corre en `compute-0-2`; `compute-0-1` está excluido.
Límites: 32 CPU, 64 GB y una A100 de 40 GB. Se selecciona el perfil menor dentro
del 5 % del más rápido; GPU sólo con >5 % de ventaja total, al menos tres muestras
activas y utilización GPU activa media >50 %. CPU requiere eficiencia >50 %
durante la ventana de cómputo, separada de I/O y startup. RAM es pico
medido/estimado +20 %, redondeada a 2 GB; 64 GB es un techo, no un objetivo.
`prepare` usa un CPU y 10 GB, derivados del pico observado más 20 %, porque más
threads no aceleraron esa ruta serial. El muestreo GPU se hace cada segundo para
no perder fits cortos. No se llena memoria artificialmente.

La mezcla del benchmark aproxima la carga planificada: 10/2 = 5 fits XGBoost
por ajuste Koopman frente a 278/49 = 5.67 en el run completo. Un benchmark de
un solo XGBoost habría sobreponderado la transformación y no es aceptable para
elegir CPU frente a GPU.

Las recomendaciones sobre citas de la revista, título, Abstract, contribuciones,
Discussion, Conclusion, trabajos futuros y special issues quedan
`DEFERRED_TO_MANUSCRIPT_PHASE`; este experimento sólo produce la evidencia que
permitirá atenderlas sin reescribir aún el paper.

Cada artefacto final queda ligado por SHA-256 en la cadena raw → harmonized →
features/target → folds → inner selection → OOF/model/calibration/threshold →
ablaciones/semillas/balance → inference/transport/DCA → gate/report. Un fallo no
promueve resultados parciales.

## 13. Criterio de interpretación

Éxito computacional no implica mejora científica. Sólo los artefactos del run
actual determinan el estado. No se compararán resultados para recuperar cifras
históricas. Tampoco se modificará el paper en esta fase.

## Referencias IEEE verificadas

[1] R. van de Water, H. Schmidt, P. Elbers, P. Thoral, B. Arnrich, and P.
Rockenschaub, “Yet Another ICU Benchmark: A Flexible Multi-Center Framework for
Clinical ML,” in *Proc. 12th Int. Conf. Learning Representations (ICLR)*, 2024.
[Online]. Available: https://proceedings.iclr.cc/paper_files/paper/2024/hash/c26a89073d972f2e6643617b0f3a9e8a-Abstract-Conference.html

[2] Y. Liu, C. Li, J. Wang, and M. Long, “Koopa: Learning Non-stationary Time
Series Dynamics with Koopman Predictors,” in *Advances in Neural Information
Processing Systems 36*, 2023, doi: https://doi.org/10.52202/075280-0538.

[3] A. Mallen, C. A. Keller, and J. N. Kutz, “Koopman-inspired approach for
identification of exogenous anomalies in nonstationary time-series data,”
*Machine Learning: Science and Technology*, vol. 4, no. 2, Art. no. 025033,
2023, doi: https://doi.org/10.1088/2632-2153/acdd50.

[4] A. F. Ansari, A. Heng, A. Lim, and H. Soh, “Neural Continuous-Discrete
State Space Models for Irregularly-Sampled Time Series,” in *Proc. 40th Int.
Conf. Machine Learning*, PMLR, vol. 202, pp. 926–951, 2023. [Online]. Available:
https://proceedings.mlr.press/v202/ansari23a.html

[5] L. L. Guo *et al.*, “Evaluation of domain generalization and adaptation on
improving model robustness to temporal dataset shift in clinical medicine,”
*Scientific Reports*, vol. 12, Art. no. 2726, 2022, doi:
https://doi.org/10.1038/s41598-022-06484-1.

[6] M. Londschien, M. Burger, G. Rätsch, and P. Bühlmann, “Domain
generalization and adaptation in intensive care with anchor regression,” *RSS:
Data Science and Artificial Intelligence*, vol. 2, no. 1, Art. no. udag001,
2026, doi: https://doi.org/10.1093/rssdat/udag001.

[7] F. Tranchellini, Y. Farag, C. Jutzeler, and L. Meegahapola, “Evaluating
deep learning sepsis prediction models in ICUs under distribution shift: a
multi-centre retrospective cohort study,” *npj Digital Medicine*, vol. 9, Art.
no. 306, 2026, doi: https://doi.org/10.1038/s41746-026-02364-4.

[8] J. Backes, A. Tsanda, T. Knopp, W. Renz, and E. Schöll, “Combining machine
learning and physiological network models for sepsis prediction,” *Frontiers in
Network Physiology*, vol. 6, Art. no. 1852577, 2026, doi:
https://doi.org/10.3389/fnetp.2026.1852577.

[9] B. Zhang, “Koopman framework with self-supervised spectral alignment for
multi-domain time-series modeling and prediction,” *Neurocomputing*, vol. 652,
Art. no. 131109, 2025, doi: https://doi.org/10.1016/j.neucom.2025.131109.

[10] Y. Qin, S. Yin, J. Liu, W. Qian, Y. Cao, and L. Cao, “Slow–fast
dynamics-assisted Koopman network for anomaly detection in non-stationary
industrial processes with time-lagged variables,” *Journal of Process Control*,
Art. no. 103678, 2026, doi:
https://doi.org/10.1016/j.jprocont.2026.103678.

Formato de referencias: **IEEE**. Todas las fuentes principales están dentro de
2021–2026; la única excepción temporal permitida es la documentación/scorer
oficial PhysioNet/CinC 2019 usada para compatibilidad secundaria.
