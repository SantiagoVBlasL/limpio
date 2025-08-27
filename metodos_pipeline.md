# Métodos

## Datos y particiones

Partimos de un conectoma por sujeto $X\in\mathbb{R}^{C\times R\times R}$ con $C=4$ canales y $R=131$ ROIs. Los canales seleccionados son:

1. **Pearson\_Full\_FisherZ\_Signed**
2. **MI\_KNN\_Symmetric**
3. **dFC\_StdDev**
4. **DistanceCorr**

Para **clasificación AD vs CN** hay 184 sujetos (CN=89, AD=95). La **validación externa** usa $K=5$ folds estratificados por *ResearchGroup\_Mapped*, *Sex* y *Manufacturer*. En el **Fold 1**, el test contiene **N=37** (AD=19, CN=18). El **pool de entrenamiento del VAE** incluye AD, CN y MCI (N=394) para robustez representacional.

### Normalización inter-canal

Aplicamos *zscore\_offdiag* **por canal** usando exclusivamente los sujetos de entrenamiento del VAE (N≈315). Cada matriz de un canal se vectoriza tomando solo elementos fuera de la diagonal (para evitar sesgos por autoconexiones) y se normaliza con $\mu_c,\,\sigma_c$ del *train*; test/val heredan esos parámetros (sin *leakage*).

---

## Etapa 1 — Autoencoder Variacional Convolucional (VAE)

Sea $E$ el *encoder* y $D$ el *decoder*. El *encoder* (4 bloques convolucionales + capas densas con *dropout* y *layer norm* en FC) mapea $X$ a los parámetros de una gaussiana latente $q_\phi(z\mid X)=\mathcal{N}(\mu(X),\,\mathrm{diag}(\sigma^2(X)))$ en $\mathbb{R}^{D}$, con **dimensión latente** $D=48$. Usamos **$\mu$** como *embedding* para clasificación.

**Función de pérdida (β-VAE):**

$$
\mathcal{L}_{\text{VAE}}(X) = \underbrace{\lVert X - \hat X \rVert_2^2}_{\text{reconstrucción}}\; +\; \beta\,\underbrace{\mathrm{KL}\big(q_\phi(z\mid X)\,\Vert\,\mathcal{N}(0,I)\big)}_{\text{regularización}}
$$

con *scheduler* cíclico de $\beta$ (1 ciclo) y *CosineAnnealingWarmRestarts* para el *learning rate*. *Early stopping* sobre la pérdida de validación interna.

---

## Etapa 2 — Clasificación en el espacio latente y SHAP

Construimos el vector de *features*

$$
\mathbf{f}(X) = \big[\,\mu_1(X),\dots,\mu_{48}(X),\;\text{Age},\;\text{Sex},\;\text{Manufacturer}\,\big].
$$

Entrenamos varios clasificadores (XGB, SVM, LogReg, LGBM-GB) con *búsqueda bayesiana/Optuna* anidada (5-fold interna sobre el *train/dev* de la CV externa). *Scoring*: AUC-ROC. Calibración posterior (si aplica). Selección de umbral con el índice de Youden sobre *dev*.

Para interpretabilidad, fijamos el mejor *pipeline* del *train/dev* y computamos **SHAP** (background del *train*). Para cada dimensión $k$ del latente definimos el **peso discriminativo**

$$
 w_k \;=\; \mathbb{E}\big[\phi_k\;\big|\;\text{AD}\big]\;\;-\;\mathbb{E}\big[\phi_k\;\big|\;\text{CN}\big],
$$

donde $\phi_k$ es la contribución SHAP de la *feature* $\mu_k$.

Definimos el **índice escalar de firma AD** para un sujeto $X$:

$$
S(X)\;=\;\sum_{k=1}^{D} w_k\,\mu_k(X)\;=\;\mathbf{w}^{\top}\,\mu(X).
$$

Este escalar, lineal en el latente, reemplaza al clasificador completo para auditar saliencia con métodos basados en gradiente.

---

## Etapa 3 — Retroproyección a conexiones con Integrated Gradients

Sea $\mathbf{x}'$ la **línea base** (tensor nulo tras normalización por canal). La atribución **IG** por conexión $(c,i,j)$ viene dada por

$$
\mathrm{IG}_{cij}(X) \;=\; \big(X_{cij}-x'_{cij}\big)\;\int_{\alpha=0}^{1} \frac{\partial S\big(\mathbf{x}'+\alpha\,(X-\mathbf{x}')\big)}{\partial X_{cij}}\,d\alpha.
$$

Agrupamos en el plano ROI–ROI sumando $|\mathrm{IG}_{cij}|$ sobre canales.

### Saliencia diferencial por grupo

Promediamos por grupo y contrastamos **magnitud** de IG:

$$
\Delta S_{ij} \;=\; \mathbb{E}\big[\,|\mathrm{IG}_{ij}(X)|\,\big|\,\text{AD}\big]\; -\; \mathbb{E}\big[\,|\mathrm{IG}_{ij}(X)|\,\big|\,\text{CN}\big].
$$

Signo e intensidad de $\Delta S_{ij}$ producen el mapa final (rojo: pro-AD; azul: pro-CN).

### Contribución por canal

El peso porcentual por canal $c$ se calcula como

$$
\mathrm{Pct}(c)=\frac{\sum_{i<j}\,|\mathrm{IG}_{cij}(X)|}{\sum_{c'}\sum_{i<j}|\mathrm{IG}_{c'ij}(X)|}\times 100,\quad\text{(promedio sobre sujetos)}.
$$

En el Fold 1, los porcentajes observados fueron coherentes con la hipótesis de mayor relevancia de **dFC\_StdDev** y **Pearson**, seguidos por **MI** y **DistanceCorr**.

---

## Métricas y reporte

En cada fold de la CV externa reportamos en **test**: AUC-ROC, *balanced accuracy*, sensibilidad, especificidad y curvas calibradas. En *dev* registramos el umbral óptimo (Youden) y guardamos el *background* SHAP para reproducibilidad.

---

## Controles y *sanity checks*

* **No *leakage***: normalización por canal y SHAP *background* calculados solo con *train*.
* **Efectos de escáner/sitio**: estratificación por *Manufacturer*; análisis de sensibilidad reportando métricas por fabricante.
* **Robustez de IG**: prueba de randomización (sustituir $\mathbf{w}$ por permutaciones; la saliencia debe colapsar).
* **Ablaciones**: (i) sin metadatos; (ii) solo Pearson; (iii) $\mu$ vs. $\mu\!+\!\sigma$; (iv) β y *dropout*.
* **Estadística en $\Delta S_{ij}$**: *permutation testing* por arista y FDR a nivel red; resumen por comunidades (DMN, límbico, etc.).

---

## Implementación

Entrenamiento en GPU con *PyTorch* (VAE) y *scikit-learn/LightGBM/XGBoost* (clasificadores). *Optuna* para búsqueda de hiperparámetros. *Seed* fija para reproducibilidad. Se guardan: pesos del VAE, *pipelines* de clasificación, *background* SHAP y matrices de saliencia por sujeto.

---

## Algoritmo 1 — Interpretabilidad Latente→Conexiones

```pseudo
Input: X (conectoma C×R×R), metadatos m, encoder E, pesos w (de SHAP), baseline x'=0
1: (μ, σ) ← E(X)
2: S ← w^T μ
3: Inicializar IG ← 0 (C×R×R)
4: for α in linspace(0,1,M):
5:     X_α ← x' + α·(X - x')
6:     μ_α ← E_μ(X_α)
7:     S_α ← w^T μ_α
8:     g_α ← ∂S_α/∂X_α   # backprop
9:     IG ← IG + g_α
10: IG ← (X - x') ⊙ IG · ((1/M))
11: Agregar |IG| por canales y promediar por grupo para ΔS
Output: mapas ΔS (ROI×ROI), % por canal
```

---

## Limitaciones

(1) La linealización vía $S(X)$ aproxima la frontera del clasificador; (2) IG depende de la línea base elegida; (3) los resultados por canal pueden estar influidos por la escala residual tras la normalización; (4) tamaño muestral por fabricante.

---

## Buenas prácticas de figura

* **Fig.1**: Esquema del pipeline (X→VAE→μ→SHAP→w→IG→ΔS).
* **Fig.2**: Barras de % por canal + barras de SHAP por dimensión latente top-k.
* **Fig.3**: Mapa $\Delta S_{ij}$ (triángulo superior) + *connectogram* resumido por redes.

