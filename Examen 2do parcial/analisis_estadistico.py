"""
Examen Práctico 2do Parcial - Análisis Estadístico
Datos de: peso, altura, velocidad, color
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# 1. DATOS
# ─────────────────────────────────────────────
data = {
    'peso':      [7.2, 8.5, 9.8, 6.5, 7.5, 10.1, 11.0, 11.0, 11.1, 11.2,
                  11.3, 11.4, 11.4, 11.7, 12.0, 12.9, 12.9, 10.3,  9.7, 10.8,
                  11.0, 10.2, 10.5,  6.5,  6.3,  7.3,  7.5,  7.9,  8.2],
    'altura':    [50, 66, 73, 72, 81, 73, 66, 75, 70, 75,
                  69, 76, 76, 69, 75, 64, 55, 76, 71, 64,
                  78, 70, 74, 72, 77, 51, 62, 60, 70],
    'velocidad': [10.3, 10.3, 10.2, 16.4, 18.8, 19.7, 15.6, 21.2, np.nan, 19.9,
                  24.2, 21.0, 21.4, 21.3, np.nan, 22.2, 33.8, 27.4, 25.7, 24.9,
                  23.1, 31.7, 36.3, 38.3, 42.6, 55.4, np.nan, 58.3, np.nan],
    'color':     ['Blanco','Amarillo','Verde','Verde','Verde','Verde','Blanco','Amarillo',
                  'NA','Blanco','Amarillo','Blanco','Verde','Verde','Amarillo','Amarillo',
                  'Blanco','Amarillo','Verde','Verde','Amarillo','Amarillo','Verde','Verde',
                  'Verde','Blanco','Blanco','Amarillo','Verde']
}

df = pd.DataFrame(data)
# Limpiar color NA
df['color'] = df['color'].replace('NA', np.nan)

print("="*60)
print("   ANÁLISIS ESTADÍSTICO COMPLETO")
print("="*60)
print(f"\nTotal de registros: {len(df)}")
print("\nPrimeras filas:")
print(df.head())

# ─────────────────────────────────────────────
# 2. FRECUENCIAS ABSOLUTAS - todas las variables
# ─────────────────────────────────────────────
def frecuencias_absolutas(serie, nombre, bins=6):
    """Calcula frecuencias absolutas para variable numérica o categórica."""
    if serie.dtype == object or serie.name == 'color':
        fa = serie.dropna().value_counts().sort_index()
        return fa
    else:
        s = serie.dropna()
        counts, edges = np.histogram(s, bins=bins)
        labels = [f"[{edges[i]:.1f}-{edges[i+1]:.1f})" for i in range(len(edges)-1)]
        fa = pd.Series(counts, index=labels, name=nombre)
        return fa

def frecuencias_relativas(fa):
    return fa / fa.sum()

def frecuencias_acumuladas(fa):
    return fa.cumsum()

print("\n" + "="*60)
print("FRECUENCIAS POR VARIABLE")
print("="*60)

variables_num = ['peso', 'altura', 'velocidad']
resumen = {}

for var in variables_num:
    s = df[var].dropna()
    fa = frecuencias_absolutas(s, var)
    fr = frecuencias_relativas(fa)
    fac = frecuencias_acumuladas(fa)
    resumen[var] = {'fa': fa, 'fr': fr, 'fac': fac}

    print(f"\n{'─'*50}")
    print(f"  Variable: {var.upper()}")
    print(f"{'─'*50}")
    tabla = pd.DataFrame({
        'F. Absoluta': fa,
        'F. Relativa': fr.round(4),
        'F. Acumulada': fac
    })
    print(tabla.to_string())

# Color (categórica)
s_color = df['color'].dropna()
fa_color = s_color.value_counts().sort_index()
fr_color = fa_color / fa_color.sum()
fac_color = fa_color.sort_index().cumsum()
resumen['color'] = {'fa': fa_color, 'fr': fr_color, 'fac': fac_color}

print(f"\n{'─'*50}")
print(f"  Variable: COLOR")
print(f"{'─'*50}")
tabla_color = pd.DataFrame({
    'F. Absoluta': fa_color,
    'F. Relativa': fr_color.round(4),
    'F. Acumulada': fac_color
})
print(tabla_color.to_string())

# ─────────────────────────────────────────────
# 3. MEDIA, MEDIANA, MODA
# ─────────────────────────────────────────────
print("\n" + "="*60)
print("MEDIDAS DE TENDENCIA CENTRAL")
print("="*60)

for var in variables_num:
    s = df[var].dropna()
    media   = s.mean()
    mediana = s.median()
    moda    = s.mode().values
    print(f"\n  {var.upper()}:")
    print(f"    Media   = {media:.4f}")
    print(f"    Mediana = {mediana:.4f}")
    print(f"    Moda    = {moda}")

s_c = df['color'].dropna()
print(f"\n  COLOR:")
print(f"    Moda = {s_c.mode().values}")

# ─────────────────────────────────────────────
# 4. GRÁFICAS
# ─────────────────────────────────────────────
plt.rcParams.update({
    'figure.facecolor': '#0f1117',
    'axes.facecolor':   '#1a1d27',
    'axes.edgecolor':   '#3a3d4f',
    'axes.labelcolor':  '#e0e4f0',
    'xtick.color':      '#b0b4c8',
    'ytick.color':      '#b0b4c8',
    'text.color':       '#e0e4f0',
    'grid.color':       '#2a2d3f',
    'grid.linewidth':   0.7,
    'font.family':      'monospace',
    'axes.titlesize':   11,
    'axes.titleweight': 'bold',
    'axes.titlecolor':  '#7dd3fc',
})

COLORS_CAT = {'Blanco': '#f0f4ff', 'Amarillo': '#fde68a', 'Verde': '#6ee7b7'}
PALETTE = ['#7dd3fc', '#f472b6', '#a78bfa', '#34d399', '#fb923c']

# ── Figura 1: Frecuencias Absolutas (Barras) ──────────────────
fig1, axes1 = plt.subplots(2, 2, figsize=(14, 10))
fig1.suptitle("FRECUENCIAS ABSOLUTAS — Gráfica de Barras",
              fontsize=14, color='#7dd3fc', fontweight='bold', y=1.01)
fig1.patch.set_facecolor('#0f1117')

vars_all = ['peso', 'altura', 'velocidad', 'color']
for ax, var, col in zip(axes1.flat, vars_all, PALETTE):
    fa = resumen[var]['fa']
    bars = ax.bar(range(len(fa)), fa.values, color=col, alpha=0.85,
                  edgecolor='#ffffff22', linewidth=0.5, width=0.6)
    ax.set_xticks(range(len(fa)))
    ax.set_xticklabels(fa.index, rotation=35, ha='right', fontsize=8)
    ax.set_title(var.upper())
    ax.set_ylabel("Frecuencia Absoluta")
    ax.grid(axis='y', alpha=0.4)
    for bar, val in zip(bars, fa.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                str(val), ha='center', va='bottom', fontsize=8, color='white')

plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/01_frecuencias_absolutas_barras.png',
            dpi=150, bbox_inches='tight', facecolor='#0f1117')
plt.close()
print("\n✓ Guardado: 01_frecuencias_absolutas_barras.png")

# ── Figura 2: Frecuencias Relativas (Pastel / Polar) ──────────
fig2, axes2 = plt.subplots(2, 2, figsize=(14, 10), subplot_kw=dict(aspect='equal'))
fig2.suptitle("FRECUENCIAS RELATIVAS — Diagrama de Pastel",
              fontsize=14, color='#f472b6', fontweight='bold')
fig2.patch.set_facecolor('#0f1117')

for ax, var in zip(axes2.flat, vars_all):
    fr = resumen[var]['fr']
    c_list = (list(COLORS_CAT.values()) if var == 'color'
              else PALETTE[:len(fr)])
    wedges, texts, autotexts = ax.pie(
        fr.values,
        labels=fr.index,
        autopct='%1.1f%%',
        colors=c_list,
        startangle=90,
        pctdistance=0.75,
        wedgeprops=dict(edgecolor='#0f1117', linewidth=1.5)
    )
    for t in texts:
        t.set_color('#c8cce0')
        t.set_fontsize(8)
    for at in autotexts:
        at.set_color('white')
        at.set_fontsize(8)
        at.set_fontweight('bold')
    ax.set_title(var.upper(), color='#f472b6')

plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/02_frecuencias_relativas_pastel.png',
            dpi=150, bbox_inches='tight', facecolor='#0f1117')
plt.close()
print("✓ Guardado: 02_frecuencias_relativas_pastel.png")

# ── Figura 3: Frecuencias Acumuladas ──────────────────────────
fig3, axes3 = plt.subplots(1, 3, figsize=(16, 5))
fig3.suptitle("FRECUENCIAS ACUMULADAS", fontsize=14, color='#a78bfa', fontweight='bold')
fig3.patch.set_facecolor('#0f1117')

for ax, var, col in zip(axes3, variables_num, PALETTE):
    fac = resumen[var]['fac']
    ax.step(range(len(fac)), fac.values, where='post', color=col, linewidth=2.5)
    ax.fill_between(range(len(fac)), fac.values, step='post', alpha=0.25, color=col)
    ax.scatter(range(len(fac)), fac.values, color=col, zorder=5, s=50)
    ax.set_xticks(range(len(fac)))
    ax.set_xticklabels(fac.index, rotation=40, ha='right', fontsize=7)
    ax.set_title(var.upper())
    ax.set_ylabel("F. Acumulada")
    ax.grid(alpha=0.4)

plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/03_frecuencias_acumuladas.png',
            dpi=150, bbox_inches='tight', facecolor='#0f1117')
plt.close()
print("✓ Guardado: 03_frecuencias_acumuladas.png")

# ── Figura 4: Polígono de Frecuencias ─────────────────────────
fig4, axes4 = plt.subplots(1, 3, figsize=(16, 5))
fig4.suptitle("POLÍGONO DE FRECUENCIAS", fontsize=14, color='#34d399', fontweight='bold')
fig4.patch.set_facecolor('#0f1117')

for ax, var, col in zip(axes4, variables_num, PALETTE):
    fa = resumen[var]['fa']
    x = np.arange(len(fa))
    ax.plot(x, fa.values, color=col, linewidth=2.5, marker='o',
            markersize=7, markerfacecolor='white', markeredgecolor=col, zorder=5)
    ax.fill_between(x, fa.values, alpha=0.20, color=col)
    ax.bar(x, fa.values, color=col, alpha=0.25, width=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(fa.index, rotation=40, ha='right', fontsize=7)
    ax.set_title(var.upper())
    ax.set_ylabel("Frecuencia")
    ax.grid(alpha=0.4)

plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/04_poligono_frecuencias.png',
            dpi=150, bbox_inches='tight', facecolor='#0f1117')
plt.close()
print("✓ Guardado: 04_poligono_frecuencias.png")

# ── Figura 5: Media / Mediana / Moda ──────────────────────────
fig5, axes5 = plt.subplots(1, 3, figsize=(16, 5))
fig5.suptitle("MEDIA · MEDIANA · MODA", fontsize=14, color='#fb923c', fontweight='bold')
fig5.patch.set_facecolor('#0f1117')

for ax, var, col in zip(axes5, variables_num, PALETTE):
    s = df[var].dropna()
    ax.hist(s, bins=8, color=col, alpha=0.5, edgecolor='white', linewidth=0.5)
    media   = s.mean()
    mediana = s.median()
    moda    = s.mode()[0]
    ax.axvline(media,   color='#f87171', linewidth=2, linestyle='--', label=f'Media={media:.1f}')
    ax.axvline(mediana, color='#fde68a', linewidth=2, linestyle='-.',  label=f'Mediana={mediana:.1f}')
    ax.axvline(moda,    color='#6ee7b7', linewidth=2, linestyle=':',   label=f'Moda={moda:.1f}')
    ax.legend(fontsize=8, facecolor='#1a1d27', edgecolor='#3a3d4f')
    ax.set_title(var.upper())
    ax.set_xlabel("Valor")
    ax.set_ylabel("Frecuencia")
    ax.grid(alpha=0.4)

plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/05_media_mediana_moda.png',
            dpi=150, bbox_inches='tight', facecolor='#0f1117')
plt.close()
print("✓ Guardado: 05_media_mediana_moda.png")

print("\n" + "="*60)
print("  ✅ ANÁLISIS COMPLETO FINALIZADO")
print("  Archivos generados en /mnt/user-data/outputs/")
print("="*60)
