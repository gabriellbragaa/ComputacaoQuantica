import json
import matplotlib.pyplot as plt


# ==============================
# 1. Ler resultado do QAOA
# ==============================

with open("resultado_qaoa.json", "r", encoding="utf-8") as arquivo:
    resultado = json.load(arquivo)


numero_qubits = resultado["numero_qubits"]
profundidade = resultado["profundidade"]
melhor_bitstring = resultado["melhor_bitstring"]
melhor_probabilidade = resultado["melhor_probabilidade"]
probabilidades = resultado["probabilidades"]


# ==============================
# 2. Criar imagem
# ==============================

fig, ax = plt.subplots(figsize=(12, 7))

ax.axis("off")


# Título
ax.text(
    0.5,
    0.92,
    "Resultado do QAOA - MAXCUT",
    ha="center",
    fontsize=24,
    fontweight="bold"
)


# Informações
ax.text(
    0.5,
    0.80,
    f"Qubits: {numero_qubits}",
    ha="center",
    fontsize=16
)

ax.text(
    0.5,
    0.74,
    f"Profundidade QAOA: {profundidade}",
    ha="center",
    fontsize=16
)


# Melhor solução
ax.text(
    0.5,
    0.62,
    "Melhor solução encontrada",
    ha="center",
    fontsize=18,
    fontweight="bold"
)

ax.text(
    0.5,
    0.54,
    melhor_bitstring,
    ha="center",
    fontsize=30,
    fontweight="bold"
)

ax.text(
    0.5,
    0.46,
    f"Probabilidade: {melhor_probabilidade:.2%}",
    ha="center",
    fontsize=18
)


# ==============================
# 3. Mostrar probabilidades
# ==============================

ax.text(
    0.5,
    0.34,
    "Probabilidades das soluções",
    ha="center",
    fontsize=18,
    fontweight="bold"
)

# Mostrar apenas as maiores probabilidades
indices = sorted(
    range(len(probabilidades)),
    key=lambda i: probabilidades[i],
    reverse=True
)[:8]

y = 0.27

for indice in indices:

    bitstring = format(indice, f"0{numero_qubits}b")
    probabilidade = probabilidades[indice]

    ax.text(
        0.35,
        y,
        bitstring,
        fontsize=14
    )

    ax.text(
        0.60,
        y,
        f"{probabilidade:.2%}",
        fontsize=14
    )

    y -= 0.035


# ==============================
# 4. Salvar imagem
# ==============================

plt.savefig(
    "resultado_qaoa.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("Imagem criada: resultado_qaoa.png")