import json 
# ============================================================
# CONFIGURAÇÃO DE CPU
# ============================================================

import os

numero_threads = os.cpu_count()

os.environ["OMP_NUM_THREADS"] = str(numero_threads)
os.environ["MKL_NUM_THREADS"] = str(numero_threads)
os.environ["OPENBLAS_NUM_THREADS"] = str(numero_threads)
os.environ["NUMEXPR_NUM_THREADS"] = str(numero_threads)

print("=" * 60)
print("CONFIGURAÇÃO DA CPU")
print("=" * 60)

print(f"Threads disponíveis: {numero_threads}")
print(f"OMP_NUM_THREADS: {os.environ['OMP_NUM_THREADS']}")
print(f"MKL_NUM_THREADS: {os.environ['MKL_NUM_THREADS']}")
print(f"OPENBLAS_NUM_THREADS: {os.environ['OPENBLAS_NUM_THREADS']}")
print(f"NUMEXPR_NUM_THREADS: {os.environ['NUMEXPR_NUM_THREADS']}")

# ============================================================
# QAOA - MAXCUT
# Versão didática com mensagens no terminal
# ============================================================

import pennylane as qml
from pennylane import numpy as np
import networkx as nx
from matplotlib import pyplot as plt
import random

# PASSO 1 - CRIAÇÃO DO GRAFO


print("\n" + "=" * 60)
print("PASSO 1 - CRIAÇÃO DO GRAFO")
print("=" * 60)

# Fixamos a semente para que o mesmo grafo seja gerado
# sempre que o programa for executado.

random.seed(42)
np.random.seed(42)

numero_de_vertices = 6

print(f"\nNúmero de vértices: {numero_de_vertices}")
print("Criando um grafo aleatório...")

grafo = nx.fast_gnp_random_graph(
    numero_de_vertices,
    0.5
)

print(f"Número de vértices criados: {grafo.number_of_nodes()}")
print(f"Número de arestas criadas: {grafo.number_of_edges()}")

print("\nArestas do grafo:")

for aresta in grafo.edges():
    print(f"  {aresta[0]} -- {aresta[1]}")


# ============================================================
# DESENHANDO O GRAFO
# ============================================================

print("\nGerando imagem do grafo...")

plt.figure(figsize=(5, 3))

posicao = nx.spring_layout(grafo)

nx.draw(
    grafo,
    with_labels=True,
    node_size=700,
    pos=posicao
)

plt.savefig("example_graph.png")

print("Imagem salva como: example_graph.png")


# ============================================================
# PASSO 2 - CRIAÇÃO DOS HAMILTONIANOS
# ============================================================

print("\n" + "=" * 60)
print("PASSO 2 - HAMILTONIANOS DO QAOA")
print("=" * 60)

print("\nCalculando os Hamiltonianos do problema MAXCUT...")

Hamiltoniano_custo, Hamiltoniano_mixer = qml.qaoa.maxcut(
    grafo
)

print("\nHamiltoniano de custo H_C:")
print(Hamiltoniano_custo)

print("\nHamiltoniano mixer H_M:")
print(Hamiltoniano_mixer)


# ============================================================
# PASSO 3 - CONFIGURAÇÃO DO QAOA
# ============================================================

print("\n" + "=" * 60)
print("PASSO 3 - CONFIGURAÇÃO DO QAOA")
print("=" * 60)

profundidade = 10

qubits = range(numero_de_vertices)

print(f"\nNúmero de qubits: {numero_de_vertices}")
print(f"Profundidade do circuito: {profundidade}")

print("\nCada camada QAOA possui:")

print("  1. Cost Layer")
print("  2. Mixer Layer")


# ============================================================
# PASSO 4 - DEFININDO O MIXER
# ============================================================

def camada_mixer(beta):

    qml.qaoa.mixer_layer(
        beta,
        Hamiltoniano_mixer / grafo.number_of_edges()
    )


# ============================================================
# PASSO 5 - DEFININDO A CAMADA DE CUSTO
# ============================================================

def camada_custo(gamma):

    qml.qaoa.cost_layer(
        gamma,
        Hamiltoniano_custo / grafo.number_of_edges()
    )


# ============================================================
# PASSO 6 - DEFININDO UMA CAMADA QAOA
# ============================================================

def camada_qaoa(gamma, beta):

    # Primeiro aplica o Hamiltoniano de custo
    camada_custo(gamma)

    # Depois aplica o Hamiltoniano mixer
    camada_mixer(beta)


# ============================================================
# PASSO 7 - CRIANDO O CIRCUITO QAOA
# ============================================================

def circuito_qaoa(parametros):

    # --------------------------------------------------------
    # Estado inicial
    # --------------------------------------------------------
    #
    # Cada qubit começa no estado |0>.
    #
    # A porta Hadamard transforma:
    #
    # |0> → |+>
    #
    # onde:
    #
    # |+> = (|0> + |1>) / sqrt(2)
    #
    # Assim criamos uma superposição de todos os bitstrings.
    # --------------------------------------------------------

    for qubit in qubits:

        qml.Hadamard(
            wires=qubit
        )

    # --------------------------------------------------------
    # Aplicação das camadas QAOA
    # --------------------------------------------------------

    qml.layer(
        camada_qaoa,
        profundidade,
        parametros[0],
        parametros[1]
    )


# ============================================================
# PASSO 8 - CRIANDO O DISPOSITIVO QUÂNTICO
# ============================================================

print("\n" + "=" * 60)
print("PASSO 4 - DISPOSITIVO QUÂNTICO")
print("=" * 60)

print("\nCriando o simulador quântico...")

dispositivo = qml.device(
    "default.qubit",
    wires=qubits
)

print("Simulador: default.qubit")
print(f"Quantidade de qubits: {numero_de_vertices}")


# ============================================================
# PASSO 9 - FUNÇÃO DE CUSTO
# ============================================================

@qml.qnode(dispositivo)
def funcao_custo(parametros):

    circuito_qaoa(parametros)

    return qml.expval(
        Hamiltoniano_custo
    )


print("\nFunção de custo criada.")

print(
    "Ela mede o valor esperado do Hamiltoniano de custo."
)


# ============================================================
# PASSO 10 - PARÂMETROS INICIAIS
# ============================================================

print("\n" + "=" * 60)
print("PASSO 5 - PARÂMETROS INICIAIS")
print("=" * 60)

# Otimizador Adam
otimizador = qml.AdamOptimizer()

# Número de vezes que os parâmetros serão atualizados
numero_de_passos = 1000

# Parâmetro utilizado para gerar os valores iniciais
T = 7.5


# ------------------------------------------------------------
# Valores iniciais de beta
# ------------------------------------------------------------

beta = [
    -(1 - i / profundidade) * T / profundidade
    for i in range(profundidade)
]


# ------------------------------------------------------------
# Valores iniciais de gamma
# ------------------------------------------------------------

gamma = [
    (i / profundidade) * T / profundidade
    for i in range(profundidade)
]


# ------------------------------------------------------------
# Junta beta e gamma em um único vetor de parâmetros
# ------------------------------------------------------------

parametros = np.array(
    [beta, gamma],
    requires_grad=True
)


print("\nParâmetros beta:")

for i, valor in enumerate(beta):
    print(f"  beta[{i}] = {valor:.6f}")


print("\nParâmetros gamma:")

for i, valor in enumerate(gamma):
    print(f"  gamma[{i}] = {valor:.6f}")


# ============================================================
# PASSO 11 - OTIMIZAÇÃO
# ============================================================

print("\n" + "=" * 60)
print("PASSO 6 - OTIMIZAÇÃO DOS PARÂMETROS")
print("=" * 60)

print("\nComeçando a otimização...")
print(f"Número de iterações: {numero_de_passos}")

print(
    "\nDurante a otimização, o Adam irá alterar "
    "beta e gamma para tentar minimizar a função de custo."
)

lista_custos = []

custo_inicial = None
custo_final = None


# ------------------------------------------------------------
# LOOP DE OTIMIZAÇÃO
# ------------------------------------------------------------

for passo in range(numero_de_passos):

    parametros, custo = otimizador.step_and_cost(
        funcao_custo,
        parametros
    )

    lista_custos.append(custo)

    # Guarda o primeiro custo
    if passo == 0:

        custo_inicial = custo

        print(
            f"\nCusto inicial: {custo:.6f}"
        )

    # Mostra o progresso a cada 50 iterações
    if passo % 50 == 0 or passo == numero_de_passos - 1:

        print(
            f"Iteração {passo + 1:4d}/{numero_de_passos} "
            f"| Custo: {custo:.6f}"
        )


custo_final = custo


print("\nOtimização finalizada!")

print(
    f"\nCusto inicial: {custo_inicial:.6f}"
)

print(
    f"Custo final:   {custo_final:.6f}"
)


# ============================================================
# PASSO 12 - MOSTRANDO OS PARÂMETROS FINAIS
# ============================================================

print("\n" + "=" * 60)
print("PASSO 7 - PARÂMETROS OTIMIZADOS")
print("=" * 60)

print("\nValores finais de beta:")

for i, valor in enumerate(parametros[0]):
    print(
        f"  beta[{i}] = {float(valor):.6f}"
    )


print("\nValores finais de gamma:")

for i, valor in enumerate(parametros[1]):
    print(
        f"  gamma[{i}] = {float(valor):.6f}"
    )


# ============================================================
# PASSO 13 - GRÁFICO DA OTIMIZAÇÃO
# ============================================================

print("\n" + "=" * 60)
print("PASSO 8 - GRÁFICO DO CUSTO")
print("=" * 60)

plt.figure(figsize=(7, 4))

plt.plot(lista_custos)

plt.ylabel("Custo")
plt.xlabel("Iteração")

plt.title("Otimização do QAOA")

plt.tight_layout()

plt.savefig(
    "cost_minimization.png"
)

print(
    "Gráfico salvo como: cost_minimization.png"
)


# ============================================================
# PASSO 14 - CIRCUITO DE PROBABILIDADES
# ============================================================

print("\n" + "=" * 60)
print("PASSO 9 - MEDINDO OS QUBITS")
print("=" * 60)

print(
    "\nAgora vamos executar o circuito novamente "
    "usando os parâmetros otimizados."
)


@qml.qnode(dispositivo)
def circuito_probabilidade(gamma, beta):

    circuito_qaoa(
        [gamma, beta]
    )

    return qml.probs(
        wires=qubits
    )


print("\nExecutando o circuito quântico...")

probabilidades = circuito_probabilidade(
    parametros[0],
    parametros[1]
)

print("Medição concluída.")


# ============================================================
# PASSO 15 - RESULTADOS DAS MEDIÇÕES
# ============================================================

print("\n" + "=" * 60)
print("PASSO 10 - RESULTADOS DAS MEDIÇÕES")
print("=" * 60)

print(
    "\nCada resultado possui "
    f"{numero_de_vertices} bits."
)

print(
    f"Como temos {numero_de_vertices} qubits, "
    f"existem 2^{numero_de_vertices} = "
    f"{2 ** numero_de_vertices} possíveis resultados."
)

print("\nProbabilidade de cada bitstring:\n")


melhor_bitstring = None
maior_probabilidade = 0


for numero, probabilidade in enumerate(probabilidades):

    bitstring = format(
        numero,
        f"0{numero_de_vertices}b"
    )

    probabilidade_float = float(probabilidade)

    print(
        f"  |{bitstring}> "
        f"-> {probabilidade_float:.6f} "
        f"({probabilidade_float * 100:.2f}%)"
    )

    # Verifica se este é o resultado mais provável
    if probabilidade_float > maior_probabilidade:

        maior_probabilidade = probabilidade_float

        melhor_bitstring = bitstring


# ============================================================
# PASSO 16 - MELHOR RESULTADO
# ============================================================

print("\n" + "=" * 60)
print("PASSO 11 - MELHOR SOLUÇÃO ENCONTRADA")
print("=" * 60)

print(
    f"\nMelhor bitstring encontrado: "
    f"|{melhor_bitstring}>"
)

print(
    f"Probabilidade: "
    f"{maior_probabilidade:.6f}"
)

print(
    f"Probabilidade: "
    f"{maior_probabilidade * 100:.2f}%"
)


# ============================================================
# PASSO 17 - INTERPRETANDO O BITSTRING
# ============================================================

print("\n" + "=" * 60)
print("PASSO 12 - INTERPRETAÇÃO DO MAXCUT")
print("=" * 60)

print(
    "\nCada bit representa o grupo ao qual um vértice pertence:"
)

print(
    "  0 → Grupo 0"
)

print(
    "  1 → Grupo 1"
)

print(
    f"\nSolução encontrada: {melhor_bitstring}"
)

print(
    "\nVértices no Grupo 0:"
)

grupo_zero = []

for vertice, bit in enumerate(melhor_bitstring):

    if bit == "0":
        grupo_zero.append(vertice)

print(
    grupo_zero
)


print(
    "\nVértices no Grupo 1:"
)

grupo_um = []

for vertice, bit in enumerate(melhor_bitstring):

    if bit == "1":
        grupo_um.append(vertice)

print(
    grupo_um
)


# ============================================================
# PASSO 18 - GRÁFICO DAS PROBABILIDADES
# ============================================================

print("\n" + "=" * 60)
print("PASSO 13 - GRÁFICO DAS PROBABILIDADES")
print("=" * 60)


bitstrings = [
    format(
        i,
        f"0{numero_de_vertices}b"
    )
    for i in range(2 ** numero_de_vertices)
]


plt.figure(figsize=(10, 4))

plt.style.use("seaborn-v0_8")


plt.bar(
    bitstrings,
    probabilidades
)

plt.ylabel("Probabilidade")

plt.xlabel("Bitstrings")

plt.title(
    "Probabilidades dos resultados do QAOA"
)

plt.xticks(
    rotation=90
)

plt.tight_layout()

plt.savefig(
    "measurement_probabilities.png"
)

print(
    "Gráfico salvo como: "
    "measurement_probabilities.png"
)


# ============================================================
# PASSO 19 - RESUMO FINAL
# ============================================================

print("\n" + "=" * 60)
print("EXECUÇÃO FINALIZADA")
print("=" * 60)

print("\nResumo da execução:")

print(
    f"  • Vértices: {numero_de_vertices}"
)

print(
    f"  • Arestas: {grafo.number_of_edges()}"
)

print(
    f"  • Qubits: {numero_de_vertices}"
)

print(
    f"  • Camadas QAOA: {profundidade}"
)

print(
    f"  • Iterações: {numero_de_passos}"
)

print(
    f"  • Custo inicial: {custo_inicial:.6f}"
)

print(
    f"  • Custo final: {custo_final:.6f}"
)

print(
    f"  • Melhor bitstring: |{melhor_bitstring}>"
)

print(
    f"  • Probabilidade: "
    f"{maior_probabilidade * 100:.2f}%"
)


print("\nArquivos gerados:")

print(
    "  • example_graph.png"
)

print(
    "  • cost_minimization.png"
)

print(
    "  • measurement_probabilities.png"
)


print("\nFim do programa.")

import json

# Encontrar o melhor resultado
melhor_indice = int(np.argmax(probabilidades))
melhor_bitstring = format(melhor_indice, f"0{numero_de_vertices}b")
melhor_probabilidade = float(probabilidades[melhor_indice])

resultado = {
    "numero_qubits": numero_de_vertices,
    "profundidade": profundidade,
    "melhor_bitstring": melhor_bitstring,
    "melhor_probabilidade": melhor_probabilidade,
    "probabilidades": [float(p) for p in probabilidades]
}

with open("resultado_qaoa.json", "w", encoding="utf-8") as arquivo:
    json.dump(resultado, arquivo, indent=4, ensure_ascii=False)

print("\n\nResultado salvo em resultado_qaoa.json")
print(f"Melhor bitstring: {melhor_bitstring}")
print(f"Probabilidade: {melhor_probabilidade:.6f}")