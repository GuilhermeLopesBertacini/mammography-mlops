# 08. Análise Exploratória e Profiling de Dados: VinDr-Mammo

Este documento consolida o perfilamento estatístico e radiológico detalhado dos metadados e anotações do dataset **VinDr-Mammo**, realizado a partir dos arquivos oficiais:
* [`data/metadata.csv`](data/metadata.csv) (20.000 registros)
* [`data/breast-level_annotations.csv`](data/breast-level_annotations.csv) (20.000 registros)
* [`data/finding_annotations.csv`](data/finding_annotations.csv) (20.486 registros)

---

## 1. Perfilamento de Hardware e Imagens (DICOM Metadata)

### 1.1 Fabricantes e Modelos de Mamógrafos
O dataset reúne exames de 3 fabricantes com distribuições desiguais:
* **SIEMENS (Mammomat Inspiration):** 15.244 imagens ($76.22\%$)
* **Planmed (Planmed Nuance):** 3.796 imagens ($18.98\%$)
* **IMS s.r.l. / IMS GIOTTO S.p.A. (GIOTTO CLASS / 3DL):** 960 imagens ($4.80\%$)

### 1.2 O Risco Crítico da Fotometria (`Photometric Interpretation`)
* **`MONOCHROME2`:** 16.204 imagens ($81.02\%$) — Padrão digital onde maior valor numérico = maior atenuação radiológica (tecido denso/nódulo branco).
* **`MONOCHROME1`:** 3.796 imagens ($18.98\%$) — **Exatamente todas as imagens do fabricante Planmed**.
* **Implicação para Engenharia:** Se as imagens forem carregadas sem inversão dinâmica baseada na tag DICOM, quase $19\%$ da base terá os valores invertidos (mama preta em fundo branco), colapsando a extração de features convolucionais.

### 1.3 Resolução Nativa e Escala Espacial
* **Altura (Rows):** Mínimo $2.812\,\text{px}$, Máximo $3.580\,\text{px}$, Média $\mathbf{3.387\,\text{px}}$.
* **Largura (Columns):** Mínimo $2.012\,\text{px}$, Máximo $2.812\,\text{px}$, Média $\mathbf{2.647\,\text{px}}$.
* **Matriz Média:** Cerca de $9.0$ Megapixels por mamografia ($3387 \times 2647$).
* **Janelamento de Fábrica (`Window Center` e `Window Width`):** Preenchidos em $100\%$ das imagens (ex: $c \approx 1600, w \approx 1500$), servindo como baseline de apresentação radiológica.

---

## 2. Perfilamento Clínico (Nível de Mama e Exame)

### 2.1 Estrutura dos Exames
* **Total de Estudos:** $5.000$ pacientes/estudos únicos.
* **Projeções por Estudo:** Exatamente $4$ imagens por paciente ($20.000$ imagens):
  * $5.001$ imagens L-CC e $4.999$ imagens L-MLO (mama esquerda).
  * $5.000$ imagens R-CC e $5.000$ imagens R-MLO (mama direita).
* **Split Oficial (Anti-Leakage por Paciente):**
  * **Training Split:** $4.000$ estudos ($16.000$ imagens) — $80.0\%$.
  * **Test Split:** $1.000$ estudos ($4.000$ imagens) — $20.0\%$.
  * Nenhum estudo/paciente do conjunto de teste está presente no treino.

### 2.2 Distribuição de Densidade Mamária (ACR Density)
* **DENSITY A** (Predominantemente Gordurosa): 100 imagens ($0.50\%$)
* **DENSITY B** (Fibroglandular Dispersa): 1.908 imagens ($9.54\%$)
* **DENSITY C** (Heterogeneamente Densa): 15.292 imagens ($\mathbf{76.46\%}$)
* **DENSITY D** (Extremamente Densa): 2.700 imagens ($\mathbf{13.50\%}$)

> [!IMPORTANT]
> **Quase 90% do dataset compreende mamas densas (C e D)**. Em mamas densas, o parênquima glandular atua como máscara sobreposta a lesões isodensas. Este perfil epidemiológico torna indispensável a aplicação de realce adaptativo de contraste local (CLAHE) e o uso de resoluções suficientes para distinguir bordas espiculadas.

### 2.3 Classificação BI-RADS Global por Mama
* **BI-RADS 1** (Normal / Negativo): 13.406 ($67.03\%$)
* **BI-RADS 2** (Achados Benignos): 4.676 ($23.38\%$)
* **BI-RADS 3** (Provavelmente Benigno): 930 ($4.65\%$)
* **BI-RADS 4** (Suspeito de Malignidade): 762 ($3.81\%$)
* **BI-RADS 5** (Altamente Sugestivo de Malignidade): 226 ($1.13\%$)

---

## 3. Perfilamento de Lesões e Bounding Boxes (Object Detection)

### 3.1 Proporção de Imagens com Lesão vs Background
* **Imagens sem achados focais (`['No Finding']`):** 18.232 imagens ($\mathbf{91.16\%}$).
* **Imagens com Bounding Box anotada:** 1.768 imagens ($\mathbf{8.84\%}$), somando **2.254 anotações de lesão**.
* Distribuição de caixas por imagem positiva:
  * 1 lesão: 1.430 imagens ($80.9\%$)
  * 2 lesões: 239 imagens ($13.5\%$)
  * 3 a 7 lesões: 99 imagens ($5.6\%$)

### 3.2 Categorias de Achados Locais
* **Massas (`Mass`):** 1.123 anotações isoladas ($49.8\%$) + 82 associadas a calcificações.
* **Calcificações Suspeitas (`Suspicious Calcification`):** 402 anotações ($17.8\%$).
* **Assimetria Focal (`Focal Asymmetry`):** 232 anotações ($10.3\%$).
* **Distorção Arquitetural (`Architectural Distortion`):** 95 anotações ($4.2\%$).
* **Assimetria Global / Linfonodos / Alterações Cutâneas:** 402 anotações restantes.

### 3.3 Geometria Espacial das Bounding Boxes (Por que 640x640 colapsa?)

| Métrica | Largura ($px$) | Altura ($px$) | Área Relativa à Imagem (%) |
| :--- | :--- | :--- | :--- |
| **Mínimo** | $24.2\,\text{px}$ | $16.2\,\text{px}$ | $0.0053\%$ |
| **Mediana Geral** | $227.9\,\text{px}$ | $244.0\,\text{px}$ | $\mathbf{0.614\%}$ |
| **Média Geral** | $266.7\,\text{px}$ | $301.0\,\text{px}$ | $1.257\%$ |
| **Máximo** | $1120.5\,\text{px}$ | $1982.4\,\text{px}$ | $27.64\%$ |

#### Dimensões por Tipo de Lesão:
* **Calcificações Suspeitas:** Mediana de $141 \times 147\,\text{px}$ (Área relativa: $\mathbf{0.231\%}$).
* **Massas Tumorais:** Mediana de $227 \times 234\,\text{px}$ (Área relativa: $\mathbf{0.578\%}$).
* **Distorções Arquiteturais:** Mediana de $187 \times 195\,\text{px}$ (Área relativa: $\mathbf{0.381\%}$).

#### Demonstração Matemática do Colapso de Downscaling Linear:
Ao comprimir uma mamografia nativa média de $3387 \times 2647$ para $640 \times 640$, a matriz sofre uma redução dimensional de:
$$\text{Fator de Escala } s = \frac{640}{3387} \approx 0.189 \quad (\text{redução de } \sim 5.3\times \text{ em cada eixo})$$

* Uma calcificação de $141 \times 147\,\text{px}$ é reduzida para apenas **$26 \times 27\,\text{px}$**.
* A menor lesão anotada no dataset ($24.2 \times 16.2\,\text{px}$) colapsa para **$4.5 \times 3.0\,\text{px}$**.
* Em uma pirâmide de features (FPN P3/P4 com stride 8 e 16), uma anomalia de $4$ pixels sequer gera um vetor representativo de ativação convolucional no mapa de features, justificando a queda catastrófica de sensibilidade observada nos experimentos anteriores da IC.
