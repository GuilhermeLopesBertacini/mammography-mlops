# 02. Engenharia de Pré-processamento e Tratamento de Imagens Mamográficas

O pré-processamento em mamografia não é uma etapa cosmética de ajuste visual: é uma **transformação matemática com impacto direto na convergência de gradientes e na preservação de invariantes anatômicas**. 

Neste documento, dissecamos as técnicas essenciais de processamento de imagens radiológicas, seus trade-offs, potenciais armadilhas e a formulação matemática de cada etapa.

---

## 1. Tratamento Radiológico do Contêiner DICOM

Arquivos DICOM de mamografia digital (FFDM) contêm particularidades que diferem drasticamente de imagens naturais padrão RGB (JPEG/PNG):

### 1.1 Fotometria: `MONOCHROME1` vs `MONOCHROME2`
* Na tag DICOM `(0028, 0004) Photometric Interpretation`:
  * **`MONOCHROME1`**: Os menores valores de pixel representam o branco (alta radiodensidade), e os maiores representam o preto (ar/fundo). É o padrão tradicional de exibição em negatoscópios radiológicos antigos.
  * **`MONOCHROME2`**: Os menores valores representam o preto (fundo/ar), e os maiores representam o branco (tecido denso, ossos, tumores).
* **Armadilha:** Converter um lote de imagens sem checar essa tag inverterá as anomalias teciduais. Em imagens `MONOCHROME1`, o tecido mamário se tornará escuro e o fundo ficará claro, arruinando qualquer padrão convolucional.
* **Normalização Padronizada:**
  $$\text{Pixel}_{\text{norm}}(x, y) = \begin{cases} 
  \text{MaxVal} - \text{Pixel}(x, y), & \text{se } \text{Photometric} = \text{'MONOCHROME1'} \\ 
  \text{Pixel}(x, y), & \text{se } \text{Photometric} = \text{'MONOCHROME2'} 
  \end{cases}$$

### 1.2 Profundidade de Bits e Faixa Dinâmica
* Mamógrafos digitais capturam imagens em **12, 14 ou 16 bits** por pixel ($4.096$ a $65.536$ níveis de cinza), enquanto a maioria das bibliotecas de visão (OpenCV, PIL) opera por padrão em **8 bits** ($256$ níveis).
* **Downscaling linear ingênuo para 8 bits ($I / 256$):** Destrói sutis variações de atenuação de raios-X em massas isodensas envoltas em tecido glandular fibroso.
* **Aplicação de VOI LUT (Value of Interest Look-Up Table):**
  A janela radiológica linear padrão mapeia a faixa dinâmica de interesse $[c - w/2, c + w/2]$ para $[0, 255]$:
  $$I_{\text{win}}(x, y) = \text{clip}\left( \frac{\text{Pixel}(x, y) - (c - 0.5)}{w - 1} + 0.5, \ 0, \ 1 \right) \times 255$$
  onde $c = \text{Window Center}$ e $w = \text{Window Width}$.

### 1.3 Cálculo Dinâmico e Robusto de Window Center ($c$) e Window Width ($w$)

Na ausência das tags `(0028, 1050)` e `(0028, 1051)` — ou quando a janela do fabricante é genérica demais —, calcular $c$ e $w$ sobre a matriz inteira é um erro grave: o fundo (ar) e a gordura subcutânea/retromamária formam modas dominantes de baixa atenuação que puxam o centro para baixo, comprimindo o contraste do tecido fibroglandular e das massas tumorais.

> [!NOTE]
> Mamografia é uma projeção 2D de atenuação planar de raios-X, não Tomografia Computadorizada (TC). Portanto, os valores nativos dos pixels não são rigorosamente calibrados em Unidades Hounsfield (onde água $= 0$, ar $= -1000$ e gordura $\approx -120$ a $-90$ HU), mas seguem curvas de atenuação análogas em níveis de cinza brutos ($GL$). A mesma lógica de filtragem tecidual aplica-se isolando as densidades radiológicas relativas.

#### Algoritmo de Cálculo Focado no Tecido de Interesse:

```text
[Matriz Bruta 16-bit] ──► [Máscara Parênquima M(x,y)] ──► [Vetor P_parenquima]
                                                                  │
              ┌───────────────────────────────────────────────────┘
              ▼
   [Filtragem da Moda de Gordura / Limiar Inferior]
              │
              ▼ [Subconjunto P_denso]
   [Percentis Robustos p_low (ex: 2%) e p_high (ex: 99%)]
              │
              ├─► Window Center: c = (p_high + p_low) / 2
              └─► Window Width:  w = p_high - p_low
```

1. **Segmentação da Região Anatômica:**
   Calcula-se uma máscara booleana $M(x, y)$ da silhueta mamária (descartando o fundo de ar onde $\text{Pixel}(x, y) \le \text{Threshold}_{\text{ar}}$).
   $$P_{\text{mama}} = \{ \text{Pixel}(x, y) \mid M(x, y) = 1 \}$$

2. **Identificação e Supressão da Componente de Gordura:**
   No histograma de $P_{\text{mama}}$, o tecido adiposo forma o primeiro pico (moda) proeminente de menor densidade. Duas estratégias são empregadas:
   * **Abordagem por Moda Adiposa ($\mu_{\text{fat}}, \sigma_{\text{fat}}$):**
     O histograma global da mama é uma **mistura bimodal (Gaussian Mixture Model - GMM)**: a distribuição normal da gordura $\mathcal{N}(\mu_{\text{fat}}, \sigma_{\text{fat}}^2)$ sobrepõe-se à distribuição do parênquima $\mathcal{N}(\mu_{\text{gland}}, \sigma_{\text{gland}}^2)$.
     Define-se o limiar de corte inferior por:
     $$T_{\text{fat}} = \mu_{\text{fat}} + k \cdot \sigma_{\text{fat}}$$
     * **Por que $k \in [1.0, 1.5]$ e não $k = 3.0$?**
       * Se $k = 3.0$, elimina-se teoricamente $99.7\%$ da distribuição normal da gordura isolada. Porém, as distribuições de gordura e tecido glandular não são separadas por um vale nulo; há uma **área de sobreposição anatômica real** (tecido fibroglandular de baixa densidade, ductos e lesões incipientes imersas em estroma adiposo).
       * Com $k = 3.0$, o limiar invade profundamente o pico do tecido fibroglandular, descartando lesões isodensas.
       * Com $k \in [1.0, 1.5]$, como o corte é unilateral à direita da média, remove-se toda a cauda esquerda e o centro da gordura ($\ge 84.1\%$ a $93.3\%$ do tecido puramente adiposo), fixando o ponto de operação exatamente na região de transição com máxima especificidade, sem suprimir parênquima útil.
     Pixels com valor abaixo de $T_{\text{fat}}$ são descartados do cálculo dos parâmetros de janela.
   * **Abordagem por Percentil de Descarte de Gordura:**
     Como a mama contém entre 40% e 70% de tecido adiposo dependendo da densidade BIRADS (A a D), estabelece-se um piso $p_{\text{fat}} = \text{Percentil}(P_{\text{mama}}, \alpha)$, com $\alpha \in [20\%, 40\%]$.

3. **Derivação de $c$ e $w$ sobre o Tecido Fibroglandular Restante ($P_{\text{denso}}$):**
   Com o vetor filtrado $P_{\text{denso}} = \{ v \in P_{\text{mama}} \mid v \ge T_{\text{fat}} \}$, calculam-se os percentis de cauda para suprimir ruídos espúrios e saturação:
   $$V_{\text{min}} = \text{Percentil}(P_{\text{denso}}, \ 1.0)$$
   $$V_{\text{max}} = \text{Percentil}(P_{\text{denso}}, \ 99.5)$$

   Os parâmetros da janela radiológica são formalizados por:
   $$w = V_{\text{max}} - V_{\text{min}}$$
   $$c = \frac{V_{\text{max}} + V_{\text{min}}}{2}$$

Esse cálculo garante que 100% dos $256$ níveis de cinza da imagem convertida (8 bits) sejam dedicados a discriminar variações de densidade do parênquima glandular e nódulos, impedindo que ar e gordura consumam faixa dinâmica útil.

---

## 2. Isolamento do Parênquima Mamário e Remoção de Artefatos

Mamografias brutas contêm entre 40% e 70% de pixels de fundo totalmente escuro, além de anotações radiopacas (etiquetas de chumbo como "L", "R", "CC", "MLO"), fitas adesivas e ruído eletrônico periférico.

```text
[Imagem Bruta] ──────► [Binarização Otsu] ──────► [Análise de Componentes Conexos] ──────► [Máscara de Mama]
                                                          │ (descarta labels e ruído)
                                                          ▼
[Imagem Recortada] ◄── [Recorte de Caixa Mínima (ROI)] ◄──┴─────────────────────────────┘
```

### 2.1 Algoritmo de Extração de Silhueta Baseado em Componentes Conexos (CCA)
Em vez de escaneamentos linha-a-linha com limiares fixos (sensíveis a variações de exposição), a abordagem padrão de engenharia é:

1. **Binarização Adaptativa / Otsu:** Gera uma máscara binária inicial $B(x, y)$ separando o ar do tecido corporal.
2. **Operações Morfológicas:** 
   * Abertura morfológica (erosão seguida de dilatação com elemento estruturante elíptico) para desconectar pequenas etiquetas radiográficas unidas à pele por ruído.
3. **Seleção do Maior Componente Conexo (Largest Connected Component):**
   * Em uma mamografia válida, o maior objeto contíguo em área é invariavelmente a mama.
   * Todos os demais componentes conexos (etiquetas de texto, artefatos de poeira) são zerados na máscara.
4. **Recorte Dinâmico (Cropping) com Margem de Segurança:**
   * Extrai a caixa delimitadora mínima $(x_{\text{min}}, y_{\text{min}}, x_{\text{max}}, y_{\text{max}})$ que circunscreve a máscara da mama.
   * Adiciona-se uma margem de segurança $\delta$ (ex: 50 pixels) para evitar corte inadvertido da pele ou prolongamento axilar.

### 2.2 Reajuste Geométrico das Bounding Boxes (Invariante Espacial)
Ao aplicar o corte com deslocamento horizontal $x_{\text{offset}}$ e vertical $y_{\text{offset}}$, as anotações $B = [x, y, w, h]$ são transladadas rigorosamente:

$$x_{\text{novo}} = \max(0, \ x - x_{\text{offset}})$$
$$y_{\text{novo}} = \max(0, \ y - y_{\text{offset}})$$
$$w_{\text{novo}} = \min(W_{\text{recorte}} - x_{\text{novo}}, \ w)$$
$$h_{\text{novo}} = \min(H_{\text{recorte}} - y_{\text{novo}}, \ h)$$

Se $w_{\text{novo}} \le 0$ ou $h_{\text{novo}} \le 0$, a anotação perdeu coerência e deve disparar uma exceção de validação de domínio.

---

## 3. Segmentação e Tratamento do Músculo Peitoral (Incidência MLO)

Na incidência Médio-Lateral Oblíqua (MLO), o músculo peitoral maior aparece como uma estrutura triangular de alta radiodensidade no canto superior (esquerdo ou direito, dependendo da lateralidade).

### O Desafio Clínico e Algorítmico
* **Problema:** A intensidade de atenuação de raios-X do músculo peitoral é praticamente idêntica à de adenomas, cistos densos e carcinomas. Redes neurais frequentemente ativam âncoras na borda do músculo peitoral, gerando falso-positivos de lesão.
* **Risco da Remoção Agressiva:** Carcinomas em estágios avançados podem infiltrar o plano fascial do músculo peitoral. Cortar o músculo peitoral indiscriminadamente elimina a visualização de invasão tumoral axilar profunda.
* **Abordagens de Engenharia:**
  1. **Transformada de Hough para Retas:** Assume que a borda do músculo peitoral se aproxima de uma reta angular no quadrante superior. É rápida, mas falha em músculos com curvatura convexa pronunciada.
  2. **Segmentação por Redes Leves (UNet):** Treinar uma mini-UNet dedicada exclusivamente a gerar a máscara binária do músculo peitoral.
  3. **Preservação Guiada por Atenuação:** Em vez de deletar o músculo (preenchendo com preto), aplica-se uma atenuação de gradiente ou mantém-se a estrutura, permitindo que a rede aprenda a diferenciar a textura muscular da arquitetura desorganizada de um tumor.

---

## 4. Técnicas de Realce de Contraste Local

### 4.1 CLAHE (Contrast Limited Adaptive Histogram Equalization)
* O CLAHE opera dividindo a imagem em grade de células contextuais (*tiles*, tipicamente $8 \times 8$ ou $16 \times 16$).
* O histograma de cada bloco é equalizado de forma independente.
* **Parâmetro Crítico (`clipLimit`):** Controla o limite máximo de amplificação local de frequências. Valores entre $1.5$ e $3.0$ realçam bordas sem ruído. Valores acima de $4.0$ geram granulação artificial que mimetiza falsas microcalcificações.
* **Interpolação Bilinear:** Os limites entre os blocos são suavizados por interpolação para eliminar descontinuidades em formato de grade (*grid artifacts*).

### 4.2 Unsharp Masking (Máscara de Desfoque)
* Fórmula: $I_{\text{sharp}} = I + \alpha \cdot (I - G_\sigma(I))$, onde $G_\sigma$ é um filtro Gaussiano de desfoque.
* Destaca bordas espiculadas de tumores malignos, porém possui menor adaptabilidade espacial do que o CLAHE.

---

## 5. O Dilema da Resolução: Redimensionamento Global vs Tiling (Patching)

Este foi o **gargalo técnico determinante** do experimento anterior na Jetson (redimensionamento direto para $640 \times 640$).

### Comparação Numérica de Escala
* Uma imagem nativa de mamógrafo possui cerca de $3000 \times 4000 = 12.000.000$ pixels.
* Uma microcalcificação típica tem entre $100\,\mu\text{m}$ e $300\,\mu\text{m}$, ocupando cerca de $4 \times 4$ a $10 \times 10$ pixels na resolução original.
* Ao comprimir a imagem inteira para $640 \times 640$ (redução de ~30x no número total de pixels), um aglomerado de microcalcificações é colapsado para um único pixel médio atenuado ou desaparece por completo durante a interpolação.

```
Estratégia A: Redimensionamento Global (640x640 ou 1024x1024)
┌──────────────────────────────────────┐
│ Mama inteira comprimida              │  => Perda total de microestruturas
│ (Contexto global alto, detalhe baixo)│  => Baixa demanda de VRAM
└──────────────────────────────────────┘

Estratégia B: Tiling / Janelas Deslizantes com Overlap
┌──────────┬──────────┬──────────┐
│ Patch 1  │ Patch 2  │ Patch 3  │  => Resolução espacial nativa 100% preservada
│ (1024px) │ (1024px) │ (1024px) │  => Microcalcificações perfeitamente visíveis
├──────────┼──────────┼──────────┤  => Requer algoritmo de fusão/NMS pós-inferência
│ Patch 4  │ Patch 5  │ Patch 6  │
└──────────┴──────────┴──────────┘
```

### A Abordagem de Alta Precisão: Pipeline de Tiling (Janelas Deslizantes)
1. **Recorte de Patches com Sobreposição (*Overlap*):**
   * Extrair blocos de $1024 \times 1024$ pixels com $20\%$ de sobreposição (para evitar que uma lesão seja cortada exatamente na borda do patch).
2. **Mapeamento de Ground-Truth Bounding Boxes:**
   * Bounding boxes que caem inteiramente dentro do patch são mantidas com coordenadas locais.
   * Caixas que cruzam a borda têm a área recalculada; se mais de $50\%$ da caixa residir no patch, ela é retida com corte (*clipping*); caso contrário, é ignorada naquele patch específico.
3. **Filtragem de Patches Vazio de Fundo:**
   * Patches contendo mais de $90\%$ de pixels de fundo preto são descartados do treino, economizando até $60\%$ do tempo computacional.
4. **Incorrência Pós-Treino (Stitching & Soft-NMS):**
   * Durante a inferência, as caixas detectadas em cada patch são projetadas de volta nas coordenadas globais da mamografia.
   * Aplica-se **Soft-NMS** global para fundir detecções redundantes geradas na área de sobreposição.
