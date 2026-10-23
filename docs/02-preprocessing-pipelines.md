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
  Utilizar as tags DICOM de janela radiológica `Window Center (0028, 1050)` e `Window Width (0028, 1051)`:
  $$I_{\text{win}}(x, y) = \text{clip}\left( \frac{\text{Pixel}(x, y) - (c - 0.5)}{w - 1} + 0.5, \ 0, \ 1 \right) \times 255$$
  onde $c = \text{Window Center}$ e $w = \text{Window Width}$. Alternativamente, quando as tags não estiverem disponíveis, aplica-se corte por percentis robustos ($p_{0.5}$ e $p_{99.5}$) para eliminar valores espúrios de saturação do sensor.

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
