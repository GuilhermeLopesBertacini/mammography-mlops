# 04. Benchmark e Seleção de Arquiteturas de Detecção

A escolha da arquitetura do detector é um trade-off entre **precisão de localização espacial em limiares estritos de IoU**, **sensibilidade a pequenas anomalias ($mAP_s$)**, **estabilidade de convergência** e **demanda de memória de vídeo (VRAM)**.

Neste documento, analisamos as famílias de arquiteturas de visão computacional aplicadas à mamografia.

---

## 🔬 Análise Comparativa de Famílias de Detectores

| Família | Exemplos de Modelos | Mecanismo de Proposta | Sensibilidade a Lesões Pequenas ($mAP_s$) | Taxa de Falsos-Positivos | Requisito de VRAM | Estabilidade de Treinamento |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Dois Estágios (Two-Stage)** | **Cascade R-CNN**, Faster R-CNN | RPN denso $\to$ RoIAlign $\to$ Múltiplos estágios sequenciais com IoUs crescentes | **Alta a Excelente** (se âncoras calibradas) | **Muito Baixa** (alta especificidade) | Moderada a Alta ($12 - 24$ GB) | **Muito Alta** (gradientes bem condicionados) |
| **Estágio Único Clássico** | **RetinaNet**, FCOS | Propostas densas diretas + Focal Loss | Baixa a Moderada | Moderada (muitos falsos alarmes em tecido denso) | Baixa ($8 - 12$ GB) | Alta |
| **Estágio Único Moderno** | **YOLOv8 / YOLOv11** | Decoupled Anchor-Free Heads + Task-Aligned Assigner | Moderada | Moderada | Baixa ($6 - 12$ GB) | Alta |
| **Transformers (DETR)** | **Deformable DETR**, **RT-DETR**, Co-DETR | Atenção Deformável Multiescala + Bipartite Matching (sem NMS) | **Excelente** (atenção global e local em alta resolução) | Baixa | Alta ($16 - 32$ GB) | Moderada (requer mais épocas para convergir) |

---

## 1. Detectores de Dois Estágios: Cascade R-CNN em Profundidade

O estudo original da IC demonstrou que o **Cascade R-CNN** é conceitualmente ideal para mamografia, mas sua implementação falhou em lesões diminutas devido à resolução e calibração de âncoras.

### Por que o Cascade R-CNN se destaca em diagnóstico radiológico?
1. **O Paradoxo do Limiar de IoU:**
   * Em detectores convencionais (como Faster R-CNN com limiar fixo $\text{IoU} = 0.5$), propostas com sobreposição de $50\%$ com o tumor já são consideradas positivas. Para radiologia, uma caixa com $50\%$ de IoU frequentemente corta a margem espiculada de um tumor maligno ao meio, inutilizando a análise de contorno.
   * Se aumentarmos o limiar diretamente para $\text{IoU} = 0.7$, a quantidade de propostas positivas despenca dramaticamente, provocando desbalanceamento severo e colapso do classificador (*overfitting geométrico*).
2. **A Solução em Cascata:**
   * O Cascade R-CNN encadeia 3 classificadores/regressores sucessivos treinados com limiares progressivos:
     $$\text{Estágio 1: } \text{IoU} = 0.5 \quad \longrightarrow \quad \text{Estágio 2: } \text{IoU} = 0.6 \quad \longrightarrow \quad \text{Estágio 3: } \text{IoU} = 0.7$$
   * A saída do regressor do Estágio 1 vira a entrada do Estágio 2, refinando progressivamente a posição da caixa delimitadora sem empobrecer o conjunto de treinamento.

```text
Entrada FPN ──► [RoIAlign] ──► [Box Head 1 (IoU 0.5)] ──► BBox Refinada 1
                                                              │
               ┌──────────────────────────────────────────────┘
               ▼
           [RoIAlign] ──► [Box Head 2 (IoU 0.6)] ──► BBox Refinada 2
                                                         │
               ┌─────────────────────────────────────────┘
               ▼
           [RoIAlign] ──► [Box Head 3 (IoU 0.7)] ──► Predição Final de Alta Qualidade
```

---

## 2. Detectores Modernos Baseados em Transformer: DETR e Variantes

### 2.1 O Problema do DETR Vanilla em Imagens Médicas
O DETR original aplica auto-atenção sobre a imagem inteira. O custo computacional da atenção total é quadrático: $\mathcal{O}(H^2 W^2)$. Em mamografias de alta resolução ($3000 \times 4000$), isso causaria estouro imediato de memória (*Out of Memory*), mesmo em GPUs industriais de 80 GB.

### 2.2 Deformable DETR e RT-DETR (A Solução Eficiente)
* **Atenção Deformável:** Em vez de calcular atenção com todos os pixels do mapa de features, cada consulta foca apenas em um conjunto pequeno e fixo de pontos de amostragem aprendidos ao redor de uma coordenada de referência (tipicamente 4 pontos por escala).
* **Vantagem Clínica:** Permite processar pirâmides de features em resoluções mais altas (P3, P4, P5), capturando simultaneamente calcificações diminutas e grandes distorções teciduais.
* **Eliminação do NMS (Non-Maximum Suppression):** O bipartite matching (Hungarian Algorithm) força o modelo a prever exatamente uma caixa por lesão, evitando a necessidade de ajuste manual de limiares de supressão de sobreposição.

---

## 3. Otimização de Funções de Perda (Loss Functions) para Assimetria Extrema

O desafio mais severo em mamografia é a **esparsidade de lesões**: em média, mais de $99.8\%$ da área tecidual de uma mamografia corresponde a tecido normal (benigno/negativo).

### 3.1 Classificação: Focal Loss vs Cross-Entropy
A Cross-Entropy padrão é facilmente dominada pela soma cumulativa de gradientes vindos de milhares de amostras negativas fáceis (fundo ou parênquima homogêneo). A **Focal Loss** resolve este problema adicionando um termo modulador:

$$\text{FL}(p_t) = -\alpha_t (1 - p_t)^\gamma \log(p_t)$$

onde $\gamma \approx 2.0$ atenua drasticamente a contribuição de amostras fáceis ($p_t > 0.8$), forçando o modelo a focar exclusivamente em regiões fronteiriças e de difícil discriminação.

### 3.2 Regressão de Bounding Box: CIoU e EIoU
Em vez de regressão linear Smooth-L1 ($|x - \hat{x}|$), utilizamos perdas baseadas em métricas geométricas completas:
* **CIoU (Complete IoU):** Penaliza a distância entre centros de caixas, a sobreposição de área e a discrepância na razão de aspecto (*aspect ratio*).
* **EIoU (Efficient IoU):** Decompõe explicitamente a perda de largura e altura de forma desacoplada, acelerando o ajuste fino de margens tumorais espiculadas.

---

## 4. Recomendações Experimentais para o Repositório

Como o objetivo é testar múltiplas hipóteses com rastreamento no MLflow, definimos a seguinte matriz de experimentos arquiteturais:

1. **Baseline Robusto:** **Cascade R-CNN** com backbone ResNet-50 / ConvNeXt-Small e FPN, ajustando os tamanhos de âncoras para a escala médica mamográfica (âncoras menores na base da pirâmide: $[4, 8, 16, 32, 64]$).
2. **Candidato SOTA Transformer:** **RT-DETR (Real-Time DETR)** ou **Deformable DETR**, avaliando a precisão de localização sem dependência de âncoras ou NMS heurístico.
3. **Candidato Leve:** **YOLOv11** treinado com entrada de alta resolução em patches, avaliando o trade-off de velocidade e acurácia.
