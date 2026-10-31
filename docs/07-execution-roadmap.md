# Roadmap de Execução: Da Ingestão dos Dados ao Treinamento em MLOps

Este documento detalha o plano de ação técnico estruturado para guiar o desenvolvimento do projeto a partir do momento em que o dataset **VinDr-Mammo** estiver baixado no servidor de treino.

---

## 🗺️ Visão Geral em Fases

```text
[Dataset Baixado] 
       │
       ▼
[Fase 1: Profiling de Dados & EDA Radiológica] 
       │ (Inspeção de DICOMs, CSVs, tags VOI LUT, distribuição BI-RADS e lesões)
       ▼
[Fase 2: Modelagem de Domínio & Adapters de Leitura (DDD)]
       │ (Entidades puras, parser VinDr, partição anti-leakage por paciente)
       ▼
[Fase 3: Laboratório de Pré-Processamento: Candidatos & Benchmark]
       │ (Comparativo: Clássico vs Deep Segmentation; VOI LUT adaptativo vs Tags)
       ▼
[Fase 4: Pipeline de Extração de Features & Amostragem (Tiling/Patches)]
       │ (Preservação de escala nativa para microcalcificações, filtragem de fundo)
       ▼
[Fase 5: Model Runner & Integração MLOps (MLflow Remoto)]
       │ (Treinamento do baseline Cascade R-CNN / DETR com logging remoto)
```

---

## 📌 Fase 1: Profiling de Dados e EDA Radiológica

O primeiro passo prático após o download é realizar uma **Análise Exploratória Orientada a Domínio (EDA)** para entender a realidade dos dados brutos, evitando suposições incorretas.

### Tarefas Chave:
1. **Auditoria de Integridade dos Metadados:**
   * Inspecionar `finding_annotations.csv` e `breast-level_annotations.csv`.
   * Analisar discrepâncias inter-observador: identificar como lidar com as anotações independentes dos 3 radiologistas (ex: aplicar *Majority Voting* ou União/Interseção de caixas via *Weighted Boxes Fusion - WBF*).
   * Mapear a distribuição das classes de lesão: Massa (*Mass*), Calcificação (*Calcification*), Assimetria (*Asymmetry*), Distorção Arquitetural (*Architectural Distortion*).
2. **Inspeção Técnica dos Arquivos DICOM:**
   * Script de amostragem estatística sobre os cabeçalhos DICOM (`pydicom`):
     * Tags `PhotometricInterpretation`: verificar a proporção de `MONOCHROME1` vs `MONOCHROME2`.
     * Presença e confiabilidade de `WindowCenter` `(0028, 1050)` e `WindowWidth` `(0028, 1051)`.
     * Relação entre dimensões das matrizes (resolução nativa média: ex. $2800 \times 3500$).
     * Profundidade real de bits (`BitsStored` vs `BitsAllocated`: 12, 14 ou 16 bits).

---

## 📌 Fase 2: Domínio Puro & Contratos (Ports and Adapters)

Antes de processar imagens em massa, estabilizamos a camada de Domínio e as fronteiras do sistema.

### Tarefas Chave:
1. **Modelagem de Entidades e Value Objects (`src/domain/`):**
   * `MammogramImage`, `PatientCase`, `BoundingBox`, `LesionAnnotation`, enums de lateralidade (`L`, `R`) e incidência (`CC`, `MLO`).
   * Invariantes de validação geométrica embutidas nas entidades (exceções explícitas para caixas fora dos limites).
2. **Implementação do `VinDrDatasetReaderAdapter` (`src/infrastructure/datasets/`):**
   * Parser que consome os CSVs e diretórios DICOM do VinDr-Mammo e instancia os agregados `PatientCase`.
3. **Divisão Estratificada Anti-Leakage:**
   * Algoritmo de particionamento (Treino / Validação / Teste) agrupado estritamente por `patient_id`.
   * Estratificação considerando a presença de lesões positivas e distribuição BI-RADS para evitar desbalanceamento entre os splits.

---

## 📌 Fase 3: Laboratório de Candidatos de Pré-Processamento

Em vez de eleger arbitrariamente um método único de pré-processamento, estruturaremos um **benchmark comparativo controlado** entre candidatos com perfis distintos de complexidade.

### Comparativo de Candidatos:

| Componente | Candidato A (Abordagem Clássica CV) | Candidato B (Abordagem Deep Learning) | Critério de Decisão |
| :--- | :--- | :--- | :--- |
| **Segmentação da Silhueta** | Otsu + Morfologia Matemática + CCA (*Connected Component Analysis*) para manter o maior componente. | U-Net leve pré-treinada ou segmentador leve binário. | Custo computacional vs precisão de borda. (A abordagem clássica costuma ser determinística, rápida e suficiente para a mama externa). |
| **Remoção do Músculo Peitoral (MLO)** | Transformada de Hough para detecção de retas / crescimento de regiões a partir do vértice superior. | Rede de segmentação semântica (U-Net) treinada para máscara peitoral. | Preservação de tecido fibroglandular adjacente vs falsos-positivos de massa. |
| **Janelamento Radiológico (VOI LUT)** | Tags de fábrica do DICOM (`WindowCenter` / `WindowWidth`). | Algoritmo adaptativo baseado em supressão da moda de gordura (GMM) e percentis do parênquima denso. | Contraste de lesões isodensas e estabilidade de histograma. |
| **Realce de Contraste** | Imagem em escala de cinza normalizada sem filtros. | CLAHE adaptativo com controle estrito de `clipLimit` ($1.5 - 2.5$). | Ganho em detecção de massa vs introdução de ruído granular espúrio. |

---

## 📌 Fase 4: Engenharia de Resolução & Pipeline de Tiling

Esta etapa aborda a causa raiz das falhas em modelos anteriores de mamografia.

### Tarefas Chave:
1. **Pipeline de Tiling com Overlap:**
   * Gerador de janelas deslizantes (ex: $1024 \times 1024$ com $20\%$ de sobreposição).
   * Algoritmo de projeção e recorte de Bounding Boxes com regras de preservação de lesões fronteiriças.
   * Filtro de descarte de patches compostos exclusivamente por ar ($>90\%$ de pixels escuros).
2. **Reversão e Fusão na Inferência:**
   * Módulo de *Stitching* e aplicação de Soft-NMS / WBF para reconstruir a imagem completa na predição final.

---

## 📌 Fase 5: Setup de MLOps & Execução Experimental

Com os dados tratados e os artefatos estruturados:

### Tarefas Chave:
1. **Conexão do `MLflowTrackerAdapter`:**
   * Teste de conectividade com o túnel seguro da Cloudflare do seu servidor caseiro.
   * Registro automatizado de parâmetros de treino, tags de Git commit e métricas escalonadas ($mAP_{50}$, $mAP_{75}$, $mAP_s$, $mAP_m$, $mAP_l$).
2. **Treinamento do Modelo Baseline:**
   * Configuração de um detector de referência (ex: Cascade R-CNN com âncoras reescaladas para patches médicos).
   * Execução de *Smoke Run* (5 épocas em subconjunto mínimo) para validar se o pipeline completo funciona sem falhas de I/O ou estouro de memória de GPU.
3. **Experimentação Sistemática:**
   * Variar hiperparâmetros, backbones (ResNet-50 vs ConvNeXt vs Swin/Transformer) e pipelines de pré-processamento registrados no MLflow.
