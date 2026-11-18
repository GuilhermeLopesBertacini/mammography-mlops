# ADR-0002: Stack de Treinamento (ConvNeXt + Cascade R-CNN), Estratégia de Dados e Observabilidade

* **Status:** Aceito
* **Data:** 2026-11-17
* **Decisores:** Guilherme Lopes Bertacini

---

## 1. Contexto

Com a base hexagonal implementada (`domain`, `ports`, `application/tiling_engine.py`, `infrastructure/datasets/vindr_reader.py`) e o dataset VinDr-Mammo montado em `/dados/` na VM `liaa-server-01` (GPU `NVIDIA GeForce RTX 5060 Ti 16GB`, 5.000 estudos / 20.000 imagens DICOM), é necessário fechar as decisões técnicas para o pipeline de treinamento fim-a-fim.

### Profiling de Anotações (`data/finding_annotations.csv`)
- **Total de linhas:** `20.486` (`16.391` em `split == "training"`, `4.095` em `split == "test"`).
- **Cobertura de Bounding Boxes:** Apenas `1.768` de `20.000` imagens (**8,84%**) e `924` de `5.000` estudos possuem ao menos 1 bbox (`2.254` bboxes no total).
- **Estrutura de `finding_categories`:** Lista multi-label por bbox (ex: `['Suspicious Calcification', 'Mass']` ocorre em 82 caixas).
- **Distribuição das 10 categorias em linhas com bbox (`training` / `test`):**
  - `Mass`: 989 / 237
  - `Suspicious Calcification`: 428 / 115
  - `Focal Asymmetry`: 216 / 53
  - `Architectural Distortion`: 95 / 24
  - `Asymmetry`: 77 / 20
  - `Suspicious Lymph Node`: 46 / 11
  - `Skin Thickening`: 45 / 12
  - `Nipple Retraction`: 30 / 7
  - `Global Asymmetry`: 20 / 6
  - `Skin Retraction`: 15 / 3
- **Linhas sem bbox (`18.232` linhas):** 100% marcadas como `['No Finding']`, sendo majoritariamente `BI-RADS 1` (13.406) e `BI-RADS 2` (4.664), mas contendo `BI-RADS 3` (126), `BI-RADS 4` (34) e `BI-RADS 5` (2).

---

## 2. Decisões

### 2.1. Orquestração, Treino e Arquitetura do Modelo
1. **Hydra + Pydantic (Fail-Fast):** Configurações modulares em `configs/` (`data`, `preprocessing`, `model`, `trainer`). O `DictConfig` do Hydra é validado por schemas Pydantic na fronteira (`cli/train.py`) antes de invocar casos de uso.
2. **PyTorch Lightning (`bf16-mixed`):** Gerenciamento de loop, AMP nativo da arquitetura Blackwell (RTX 5060 Ti) e logging via `MLFlowLogger` usando credenciais de `.env`.
3. **Backbone + Detector (`convnext_small` + FPN + Cascade R-CNN):**
   - `timm` (`convnext_small`, ~50M params) com `features_only=True` como extrator multi-escala.
   - `FPN` + 3 estágios de `CascadeRoIHead` (limiares IoU `0.5 → 0.6 → 0.7`) instanciados como `nn.Module` puro dentro do `LightningModule` (sem `mmengine.Runner`), evitando conflito de trainers e problemas de compilação do `mmcv`.
4. **Normalização Agnóstica a Micro-Batch:** Proibido o uso de `BatchNorm2d` não-congelado na FPN e nas cabeças Cascade; utilizar `GroupNorm`, `LayerNorm` ou `FrozenBatchNorm2d` para compatibilidade com `accumulate_grad_batches` e micro-batches de tamanho `2–4` (batch efetivo `16` ou `32`).

### 2.2. Pré-processamento, Tiling e Estratégia de Batches
1. **Cache Offline em `/dados/`:** Crop de fundo preto (com translação de coordenadas das bboxes) + janelamento em 3 canais (ex: linear + CLAHE) + fatiamento via `TilingEngine` (`patch_size=1024`, sem downsampling) executados **uma única vez offline**. Proibido ler DICOM ou aplicar CLAHE on-the-fly durante o `DataLoader`.
2. **Particionamento Anti-Leakage:** Manter `split == "test"` oficial intocado. Dividir `split == "training"` em `train` / `val` agrupado estritamente por `study_id` (`StratifiedGroupKFold`) com `seed` fixa registrada no Hydra/MLflow.
3. **Balanceamento de Patches e Tratamento de `No Finding`:**
   - **Treino:** Utilizar 100% dos patches positivos + amostragem controlada de patches negativos (razão `1:1` a `1:2` pos:neg). Patches negativos devem ser extraídos apenas de tecido mamário em estudos `BI-RADS 1`/`2` (excluindo `BI-RADS 4`/`5` sem bbox para evitar falsos negativos no rótulo).
   - **Validação/Teste:** Avaliar sobre todos os patches do split para medir taxa real de falsos positivos.
4. **Augmentation Conservador (`albumentations`):** `HorizontalFlip`, `ShiftScaleRotate` sutil (`±5–10°`) e ajuste leve de brilho/contraste. Proibido `VerticalFlip` e ruído gaussiano agressivo.

### 2.3. Observabilidade (MLflow vs. Grafana)
- **MLflow (`MLFLOW_TRACKING_URI` via `.env`):** Fonte da verdade para hiperparâmetros, métricas por época (`loss`, `mAP`) e checkpoints.
- **Grafana (`../observability-infra`):** Responsável exclusivamente por métricas de infraestrutura/hardware da VM (`node_exporter` + `dcgm-exporter`), sem acoplamento no código Python de treinamento.

---

## 3. Topologia de Dados e Persistência

| Camada / Artefato | Localização | Propriedade | Comportamento e Regras |
| :--- | :--- | :--- | :--- |
| **Dataset Bruto (DICOM)** | `/dados/vindr-mammo/.../images` | Read-only | 5.000 estudos em NVMe dedicado (`/dev/nvme1n1p2`). Imutável. |
| **Metadados & CSVs** | `data/*.csv` | Read-only | Anotações oficiais e splits. |
| **Cache de Patches** | `/dados/vindr-cache/` | Read/Write | Patches pré-processados (crop + CLAHE + tiling) + manifesto Parquet com `study_id` e split. |
| **Tracking Remoto** | Servidor MLflow (`.env`) | Append-only | Hiperparâmetros Hydra, métricas e artefatos de checkpoint. |

---

## 4. Impasses Conhecidos e Riscos

- **Multi-label vs. Single-label em `finding_categories`:** Detectores R-CNN padrão assumem 1 classe por bbox, enquanto ~7% das linhas com bbox no VinDr-Mammo possuem múltiplas categorias simultâneas (além de classes raras como `Skin Retraction` com apenas 15 exemplos em treino). A decisão entre **baseline binário (`Lesion` vs `Background`)** ou **multiclasse** (com critério de desempate/agrupamento) deve ser fechada antes da definição final do `num_classes` no YAML do modelo.
- **Concorrência na GPU da VM:** O serviço `llama-server` (~10 GB VRAM) deve ser pausado antes de execuções de treino na RTX 5060 Ti (16 GB).
- **Schedulers com `accumulate_grad_batches`:** Todo `LR Scheduler` no Lightning deve usar `self.trainer.estimated_stepping_batches` para não contar micro-batches em vez de passos reais do otimizador.
