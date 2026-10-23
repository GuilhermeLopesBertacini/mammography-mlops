# Mammography MLOps: Detecção e Diagnóstico Assistido de Lesões Mamárias

Pipeline profissional de Machine Learning Operations (MLOps) e Visão Computacional para detecção, localização e classificação de lesões e massas tumorais em mamografias digitais (FFDM).

O projeto é estruturado sob os princípios de **Domain-Driven Design (DDD)**, arquitetura **Ports and Adapters (Hexagonal)**, e princípios **SOLID**, desacoplando a lógica de domínio radiológico das ferramentas e frameworks de infraestrutura (PyTorch, MMDetection, MLflow, OpenCV, DVC).

---

## 🎯 Objetivo da Pesquisa e Engenharia

1. **Rigor Clínico e Teórico:** Transcender a modelagem ingênua de visão computacional, incorporando o conhecimento do domínio radiológico (incidências CC/MLO, remoção de artefatos, análise de simetria bilateral e tratamento de microcalcificações vs massas).
2. **Reprodutibilidade e Rastreabilidade Completa:** Rastrear 100% dos hiperparâmetros, linhagem de dados (*data lineage*), curvas de aprendizado, matrizes de confusão e checkpoints via **MLflow** remoto e **Grafana**.
3. **Escalabilidade Arquitetural:** Suportar ingestão agnóstica de múltiplos datasets (INbreast, CBIS-DDSM, VinDr-Mammo) e alternância entre múltiplos backends de treinamento (PyTorch nativo, PyTorch Lightning, MMDetection).

---

## 📂 Mapa de Estudos e Documentação Técnica

Antes da implementação do código de execução, o repositório estabelece a fundamentação teórica e técnica detalhada em [`docs/`](docs/):

* [**01. Benchmark e Seleção de Datasets**](docs/01-datasets-benchmark.md): Análise comparativa entre INbreast, CBIS-DDSM, VinDr-Mammo, EMBED e CMMD (volume, anotações, licenças e desafios).
* [**02. Engenharia de Pré-processamento**](docs/02-preprocessing-pipelines.md): Estudo aprofundado sobre extração de silhueta, remoção de músculo peitoral, equalização de contraste (CLAHE), resolução espacial e amostragem por *tiling* (patching).
* [**03. Representação de Features e Raciocínio Multi-View**](docs/03-feature-representation-and-multiview.md): Fusão de incidências ipsilaterais (CC + MLO), simetria bilateral e pré-treinamento específico em domínio médico (RadImageNet vs ImageNet).
* [**04. Benchmark de Arquiteturas de Detecção**](docs/04-model-architectures-benchmark.md): Comparação sistemática entre detectores de dois estágios (Cascade R-CNN), detectores baseados em Transformer (Deformable DETR, RT-DETR, Co-DETR) e de estágio único (YOLOv11, RetinaNet).
* [**05. Avaliação de Frameworks de Treinamento**](docs/05-training-frameworks-and-tooling.md): Trade-offs entre PyTorch nativo, PyTorch Lightning e MMDetection.
* [**06. Arquitetura de Software (DDD & Ports and Adapters)**](docs/06-mlops-architecture-and-ddd.md): Desenho estrutural das camadas de Domínio, Aplicação e Infraestrutura para Machine Learning.
* [**Decisões Arquiteturais (ADRs)**](docs/adr/): Registro histórico das decisões fundamentais tomadas no projeto.

---

## 🏗️ Estrutura do Monorepo

```text
mammography-mlops/
├── README.md
├── docs/                           # Documentação e estudos teóricos aprofundados
│   ├── 01-datasets-benchmark.md
│   ├── 02-preprocessing-pipelines.md
│   ├── 03-feature-representation-and-multiview.md
│   ├── 04-model-architectures-benchmark.md
│   ├── 05-training-frameworks-and-tooling.md
│   ├── 06-mlops-architecture-and-ddd.md
│   └── adr/                        # Architectural Decision Records
├── configs/                        # Configurações declarativas (Hydra / YAML)
│   ├── data/
│   ├── experiment/
│   └── model/
├── src/                            # Código fonte sob Ports & Adapters
│   ├── domain/                     # Entidades, Value Objects e Regras puras
│   ├── ports/                      # Contratos abstratos (Interfaces)
│   ├── application/                # Casos de uso (Pipelines de dados e treino)
│   └── infrastructure/             # Adapters concretos (MLflow, OpenCV, Torch)
└── tests/                          # Testes unitários e de integração
```
