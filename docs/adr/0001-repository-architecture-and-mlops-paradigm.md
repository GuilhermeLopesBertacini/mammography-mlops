# ADR-0001: Adoção de Monorepo Modular e Arquitetura Hexagonal (Ports and Adapters)

* **Status:** Aceito
* **Data:** 2026-10-23
* **Decisores:** Guilherme Lopes Bertacini

---

## 1. Contexto
O projeto de Iniciação Científica original acumulou débitos técnicos comuns em pesquisas de visão computacional: acoplamento direto com o framework MMDetection v3, scripts procedurais espaguete, ausência de testes automatizados e dependência de hardware restrito (Jetson Orin 8GB com downsampling forçado para 640x640). 

Com a transição para treinamento em servidor dedicado na universidade e integração com infraestrutura remota de observabilidade (MLflow e Grafana via túnel Cloudflare), o projeto requer uma base de software que garanta reprodutibilidade, desacoplamento e rigor de engenharia de machine learning.

---

## 2. Decisão

1. **Adoção de Monorepo Modular:** O código-fonte de ingestão, pré-processamento, modelagem e integração de infraestrutura residirá em um único repositório Git, organizado em camadas concêntricas estritas com fronteiras bem definidas.
2. **Separação por Ports and Adapters (Arquitetura Hexagonal):**
   * **Domínio:** Modelagem de entidades radiológicas puras (`PatientCase`, `MammogramImage`, `BoundingBox`) sem dependências de PyTorch, OpenCV ou MLflow.
   * **Aplicação:** Casos de uso orquestradores (`PreprocessDatasetUseCase`, `TrainModelUseCase`).
   * **Portas:** Interfaces abstratas (`DatasetReaderPort`, `PreprocessorPort`, `ModelRunnerPort`, `ExperimentTrackerPort`).
   * **Adapters:** Implementações concretas de tecnologia (`INbreastAdapter`, `VinDrMammoAdapter`, `OpenCVAdapter`, `MLflowAdapter`, `MMDetectionAdapter`, `LightningAdapter`).
3. **Gerenciamento de Dependências com `uv`:** O ambiente Python será gerenciado via `uv` com arquivo `pyproject.toml` e lockfile determinístico (`uv.lock`).

---

## 3. Consequências

### Positivas
* **Agnosticismo de Framework:** A troca ou comparação entre MMDetection, PyTorch Lightning ou detectores baseados em Transformer (DETR) não exigirá reescrita de código de dados ou de avaliação.
* **Testabilidade Real:** Toda a lógica de domínio (como recálculo geométrico de bounding boxes pós-recorte e divisão de folds anti-leakage por paciente) pode ser testada em milissegundos com `pytest` sem carregar modelos pesados em GPU.
* **Isolamento de Falhas:** Problemas de conectividade com o túnel do MLflow são contornados pelo Adapter de rastreamento com buffer local, sem quebrar o laço de treinamento.

### Negativas / Custos
* Maior investimento inicial em boilerplate de interfaces e DTOs em comparação com a execução de um script procedural único. Esse custo é aceito e compensado pela estabilidade e manutenibilidade a longo prazo.
