# 06. Arquitetura de Software: DDD e Ports and Adapters para MLOps

O fracasso mais comum em projetos de inteligência artificial aplicados à saúde é o **código espaguete experimental**: scripts que misturam leitura de arquivos DICOM, operações do OpenCV, chamadas do PyTorch e escrita de logs em um único arquivo procedural.

Neste documento, estabelecemos a arquitetura de software profissional do repositório, fundamentada em **Domain-Driven Design (DDD)** e **Arquitetura Hexagonal (Ports and Adapters)**.

---

## 🏛️ 1. O Modelo de Domínio (DDD) em Visão Computacional Médica

O Domínio é o núcleo do sistema: contém a lógica médica e matemática pura, sem nenhuma dependência de bibliotecas de infraestrutura (como OpenCV, PyTorch, MLflow ou banco de dados).

```text
[Domínio Puro - Python Standard Library]
├── Value Objects:
│   ├── Coordinate(x: float, y: float)
│   ├── BoundingBox(x_min: float, y_min: float, width: float, height: float)
│   ├── Laterality (Enum: LEFT, RIGHT)
│   ├── ViewProjection (Enum: CC, MLO)
│   └── LesionType (Enum: MASS, CALCIFICATION, ASYMMETRY, DISTORTION)
│
├── Entidades:
│   └── MammogramImage (id, path, laterality, view, dimensions, annotations)
│
└── Agregado (Aggregate Root):
    └── PatientCase (patient_id, study_date, images: List[MammogramImage])
```

### Invariantes de Domínio (Regras de Negócio Médicas que Nunca Podem Ser Violadas):
1. **Invariante Espacial de Bounding Box:** Uma anotação não pode ter coordenadas negativas nem extrapolar a largura e altura da matriz da imagem:
   $$x_{\text{min}} \ge 0, \quad y_{\text{min}} \ge 0, \quad x_{\text{min}} + w \le W_{\text{imagem}}, \quad y_{\text{min}} + h \le H_{\text{imagem}}$$
2. **Invariante de Isolamento do Paciente (*Anti-Leakage*):** Todos os exames pertencentes ao mesmo `PatientCase` devem residir obrigatoriamente no mesmo conjunto de partição (todos em Treino OU todos em Validação OU todos em Teste). O particionamento aleatório por imagem é proibido por violar a independência estatística clínica.
3. **Invariante de Preservação Anatômica pós-Crop:** Se uma operação geométrica descartar mais de $50\%$ da área de uma lesão rotulada, o corte é invalidado ou a anotação deve ser explicitamente rejeitada com log de auditoria.

---

## 🔌 2. Arquitetura Hexagonal (Ports and Adapters)

A arquitetura hexagonal separa o sistema em três círculos concêntricos: **Domínio**, **Aplicação** e **Infraestrutura**. A regra de dependência é estrita: as dependências sempre apontam para dentro.

```text
               ┌────────────────────────────────────────────────────────┐
               │                     INFRAESTRUTURA                     │
               │                                                        │
               │   [INbreastReader]       [VinDrMammoReader]            │
               │          │                        │                    │
               │          ▼                        ▼                    │
               │ ┌────────────────────────────────────────────────────┐ │
               │ │                     APLICAÇÃO                      │ │
               │ │                                                    │ │
               │ │   [IngestData]  [PreprocessPipeline]  [TrainModel] │ │
               │ │         │                │                 │       │ │
               │ │         ▼                ▼                 ▼       │ │
               │ │       ┌────────────────────────────────────┐       │ │
               │ │       │              DOMÍNIO               │       │ │
               │ │       │                                    │       │ │
               │ │       │  PatientCase, MammogramImage,      │       │ │
               │ │       │  BoundingBox, Invariantes Clínicas │       │ │
               │ │       └────────────────────────────────────┘       │ │
               │ │         ▲                ▲                 ▲       │ │
               │ │         │                │                 │       │ │
               │ │      [DatasetPort] [PreprocessorPort] [TrackerPort]│ │
               │ └────────────────────────────────────────────────────┘ │
               │          ▲                        ▲                    │
               │          │                        │                    │
               │   [OpenCVAdapter]          [MLflowTrackerAdapter]      │
               │                                                        │
               └────────────────────────────────────────────────────────┘
```

### 2.1 As Portas (Interfaces Abstratas)
As portas são classes abstratas puras (definidas no módulo `ports/`) que declaram contratos sem conhecer a implementação:

* **`DatasetReaderPort`**:
  ```python
  class DatasetReaderPort(ABC):
      @abstractmethod
      def load_cases(self, source_path: Path) -> List[PatientCase]: ...
  ```
* **`PreprocessorPort`**:
  ```python
  class PreprocessorPort(ABC):
      @abstractmethod
      def process(self, image: MammogramImage) -> Tuple[MammogramImage, np.ndarray]: ...
  ```
* **`ExperimentTrackerPort`**:
  ```python
  class ExperimentTrackerPort(ABC):
      @abstractmethod
      def start_run(self, run_name: str, tags: Dict[str, str]) -> None: ...
      @abstractmethod
      def log_params(self, params: Dict[str, Any]) -> None: ...
      @abstractmethod
      def log_metrics(self, metrics: Dict[str, float], step: int) -> None: ...
      @abstractmethod
      def log_artifact(self, local_path: Path, artifact_path: Optional[str] = None) -> None: ...
      @abstractmethod
      def end_run(self) -> None: ...
  ```

---

## 🌐 3. MLOps e Integração com Servidor Remoto MLflow

O ambiente de execução foi desenhado para conectar o computador da universidade (responsável pelo processamento pesado em GPU) ao seu servidor caseiro de rastreamento.

```text
[Servidor de Treinamento da Universidade]
  │  GPU Executando Treino
  │  MLflowTrackerAdapter
  │
  ▼ Requisições HTTPS com Auth Token
[Túnel Seguro Cloudflare]
  ▼
[Servidor Caseiro]
  ├── MLflow Tracking Server (Porta 5000)
  ├── MinIO / Local Artifact Store (Armazenamento de Pesos .pth/.pt)
  └── Grafana Dashboard (Visualização de Recursos e Telemetria)
```

### Boas Práticas do Tracker Adapter:
1. **Resiliência a Falhas de Rede:** O túnel Cloudflare pode sofrer quedas temporárias de conexão. O `MLflowTrackerAdapter` deve armazenar métricas em buffer local na memória e executar tentativas com recuo exponencial (*exponential backoff retry*) antes de interromper o treinamento.
2. **Registro Completo de Linhagem:** Cada execução no MLflow deve registrar:
   * **Parâmetros:** Arquitetura do backbone, taxa de aprendizado, limiares de IoU do Cascade R-CNN, clip limit do CLAHE, tamanho do patch.
   * **Métricas:** Perda total, perdas de classificação e regressão por estágio, $mAP_{50}$, $mAP_{75}$, $mAP_s$, $mAP_m$, $mAP_l$.
   * **Artefatos:** Curvas de precisão-revocação (PR Curves), matrizes de confusão e predições visuais com as bounding boxes projetadas sobre os patches de validação.
   * **Metadata de Versionamento:** Hash do commit do Git (`git rev-parse HEAD`), hash da versão dos dados e ambiente de hardware.
