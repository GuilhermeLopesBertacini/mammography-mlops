# 05. Avaliação de Frameworks de Treinamento e Tooling

A seleção do ecossistema de treinamento deve equilibrar **reuso de arquiteturas consagradas do estado da arte** com **manutenibilidade, transparência de código e facilidade de depuração**.

Neste documento, avaliamos os principais frameworks de visão computacional e detalhamos o tooling de engenharia adotado.

---

## ⚖️ Comparativo de Frameworks de Treinamento

| Framework | Ponto Forte | Ponto Fraco | Aderência à Arquitetura Hexagonal |
| :--- | :--- | :--- | :--- |
| **MMDetection v3** (OpenMMLab) | Suporte nativo e catálogo completo de modelos complexos (Cascade R-CNN, Deformable DETR, Co-DETR). Otimizado para benchmark COCO. | Alto acoplamento interno ao ecossistema OpenMMLab (`mmengine`, `mmcv`). Dificuldade para debugar pilhas de execução profundas. Sistema de configuração rígido via dicts/arquivos Python. | **Média:** Exige encapsulamento rigoroso dentro de um Adapter para não contaminar o domínio. |
| **PyTorch Lightning** | Separação explícita entre código de modelagem (`LightningModule`) e infraestrutura de hardware (`Trainer`). Suporte nativo e limpo a MLflow callbacks, multi-GPU e precisão mista (AMP fp16/bf16). | Não possui modelos prontos de detecção de objetos de dois estágios out-of-the-box (exige importar backbones do `timm` ou `torchvision`). | **Alta:** O `Trainer` se integra de forma limpa como um adapter de execução. |
| **Torchvision / PyTorch Nativo** | 100% transparente, sem dependências mágicas ou registros ocultos. Facilidade total para criar perdas customizadas e arquiteturas multi-view exclusivas. | Não possui implementações nativas do Cascade R-CNN ou Deformable DETR prontas (apenas Faster R-CNN e RetinaNet). | **Altíssima:** Controle total sobre cada tensor e fluxo de gradiente. |
| **Ultralytics (YOLOv8 / YOLOv11)** | Extremamente rápido para prototipagem de pipelines de estágio único em edge devices. | Forte acoplamento com CLI proprietário; abstração rígida que dificulta incorporar lógica médica complexa (fusão CC + MLO). | **Baixa a Média.** |

---

## 🛠️ Decisão Arquitetural: O Padrão Adapter para Model Runners

Para não amarrar o projeto a nenhum framework específico e permitir a evolução contínua da pesquisa, a camada de aplicação interagirá exclusivamente com a interface abstrata `ModelRunnerPort`:

```python
from abc import ABC, abstractmethod
from typing import Dict, Any

class ModelRunnerPort(ABC):
    @abstractmethod
    def train(
        self, 
        config: Dict[str, Any], 
        train_data_path: str, 
        val_data_path: str,
        tracker: "ExperimentTrackerPort"
    ) -> Dict[str, float]:
        """Executa o pipeline de treino e avaliação, reportando métricas ao tracker."""
        pass
```

### Adapters Concretos:
1. **`MMDetectionAdapter`**: Permite rodar o **Cascade R-CNN** com todas as configurações avançadas de FPN e cabeçotes sequenciais sem reinventar a roda da arquitetura.
2. **`LightningRunnerAdapter`**: Permite rodar experimentos customizados, como fusão multi-view (CC + MLO) com cross-attention, mantendo o controle total dos gradientes e callbacks diretos do MLflow.

---

## 🧰 Stack de Tooling de Engenharia de Software

Para manter um padrão profissional no repositório:

* **Gerenciador de Pacotes e Ambientes: `uv`**
  * Desenvolvido em Rust pela Astral, é de $10\times$ a $100\times$ mais rápido que `pip`, `conda` e `poetry`.
  * Cria ambientes virtuais isolados e locks determinísticos (`uv.lock`), garantindo 100% de reprodutibilidade entre o computador local e o servidor da universidade.
* **Linter e Formatador: `ruff`**
  * Unifica Black, Flake8 e isort em uma única ferramenta instantânea.
* **Tipagem Estática: `mypy`**
  * Verificação estrita de tipos para garantir que contratos e interfaces de domínio não recebam estruturas inválidas em tempo de execução.
* **Configurações Declarativas: `Hydra` + YAML**
  * Separação estrita de configurações de dados, hiperparâmetros de modelos e credenciais de infraestrutura. Permite sobrescrever parâmetros via CLI (`python run.py model=cascade_rcnn data.batch_size=8`).
