# Índice de Estudos e Fundamentação Técnica

Esta pasta reúne a documentação de engenharia e a base teórica necessária para o desenvolvimento do pipeline de **Mammography MLOps**. 

Para evitar desenvolvimento ad-hoc e escolhas arbitrárias de bibliotecas e algoritmos, cada etapa do ciclo de vida de Machine Learning foi dissecada e avaliada sob a ótica de trade-offs de desempenho, viabilidade computacional e rigor clínico.

---

## 📑 Roteiro de Leitura

1. [**01. Benchmark e Seleção de Datasets**](01-datasets-benchmark.md)
   * Comparação entre INbreast, CBIS-DDSM, VinDr-Mammo, EMBED e CMMD.
   * Resolução nativa, anotações (BI-RADS, bounding box, polígonos), formatos (DICOM vs PNG) e processo de solicitação de acesso.

2. [**02. Engenharia de Pré-processamento**](02-preprocessing-pipelines.md)
   * Tratamento de profundidade de bits (12/16-bit DICOM vs 8-bit) e fotometria (`MONOCHROME1` vs `MONOCHROME2`).
   * Algoritmos de segmentação de contorno mamário e eliminação de artefatos.
   * Métodos para remoção do músculo peitoral na incidência MLO.
   * Filtros de contraste (CLAHE, Unsharp Masking) e trade-offs de introdução de artefatos.
   * Resolução e o dilema espacial: Redimensionamento Global vs *Tiling* (Patching) em alta resolução.

3. [**03. Representação de Features e Raciocínio Multi-View**](03-feature-representation-and-multiview.md)
   * Anatomia da mamografia: incidências Crânio-Caudal (CC) e Médio-Lateral Oblíqua (MLO).
   * Simetria bilateral como heurística diagnóstica.
   * Modelos siameses e fusão multi-view (*early*, *intermediate* e *late fusion*).
   * Transfer Learning em imagens médicas: limitações do ImageNet e vantagens de backbones pré-treinados em radiologia (RadImageNet).

4. [**04. Benchmark de Arquiteturas de Detecção**](04-model-architectures-benchmark.md)
   * Detectores de dois estágios: Faster R-CNN e Cascade R-CNN (por que múltiplos limiares de IoU funcionam para tumores).
   * Detectores modernos de estágio único: RetinaNet (Focal Loss) e YOLOv11/v12.
   * Arquiteturas baseadas em Transformers: Deformable DETR, RT-DETR e Co-DETR.
   * Matriz de decisão considerando sensibilidade a pequenas lesões ($mAP_s$), custo de VRAM e tempo de convergência.

5. [**05. Avaliação de Frameworks de Treinamento**](05-training-frameworks-and-tooling.md)
   * Comparação técnica: MMDetection v3 vs PyTorch Lightning vs Torchvision vs PyTorch Nativo.
   * Facilidade de extensão, customização de *loss functions*, acoplamento arquitetural e integração com tracking remoto.

6. [**06. Arquitetura de Software (DDD & Ports and Adapters)**](06-mlops-architecture-and-ddd.md)
   * Modelagem de Domínio para Visão Computacional Médica.
   * Definição de Portas (Interfaces) e Adapters concretos.
   * Estratégia de rastreamento remoto via servidor caseiro MLflow exposto via túnel Cloudflare.
   * Pipeline de ingestão, particionamento estratificado e prevenção de *data leakage* por paciente.

7. [**ADRs - Architectural Decision Records**](adr/)
   * Registro histórico de decisões definitivas tomadas durante o projeto.
