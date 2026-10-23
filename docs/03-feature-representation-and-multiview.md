# 03. Representação de Features e Raciocínio Multi-View

No diagnóstico clínico real, **nenhum radiologista analisa uma mamografia como uma imagem isolada**. O processo de leitura de um exame mamográfico depende intrinsecamente de correlações espaciais tridimensionais e comparações anatômicas comparadas.

Neste documento, exploramos como transpor esse raciocínio clínico para a engenharia de features e arquitetura de redes profundas.

---

## 1. As Quatro Incidências Padrão do Exame Mamográfico

Um exame de rastreamento completo consiste obrigatoriamente em quatro imagens:
* **L-CC (Left Cranio-Caudal):** Projeção axial superior da mama esquerda.
* **R-CC (Right Cranio-Caudal):** Projeção axial superior da mama direita.
* **L-MLO (Left Mediolateral Oblique):** Projeção angular (em ~45°) incluindo o músculo peitoral e axila da mama esquerda.
* **R-MLO (Right Mediolateral Oblique):** Projeção angular da mama direita.

```text
┌──────────────┐  ┌──────────────┐     ┌──────────────┐  ┌──────────────┐
│     L-CC     │  │     R-CC     │     │    L-MLO     │  │    R-MLO     │
│              │  │              │     │  \           │  │           /  │
│              │  │              │     │   \ Peitoral │  │ Peitoral /   │
└──────────────┘  └──────────────┘     └──────────────┘  └──────────────┘
  └───────── Ipsilateral ─────────┘       └───────── Contralateral ───────┘
     (Mesma mama, 2 ângulos)                 (Comparação de Simetria)
```

---

## 2. Paradigmas de Raciocínio Clínico

### 2.1 Análise Ipsilateral (Correlação CC + MLO)
* **Artefato de Sobreposição (*Summation Artifact*):** Em uma projeção 2D de um volume 3D, tecidos glandulares normais em planos de profundidade diferentes podem se sobrepor na imagem, aparentando uma massa radiopaca suspeita.
* **Validação Cruzada:** Uma lesão é considerada verdadeira apenas se for identificada **simultaneamente** nas duas incidências ortogonais (CC e MLO). Uma anomalia presente em apenas uma projeção é candidata imediata a falso-positivo por sobreposição.
* **Triangulação:** A distância da lesão até o mamilo em CC e MLO permite mapear a coordenada 3D exata do nódulo no quadrante mamário.

### 2.2 Análise Contralateral (Simetria Bilateral)
* A anatomia glandular de um indivíduo tende a ser simétrica entre a mama esquerda e a direita.
* Radiologistas buscam ativamente por **assimetrias em espelho** (*mirror comparison*). Uma densidade tecidual densa é tolerada se existir de forma idêntica e simétrica na mama oposta; caso esteja presente apenas em um lado, a probabilidade de malignidade (*BI-RADS 3 ou 4*) se eleva substancialmente.

---

## 3. Estratégias de Fusão Multi-View em Redes Neurais

Para integrar essas informações nas redes neurais convolucionais e em Vision Transformers, existem três abordagens arquiteturais:

```text
1. Early Fusion (Canal)        2. Feature-Level Fusion (Atenção Cruzada)
   [CC]  ──┐                      [CC]  ──► [Backbone A] ──┐
           ├─► [Canal 2] ─► [CNN]                           ├─► [Cross-Attention] ─► [Head]
   [MLO] ──┘                      [MLO] ──► [Backbone B] ──┘
```

| Paradigma | Mecanismo | Vantagens | Desvantagens / Desafios |
| :--- | :--- | :--- | :--- |
| **Early Fusion** (Fusão Precoce) | Concatenar as imagens CC e MLO como canais de entrada (ex: Tensor $[2, H, W]$). | Arquitetura simples, utiliza modelos pré-existentes sem alterações profundas. | **Inviável na prática:** A compressão mecânica deforma o parênquima de forma completamente não-linear entre CC e MLO; não há correspondência pixel-a-pixel para convolução direta. |
| **Feature Fusion** (Cross-Attention) | Dois backbones siameses com pesos compartilhados que extraem mapas de features $F_{\text{CC}}$ e $F_{\text{MLO}}$, seguidos por módulos de atenção cruzada (*Cross-View Transformer*). | Modela relações conceituais e semânticas de alto nível sem exigir alinhamento pixel-a-pixel. | Maior consumo de memória de vídeo e complexidade de convergência. |
| **Late Fusion** (Fusão Tardia) | Dois detectores independentes emitem propostas de bounding box; um módulo final de decisão (GNN ou Regras Bayesianas) cruza as predições. | Baixo acoplamento; permite treinar e validar os modelos de visão de forma modular. | Não permite que a extração de features em uma incidência seja guiada pela pista visual da outra. |

---

## 4. Transfer Learning: ImageNet vs RadImageNet

A maioria dos repositórios de visão computacional inicializa pesos a partir do **ImageNet-1k** (1.2 milhão de fotos coloridas de objetos cotidianos). Em radiologia médica, essa premissa introduz vieses severos:

### As Diferenças de Domínio (*Domain Gap*)
1. **Espectro de Cor vs Intensidade Contínua:** ImageNet foca em texturas cromáticas RGB ($3$ canais). Mamografias são matrizes de raios-X em escala de cinza mono-canal onde a intensidade reflete a densidade física de elétrons do tecido.
2. **Escala de Objetos:** Objetos no ImageNet costumam ocupar o centro da imagem e cobrir frações amplas do enquadramento ($20\%$ a $80\%$ da área). Uma microcalcificação em mamografia pode ocupar menos de $0.001\%$ da matriz total.

### Alternativas de Pré-Treinamento Específico
* **RadImageNet:** Base pública contendo mais de $1.35$ milhão de imagens radiológicas reais (Raio-X, Tomografia Computadorizada, Ressonância Magnética e Ultrassom). Modelos inicializados com pesos do RadImageNet convergem até **3x mais rápido** em detecção de lesões e demonstram gradientes significativamente mais estáveis em transfer learning.
* **Autoencoders Mascarados Médicos (MAE):** Pré-treinamento não supervisionado no próprio domínio através de reconstrução de patches mascarados em mamografias não rotuladas antes do ajuste supervisionado.
