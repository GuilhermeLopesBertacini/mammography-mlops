# 01. Benchmark e Seleção de Datasets para Mamografia

A escolha da base de dados é a decisão mais determinante para o sucesso ou fracasso de um modelo em visão computacional médica. Modelos treinados em datasets pequenos com baixa diversidade de pacientes sofrem de **generalização ilusória** e falham severamente sob deslocamento de distribuição (*domain shift*) entre diferentes fabricantes de mamógrafos (Hologic, GE, Siemens).

Neste documento, analisamos as principais bases de dados públicas e acadêmicas disponíveis para detecção e classificação de lesões mamárias.

---

## 📊 Matriz Comparativa de Datasets

| Dataset | Modalidade | N° Casos / Pacientes | N° Imagens | Tipo de Anotação | Formato Original | Nível de Acesso | Relevância / Recomendação |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **INbreast** | FFDM (Digital) | 115 pacientes | 410 imagens | Polígonos finos (massas, calcificações, assimetrias) | DICOM (16-bit) | Acadêmico (requer e-mail/solicitação) | **Baseline histórico**, excelente para validação de alta precisão, mas insuficiente para treinar detectores profundos sozinhos. |
| **CBIS-DDSM** | Film Scanned (Digitalizado) | 1.566 pacientes | 6.671 imagens | Bounding boxes e máscaras binárias de contorno | DICOM / TCIA | **Público irrestrito** (TCIA) | Amplo volume, mas são filmes digitalizados antigos (não é FFDM nativo). Ruído óptico e artefatos de escaneamento. |
| **VinDr-Mammo** | FFDM (Digital) | 5.000 estudos | 20.000 imagens | Bounding boxes + BI-RADS (1 a 5) + Densidade (A-D) | DICOM (16-bit) | Credenciado (PhysioNet - aprovação rápida) | **Padrão ouro moderno**. Dados recentes de FFDM (Siemens/Planmed), anotados por múltiplos radiologistas especialistas. |
| **CMMD** | FFDM (Digital) | 1.775 pacientes | 3.744 imagens | Rótulos de imagem/estudo (Benigno/Maligno, Biópsia) | DICOM (TCIA) | **Público irrestrito** (TCIA) | Bom volume de FFDM nativo, porém as anotações são a nível de imagem/caso, sem coordenadas espaciais para detecção direta de objetos. |
| **EMBED** | FFDM (Digital) | 110.000 pacientes | ~3.400.000 imagens | Laudos clínicos estruturados, BI-RADS, histologia | DICOM (TCIA / Emory) | Credenciado (DUA restrito) | Escala massiva industrial. Complexidade de armazenamento (dezenas de Terabytes), inviável para iteração rápida inicial. |

---

## 🔍 Análise Detalhada dos Datasets Principais

### 1. INbreast (Base Histórica da IC)
* **Origem:** Centro Hospitalar de São João (CHSJ), Porto, Portugal (2012).
* **Vantagens:** 
  * As anotações foram feitas por radiologistas especialistas utilizando contornos vetoriais exatos (arquivos XML com pontos cartesianos $x, y$).
  * Contém todos os achados clínicos detalhados: massas circunscritas, espiculadas, microcalcificações individuais e agrupadas, assimetrias focais e distorções arquiteturais.
* **Desvantagens:**
  * **Tamanho amostral ínfimo:** Apenas 410 imagens de 115 pacientes. Se separado em treino/validação/teste sem vazamento de paciente (*patient-level split*), o conjunto de treino conta com menos de 300 imagens.
  * O overfitting é praticamente inevitável sem técnicas agressivas de pré-treinamento e regularização.
* **Formato e Características:**
  * Equipamento: Mamógrafo digital Siemens MammoNovation.
  * Resolução nativa: $3328 \times 4084$ ou $2560 \times 3328$ pixels.
  * Profundidade: 14 bits salvos em contêineres DICOM de 16 bits.

---

### 2. VinDr-Mammo (Recomendação Principal para Expansão)
* **Origem:** Vingroup Big Data Institute (VinBigData), Vietnã (2022).
* **Vantagens:**
  * É atualmente a base aberta de mamografia digital de campo total (FFDM) mais moderna e com anotações de localização de maior qualidade.
  * **20.000 imagens de 5.000 exames**, contendo as quatro incidências padrão por paciente (L-CC, R-CC, L-MLO, R-MLO).
  * Cada imagem foi avaliada de forma independente por **três radiologistas**, garantindo consenso interobservador.
  * Fornece anotações no nível de lesão (Bounding Boxes para: Massa, Calcificação, Assimetria, Espessamento de Pele, Retração de Pele, Distorção Arquitetural) e no nível de exame (BI-RADS 1 a 5 e densidade ACR A a D).
* **Desvantagens:**
  * Requer registro no PhysioNet e aprovação de credenciamento (completar o treinamento básico do CITI sobre pesquisa com seres humanos, gratuito para estudantes e pesquisadores).
* **Acesso:**
  * Disponível em: `https://physionet.org/content/vindr-mammo/1.0.0/`
  * Processo: Criar conta no PhysioNet $\to$ assinar o Data Use Agreement (DUA) $\to$ submeter certificado CITI. O acesso costuma ser liberado em 48-72h.

---

### 3. CBIS-DDSM (Curated Breast Imaging Subset of DDSM)
* **Origem:** University of South Florida / Massachusetts General Hospital (2017).
* **Vantagens:**
  * Download direto e imediato via cliente TCIA (The Cancer Imaging Archive) sem necessidade de aprovação humana prévia.
  * Anotações curadas com máscaras de segmentação e caixas delimitadoras separadas para massas (*masses*) e calcificações (*calcifications*), com rótulos histológicos (benigno vs maligno).
* **Desvantagens Críticas:**
  * Não é mamografia digital direta (FFDM); são filmes radiológicos analógicos digitalizados por scanners ópticos nas décadas de 1990 e 2000.
  * Presença massiva de artefatos de poeira no scanner, riscos no filme, fitas adesivas e marcadores de chumbo radiopacos.
  * O ruído da granulação do filme dificulta o aprendizado de representações transferíveis para exames de mamógrafos digitais modernos.

---

## 🎯 Estratégia de Dataset para o Repositório

Para estruturar um projeto de nível profissional e viabilizar tanto a pesquisa científica quanto a engenharia de dados, adotaremos uma **estratégia em duas fases**:

1. **Fase 1 (Desenvolvimento e Validação Rápida):**
   * Utilizar o dataset **INbreast** para estruturar e validar as portas (*Ports*) do pipeline, os algoritmos de pré-processamento, os testes unitários e a integração remota com o MLflow.
   * Por ser um dataset compacto (~410 imagens), permite iterar com agilidade em execuções de fumaça (*smoke tests*) e validação de ponta a ponta sem sobrecarregar a rede ou a máquina.

2. **Fase 2 (Treinamento Robusto e Benchmark Científico):**
   * Solicitar o acesso ao **VinDr-Mammo** via PhysioNet.
   * Implementar o `VinDrMammoReaderAdapter` que ingere o formato nativo do VinDr e mapeia para a mesma entidade `MammogramImage` do domínio.
   * Executar os experimentos definitivos com validação cruzada estratificada em 5 folds garantindo que exames de um mesmo paciente nunca estejam simultaneamente no treino e na validação.
