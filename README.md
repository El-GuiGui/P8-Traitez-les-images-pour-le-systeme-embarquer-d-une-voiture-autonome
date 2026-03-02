# P8 - Segmentation d'images pour un système embarqué de véhicule autonome

## Contexte

Ce projet s'inscrit dans le cadre d'un système embarqué de vision par ordinateur pour véhicules autonomes, développé pour l'entreprise Future Vision Transport. La chaîne complète du système est composée de quatre blocs : acquisition des images, traitement des images, segmentation des images, et système de décision. Ce dépôt couvre le bloc de segmentation (bloc 3), qui reçoit en entrée des images prétraitées et produit des masques de segmentation sémantique destinés au système de décision en aval.

L'objectif est d'entraîner plusieurs modèles de segmentation sémantique sur le jeu de données Cityscapes, réduit à 8 catégories principales, puis de les exposer via une API de prédiction et une application de démonstration déployées sur le cloud.

---

## Jeu de données

Le jeu de données utilisé est [Cityscapes](https://www.cityscapes-dataset.com/), composé d'images de caméras embarquées en milieu urbain, accompagnées de leurs masques d'annotation sémantique.

Les 32 sous-catégories de Cityscapes ont été regroupées en 8 catégories principales :

| Indice | Catégorie    | Couleur dans les visualisations |
| ------ | ------------ | ------------------------------- |
| 0      | void         | Noir                            |
| 1      | flat         | Violet (chaussée, trottoir)     |
| 2      | construction | Gris foncé (bâtiments, murs)    |
| 3      | object       | Gris clair (poteaux, panneaux)  |
| 4      | nature       | Vert (végétation, terrain)      |
| 5      | sky          | Bleu                            |
| 6      | human        | Rouge (piétons, cyclistes)      |
| 7      | vehicle      | Bleu foncé (voitures, bus)      |

Le remapping des 34 label IDs d'origine vers ces 8 groupes est géré dans `scripts/preprocessing.py` via une LUT (Look-Up Table).

---

## Architecture du projet

```
P8/
├── data/
│   └── raw/
│       └── cityscapes/
│           ├── leftImg8bit/        # Images RGB
│           └── gtFine/             # Masques d'annotation
│
├── models/
│   └── best_model.keras            # Meilleur modèle sauvegardé
│
├── notebooks/
│   └── *.ipynb                     # Notebooks d'entraînement et d'analyse
│
├── out/
│   ├── experiments/                # Fichiers .keras et .json de chaque run
│   └── cityscapes_split_testXtrainXval.csv
│
├── logs/
│
├── scripts/
│   ├── __init__.py
│   ├── config.py                   # Chemins et configuration du projet
│   ├── preprocessing.py            # Remapping labels, colorisation, overlay
│   ├── datagen.py                  # CityscapesSequence (générateur Keras)
│   ├── augmentations.py            # Pipeline d'augmentation (Albumentations)
│   ├── augmentations_alternative.py
│   ├── models.py                   # Définition des architectures
│   ├── losses_metrics.py           # Dice loss, MeanIoU custom
│   ├── training.py                 # Boucle d'entraînement et callbacks
│   ├── inference.py                # Prédiction à partir d'une image PIL
│   ├── viz.py                      # Helpers de visualisation
│   └── seed.py                     # Reproducibilité
│
├── api/
│   ├── main.py
│   └── __init__.py
│
├── app/
│   ├── streamlit_app.py            # Dashboard de démonstration
│   └── __init__.py
│
├── test/
│   ├── test_api_health.py
│   ├── test_imports.py
│   ├── test_models_shapes.py
│   └── __init__.py
│
│
├── start.sh                        # Script de lancement
├── requirements.txt
├── .replit
├── .gitignore
├── pytest.ini
└── TEST_IMPORTS.py                 # Vérification de l'environnement
```

---

## Modèles implémentés

Toutes les architectures suivent le même paradigme encoder-decoder de type U-Net. Elles partagent la même boucle d'entraînement, les mêmes callbacks et les mêmes métriques, ce qui rend les comparaisons directement équitables.

### U-Net from scratch

Architecture U-Net légère entraînée sans préentraînement. Sert de baseline absolue pour évaluer l'apport du transfert learning.

### U-Net + VGG16

Encodeur VGG16 préentraîné sur ImageNet, gelé en phase initiale. Le décodeur est constitué de blocs convolutifs avec skip connections reprises depuis les couches `block1_conv2` à `block5_conv3`.

### U-Net + ResNet50

Encodeur ResNet50 préentraîné sur ImageNet. Les skip connections sont extraites aux sorties des blocs résiduels `conv1_relu`, `conv2_block3_out`, `conv3_block4_out`, `conv4_block6_out`. Un prétraitement dédié (`ResNet50Preprocess`) est intégré directement dans le graphe Keras pour éviter toute fuite entre entraînement et inférence.

---

## Pipeline d'entraînement

### Générateur de données

`CityscapesSequence` est un générateur Keras (`tf.keras.utils.Sequence`) qui charge les images et masques à la volée, les redimensionne à la résolution cible (256x256 par défaut) et applique les augmentations.

### Augmentations

Gérées via [Albumentations](https://albumentations.ai/) :

- Flip horizontal (p=0.5)
- Ajustement aléatoire de la luminosité et du contraste (p=0.5)
- Variation teinte/saturation/valeur (p=0.3)
- Flou gaussien léger (p=0.15)
- Bruit gaussien (p=0.2)
- Décalage, mise à l'échelle, rotation légère (p=0.35)

### Fonction de perte

Combinaison de la Cross-Entropie Catégorielle Sparse et de la Dice Loss (`ce_dice`), avec masquage des pixels labellisés `IGNORE_LABEL` (valeur 255). Le coefficient de la Dice Loss est de 0.5 ou 1 selon entrainement.

### Métrique principale

`MeanIoUArgmax` : calcul du mIoU (Mean Intersection over Union) après argmax sur les prédictions, en excluant les pixels `IGNORE_LABEL`.

### Callbacks

- `ModelCheckpoint` : sauvegarde du meilleur modèle selon le `val_mIoU`
- `EarlyStopping` : patience de 8 epochs sur le `val_mIoU`
- `ReduceLROnPlateau` : réduction du learning rate par facteur 0.5, patience de 3 epochs

### Optimiseur

Adam, learning rate initial de 1e-3, plancher à 1e-5.

---

## Reproductibilité

Les seeds sont fixées avant chaque run. La session Keras est réinitialisée entre deux entraînements pour éviter les fuites de mémoire et garantir l'isolation des runs.

---

## Inférence

Le module `scripts/inference.py` expose une fonction `predict_from_pil` qui prend une image PIL en entrée et retourne :

- le masque brut (indices de classe, `uint8`)
- le masque colorisé (RGB)
- l'overlay image/masque
- l'image redimensionnée

Le modèle est chargé une seule fois en mémoire (singleton). S'il est absent localement, il est téléchargé automatiquement depuis Hugging Face Hub :
[GuiLL-L/my-best-model-unet-for-proj8-cityscapes](https://huggingface.co/GuiLL-L/my-best-model-unet-for-proj8-cityscapes)

---

## Dashboard Streamlit

L'application `streamlit_app.py` est le point d'entrée de démonstration. Elle propose :

- Un moteur de prédiction permettant de sélectionner une image du dataset ou d'en uploader une
- La comparaison visuelle entre le masque ground truth et le masque prédit
- La comparaison des performances entre les runs

Le dashboard est déployé sur le cloud et accessible publiquement avec Replit :

- https://d216f258-90bd-4f70-914b-20f258543b19-00-2hjmwjb4ufx1j.picard.replit.dev:8000/docs
- https://d216f258-90bd-4f70-914b-20f258543b19-00-2hjmwjb4ufx1j.picard.replit.dev/

---

## Installation

### Prérequis

- Python 3.10+
- GPU recommandé (les entraînements ont été réalisés sur une NVIDIA GTX 1080 Ti)
- TensorFlow 2.x
- keras_hub (pour éventuellement SegFormer et ConvNeXt à l'avenir)

### Installation des dépendances

```bash
pip install tensorflow keras-hub albumentations streamlit plotly pillow requests pandas numpy
```

ou via le requirements.txt

### Configuration des chemins

Le projet détecte automatiquement sa racine en remontant l'arborescence depuis le répertoire courant à la recherche des dossiers `scripts/` et `data/`. Il est également possible de surcharger ce comportement via la variable d'environnement :

```bash
export PROJ8_ROOT=/chemin/absolu/vers/P8
```

### Données

Télécharger le jeu de données Cityscapes (images `leftImg8bit_trainvaltest.zip` et annotations `gtFine_trainvaltest.zip`) depuis le site officiel et les déposer dans :

```
data/raw/cityscapes/leftImg8bit/
data/raw/cityscapes/gtFine/
```

---

## Lancer le dashboard et l'api

```bash
bash start.sh
```

ou directement :

```bash
streamlit run streamlit_app.py
```

---

## Vérifier l'environnement

```bash
python TEST_IMPORTS.py
```

Ce script vérifie que toutes les dépendances critiques sont correctement installées et importables.

---

## Résultats

Les résultats de chaque run sont sauvegardés dans `out/experiments/` sous forme de fichiers `.keras` (poids), `.json` (métriques et historique d'entraînement) et `.png` (métriques et graph).

Les métriques de référence sont le `mIoU` sur les ensembles de validation et de test, calculés sur les 8 catégories avec exclusion des pixels ambigus.

---
