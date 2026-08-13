# **Rapport de Projet — Atlantic Haven Hotels**

## **Examen Final Machine Learning & Data Science — M1**

Réalisé au sein de **ISPM — Madagascar** ([www.ispm-edu.com](https://www.ispm-edu.com))

---

### **1. Informations sur le Groupe**

#### Membre 1

- nom : RAKOTOMALALA
- prénom(s) : Princy
- classe : IGGLIA 4
- numéro : 4
- rôle : développeur

#### Membre 2

- nom : RANDRIAMAMONJISOA
- prénom(s) : Tokindrainy
- classe : IGGLIA 4
- numéro : 17
- rôle : responsable de la modélisation

#### Membre 3

- nom : RANDRIATAHINARIMANANA
- prénom(s) : Tendry Ny Avo Gabriel
- classe : IGGLIA 4
- numéro : 18
- rôle :  analyste

#### Membre 4

- nom : RABENJARISON
- prénom(s) : Fenomalala Safidy
- classe : IGGLIA 4
- numéro : 23
- rôle : développeur

#### Membre 5

- nom : RANDRIANANTOANDRO
- prénom(s) : Radoniaina Francky
- classe : IGGLIA 4
- numéro : 27
- rôle : responsable de la modélisation

#### Membre 6

- nom :  ANDRIATIANA
- prénom(s) : Antsa Eliot
- classe : IGGLIA 4
- numéro : 31
- rôle :  analyste

#### Membre 7

- nom : RANDY
- prénom(s) : Hajaniaina Cédric
- classe : IGGLIA 4
- numéro : 39
- rôle : Feature Engineering

---

### **2. Résumé du Travail**

#### Problématique

Atlantic Haven Hotels fait face à un taux d'annulation élevé (25,8 % dans les données d'entraînement), ce qui entraîne des pertes financières et logistiques majeures (chambres inoccupées en dernière minute, mauvaise planification). L'objectif de ce projet est de concevoir un système de prédiction robuste capable d'anticiper ces annulations suffisamment tôt afin que l'hôtel puisse réagir (confirmer le séjour, ajuster sa surréservation ou proposer des offres).

#### Méthodologie adoptée

Nous avons mis en place une méthodologie rigoureuse structurée en cinq étapes principales :
1. **EDA & Nettoyage** : Analyse des types de variables, des valeurs manquantes, et découverte d'une formule mathématique exacte pour reconstruire le prix moyen par nuit sans bruit ni imputation naïve.
2. **Feature Engineering** :  Création de variables temporelles (mois d'arrivée, jour de semaine, délai catégorisé), historiques (taux d'annulation passée, réservation directe) et financières (écart de prix régional, croisement acompte/remboursabilité), avec gestion anti-fuite stricte (statistiques calculées sur train uniquement).
3. **Validation Temporelle** : Séparation chronologique des données pour éviter les fuites d'information (data leakage) et mimer la structure temporelle du jeu de test réel.
4. **Modélisation & Tuning** : Entraînement de plusieurs modèles (Régression Logistique baseline, Random Forest, LightGBM, XGBoost) optimisés par recherche d'hyperparamètres (Optuna).
5. **Ajustement du Seuil** : Recherche linéaire systématique du seuil de classification optimal pour maximiser le F1-score de la classe « Annulé », privilégiant un rappel élevé pour capter la majorité des pertes potentielles.

#### Résultats obtenus

Les données étant ordonnées dans le temps (le jeu de test contient des réservations 
plus récentes que le train), un split aléatoire ou une validation croisée classique 
donnerait une estimation trop optimiste (fuite d'information du futur vers le passé). 
Nous avons donc trié les données par `date_reservation` et conservé les 80% les plus 
anciennes pour l'entraînement (Train_sub) et les 20% les plus récentes pour la 
validation (Val) :

- Train_sub : 6 400 lignes (2023-01-01 → 2024-11-28)
- Val : 1 600 lignes (2024-11-28 → 2025-05-24)

*Meilleur F1-score obtenu sur le jeu de validation 
> F1-score (classe annulation) = 0.454**

*Métriques complémentaires

| Métrique (classe annulation) | Valeur 
| Précision                    | 0.361 
| Rappel (recall)              | 0.611 
| Accuracy globale             | 0.606 
| F1-score (classe maintenue)  | 0.692 

*Découverte importante issue de l'analyse

La cible est déséquilibrée (~26% d'annulations). Sans pondération, un modèle naïf 
tend à sous-prédire la classe annulation. Nous avons utilisé `class_weight="balanced"`, 
ce qui améliore fortement le rappel (0.611) au prix de la précision (0.361) — un 
compromis pertinent puisque manquer une annulation (faux négatif) coûte plus cher 
opérationnellement à l'hôtel qu'une fausse alerte (faux positif).

Par ailleurs, `agent_id` présente ~42% de valeurs manquantes et `hotel_id`/`agent_id` 
ont une cardinalité trop élevée pour un simple One-Hot Encoding — ces variables ont 
été exclues de la baseline et sont recommandées comme pistes de feature engineering 
(ex : taux d'annulation historique par hôtel/agent).


#### Mots-clés

classification binaire, annulation, validation temporelle, F1-score, feature engineering, optimisation.

---

### **3. Contenu du Repository**

Voici la liste des fichiers et liens importants permettant d’évaluer votre travail :

- **notebook.ipynb** : code complet de l’EDA, du prétraitement, de la modélisation et de l’évaluation ;
- **submission.csv** : prédictions sur `reservations_test.csv` ;
- **README.md** : présent rapport complété ;
- **prep_common.py** : sert à la phase de tuning/comparaison ;
- **final_model_lgbm.py** : sert à générer la soumission ;
- **requirements.txt** : dépendances nécessaires à la reproduction du projet *(si nécessaire)* ;

- *(ajoutez ici les autres fichiers utiles sans inclure les fichiers temporaires).* 

**🔗 Liens utiles :**

- [**LIEN VERS LA VIDÉO DE PRÉSENTATION** — Google Drive ou YouTube](https://www.youtube.com/)
- [Lien vers le dépôt GitHub](https://github.com/Tokindrainy/AtlanticHead_CyberVisor.git)


---

### **4. Résultats de Modélisation**

Voici les résultats obtenus sur **le même jeu de validation** afin que la comparaison soit valide.

 Modèle                            | Paramètres principaux  | F1-score | Précision | Rappel | ROC-AUC |

| Régression logistique — baseline | class_weight="balanced"|   0.454  |   0.361   |0.611   | 0.663   |
                                    ,seuil=0.5,random_state=42 

| Modèle 2 : CatBoost|depth, l2_leaf_reg, 
                        scale_pos_weight(Optuna, 30 essais)  | 0.483   | -         |  -     |  0.50   |

| Modèle 3 : XGBoost |max_depth, min_child_weight, 
                        scale_pos_weight(Optuna, 40 essais)  | 0.470   | -         |  -     |  0.46   |

| Modèle final : LightGBM|num_leaves=36, max_depth=4, 
                        learning_rate = 0.0145,
                        n_estimators = 256,    
                    scale_pos_weight=3.66(Optuna, 40 essais) | 0.484   | 0.354*    |  0.782*|  0.50   |
|  |  |  |  |  |

**Seuil de décision retenu :** 0.51 (issu d'une recherche linéaire sur [0.05, 0.95] par pas de 0.01, maximisant le F1 sur la classe annulation).

**Justification du choix du modèle final :**
LightGBM et CatBoost obtiennent des F1 quasi identiques (0.484 vs 0.483), 
largement devant XGBoost (0.470) et la baseline (0.454). 
Nous avons retenu LightGBM pour trois raisons : 
(1) temps d'entraînement nettement plus court, un facteur non négligeable en hackathon de 8h avec recherche d'hyperparamètres ; 
(2) gestion native des variables catégorielles sans encodage manuel lourd, contrairement à XGBoost ; 
(3) le seuil optimal (0.51) reste très proche de 0.5, signe d'un modèle stable et peu sensible à ce réglage, contrairement à XGBoost dont le seuil optimal dévie davantage (0.46). 
Le compromis retenu privilégie le rappel (78%) au détriment de la précision (35%) : cohérent avec le constat de Q2, un faux négatif (annulation manquée) coûte plus cher à l'hôtel qu'une fausse alerte non intrusive.
---

### **5. Réponses aux Questions d’Analyse**

*Répondez précisément aux questions ci-dessous. Utilisez des chiffres, tableaux ou références à vos graphiques pour justifier vos réponses.*

#### **Q1. Pourquoi utilise-t-on principalement le F1-score plutôt que l’accuracy pour cette tâche ?**

Le jeu de données est déséquilibré : environ 74 % des réservations sont maintenues et seulement 26 % sont annulées. Si nous utilisions l'accuracy, un modèle simpliste prédisant constamment « non annulé » obtiendrait une précision globale (accuracy) de 74 %. Cependant, ce modèle serait totalement inutile d'un point de vue opérationnel car il ne détecterait aucune annulation. Le **F1-score** combine la précision (capacité à ne pas générer de fausses alertes) et le rappel (capacité à capter toutes les annulations), ce qui en fait la métrique idéale pour évaluer la détection de la classe minoritaire d'annulation.

#### **Q2. Dans ce contexte, qu’est-ce qui est le plus grave : un faux positif ou un faux négatif ?**

* **Faux Positif (FP)** : Le modèle prédit qu'une réservation sera annulée, mais le client se présente. Si l'hôtel a surréservé la chambre en conséquence, cela crée une situation de surréservation réelle (pas de chambre disponible pour le client), forçant l'hôtel à le reloger à ses frais et dégradant fortement son image de marque.
* **Faux Négatif (FN)** : Le modèle prédit que le client viendra, mais il annule tardivement. La chambre reste inoccupée et l'hôtel perd la totalité du revenu associé.
* **Gravité relative** : Les faux négatifs représentent une perte financière immédiate et certaine. Par conséquent, il est préférable de privilégier un rappel élevé (minimiser les FN) en acceptant un certain nombre de faux positifs, à condition que les actions prises par l'hôtel en cas d'alerte (FP) restent non intrusives (ex: e-mail de courtoisie plutôt qu'une annulation forcée de la part de l'hôtel).


#### **Q3. Quelles variables créées par feature engineering ont le plus amélioré votre modèle par rapport à la régression logistique de référence ?**
L'ajout des 13 variables dérivées (voir section Feature Engineering) fait passer le F1-score de la baseline de 0.454 à 0.4563 (+0.0023), avec un rappel amélioré (0.611 → 0.639) au prix d'une précision quasi stable (0.361 → 0.355). Le gain reste modeste sur ce modèle linéaire : nos deux variables les plus discriminantes prises isolément (tarif_remboursable_bin, is_reservation_directe, voir analyse en section Feature Engineering) apportent l'essentiel du signal, tandis que les variables historiques et temporelles ont un effet plus marginal sur une régression logistique. Elles restent pertinentes à tester sur les modèles non-linéaires (Random Forest, XGBoost), susceptibles de mieux capter des interactions entre variables qu'une régression logistique linéaire ne peut pas exploiter.
#### **Q4. Pourquoi un découpage aléatoire simple peut-il produire une évaluation trompeuse sur ce dataset ?**

Les données possèdent une structure temporelle forte. Le jeu de test contient des réservations créées à des dates futures par rapport au jeu d'entraînement. Si nous faisions un découpage aléatoire (ex: K-Fold standard), le modèle s'entraînerait sur des réservations futures pour prédire des réservations passées. Cela introduirait des fuites de données temporelles (data leakage). Notre stratégie de **validation temporelle** (séparation avant/après le 2024-11-28) respecte la chronologie réelle et fournit une estimation honnête de la capacité de généralisation du modèle sur le futur.

#### **Q5. Quels profils ou scénarios de réservation sont les plus fréquemment associés aux annulations dans vos analyses ?**

Calculé sur l'ensemble du train (taux d'annulation global = 25.8%) :

scénario 1 : Acompte nul + réservation longtemps à l'avance : type_acompte="aucun" combiné à delai_categorise="long_terme" (>90 jours avant l'arrivée) atteint 48.6% d'annulation (n=290), soit près du double de la moyenne. L'absence d'engagement financier associée à un long délai de réflexion semble être le facteur combiné le plus à risque.
sccénario 2 : Tarif remboursable sans acompte : les réservations tarif_remboursable="oui" (31.3%, n=5481) annulent plus du double des non-remboursables (14.0%, n=2519). Sans engagement financier en cas d'annulation, le client n'a rien à perdre à se désister.
scénario 3 : Acompte total : à l'inverse, un acompte total est le facteur protecteur le plus net, quel que soit le délai (9-14% d'annulation contre 23-49% pour "aucun" acompte) — cohérent d'un point de vue économique (coût irrécupérable si annulation).
scénario 4 : Canal de réservation : les réservations via plateforme_en_ligne (30.4%) et agence (27.8%) annulent nettement plus que celles passées en entreprise (14.5%) ou directement au site_hotel (21.5%) — les canaux à faible engagement personnel montrent un taux plus élevé.
scénario 5 : Composition du groupe : les réservations de type groupe (30.8%) et famille (28.1%) annulent plus que les réservations affaires (23.1%) ou loisirs_couple (24.8%) — plus de personnes impliquées, plus de risque qu'un imprévu fasse annuler l'ensemble du séjour.

Ce qui n'est PAS un facteur discriminant notable : le statut de fidélité (client_type, 26.1% chez les fidèles vs 26.1% chez les nouveaux — quasi identique) et le fait d'avoir déjà attendu sur liste d'attente (23.9% vs 25.9%, effet négligeable). Contrairement à l'intuition, la fidélité seule ne protège pas contre l'annulation — c'est la structure de l'engagement financier (acompte, remboursabilité) qui domine largement.

*Attention : décrivez des circonstances observables et des interactions entre variables. Ne présentez pas une région ou une population comme étant intrinsèquement à risque.*

#### **Q6. Comment votre pipeline traite-t-il les valeurs manquantes et les catégories jamais observées pendant l’entraînement ?**
(1)Valeurs manquantes :

enfants, demandes_speciales, prix_moyen_nuit_eur : imputation par la médiane (globale ou par groupe région/catégorie d'hôtel pour le prix), calculée uniquement sur le train (fit_feature_stats) et réappliquée telle quelle à la validation et au test — aucune fuite.
agent_id : conservé tel quel puis exclu des features du modèle (trop de valeurs manquantes ~42%, cardinalité élevée) ; sa présence/absence est cependant capturée indirectement via is_reservation_directe.
marche_origine : laissé tel quel (catégorielle) — LightGBM et CatBoost gèrent nativement les NaN dans les variables catégorielles sans imputation.

(2)Catégories jamais observées en test :

Nous alignons explicitement les catégories du test sur celles apprises en train (pd.Categorical(..., categories=cats) pour LightGBM/XGBoost). Une catégorie inédite en test devient automatiquement NaN, traitée nativement par le modèle plutôt que de faire planter le pipeline.
Exemple concret rencontré : canal_reservation = "assistant_vocal" apparaît dans le test mais jamais dans le train (vérifié explicitement) — traité comme valeur manquante par LightGBM, sans erreur ni fuite.
Pour CatBoost, les catégories manquantes/inédites sont explicitement recodées en une modalité "MANQUANT" plutôt que laissées en NaN, car CatBoost traite les catégorielles comme des chaînes.

#### **Q7. Selon vous, quelle action l’hôtel devrait-il entreprendre lorsqu’une réservation en cours présente une forte probabilité d’annulation ?**

Étant donné le compromis retenu (rappel élevé, précision modeste ~35%), toute action doit rester non intrusive et réversible, car environ 2 alertes sur 3 seront de fausses alertes (faux positifs). Nous recommandons une gradation d'intervention selon le niveau de probabilité, plutôt qu'une action binaire :

|Probabilité prédite|	                        |   Action recommandée|
0.51 – 0.65 (zone d'alerte, proche du seuil)	E-mail de confirmation automatique et courtois ("Nous avons hâte de vous accueillir, confirmez-vous votre séjour ?"), sans mention d'annulation
0.65 – 0.85 (risque élevé)	                    Contact proactif (SMS/email) proposant une flexibilité commerciale : option de modification gratuite des dates, ou petite offre incitative (upgrade, réduction sur options) pour renforcer l'engagement
> 0.85 (risque très élevé)	                    Priorité pour la stratégie de surréservation contrôlée : ces chambres peuvent être intégrées avec prudence dans le calcul d'overbooking de l'hôtel, en gardant une marge de sécurité

Ce qu'il ne faut pas faire : annuler ou vendre la chambre à un tiers sur simple base de la prédiction — le taux de faux positifs (65%, précision 0.355) rendrait cette pratique commercialement risquée et pourrait léser des clients qui avaient bien l'intention de venir. La probabilité doit rester un outil d'aide à la décision commerciale, jamais une décision automatique irréversible.
#### **Q8. Votre modèle présente-t-il des performances comparables selon les régions ou les types de destination ?**
F1-score du modèle final calculé région par région sur la validation temporelle (n≥20 par groupe) :

Région	           | n	|Taux d'annulation réel	|   F1      |
Campania	        169	        23.1%	            0.417
Toscana	            188	        26.1%	            0.428
Sardegna	        67	        28.4%	            0.456
Lazio	            258	        22.9%	            0.474
Veneto	            214	        27.1%	            0.483
Puglia	            99	        25.3%	            0.487
Liguria	            127	        26.8%	            0.489
Lombardia	        216	        27.8%	            0.503
Trentino-Alto-Adige	141	        29.1%	            0.554
Sicilia	            121	        37.2%	            0.556

Non, les performances ne sont pas homogènes : le F1 varie de 0.417 (Campania) à 0.556 (Sicilia), soit un écart de +0.14. On observe une corrélation visible entre le taux d'annulation réel de la région et le F1 obtenu — les régions à plus fort taux d'annulation (Sicilia 37%, Trentino 29%) sont mieux détectées, probablement car la classe positive y est moins rare, donnant plus de signal au modèle. À l'inverse, Campania et Toscana, avec les taux les plus bas (~23-26%), ont le F1 le plus faible.

Limite : plusieurs régions ont moins de 100 observations en validation (Sardegna n=67, Sicilia n=121, Puglia n=99) — ces écarts de F1 doivent être interprétés avec prudence, un intervalle de confiance serait large sur d'aussi petits sous-groupes. Nous recommandons de ne pas sur-interpréter les différences région par région comme des effets structurels, conformément à la mise en garde du sujet (les différences régionales reflètent des contextes touristiques et saisonniers, pas des caractéristiques intrinsèques).

#### **Q9. Analyse des erreurs**

5 faux positifs (modèle prédit annulation, client vient réellement) — triés par probabilité prédite décroissante :

| ID   |	Région	 | Délai résa.(j)|	Canal               |	Acompte  |	Remboursable|	Montant €  |	Historique résa./annul.|Proba   |
R002363	    Lazio	    313	            site_hotel	            aucun	    non	                791.81	    0 / 0	                 0.864
R000462	    Campania	380	            entreprise	            partiel	    oui	                240.09	    0 / 0	                 0.843
R000207	    Lombardia	243	            plateforme_en_ligne	    aucun	    oui	                912.29	    0 / 0	                 0.817
R006922	    Lazio	    45	            plateforme_en_ligne	    aucun	    oui	                669.95	    0 / 0	                 0.815
R000051	    Lombardia	402	            plateforme_en_ligne	    partiel	    oui	                517.56	    12 / 2	                 0.811


5 faux négatifs (modèle prédit maintien, client annule réellement) — triés par probabilité prédite croissante :

|   ID	  |Région	|  Délai résa. (j)|	Canal	           |Acompte	|Remboursable |Montant €	|Historique résa./annul. | Proba |
R009204	    Toscana	    29	            site_hotel	        total	        non	    452.19	       12 / 2	                0.064
R001479	    Trentino	25	            agence	            partiel	        non	    788.97	        10 / 2	                0.087
R004343	    Campania	156	            telephone	        total	        non	    525.94	        12 / 2	                0.153
R005601	    Lombardia	43	            plateforme_en_ligne	partiel	        non	    187.20	        11 / 2	                0.173
R008488	    Toscana	    37	             site_hotel	        total	        non	    421.43	        9 / 2	                0.174

Raisons possibles :
Faux positifs : réservations à très long délai (243 à 402 jours, vs 34.6 en moyenne pour les vrais négatifs) et sans aucun historique client (reservations_passees=0 dans 4 cas sur 5). Le modèle semble associer "nouveau client + réservation très en avance" à un risque élevé — un biais qui pénalise les nouveaux clients prudents plutôt que les clients réellement à risque.
Faux négatifs : à l'inverse, ce sont des clients avec un historique conséquent (9 à 12 réservations passées) qui ont déjà annulé 2 fois par le passé, et qui optent pour un acompte "total" avec tarif non-remboursable. Le modèle sous-estime leur risque, probablement parce que taux_annulation_historique (2/10 à 2/12 ≈ 17-22%) reste inférieur à la moyenne globale (26%), et qu'un acompte total/non-remboursable est perçu par le modèle comme un engagement fort — alors que ces clients annulent malgré cet engagement financier.

### **6. Conclusion et Recommandations**
Le modèle *LightGBM* optimisé, combiné à notre ingénierie de variables, offre un outil d'anticipation robuste avec un F1-score de 0,4901 et une détection de 80,6 % des annulations. Ses limites résident dans l'impossibilité de prévoir les événements imprévus de dernière minute (accidents, météo) qui génèrent des faux négatifs inévitables.

**Recommandation opérationnelle finale :**
Nous recommandons d'intégrer le modèle directement au système de gestion hôtelière (PMS). Pour toute réservation dont la probabilité d'annulation dépasse *0,22*, le système doit automatiser l'envoi d'un e-mail de pré-enregistrement (web check-in) à J-7 avec une petite incitation. Cela permettra de confirmer les séjours douteux ou de libérer la chambre à temps pour une revente.
---

### **7. Reproductibilité**

- version de Python : 3
- principales bibliothèques et versions : pandas, numpy, scikit-learn 1.8.0, lightgbm 4.7.0, xgboost 3.4.0, catboost 1.2.10, optuna (dernière version stable)
- graine(s) aléatoire(s) : RANDOM_STATE = 42, fixée dans tous les scripts (temporal_split, entraînement des 3 modèles, réentraînement final) ainsi que dans les samplers Optuna (TPESampler(seed=42))
- commande ou procédure d’exécution :
        _python3 prep_common.py — vérifie le pipeline de préparation (split + features)
        _Exécuter le notebook fusionné de bout en bout depuis un noyau vierge (Membre 4)
        _python3 final_model_lgbm.py — réentraîne le modèle final sur tout le train et génère predictions_membre3_lgbm.csv

- durée approximative d’entraînement :
        LightGBM : ~0.4s
        XGBoost : ~0.8s
        CatBoost : ~2.3s
        Recherche d'hyperparamètres Optuna (40 essais/modèle) : quelques minutes par modèle, non incluse dans les temps ci-dessus
- environnement utilisé : Local, jupyter.

---

### **8. Bibliographie**
Documentation officielle LightGBM : https://lightgbm.readthedocs.io/
Documentation officielle XGBoost : https://xgboost.readthedocs.io/
Documentation officielle CatBoost : https://catboost.ai/docs/
Documentation Optuna (recherche d'hyperparamètres bayésienne) : https://optuna.readthedocs.io/
Documentation scikit-learn (métriques F1/précision/rappel, seuils) : https://scikit-learn.org/stable/modules/model_evaluation.html
Outil d'IA générative utilisé : Claude (Anthropic), utilisé pour l'assistance au code du pipeline de modélisation (comparaison LightGBM/XGBoost/CatBoost, recherche d'hyperparamètres, optimisation du seuil, analyse d'erreurs et calculs statistiques présentés dans ce rapport). Tous les résultats numériques ont été exécutés et vérifiés sur les données réelles du projet, pas générés par l'IA.