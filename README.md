## Arborescence

notebooks/
   ├── series_utils.py                     génération des séries synthétiques
   ├── adapt_utils.py                      fenêtrage, découpages, évaluation ADAPT
   ├── generateur.ipynb                    fabrication des 6 décalages
   ├── test_adapt.ipynb                    comparaison des 9 méthodes
   ├── extraction_stl.ipynb                validation de la décomposition
   ├── statsmodels.ipynb                   vérif ARIMA
   ├── meteo/                              
   │   ├── prevision_meteo.ipynb           notebook principal
   │   ├── forecast_statsmodels.ipynb      stl forecast, ARIMA
   │   └── test_meteo.ipynb                exploration initiale
   │
   ├── har/                                
   │   ├── classification_har.ipynb        notebook principal
   │   ├── classification_har_brut.ipynb   signaux bruts
   │   ├── classification_har_brut_features.ipynb   signaux bruts + features
   │   └── test_har.ipynb                  exploration initiale
   │
   ├── offset/                              décalage de niveau
   ├── scale/                               décalage d'échelle
   ├── trend/                               décalage de la tendance
   ├── frequency/                           décalage de fréquence
   ├── noise/                               décalage du bruit
   ├── combined/                            les 5 décalages cumulés
data/
   ├── synthetic/                           source-cible générées (.npz) pour les différentes intensités
