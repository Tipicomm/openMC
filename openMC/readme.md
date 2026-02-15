
1) Générer le modèle
```
python scripts/fabriqueModele.py \
  inbox/depenses-culturelles-des-communes-total-2023.xlsx
```
Cela ajout les deux fichiers : 
````
model/depenses-culturelles-des-communes-total-2023.model.json
out/depenses-culturelles-des-communes-total-2023.preview.json
````
2) Générer le fichier export CSV nettoyé

```
python scripts/excel_to_csv.py \
  inbox/depenses-culturelles-des-communes-total-2023.xlsx \
  model/depenses-culturelles-des-communes-total-2023.model.json
  ```

  Cela permet de produire :
````
out/depenses-culturelles-des-communes-total-2023.clean.csv
```