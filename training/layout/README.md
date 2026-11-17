# Dotrénování PP-DocLayoutV3 se třídou `handwriting`

Cíl: model rozvržení se slovníkem PP-DocLayoutV3 (25 tříd) a novou třídou
`handwriting` pro ručně psané doplňky a poznámky. Výsledný model se zveřejní
a zapojí do služby `models/layout-ppdoclayoutv3` jako nová revize.

## Data

| Zdroj | Stran | Licence | Použití |
|---|---|---|---|
| [LADaS](https://huggingface.co/datasets/almanach/LADaS) (Inria, ALMANaCH), revize `2c1f39f` | 6 059 | CC-BY-4.0 | train/val/test podle LADaS |
| anotace AMČR z eScriptoria (SegmOnto) | 100 | interní | train/val/test 60/20/20 podle dokumentu |

Zóny SegmOnto se převádějí na třídy V3 podle [`mapping.yaml`](mapping.yaml)
(rozhodnutí k revizi jsou označená „REVIZE“). Ručně psané doplňky
(`MarginTextZone-ManuscriptAddendum`) → `handwriting`.

Počty objektů po převodu (17. 11. 2026):

| třída | train | val | test |
|---|---|---|---|
| text | 31 675 | 5 175 | 2 587 |
| paragraph_title | 4 570 | 830 | 472 |
| number | 3 884 | 601 | 258 |
| image | 3 404 | 541 | 326 |
| figure_title | 2 515 | 361 | 199 |
| header | 2 116 | 284 | 151 |
| **handwriting** | **1 147** | **260** | **114** |
| footnote | 685 | 80 | 40 |
| table | 511 | 95 | 63 |
| aside_text | 497 | 45 | 25 |
| seal | 445 | 76 | 40 |
| footer | 333 | 45 | 24 |
| display_formula | 226 | 46 | 15 |
| doc_title | 156 | 27 | 14 |
| ostatní třídy V3 | 0 | 0 | 0 |

Třídy, které data nemají (abstract, algorithm, chart, content, reference, …),
doplní `to_paddlex.py` pseudo-popisky z původního modelu, aby je model nezapomněl.

## Postup

```bash
# 1. datová sada COCO (~2 GB, mimo git)
python training/layout/build_dataset.py --ladas ~/.cache/aiscr-atr/datasets/ladas \
    --escriptorium EXPORT_ZK EXPORT_DS --out ~/.cache/aiscr-atr/datasets/ppdoclayoutv3-hw

# 2. formát PaddleX: polygony, pořadí čtení a pseudo-popisky z původního modelu
#    (potřebuje běžící službu rozvržení; na CPU ~1 h)
python training/layout/to_paddlex.py ~/.cache/aiscr-atr/datasets/ppdoclayoutv3-hw \
    --out ~/.cache/aiscr-atr/datasets/ppdoclayoutv3-hw-paddlex --layout-url http://127.0.0.1:18002

# 3. trénink v PaddleX (modul layout_analysis) – na stroji x86_64 s CUDA 12.6/12.9
python main.py -c paddlex/configs/modules/layout_analysis/PP-DocLayoutV3.yaml \
    -o Global.mode=check_dataset -o Global.dataset_dir=DATASET
python main.py -c ... -o Global.mode=train -o Global.dataset_dir=DATASET
python main.py -c ... -o Global.mode=evaluate -o Global.dataset_dir=DATASET

# 4. převod do safetensors pro transformers (služba rozvržení)
python main.py -c ... -o Global.mode=pdparams2safetensors \
    -o Pdparams2safetensors.input_path=output/best_model/best_model/model.pdparams \
    -o Pdparams2safetensors.output_dir=export/
```

## Kde trénovat

PaddlePaddle nemá GPU sestavení pro aarch64, takže **DGX Spark oficiálně
nepůjde** (jen vlastní překlad PaddlePaddle ze zdrojů, nepodporované).
Spolehlivá cesta je PaddleX na x86_64 s NVIDIA GPU, tedy plánovaná **4090**.
DGX Spark se hodí na kroky v PyTorch: pseudo-popisky (krok 2) a vyhodnocení
převedeného modelu přes transformers.

## Otevřené body

- Revize mapy zón (formuláře, titulní strana, `MainZone-Ab`).
- Anotace AMČR jsou interní: model na nich natrénovaný zveřejnit lze, data ne.
  Pokud by neměly jít ani do tréninku, stačí je vynechat (`--escriptorium` bez argumentů).
- Pořadí čtení se učí z předpovědí původního modelu; vlastní anotace pořadí by ho zlepšily.
- Zveřejnění: licence modelu Apache-2.0 (jako PP-DocLayoutV3), uvést LADaS (CC-BY-4.0).
