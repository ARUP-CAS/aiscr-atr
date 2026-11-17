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

# 3. trénink v PyTorch (AMD ROCm, NVIDIA CUDA i DGX Spark)
pip install -r training/layout/requirements.txt   # + torch podle GPU, viz requirements.txt
python training/layout/train_torch.py --data ~/.cache/aiscr-atr/datasets/ppdoclayoutv3-hw-paddlex \
    --out runs/baseline --eval-only                 # výchozí model pro srovnání
python training/layout/train_torch.py --data ~/.cache/aiscr-atr/datasets/ppdoclayoutv3-hw-paddlex \
    --out runs/hw-v1 --epochs 30 --batch-size 8

# 4. výsledek runs/hw-v1/best je ve formátu transformers; služba rozvržení ho načte:
#    MODEL_ID=/cesta/runs/hw-v1/best MODEL_REVISION=main
```

## Kde trénovat

Oficiální trénink PP-DocLayoutV3 je jen v PaddleX a PaddlePaddle nemá GPU sestavení
pro AMD (ROCm) ani pro ARM (DGX Spark). Proto `train_torch.py` trénuje přímo
implementaci z transformers v PyTorch:

- ztráta RT-DETR z transformers (stejná rodina modelů): třídy a polohy bloků,
  pomocné ztráty dekodéru, denoising;
- hlava tříd se rozšíří z 25 na 26; původní třídy převezmou váhy, `handwriting`
  startuje z vah třídy `text`;
- hlava pořadí čtení a hlava masek jsou zmrazené (data je nemají);
- běží na AMD RX 7900 XTX (ROCm), NVIDIA i DGX Spark; ověřeno na CPU na pár krocích.

Pořadí čtení se tak netrénuje, ale ani nezhorší přímo; změna sdílených vrstev
ho může ovlivnit, proto ho po tréninku zkontrolovat.

## Výsledky

### Běh `hw-v1` (17. 11. 2026, RX 7900 XTX, ROCm 6.4)

Učící krok 5e-5 (páteř 5e-6), dávka 8, bf16, asi 5 minut na epochu. Zastaveno po 15 epochách;
nejlepší je 4. epocha. Celkové mAP pak klesalo jen kvůli vzácným třídám s pseudo-popisky
(`algorithm`, `chart`), hlavní třídy se neměnily. Proto se nejlepší model nově vybírá podle
AP50 tříd s ručními anotacemi.

Pokrytí anotovaných oblastí na stranách AMČR mimo trénovací část
(`scripts/eval_layout_segmonto.py --only`, 38 stran, 833 oblastí):

| oblast | původní PP-DocLayoutV3 | `hw-v1` |
|---|---|---|
| rukopisný doplněk (16) | 38 % | 81 % |
| záhlaví a zápatí (17) | 24 % | 100 % |
| formulář (97) | 79 % | 99 % |
| podpis, datum, adresa (39) | 82 % | 97 % |
| nadpis (18) | 89 % | 100 % |
| odstavec (60) | 93 % | 100 % |
| celkem | 88 % | 99 % |

Z 13 nalezených ručních doplňků má 11 třídu `handwriting`. Vzorek je malý a 20 z 38 stran
bylo ve validační části, podle které se vybírala epocha (na váhách se nepodílely).

### Běh `hw-v2` (17. 11. 2026)

Učící krok 2e-5 (páteř 2e-6), 12 epoch, nejlepší model podle AP50 anotovaných tříd.
Validace roste plynule a od 5. epochy se drží (AP50 anotovaných tříd 0,744–0,751,
`handwriting` AP50 0,62–0,69); nejlepší je 7. epocha.

Srovnání obou běhů:

| | `hw-v1` | `hw-v2` |
|---|---|---|
| validace: mAP anotovaných tříd | 0,506 | 0,505 |
| validace: mAP všech tříd | 0,431 | 0,434 |
| AMČR mimo trénink: pokryto | 99 % (chybí 5) | 99 % (chybí 6) |
| AMČR: rukopisný doplněk pokryt | 81 % (11× `handwriting`) | 81 % (11× `handwriting`) |
| AMČR: přesná shoda bloku s oblastí | 324 | 259 |
| AMČR: bloků `text` / `table` | 396 / 58 | 199 / 62 |

Oba modely jsou na validaci i v pokrytí rovnocenné. `hw-v1` vrací navíc jednotlivá pole
formuláře jako textové bloky uvnitř bloku tabulky; ty úprava překryvů (`layout_refiner`)
vypustí jako vnořené. Po ní vychází formulář jako jeden blok tabulky takto
(47 formulářů na 38 stranách AMČR mimo trénink):

| | `hw-v1` | `hw-v2` |
|---|---|---|
| formulář jako jeden blok `table` | 43/47 (91 %) | 40/47 (85 %) |
| bloků uvnitř formuláře (průměr) | 1,4 | 1,7 |

Pro požadavek „formulář = jedna tabulka“ tedy v potrubí vychází o něco lépe `hw-v1`;
rozdíl jsou 3 formuláře na malém vzorku. `hw-v2` má stabilnější průběh tréninku.

## Otevřené body

- Mapa zón potvrzená 17. 11. 2026 (formuláře jako `table`).
- Anotace AMČR smí do tréninku (17. 11. 2026); zveřejní se model, ne data.
- Hlava pořadí čtení se zatím netrénuje; vlastní anotace pořadí by umožnily ji dotrénovat.
- Zveřejnění: běh `hw-v1` je na Hugging Face jako `aiscr/ArchaeoDocLayout`, značka `v0.1`
  (Apache-2.0, zatím soukromý repozitář); `hw-v2` zůstává jen na trénovacím stroji.
