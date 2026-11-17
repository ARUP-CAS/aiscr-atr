# Testovací sada

## Syntetická sada (v gitu jen generátor)

`python samples/generate.py samples/generated` vytvoří PDF a obrázky s přesně
známou geometrií: born-digital strana, otočení 90/180/270°, CropBox odlišný
od MediaBox, otočená a oříznutá strana, `/UserUnit` 2, sken, smíšené PDF
(text, sken, prázdná strana), PNG s 300 DPI a dvoustránkový TIFF.
`expected.json` u nich uvádí rozměry stran a polohu značkovacího slova.
Testy (`atr/tests/test_pdfium.py`) si sadu generují samy.

Pro české znaky potřebuje generátor písmo s diakritikou (DejaVu Sans nebo
Liberation Sans, případně cesta v `ATR_TEST_FONT`); bez něj se test češtiny přeskočí.

## Skutečné dokumenty (mimo git)

Složka `samples/corpus/` je v `.gitignore`. Patří do ní vzorek z AMČR pro
fáze 2–4 (rozvržení, směrování, rozpoznání): naskenované zprávy, born-digital
PDF, smíšená PDF, rukopis, tabulky, kresby a fotografie, ideálně s ručně
ověřeným přepisem. Zdroj: referenční vzorek z
[aiscr-management#154](https://github.com/ARUP-CAS/aiscr-management/issues/154).
