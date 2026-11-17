"""Příkazová řádka pro zkoušení komponent nad jedním souborem.

    python -m atr inspect dokument.pdf [--render-dir out/] [--dpi 150]
        geometrie stran, textová vrstva, verdikt a trasa (bez modelů)

    python -m atr convert dokument.pdf [--config config/pipeline.yaml] [--forced 3=needs_atr ...]
        celé potrubí podle konfigurace (potřebuje běžící kontejnery modelů)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from atr import config, registry
from atr.ports.errors import InputError
from atr.ports.types import DocumentInput


def _document(path: str, forced: list[str] | None = None) -> DocumentInput:
    pages = {}
    for item in forced or []:
        number, _, reason = item.partition("=")
        pages[int(number)] = reason or "forced"
    return DocumentInput(Path(path).read_bytes(), Path(path).name, forced_pages=pages)


def inspect(args: argparse.Namespace) -> int:
    source = registry.create("page_source", "pdfium")
    reader = registry.create("text_layer", "pdfium")
    judge = registry.create("text_layer_judge", "rules")
    router = registry.create("router", "rules")
    renderer = registry.create("renderer", "pdfium")
    doc = source.open(_document(args.file, args.forced))
    try:
        for g in doc.pages:
            layer = reader.read(doc, g.number)
            verdict = judge.judge(layer, g)
            decision = router.decide(g, verdict, _document(args.file, args.forced).forced_pages.get(g.number),
                                     None)
            print(f"strana {g.number}: {g.width:.1f} × {g.height:.1f} pt, otočení {g.rotation}°, "
                  f"UserUnit {g.user_unit}" + (f", obrázek {g.source_pixels} px" if g.source_pixels else ""))
            print(f"  textová vrstva: {len(layer.lines)} řádků, {len(layer.words)} slov, "
                  f"nemapovaných znaků {layer.unmapped_chars}")
            print(f"  verdikt: {verdict.verdict.value} ({verdict.reason}); trasa: {decision.route.value}")
            for line in layer.lines[: args.lines]:
                b = line.bbox
                print(f"    [{b.x0:6.1f} {b.y0:6.1f} {b.x1:6.1f} {b.y1:6.1f}] {line.text[:90]}")
            if args.render_dir:
                out = Path(args.render_dir)
                out.mkdir(parents=True, exist_ok=True)
                image = renderer.render(doc, g.number, args.dpi)
                (out / f"strana-{g.number:04d}.png").write_bytes(image.png)
    finally:
        doc.close()
    return 0


def convert(args: argparse.Namespace) -> int:
    from atr.pipeline import Pipeline, PipelineError

    cfg = config.load(args.config)
    try:
        out = Pipeline(cfg, config.build(cfg)).run(_document(args.file, args.forced))
    except PipelineError as exc:
        print(json.dumps({"reason": exc.reason, "detail": exc.detail, "pages": exc.pages}), file=sys.stderr)
        return 2
    json.dump(out, sys.stdout, ensure_ascii=False, indent=1)
    print()
    return 0


def main(argv: list[str] | None = None) -> int:
    registry.load_builtin_adapters()
    parser = argparse.ArgumentParser(prog="python -m atr")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("inspect", help="geometrie, textová vrstva a trasa stran bez modelů")
    p.add_argument("file")
    p.add_argument("--forced", nargs="*", help="vynucené strany, např. 3=needs_atr")
    p.add_argument("--lines", type=int, default=5, help="kolik řádků textové vrstvy vypsat")
    p.add_argument("--render-dir", help="uložit vykreslené strany jako PNG")
    p.add_argument("--dpi", type=float, default=150)
    p.set_defaults(func=inspect)

    p = sub.add_parser("convert", help="celé potrubí podle konfigurace")
    p.add_argument("file")
    p.add_argument("--config", default="config/pipeline.yaml")
    p.add_argument("--forced", nargs="*")
    p.set_defaults(func=convert)

    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except InputError as exc:
        print(f"{exc.reason}: {exc.detail}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
