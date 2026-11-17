"""Tok potrubí.

Pro každou stranu: geometrie → textová vrstva → verdikt → (klasifikace) →
směrování → rozvržení → text do bloků (z textové vrstvy, nebo rozpoznáním
podle typu bloku) → jazyk. Nakonec skladatel vytvoří výstup.

Tento modul se při výměně modelu nebo knihovny nemění; mění se adaptér
a `config/pipeline.yaml`.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime

from atr.config import Components, PipelineConfig
from atr.ports.errors import DependencyError
from atr.ports.types import (
    BBox,
    Block,
    BlockLine,
    BlockType,
    ComponentInfo,
    DocumentInput,
    DocumentResult,
    LayoutBlock,
    PageImage,
    PageResult,
    PositionAccuracy,
    Route,
    RouteDecision,
)
from atr.render_utils import crop


class PipelineError(RuntimeError):
    """Selhání úlohy. Neúplný výsledek se nevydává (kontrakt #17, bod 9)."""

    def __init__(self, reason: str, detail: str, pages: list[int] | None = None) -> None:
        super().__init__(f"{reason}: {detail}")
        self.reason = reason
        self.detail = detail
        self.pages = pages or []


class Pipeline:
    def __init__(self, config: PipelineConfig, components: Components) -> None:
        self.config = config
        self.c = components

    # --- veřejné API -------------------------------------------------------

    def run(self, document: DocumentInput) -> dict:
        return self.run_with_paradata(document)[0]

    def run_with_paradata(self, document: DocumentInput) -> tuple[dict, dict]:
        """Výstup `atr/docproc-json` a k němu paradata (atr.paradata)."""
        from atr import paradata

        result = self.process(document)
        output = self.c["assembler"].assemble(result)
        if self.config.validate_output:
            from atr import validation

            problems = validation.errors(output)
            if problems:
                # Neplatný výstup se nevydává; je to chyba služby, ne vstupu.
                raise PipelineError("invalid_output", "; ".join(problems[:5]))
        return output, paradata.build(result, output)

    def process(self, document: DocumentInput) -> DocumentResult:
        started = _now()
        doc = self.c["page_source"].open(document)
        try:
            known = {g.number for g in doc.pages}
            unknown = sorted(set(document.forced_pages) - known)
            if unknown:
                raise PipelineError("invalid_forced_pages", f"strany mimo dokument: {unknown}", unknown)

            pages = [self._page(doc, geometry, document.forced_pages.get(geometry.number))
                     for geometry in doc.pages]
        finally:
            doc.close()

        return DocumentResult(document=document, pages=pages, components=self._components_info(),
                              started_at=started, finished_at=_now())

    # --- jedna strana ------------------------------------------------------

    def _page(self, doc, geometry, forced_reason: str | None) -> PageResult:
        number = geometry.number
        layer = self.c["text_layer"].read(doc, number)
        judgement = self.c["text_layer_judge"].judge(layer, geometry)

        image: PageImage | None = None
        page_class = None
        classifier = self.c.get("page_classifier")
        if classifier is not None:
            image = self.c["renderer"].render(doc, number, self.config.render_dpi)
            page_class = classifier.classify(image)

        decision = self.c["router"].decide(geometry, judgement, forced_reason, page_class)
        page = PageResult(geometry=geometry, decision=decision, judgement=judgement, page_class=page_class)
        if decision.route is Route.EMPTY:
            return page

        if image is None:
            image = self.c["renderer"].render(doc, number, self.config.render_dpi)
        layout = self._layout(image, geometry)

        if decision.route is Route.OCR and not layout:
            # Strana bez textové vrstvy, na které rozvržení nic nenašlo, je prázdná.
            # Vynucená strana zůstává vynucená i bez bloků.
            page.decision = RouteDecision(number, Route.EMPTY, f"no_layout_blocks; {decision.reason}")
            return page

        if decision.route is Route.DIGITAL:
            page.blocks = self.c["text_assigner"].assign(layer, layout)
            for block in page.blocks:
                block.text_source = Route.DIGITAL
            coverage = self._text_coverage(layout, page.blocks)
            if coverage is not None and coverage < self.config.min_text_coverage:
                # Textová vrstva pokrývá jen část textových bloků (např. vložený sken,
                # nevykreslitelné písmo): strana jde celá na OCR.
                decision = RouteDecision(number, Route.OCR, f"text_layer_incomplete coverage={coverage:.2f}")
                page.decision = decision
                page.blocks = []

        if decision.route is not Route.DIGITAL:
            try:
                jobs = [(block, i) for i, block in enumerate(layout)]

                def run(job: tuple[LayoutBlock, int]) -> Block:
                    return self._recognize(image, job[0], job[1], decision.route, layout)

                if self.config.recognize_concurrency > 1 and len(jobs) > 1:
                    with ThreadPoolExecutor(self.config.recognize_concurrency) as pool:
                        page.blocks = list(pool.map(run, jobs))
                else:
                    page.blocks = [run(job) for job in jobs]
            except DependencyError:
                raise  # nedostupná služba: úlohu jde zopakovat, není to selhání rozpoznání
            except Exception as exc:
                # Každé jiné selhání rozpoznání je selhání úlohy (kontrakt #17, bod 9).
                raise PipelineError("all_engines_failed", f"{type(exc).__name__}: {exc}", [number]) from exc

        self._languages(page)
        return page

    def _text_coverage(self, layout: list[LayoutBlock], blocks: list[Block]) -> float | None:
        """Podíl plochy čitelných bloků rozvržení, do kterých textová vrstva dala text."""
        total = covered = 0.0
        for block, assigned in zip(layout, blocks):  # assign vrací bloky ve stejném pořadí
            if block.type not in self.config.recognize:
                continue  # obrázky apod. text z vrstvy mít nemusí
            total += block.bbox.area
            if assigned.text.strip():
                covered += block.bbox.area
        return covered / total if total > 0 else None

    def _layout(self, image: PageImage, geometry) -> list[LayoutBlock]:
        """Rozvržení → (úprava geometrie) → pořadí čtení."""
        blocks = self.c["layout"].detect(image, geometry)
        refiner = self.c.get("layout_refiner")
        if refiner is not None:
            blocks = refiner.refine(blocks, geometry)
        reading = self.c.get("reading_order")
        if reading is not None:
            return reading.order(blocks, geometry)
        return sorted(blocks, key=_reading_key)

    def _recognize(self, image: PageImage, block: LayoutBlock, order: int, route: Route,
                   page_blocks: list[LayoutBlock]) -> Block:
        out = Block(type=block.type, bbox=block.bbox, confidence=block.confidence,
                    order=block.order if block.order is not None else order)
        name = self.config.recognize.get(block.type)
        if name is None:
            # Typ bez rozpoznávače (obrázek, kresba): ve výstupu jen polohou.
            out.text_source = route
            return out
        content = self.c.recognizers[name].recognize(
            crop(image, block.bbox, self.config.crop_padding_pt,
                 mask=[b.bbox for b in page_blocks if b is not block and b.type is not BlockType.FIGURE]
                 if self.config.mask_neighbours else None),
            block)
        out.text, out.format, out.text_source = content.text, content.format, route
        out.lines = _derived_lines(content.text, block.bbox) if content.format == "text" else []
        return out

    def _languages(self, page: PageResult) -> None:
        detector = self.c.get("language")
        if detector is None:
            return
        for block in page.blocks:
            if block.text.strip():
                found = detector.detect(block.text)
                block.language = found[0] if found else None
        text = "\n".join(b.text for b in page.blocks if b.text.strip())
        found = detector.detect(text) if text else None
        page.language = found[0] if found else None

    def _components_info(self) -> list[ComponentInfo]:
        infos = [a.info() for a in self.c.adapters.values()]
        infos += [r.info() for r in self.c.recognizers.values()]
        return infos


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _reading_key(block: LayoutBlock) -> tuple:
    if block.order is not None:
        return (0, block.order)
    return (1, round(block.bbox.y0), block.bbox.x0)


def _derived_lines(text: str, bbox: BBox) -> list[BlockLine]:
    """Řádky z `\\n` v rozpoznaném textu; poloha je odhad (rovnoměrné rozdělení výšky bloku)."""
    rows = [r for r in text.split("\n") if r.strip()]
    if not rows:
        return []
    step = bbox.height / len(rows)
    return [BlockLine(text=row, accuracy=PositionAccuracy.DERIVED,
                      bbox=BBox(bbox.x0, bbox.y0 + i * step, bbox.x1, bbox.y0 + (i + 1) * step))
            for i, row in enumerate(rows)]
