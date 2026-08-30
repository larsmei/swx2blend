# swx2blend

**Deutsch** · [English](#english)

Blender-Add-on, das SOLIDWORKS-Teile (`.sldprt`) und -Baugruppen (`.sldasm`) **direkt** importiert. Keine SOLIDWORKS-Installation, kein Zwischenschritt über STEP.

Der Parser liest die Viewport-Tessellation (Triangle-Strips in den DisplayLists) plus Appearances und Baugruppen-Platzierungen. Das ist das Netz, das SOLIDWORKS im Viewport zeigt — keine exakte B-Rep.

Parser-Logik portiert aus [showmecad](https://github.com/larsmei/showmecad).

## Installation

1. [Release-ZIP](https://github.com/larsmei/swx2blend/archive/refs/heads/main.zip) herunterladen, **oder** dieses Repository klonen.
2. In Blender 4.2+: *Edit → Preferences → Add-ons → Install from Disk* und den Ordner `swx2blend` bzw. die ZIP wählen.
3. Add-on **swx2blend** aktivieren.

Ältere Blender-Versionen (3.6–4.1): den Ordner `swx2blend` nach `scripts/addons/` kopieren.

## Nutzung

*File → Import → SolidWorks (.sldprt/.sldasm)*

Mehrfachauswahl und Drag & Drop werden unterstützt. Option *Weld vertices* verschmilzt deckungsgleiche Punkte.

Koordinaten: SOLIDWORKS und Blender sind beide Z-up. Einheiten werden automatisch nach Metern skaliert, wenn die Tessellation in Millimetern vorliegt.

## Grenzen

- Nur das Anzeigenetz, keine parametrische Historie, keine Skizzen, keine Features.
- RealView-Bitmaps liegen oft nur als `.p2m`-Referenz in der Datei — Farben und Glanz werden übernommen, eingebettete Raster nur, soweit vorhanden.
- Muster- und Spiegelkopien in Baugruppen werden über die CompInstance-XML platziert.
- Für schärfere Kanten die Datei in SOLIDWORKS als STEP speichern und separat triangulieren.

## Entwicklung

Der Parser (`swx/`) braucht nur Python 3.10+ und NumPy (in Blender enthalten). Ohne Blender testen:

```bash
python3 -m swx.convert pfad/zu/teil.sldprt
```

## Lizenz

MIT. Siehe [LICENSE](LICENSE).

---

<a id="english"></a>

## English

Blender add-on that imports SOLIDWORKS parts (`.sldprt`) and assemblies (`.sldasm`) **directly**. No SOLIDWORKS install, no STEP round-trip.

It reads the viewport tessellation (triangle strips in DisplayLists) plus appearances and assembly placements — the mesh SOLIDWORKS shows on screen, not exact B-Rep.

Parser logic ported from [showmecad](https://github.com/larsmei/showmecad).

### Install

1. Download the [zip](https://github.com/larsmei/swx2blend/archive/refs/heads/main.zip) or clone this repo.
2. Blender 4.2+: *Edit → Preferences → Add-ons → Install from Disk*, pick the `swx2blend` folder or zip.
3. Enable **swx2blend**.

### Use

*File → Import → SolidWorks (.sldprt/.sldasm)*

Both Blender and SOLIDWORKS are Z-up. Millimetre tessellations are scaled to metres.

### Limits

Display mesh only — no feature tree. Patterned and mirrored assembly copies are stamped from CompInstance XML. For sharper edges, export STEP from SOLIDWORKS.

MIT. See [LICENSE](LICENSE).
