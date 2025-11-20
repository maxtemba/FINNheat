 
Gute Entscheidung — wir gehen jetzt gründlich vor. Kurz und ehrlich: ich habe versucht, dein `temp_model.onnx` automatisiert zu analysieren, konnte das aber **nicht** ausführen, weil die Laufzeit hier das Python-Paket `onnx` nicht installiert hat. Deshalb lief das Analyse-Script nicht.
Das ist kein Problem — im Folgenden bekommst du alles, was du brauchst:

* ein **diagnose-Script** (du kannst es lokal laufen lassen) das das Problem automatisch findet (zeigt genau, wo `80` vs `45` auftaucht),
* ein **Repair-Script**, das widersprüchliche `value_info`-Shapes entweder aktualisiert oder entfernt und das Ergebnis speichert,
* eine **detaillierte Schritt-für-Schritt-Anleitung** wie du das Problem grundsätzlich vermeidest (Best-Practices bei Export / ONNX / FINN), und
* Hinweise welche Stellen im Modell typischerweise die Diskrepanz verursachen (Reshape/Transpose/Initializers/value_info).

Führe die folgenden Schritte lokal in deiner Umgebung (oder in einem Notebook) aus — ich habe die Skripte komplett vorbereitet.

---

# 0) Vorbereitung: falls noch nicht installiert

Öffne Terminal / Notebook und installiere:

```bash
pip install onnx onnxruntime numpy
```

---

# 1) Diagnoseskript — findet jede Abweichung zwischen **inferiertem** Shape und **immanentem** value_info-Shape

Speichere als `onnx_shape_check.py` und führe aus:

```python
# onnx_shape_check.py
import onnx
from onnx import shape_inference, numpy_helper
import sys
from pprint import pprint

if len(sys.argv) < 2:
    print("Usage: python onnx_shape_check.py model.onnx")
    sys.exit(1)
fn = sys.argv[1]

m = onnx.load(fn)
print("Loaded model:", fn)
print("Nodes:", len(m.graph.node), "Initializers:", len(m.graph.initializer))

# try to run shape inference
try:
    m_si = shape_inference.infer_shapes(m)
    print("Shape inference: OK")
except Exception as e:
    print("Shape inference failed:", e)
    print("-> Will continue using original model where possible.")
    m_si = None

# helper to extract dims
def extract_shape(vi):
    try:
        dims = []
        t = vi.type.tensor_type
        for d in t.shape.dim:
            if d.HasField("dim_value"):
                dims.append(int(d.dim_value))
            elif d.HasField("dim_param"):
                dims.append(str(d.dim_param))
            else:
                dims.append(None)
        return dims
    except Exception:
        return None

# collect shapes from original graph and inferred (if available)
orig_shapes = {}
for vi in list(m.graph.value_info) + list(m.graph.input) + list(m.graph.output):
    orig_shapes[vi.name] = extract_shape(vi)

inf_shapes = {}
if m_si:
    for vi in list(m_si.graph.value_info) + list(m_si.graph.input) + list(m_si.graph.output):
        inf_shapes[vi.name] = extract_shape(vi)

# find mismatches between original value_info shapes and inferred shapes
mismatches = []
for name, oshape in orig_shapes.items():
    if name in inf_shapes and inf_shapes[name] is not None and oshape is not None:
        # compare dims where both have concrete ints
        for i,(od, idim) in enumerate(zip(oshape, inf_shapes[name])):
            if isinstance(od,int) and isinstance(idim,int) and od != idim:
                mismatches.append((name, i, od, idim))
                break

if mismatches:
    print("Found mismatches between value_info and inferred shapes:")
    for m in mismatches:
        print(f" - tensor '{m[0]}', dim index {m[1]}: value_info={m[2]} vs inferred={m[3]}")
else:
    print("No mismatches found between value_info and inferred shapes (or no inferred shapes available).")

# locate occurrences of 80 or 45 in value_info / inferred
def find_dim_occurrences(shapes, dims=(80,45)):
    out=[]
    for name,shape in shapes.items():
        if shape is None: continue
        for d in shape:
            if isinstance(d,int) and d in dims:
                out.append((name,shape))
                break
    return out

print("\nOccurrences of 80/45 in original value_info/input/output:")
pprint(find_dim_occurrences(orig_shapes))

if m_si:
    print("\nOccurrences of 80/45 in inferred shapes:")
    pprint(find_dim_occurrences(inf_shapes))

# show node context for any Transpose named Transpose_3
print("\nSearching for Transpose_3 node...")
found=False
for n in m.graph.node:
    if n.name == "Transpose_3":
        found=True
        print("Found node:", n.name, "op:", n.op_type)
        print(" inputs:", list(n.input))
        print(" outputs:", list(n.output))
        # perm attr
        for a in n.attribute:
            if a.name == "perm":
                print(" perm:", list(a.ints))
        break
if not found:
    print("No node named Transpose_3 found. Listing all Transpose nodes:")
    for n in m.graph.node:
        if n.op_type == "Transpose":
            print(" -", n.name, "inputs:", list(n.input), "outputs:", list(n.output))
```

**Was dieses Script liefert**

* Alle Stellen, wo `value_info` konkret `80` oder `45` enthält
* Mismatch-Liste der ersten Abweichung pro Tensor (z. B. `tensor X: dim1 value_info=80 vs inferred=45`)
* Details zu `Transpose_3` (Inputs/Outputs & perm)

---

# 2) Repair-Script — aktualisiert/entfernt widersprüchliche `value_info`-Einträge

Speichere als `repair_conflicting_value_info.py` und führe aus:

```python
# repair_conflicting_value_info.py
import onnx, sys
from onnx import shape_inference

if len(sys.argv) < 3:
    print("Usage: python repair_conflicting_value_info.py model.onnx repaired.onnx")
    sys.exit(1)

in_fn = sys.argv[1]
out_fn = sys.argv[2]

m = onnx.load(in_fn)
print("Loaded", in_fn)

# Try shape inference
try:
    m_si = shape_inference.infer_shapes(m)
    print("Shape inference OK.")
except Exception as e:
    print("Shape inference failed:", e)
    print("-> Clearing graph.value_info and retrying inference.")
    m.graph.value_info.clear()
    try:
        m_si = shape_inference.infer_shapes(m)
        print("Second inference OK after clearing value_info.")
    except Exception as e2:
        print("Still failing:", e2)
        print("Saving original model as-is to", out_fn)
        onnx.save(m, out_fn)
        sys.exit(1)

# Replace or update value_info entries with inferred ones
# (this avoids stale/wrong static shapes)
m.graph.value_info[:] = list(m_si.graph.value_info)

# Also attempt to update graph outputs shapes
infer_outputs = {vi.name:vi for vi in list(m_si.graph.output)}
for i,o in enumerate(m.graph.output):
    if o.name in infer_outputs:
        try:
            m.graph.output[i].type.CopyFrom(infer_outputs[o.name].type)
        except Exception:
            pass

onnx.save(m, out_fn)
print("Saved repaired model to", out_fn)
```

**Hinweis:** immer vorher ein Backup machen:

```bash
cp temp_model.onnx temp_model.onnx.bak
python repair_conflicting_value_info.py temp_model.onnx temp_model.repaired.onnx
```

---

# 3) Was genau typischerweise die Ursache ist (technisch, verständlich)

Der Fehler

```
[ShapeInferenceError] ... Inferred shape and existing shape differ in dimension 1: (80) vs (45)
```

bedeutet: **für einen bestimmten Tensor** existiert im Graph ein *statischer* Shape-Eintrag (z. B. in `graph.value_info` oder `graph.output`) mit Wert `80` an Dimension-Index 1, während ONNX die Form für denselben Tensor aus dem Rechnergraphen auf `45` schlussfolgert.

Typische Ursachen:

* **Alte/handgesetzte `value_info`-Einträge**: Du (oder ein Tool) hat manuell Shapes gesetzt oder es blieb ein alter Shape-Eintrag nach einem Modell-Änderungsschritt zurück.
* **Reshape/Reshape-Const ist falsch**: Ein `Reshape` verwendet als zweite Eingabe einen *Constant* (Initializer) mit einer Zielshape, die nicht zur tatsächlichen Datenform passt.
* **Fehlerhafte Export-Pipeline (z. B. PyTorch→ONNX)**: Bei Export wurden einige Dimensionen hartkodiert oder falsch gemappt (z. B. channel/feature dim).
* **Transpose/Flatten/Concat Logik**: Reihenfolge von Achsen (perm) führt zu einer anderen Dimension als erwartet.
* **Partielle Shape-Inference**: Manche Tools fügen value_info hinzu ohne vollständige (oder korrekte) infer_shapes anzuwenden.

---

# 4) Konkrete Maßnahmen, damit der Fehler gar nicht erst auftritt (Prävention)

1. **Direkt nach Export `shape_inference.infer_shapes()` laufen lassen**

   * Immer nach dem ONNX-Export `onnx.shape_inference.infer_shapes(model)` aufrufen und das Ergebnis (z. B. `model_inferred.onnx`) speichern. Damit sind value_info-Einträge konsistent mit dem Graph.

2. **Keine manuellen value_info-Einträge**

   * Falls du `value_info` manuell setzt, vermeide harte Zahlen für interne Tensoren. Wenn nötig, setze nur `dim_param` (z. B. `"N"`) für Batch oder lasse ONNX inferieren.

3. **Reshape-Targets als aus Nodes berechnen lassen, nicht als falsche Konstanten**

   * Wenn du Reshape benutzt, vermeide konstante target-shapes, die nicht garantiert korrekt sind. Wenn Const nötig, verifiziere seine Werte gegen Testsamples.

4. **Test mit Beispieldaten vor FINN-Streamline**

   * Führe ein kurzes ONNX-Runtime Run mit einem Test-Input (random passend zu erwarteten Input-Shape) und überprüfe Shapes der Outputs mit `print()` oder ONNX-Runtime Profiling.

5. **Be careful with Transpose / perm**

   * Prüfe Transpose perm-Attribute. Wenn Transpose Dimensionen permutiert, nachfolgende value_info muss diese Permutation widerspiegeln.

6. **Automatisierte Checks in CI/Export-Pipeline**

   * Integriere `onnx.shape_inference` + das Diagnoseskript in deine Export-Pipeline vor FINN-Schritten. Scheitern lassen, wenn Mismatch gefunden wird.

7. **Use a single source of truth for shapes**

   * Wenn du Shapes generierst/speicherst, dann nur in einem Teil der Pipeline. Doppelte, manuelle Shapeverwaltung führt schnell zu Verzweiflung.

8. **Auf Opset-Inkompatibilitäten achten**

   * Manche Opset-Versionen ändern Verhalten; halte ONNX/Export-Tooling und FINN kompatibel.

---

# 5) Wenn du FINN benutzt — extra Hinweise

* FINN ist strenger als ONNX Runtime. Wenn FINN beim Streamline meckert, heißt das meistens, dass ein unerkannter oder alter `value_info` im Graph ist.
* Nutze FINN-Hilfsfunktionen wie `tidy_up` vor Streamline, aber **verlasse dich nicht** darauf allein — führe ONNX-Shape-Inference vorher aus.
* Bei quantisierten Modellen: achte besonders auf Kanalgrößen und Feature-Maps (oft werden Kanäle falsch reshaped).

---

# 6) Nützliche Debugging-Workflows (Kurzprotokoll)

1. `onnx.shape_inference.infer_shapes` → speichern.
2. `onnx_shape_check.py` laufen lassen → falls mismatch, notiere betroffene Tensor-Namen.
3. Suche Node-Kontext (Producer/Consumer) des betroffenen Tensors — meistens Reshape/Transpose liegt direkt davor.
4. Prüfe Initializer (Konstanten) für Reshape-Ziele — oft ist hier die falsche Zahl.
5. Entweder:

   * Update den Initializer (wenn er falsch ist), oder
   * Entferne/aktualisiere `value_info` mit inferierten Shapes (Repair-Script).
6. Nochmal FINN `tidy_up` + `Streamline`.

---

# 7) Beispiel: Check für Reshape-Initializer (Snippet)

Dieses kurze Snippet listet alle Reshape-Nodes und zeigt, ob der Zielshape konstant ist:

```python
import onnx
from onnx import numpy_helper
m = onnx.load("temp_model.onnx")
for n in m.graph.node:
    if n.op_type == "Reshape":
        print("Reshape:", n.name, "inputs:", n.input, "outputs:", n.output)
        if len(n.input) > 1:
            shape_name = n.input[1]
            inits = [ini for ini in m.graph.initializer if ini.name == shape_name]
            if inits:
                arr = numpy_helper.to_array(inits[0])
                print("  -> target shape initializer:", arr)
            else:
                print("  -> target shape is dynamic (not a constant initializer).")
```

Wenn du ein konstantes Zielshape mit einer falschen `80`/`45` siehst — das ist der Übeltäter.

---

# 8) Was ich jetzt für dich tun kann (Optionen)

1. **Du führst die obigen Skripte lokal aus** (installiere `onnx`) und schickst mir die Ausgabe — ich interpretiere sie und gebe gezielte Fixes.
2. **Ich generiere noch genaueres Fix-Patch**: z. B. ein Script, das gezielt `value_info`-Einträge mit Mismatch ersetzt oder Reshape-Initializer korrigiert — wenn du mir sagst, welche Tensoren betroffen sind (oder mir die Script-Ausgabe gibst).
3. Wenn du willst, kann ich dir ein neues Notebook JSON (vollständig mit Zellen) erzeugen, das die Diagnose + Repair-Buttons enthält — du kannst es direkt in Jupyter öffnen. (Ich habe dir oben bereits ein einfaches Notebook gegeben; sag kurz, ob du es erweitert haben möchtest.)

---

# 9) Kurz-Zusammenfassung / Handlungsempfehlung jetzt

* **Wahrscheinlich**: ein `value_info` oder ein Reshape-Initializer stimmt nicht mit dem tatsächlichen Tensor überein — daher das `80 vs 45`.
* **Sofort**: pip install onnx; `python onnx_shape_check.py temp_model.onnx` — das gibt dir sofort den betroffenen Tensornamen.
* **Falls du möchtest**: poste die Ausgabe (insbesondere die Mismatch-Zeile) hier — ich sage dir dann exakt, welche Node/Initializer zu ändern ist und liefere ein 1-Kommando-Fix.

---

Wenn du willst, gebe ich dir jetzt sofort:

* die **vollständige Notebook-JSON** mit erweiterten Diagnose- und Repair-Zellen (du hattest bereits eine Version; ich kann sie auf Basis der Skripte erweitern), **oder**
* du führst `onnx_shape_check.py` lokal aus und postest die ersten paar Zeilen der Ausgabe — dann liefere ich das exakte Patch (z. B. „Ersetze value_info für `X` von 80 → 45“ oder „ändere Reshape-initializer `const_Y` von [1,80, ...] auf [1,45,...]“).

Was möchtest du als Nächstes?
