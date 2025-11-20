
### Was Sie sehen werden (Wissenschaftliche Einordnung)

Nach dem Ausführen dieser Notebooks werden Sie drei wesentliche Informationen erhalten, die für Ihre Arbeit mit dem **Kria KV260** relevant sind:

1.  **Topologische Validierung (Netron Visualisierung):**
    Der Befehl `showInNetron` öffnet eine Grafik. Hier erkennen Sie, ob FINN die Layer des neuronalen Netzes (z.B. Convolution) erfolgreich in Hardware-Instanzen (z.B. `MVAU`) abstrahiert hat. Graue Knoten deuten auf Probleme bei der Quantisierung hin, blaue Knoten signalisieren erfolgreiche Hardware-Abbildung.

2.  **Verifikationsstatus (Integrity Check):**
    Die Ausgabe `Status: ..._SUCCESS.npy` bestätigt, dass die mathematische Operation des Netzes trotz Transformation in Festkomma-Arithmetik und Dataflow-Architektur korrekt bleibt. Dies ist entscheidend für die Validität Ihres trainierten Modells.

3.  **Kria-spezifische Ressourcen (Resource Estimation):**
    Im JSON-Output sehen Sie nun möglicherweise den Eintrag **URAM** (UltraRAM). Das Kria KV260 verfügt über diese spezielle Speicherart. Wenn FINN `URAM` statt `BRAM` nutzt, haben Sie wertvollen Speicherplatz für andere Logik gespart. Dies ist ein direkter Vorteil der UltraScale+ Architektur gegenüber dem Pynq-Z1.
