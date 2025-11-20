ist nicht schlimm





Das ist ein sehr spezifisches Problem, das auftritt, wenn man mit **QONNX-Modellen** (quantisierte Modelle im FINN-Kontext) arbeitet.

### Die Ursache

Der Fehler InvalidGraph ... No opset import for domain 'qonnx.custom_op.general' bedeutet:

Ihr Modell enthält spezielle Quantisierungs-Knoten (Custom Ops), die von der Standard-Bibliothek onnxruntime nicht verstanden werden. onnxruntime kennt nur Standard-ONNX-Operationen, aber keine QONNX-Spezialitäten wie Quant.

### Die Lösung

Anstatt `onnxruntime` direkt zu nutzen, müssen wir die **Ausführungs-Engine von QONNX/FINN** verwenden. Diese ist in Python geschrieben und kann diese speziellen Knoten simulieren.

Bitte ersetzen Sie die **Zelle 3** ("Generierung von Referenzdaten") in Ihrem Notebook durch den folgenden korrigierten Code.

#### 🇩🇪 Korrigierter Code für Zelle 3

Python

```
# --- KORRIGIERTE ZELLE 3 ---
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.core.onnx_exec import execute_onnx

# Statt onnxruntime nutzen wir den ModelWrapper von QONNX
model = ModelWrapper(clean_model_path)
input_name = model.graph.input[0].name
output_name = model.graph.output[0].name
input_shape = model.get_tensor_shape(input_name)

# Zufälligen Input generieren (Float32)
# Hinweis: Shape-Handling für Batch-Size 1 falls nötig
if isinstance(input_shape[0], str): input_shape[0] = 1 
input_data = np.random.rand(*input_shape).astype(np.float32)

# Referenz-Output berechnen mittels QONNX Python-Simulation
# Dies unterstützt die speziellen Quant-Nodes
input_dict = {input_name: input_data}
output_dict = execute_onnx(model, input_dict)
expected_output = output_dict[output_name]

# Als .npy Dateien speichern (FINN erwartet diese Namen)
np.save("input.npy", input_data)
np.save("expected_output.npy", expected_output)
print("Referenzdaten (input.npy, expected_output.npy) erfolgreich generiert.")
```

---

#### 🇬🇧 Corrected Code for Cell 3

Python

```
# --- CORRECTED CELL 3 ---
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.core.onnx_exec import execute_onnx

# Instead of standard onnxruntime, we use the QONNX ModelWrapper
model = ModelWrapper(clean_model_path)
input_name = model.graph.input[0].name
output_name = model.graph.output[0].name
input_shape = model.get_tensor_shape(input_name)

# Generate random input (Float32)
# Note: Handle shape for batch size 1 explicitly if defined as string
if isinstance(input_shape[0], str): input_shape[0] = 1 
input_data = np.random.rand(*input_shape).astype(np.float32)

# Calculate reference output using QONNX Python simulation
# This natively supports the custom Quant nodes
input_dict = {input_name: input_data}
output_dict = execute_onnx(model, input_dict)
expected_output = output_dict[output_name]

# Save as .npy files (FINN expects these specific filenames)
np.save("input.npy", input_data)
np.save("expected_output.npy", expected_output)
print("Reference data (input.npy, expected_output.npy) successfully generated.")
```

### Erklärung / Explanation

🇩🇪 Deutsch:

Wir haben ort.InferenceSession durch qonnx.core.onnx_exec.execute_onnx ersetzt. Die Standard-Runtime scheitert an den Quant-Knoten, da diese keine offiziellen ONNX-Operatoren sind. Die QONNX-Bibliothek, die Teil der FINN-Umgebung ist, bringt eine eigene Python-Implementierung dieser Operatoren mit und kann den Graphen daher korrekt ausführen, um die Referenzdaten zu erzeugen.

🇬🇧 English:

We replaced ort.InferenceSession with qonnx.core.onnx_exec.execute_onnx. The standard runtime fails on Quant nodes because they are not official ONNX operators. The QONNX library, which is part of the FINN environment, provides its own Python implementation of these operators, allowing it to correctly execute the graph and generate the reference data.
