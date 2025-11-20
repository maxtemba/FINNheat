

# Bedienungsanleitung: FINN-Entwicklungsumgebung (Docker)

Diese Dokumentation beschreibt den Workflow zur Nutzung der persistenten FINN-Docker-Umgebung. Das System ist so konfiguriert, dass der Container nach dem Beenden nicht gelöscht wird, sondern gestoppt verbleibt, sodass installierte Pakete und Daten erhalten bleiben.

### 1. Initialisierung der Umgebung

Um die Arbeit zu beginnen, öffnen Sie ein Terminal auf Ihrem Host-System und navigieren Sie in das FINN-Verzeichnis. Starten Sie das Skript:

Bash

```
./run-docker.sh
```

- **Verhalten:** Sie befinden sich nun in der interaktiven Shell des Containers (`bash`).

- **Hinweis zur Persistenz:** Sollte der Container (`finn_dev_...`) bereits existieren, wird kein neuer Container erstellt. Das Skript erkennt die vorhandene Instanz und verbindet Sie automatisch wieder mit dieser.

### 2. Starten des Jupyter Notebook Servers

Da innerhalb des Containers spezifische Schreibrechte im Home-Verzeichnis eingeschränkt sind, muss der Jupyter-Server mit expliziten Pfaden für Konfigurations- und Laufzeitdateien gestartet werden.

Führen Sie im Container folgenden Befehl aus:

Bash

```
export JUPYTER_CONFIG_DIR=/tmp/jupyter_config && export JUPYTER_RUNTIME_DIR=/tmp/jupyter_runtime && jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root
```

1. Der Server startet und gibt diverse Logs aus.

2. Suchen Sie am Ende der Ausgabe nach einer URL, die folgendem Muster entspricht: `http://127.0.0.1:8888/?token=...`

3. Kopieren Sie diese URL und öffnen Sie sie in einem Webbrowser auf Ihrem Host-System.

4. **Wichtig:** Lassen Sie dieses Terminal-Fenster geöffnet. Wenn Sie den Prozess hier beenden, stoppt auch der Jupyter-Server.

### 3. Paralleles Arbeiten (Multi-Terminal-Workflow)

Da das erste Terminal durch den laufenden Jupyter-Prozess blockiert ist, empfiehlt es sich, für Kompilierungen, Dateimanagement oder Git-Operationen eine zweite Shell zu öffnen.

1. Öffnen Sie ein **neues, zweites Terminal-Fenster** auf Ihrem Host-System.

2. Verbinden Sie sich mit dem laufenden Container über folgenden Befehl:

Bash

```
docker exec -it finn_dev_kokinama bash
```

*(Hinweis: Sollte Ihr Container-Name abweichen, ersetzen Sie `finn_dev_kokinama` durch den entsprechenden Namen, der beim Start in Schritt 1 angezeigt wurde.)*

Sie können nun in diesem Fenster Befehle ausführen, deren Auswirkungen (z. B. generierte Dateien) sofort im Jupyter Notebook (Browser) sichtbar sind.

### 4. Beenden der Sitzung

- **Jupyter beenden:** Drücken Sie im ersten Terminal `Strg + C` und bestätigen Sie mit `y`.

- **Container verlassen:** Geben Sie `exit` in den Terminals ein.

- **Status:** Der Container wird gestoppt, aber **nicht gelöscht**. Beim nächsten Ausführen von `./run-docker.sh` wird der exakte Zustand wiederhergestellt.

---

### Tipp zur Effizienzsteigerung (Alias)

Um den komplexen Startbefehl für Jupyter nicht jedes Mal manuell eingeben zu müssen, können Sie einen Alias in der Konfiguration des Containers hinterlegen. Führen Sie dies einmalig innerhalb des Containers aus:

Bash

```
echo 'alias start_jupyter="export JUPYTER_CONFIG_DIR=/tmp/jupyter_config && export JUPYTER_RUNTIME_DIR=/tmp/jupyter_runtime && jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root"' >> ~/.bashrc
source ~/.bashrc
```

Künftig genügt der Befehl `start_jupyter`, um den Server zu starten.





# 🇬🇧 User Manual: FINN Development Environment (Docker)

This document outlines the workflow for using the persistent FINN Docker environment. The system is configured to ensure that the container is not deleted upon exit but remains stopped, preserving installed packages and data.

### 1. Initializing the Environment

To begin work, open a terminal on your host machine and navigate to the FINN directory. Execute the script:

Bash

```
./run-docker.sh
```

- **Behavior:** You will be placed directly into the container's interactive shell (`bash`).

- **Note on Persistence:** If the container (`finn_dev_...`) already exists, no new container is created. The script detects the existing instance and automatically reconnects you to it.

### 2. Launching the Jupyter Notebook Server

Due to restricted write permissions within the container's home directory, the Jupyter server must be launched with explicit paths for configuration and runtime files.

Execute the following command inside the container:

Bash

```
export JUPYTER_CONFIG_DIR=/tmp/jupyter_config && export JUPYTER_RUNTIME_DIR=/tmp/jupyter_runtime && jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root
```

1. The server will start and display various logs.

2. Locate the URL at the end of the output matching the pattern: `http://127.0.0.1:8888/?token=...`

3. Copy this URL and open it in a web browser on your host machine.

4. **Important:** Keep this terminal window open. Terminating the process here will stop the Jupyter server.

### 3. Parallel Workflow (Multi-Terminal Access)

Since the first terminal is blocked by the running Jupyter process, it is recommended to open a second shell for compilation, file management, or Git operations.

1. Open a **new, second terminal window** on your host machine.

2. Connect to the running container using the following command:

Bash

```
docker exec -it finn_dev_kokinama bash
```

*(Note: If your container name differs, replace `finn_dev_kokinama` with the specific name displayed during initialization in Step 1.)*

You can now execute commands in this window, and the results (e.g., generated files) will be immediately visible in the Jupyter Notebook.

### 4. Terminating the Session

- **Stopping Jupyter:** Press `Ctrl + C` in the first terminal and confirm with `y`.

- **Exiting the Container:** Type `exit` in the terminals.

- **Status:** The container is stopped but **not deleted**. Running `./run-docker.sh` again will restore the exact state.

---

### Efficiency Tip (Alias Configuration)

To avoid typing the complex Jupyter start command manually every time, you can create an alias within the container configuration. Run this once inside the container:

Bash

```
echo 'alias start_jupyter="export JUPYTER_CONFIG_DIR=/tmp/jupyter_config && export JUPYTER_RUNTIME_DIR=/tmp/jupyter_runtime && jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root"' >> ~/.bashrc
source ~/.bashrc
```

From now on, you can simply use the command `start_jupyter` to launch the server.
