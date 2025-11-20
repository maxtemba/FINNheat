# User Manual: FINN Development Environment (Docker)

This document outlines the workflow for using the persistent FINN Docker environment. The system is configured to ensure that the container is not deleted upon exit but remains stopped, preserving installed packages and data.

### 1\. Initializing the Environment

To begin work, open a terminal on your host machine and navigate to the FINN directory. Execute the script:

```bash
./run-docker.sh
```

  * **Behavior:** You will be placed directly into the container's interactive shell (`bash`).
  * **Note on Persistence:** If the container (`finn_dev_...`) already exists, no new container is created. The script detects the existing instance and automatically reconnects you to it.

### 2\. Launching the Jupyter Notebook Server

Due to restricted write permissions within the container's home directory, the Jupyter server must be launched with explicit paths for configuration and runtime files.

Execute the following command inside the container:

```bash
export JUPYTER_CONFIG_DIR=/tmp/jupyter_config && export JUPYTER_RUNTIME_DIR=/tmp/jupyter_runtime && jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root
```

1.  The server will start and display various logs.
2.  Locate the URL at the end of the output matching the pattern: `http://127.0.0.1:8888/?token=...`
3.  Copy this URL and open it in a web browser on your host machine.
4.  **Important:** Keep this terminal window open. Terminating the process here will stop the Jupyter server.

### 3\. Parallel Workflow (Multi-Terminal Access)

Since the first terminal is blocked by the running Jupyter process, it is recommended to open a second shell for compilation, file management, or Git operations.

1.  Open a **new, second terminal window** on your host machine.
2.  Connect to the running container using the following command:

<!-- end list -->

```bash
docker exec -it finn_dev_kokinama bash
```

*(Note: If your container name differs, replace `finn_dev_kokinama` with the specific name displayed during initialization in Step 1.)*

You can now execute commands in this window, and the results (e.g., generated files) will be immediately visible in the Jupyter Notebook.

### 4\. Terminating the Session

  * **Stopping Jupyter:** Press `Ctrl + C` in the first terminal and confirm with `y`.
  * **Exiting the Container:** Type `exit` in the terminals.
  * **Status:** The container is stopped but **not deleted**. Running `./run-docker.sh` again will restore the exact state.

-----

### Efficiency Tip (Alias Configuration)

To avoid typing the complex Jupyter start command manually every time, you can create an alias within the container configuration. Run this once inside the container:

```bash
echo 'alias start_jupyter="export JUPYTER_CONFIG_DIR=/tmp/jupyter_config && export JUPYTER_RUNTIME_DIR=/tmp/jupyter_runtime && jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root"' >> ~/.bashrc
source ~/.bashrc
```

From now on, you can simply use the command `start_jupyter` to launch the server.
