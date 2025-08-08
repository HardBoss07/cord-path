# `cord-path`

A high-performance CLI tool for finding an optimal travel path through a set of 2D points. `cord-path` leverages NVIDIA CUDA to accelerate distance matrix calculations, making it ideal for large datasets where speed is critical.

> **Requires:** NVIDIA driver **580.88** or later.

## 🗺️ Features

* **CUDA-Accelerated Pathfinding**: Utilizes NVIDIA CUDA for extremely fast distance matrix computations.
* **Optimized Routes**: Employs the **Nearest Neighbor** heuristic combined with **2-opt** optimization for efficient and improved path results.
* **Flexible Coordinate Support**: Handles both positive and **negative coordinates**.
* **Customizable Start**: Optional feature to specify a custom **start position**.
* **Batch Processing**: A **quiet mode** is available for use in scripts, suppressing all console output.
* **CSV Output**: Easily save the final, optimized path to a **CSV** file.

## ⬇️ Installation

Download the latest pre-compiled binary from the [Releases](./releases) page. Place the executable in a directory that is included in your system's `PATH`.

### Linux

```bash
chmod +x cord-path
sudo mv cord-path /usr/local/bin/
````

### Windows (PowerShell)

```powershell
Move-Item .\cord-path.exe "C:\Program Files\cord-path\"
# You may need to add "C:\Program Files\cord-path\" to your system's PATH
```

## 🚀 Usage

```bash
cord-path -f <file.csv> [options]
```

### Required Argument

| Flag | Description |
| :--- | :--- |
| `-f`, `--file <FILE>` | Path to a CSV file containing coordinates. Each line must be in the format: `x,y` |

### Optional Arguments

| Flag | Description |
| :--- | :--- |
| `-x`, `--start-x <X>` | Optional starting X coordinate (supports negative values). |
| `-y`, `--start-y <Y>` | Optional starting Y coordinate (supports negative values). |
| `-o`, `--output <FILE>`| Path to the output CSV file where the calculated path will be saved. |
| `-q`, `--quiet` | Suppresses all console output. **Only works when used with `--output`**. |
| `-h`, `--help` | Displays the help message and exits. |
| `-V`, `--version` | Displays the version number and exits. |

## 📄 Examples

### 1\. Simple run (console output only)

```bash
cord-path -f points.csv
```

**Output:**

```
Path length before 2-opt: 51234.77
Path length after 2-opt: 49201.12
Path order:
Point 0: (100, 250)
Point 1: (150, 200)
...
```

### 2\. With a custom starting position

```bash
cord-path -f points.csv -x 200 -y -500
```

### 3\. Outputting to a CSV file

```bash
cord-path -f points.csv -o result.csv
```

### 4\. Silent mode for scripts

```bash
cord-path -f points.csv -o result.csv -q
```

This command will save the path to `result.csv` without printing any information to the console.

## 📂 CSV File Format

The input file must be a plain CSV with two integer or float values per line, representing the X and Y coordinates.

```csv
100,200
150.5,220.8
-50,400
...
```

## ⚙️ Performance Notes

  * This tool's performance is dependent on the CUDA kernel used for distance calculations.
  * An **NVIDIA driver version 580.88 or newer** is mandatory.
  * For optimal performance, ensure your GPU is configured for high-performance mode.

## 🛠️ Compile from Source

### Prerequisites

  * **Rust** (latest stable): [https://rustup.rs](https://rustup.rs)
  * **NVIDIA CUDA Toolkit** installed and configured
  * **NVIDIA driver 580.88+**
  * A C++ compiler compatible with your CUDA version

### Build Steps

1.  **Clone the repository:**

    ```bash
    git clone [https://github.com/HardBoss07/cord-path.git](https://github.com/HardBoss07/cord-path.git)
    cd cord-path
    ```

2.  **Build the CUDA static library:**

    ```bash
    cd cuda
    nvcc -O3 --compiler-options '-fPIC' -c kernels.cu -o kernels.o
    ar rcs libcuda_kernels.a kernels.o
    cd ..
    ```

3.  **Build the Rust CLI:**

    ```bash
    cargo build --release
    ```

The final executable binary will be located at:
`target/release/cord-path`

## ⚖️ License

This project is licensed under the **GNU General Public License v3.0 (GPL-3.0)**.

You can read the full license text in the [LICENSE](https://www.google.com/search?q=LICENSE) file.