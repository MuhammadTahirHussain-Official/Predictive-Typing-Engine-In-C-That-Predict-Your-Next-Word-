# VERYX Neural Engine v10.0

> **A Secure, Interactive N-gram & Spell-Correcting Language Engine + Arcade Games**

---

**VERYX** is an advanced terminal-based neural language engine in C. It features:
- next-word and sentence prediction via n-gram learning,
- spelling correction/modeling,
- secure user authentication (roles: User/Admin),
- dynamic, full-color UI with interactive ASCII art, animations, and witty feedback,
- a mini-game language arcade for vocabulary training and fun!

---

## Table of Contents

- [Features](#features)
- [Screenshots](#screenshots)
- [Installation](#installation)
- [Usage](#usage)
- [File Structure](#file-structure)
- [Technology Overview](#technology-overview)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)

---

## Features

- **N-gram Language Model:**  
  Trains on arbitrary text files to learn vocab, bigrams, trigrams for auto-completion, prediction, and creative writing.

- **Predictive & Style-Based Sentence Generation:**  
  Generates sentences from core n-gram model or by mimicking detected topics/styles.

- **Integrated Spell Corrector:**  
  Trains from word lists or text corpora. Provides live suggestions and corrections at word-level.

- **Role-based Secure Authentication:**  
  - User/Admin split login, strong password storage (hashed), admin actions (create, view, delete users).
  - Admin "secret code" required for privileged access.

- **Interactive, Themed Terminal UI:**  
  - Full-width dynamic boxes, ASCII banners, color-coded output via ANSI escapes (with Windows/Unix support).
  - Progress bars, typing effects, witty remarks, and motivational feedback.

- **Language Arcade Mini-games:**  
   1. **Word Scramble** – Unscramble letters to form a known word.
   2. **Predict-the-Next** – Guess what the model expects as the next word.
   3. **Word Duel** – Enter a longer, valid word than the model suggests.
   - Game grading, hints, timing, and scoring per round.

- **Model Save/Load, Reset, and Statistics:**  
  - On-disk persistent storage for n-gram/spell/user models.
  - Easy viewing of model size, topics, vocabulary, and efficiency.

- **Multi-platform Support:**  
  - Runs on Linux, macOS, Windows (with minor caveats for terminal colors/input).

---

## Screenshots

> _See the UI in action — animated boxes, full-line banners, witty remarks, and interactive games!_

![VERYX Banner & UI Example](screenshots/veryx_banner_example.png)

---

## Installation

1. **Clone the repository:**
    ```sh
    git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
    cd YOUR_REPO_NAME
    ```

2. **Compile:**
    - __Linux/macOS__:  
      ```sh
      gcc -std=c99 -o veryx final.c
      ```
    - __Windows (MinGW or similar)__:  
      ```sh
      gcc -std=c99 -o veryx.exe final.c
      ```

    - _Dependencies_: Only requires a standard C environment.  
      On Windows, for full color support and beeping, run in Windows Terminal or enable **Virtual Terminal Processing**.

3. **Run:**
    ```sh
    ./veryx         # Linux/macOS
    ./veryx.exe     # Windows
    ```

---

## Usage

1. **Startup:**  
   - The system boots up with animated banners and witty remarks.

2. **Sign Up/Login:**  
   - Choose user/admin, use admin secret code for admin features.

3. **Admin Panel:**  
   - Train models (provide `.txt` files), manage users, reset/view model stats, play arcade games.

4. **User Dashboard:**  
   - Build sentences, mimic style generation, check spellings, view stats, play games.

5. **Games:**  
   - Access via the game arcade menu.

6. **Data:**  
   - User logins, N-gram, spell, and user data are persistent between runs.

---

## File Structure

```
final.c                # Main source code (this project)
users.dat              # User database (auto-created)
trained_model.txt      # Saved n-gram model (auto-created)
spell_model.txt        # Saved spell dictionary (auto-created)
screenshots/           # (optional) Example UI captures
README.md              # This file
```

---

## Technology Overview

- **Core Algorithms:**  
  - C-based hash tables for high-speed n-gram/bigram/trigram prediction.
  - Modal dialog interface, colored full-width box drawing, and dynamic terminal UI.
  - Secure password hashing for credentials.
  - In-memory and on-disk model management.

- **Model Training:**  
  - Provide a corpus (plain text) for training.
  - The system ingests, hashes, and auto-detects frequent topics and writing styles.

- **Spell Correction:**  
  - Uses Levenshtein (edit distance) and hash-based lookup for rapid correction.

---

## Configuration

- All file paths are relative; ensure working directory permissions are set.
- Colors depend on ANSI escape code support—modern terminals work best.
- Admin secret code defaults to `123a` but can easily be changed in the `#define ADMIN_SECRET_CODE` macro at the top of the source.

---

## Troubleshooting

- **Colors or Beeps not working (Windows):**  
  - Run in Windows Terminal or PowerShell 7+.
  - For classic consoles, enable legacy `Virtual Terminal Processing`.

- **Compile Errors:**  
  - Ensure `gcc` (or `clang`) is available and using C99 or newer.
  - On Windows, be sure all standard libraries are available.

- **Large Input Files:**  
  - Can handle very large files as training sets, but memory use will grow accordingly.

---

## Contributing

- Fork this repository
- Create a branch (`git checkout -b feature/your-feature`)
- Commit changes (`git commit -am 'Add some feature'`)
- Push branch (`git push origin feature/your-feature`)
- Open a Pull Request

---

## License

Released under the MIT License.

---

## Acknowledgements

- [ANSI Escape Codes](https://en.wikipedia.org/wiki/ANSI_escape_code)
- All open-source contributors and projects inspiring the modular approach.
- Created & Designed by Muhammad Tahir Hussain

---

**Unleash creative writing – power up your vocabulary – and play smart!**
