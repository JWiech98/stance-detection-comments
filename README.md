Automatische Stance Detection in Bürgerbeteiligungskommentaren mittels LLM


## Beschreibung des Projekts:

Im Rahmen einer Masterarbeit wurde ein Werkzeug entwickelt, dass bei der Auswertung von Kommentaren, die zu Beiträgen eines Beteiligungsverfahrens abgegeben wurden, unterstützt. Zunächst wird untersucht, welche planerisch wichtigen Hauptaussagen die Ursprungsbeiträge enthalten. Anschließend wird untersucht, welche Stance (Haltung) die Kommentare zu diesen jeweiligen Hauptaussagen einnehmen. Dafür werden jeweils Prompts erstellt und von einem LLM verarbeitet.  

## Inhaltsverzeichnis
1. [Installation](#installation)
2. [Verwendung](#verwendung)
3. [Beispiele](#beispiele)
4. [Lizenz](#lizenz)
5. [Autor](#autor)

## Installation

1. Klonen Sie das Repository
```bash
git clone https://github.com/JWiech98/stance-detection-comments.git
```

2. Installieren Sie die Abhängigkeiten, indem Sie die requirements.txt verwenden:
```bash
pip install -r requirements.txt
```
## Verwendung

Um das Skript auszuführen, verwenden Sie den folgenden Befehl:
```bash
 python stancedetection_code.py /pfad/zu/beitraegen.xlsx /pfad/zu/kommentaren.json /pfad/zu/api_key.txt (OPTIONAL: Huggingface Modell-ID)
```

Die Beiträge werden in Form einer Excel-Datei erwartet. Die Kommentare werden in Form einer JSON-Datei erwartet.

Der API-Key muss zuvor auf HuggingFace beantragt werden.
Das LLM kann grundsätzlich ausgetauscht werden, wodurch jedoch die Qualität und Struktur der Ergebnisse beeinflusst werden kann. Beachten Sie, dass für das jeweilig verwendete Sprachmodell Nutzungsrechte vorliegen müssen. Beantragen Sie diese auf HuggingFace.

Gleiches gilt für die verwendeten Prompts. Sie können im Code an den Anwendungskontext angepasst werden, was jedoch Auswirkungen auf Struktur und Qualität der Ergebnisse haben kann.


## Beispiele

Beispielbeitrag:
"Der Heinrich-Traun-Platz (HTP) in der Heinrich-Traun-Straße ist eigentlich ein Lieblingsort in der Nachbarschaft. Leider ist er oft in einem ungepflegten Zustand, öffentliche Sitzmöglichkeiten sind nicht vorhanden.
Eine qualitative Aufwertung würde die Aufenthaltsqualität und Beliebtheit deutlich erhöhen."

Zugehöriger Beispielkommentar:
"Vielleicht kann die Nutzung der Wiese zum Spielen UND Sitzgelegenheiten für Erwachsene möglich sein."

Ergebnisse der Stance Detection:

<table class="tg"><thead>
  <tr>
    <th class="tg-0pky">Beitragsnummer</th>
    <th class="tg-0pky">Beitragstext</th>
    <th class="tg-0pky">Hauptaussage</th>
    <th class="tg-0pky">Kommentarnummer</th>
    <th class="tg-0pky">Kommentartext</th>
    <th class="tg-0pky">Haltung</th>
  </tr></thead>
<tbody>
  <tr>
    <td class="tg-c3ow" rowspan="3">8784</td>
    <td class="tg-0pky" rowspan="3">Der Heinrich-Traun-Platz (HTP) in der Heinrich-Traun-Straße ist eigentlich ein Lieblingsort in der Nachbarschaft. Leider ist er oft in einem ungepflegten Zustand, öffentliche Sitzmöglichkeiten sind nicht vorhanden.<br>Eine qualitative Aufwertung würde die Aufenthaltsqualität und Beliebtheit deutlich erhöhen. </td>
    <td class="tg-0pky">1. Ungepflegter Zustand am Heinrich-Traun-Platz (Problem)</td>
    <td class="tg-c3ow" rowspan="3">17516</td>
    <td class="tg-0pky" rowspan="3">Vielleicht kann die Nutzung der Wiese zum Spielen UND Sitzgelegenheiten für Erwachsene möglich sein.</td>
    <td class="tg-c3ow">Neutralität</td>
  </tr>
  <tr>
    <td class="tg-0pky">2. Fehlen von öffentlichen Sitzmöglichkeiten (Problem)</td>
    <td class="tg-c3ow">Neutralität</td>
  </tr>
  <tr>
    <td class="tg-0pky">3. Potenzial für Aufwertung der Aufenthaltsqualität (Vorschlag)</td>
    <td class="tg-c3ow">Zustimmung</td>
  </tr>
</tbody></table>

## Lizenz

MIT License

Copyright (c) 2024 JWiech98

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## Autor 

JWiech98
