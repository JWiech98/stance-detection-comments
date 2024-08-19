import argparse
import csv
import html
import json
import pandas as pd
import transformers
from huggingface_hub import login
from torch import bfloat16



def importCons(con_file: str) -> dict:
    '''
    Importiert der Kommentare für die Stance Detection aus einer Excel-Datei.

    Parameters:
        file_path: Dateipfad zum Import der Beiträge

    Returns:
        contDict: Dictionary der Form {BID: Beitragsinhalt} bestehend aus Beitrags-ID und zugehörigem Beitragstext
    '''
    try:

        # Einlesen der Excel-Datei
        df = pd.read_excel(con_file)
        # Import der Beiträe (Beitrags-ID und Beitragstext) und Hinzufügen zum Beitrags-Dictionary
        df = df.sort_values(by=["contribution_id"])
        contDict = {df.at[id, "contribution_id"]: df.at[id, "contribution_content"] for id in df.index}

        return contDict
    
    except FileNotFoundError:

        # Fehlermeldung, sofern die Datei mit Beiträgen nicht gefunden werden konnte
        print(f"Die Beitrags-Datei '{con_file}' konnte nicht gefunden werden.")

        return None

def importComs(com_file: str) -> dict:
    '''
    Importiert der Kommentare für die Stance Detection aus einer JSON-Datei.
    
    Parameters:
        file_path: Dateipfad zur JSON-Datei, Kommentardatei.
    
    Returns:
        comDict: Dictionary der Form {KID: {BID: Kommentarinhalt}} bestehend aus Kommentar-ID, der Beitrags-ID,
            zu dem der Kommentar verfasst wurde, sowie dem Kommentartext.
    ''' 
    try:

        # Einlesen der JSON-Datei
        with open(com_file, "r") as commentFile:
            comments = json.load(commentFile)

        comDict = {}

        # Iteration über die Kommentare
        for com in comments:
            if com is not None:
                # Import der Kommentare (Kommentar-ID, zugehöriger Beitrags-ID (RelatedNode-ID) und Kommentartext)
                comment_id = next(iter(com))
                comment_data = com[comment_id]
                contribution_id = comment_data.get("related_node_id", "")
                content = comment_data.get("text", "")
                if content is not None:
                    content = html.unescape(content)
                # Hinzufügen der Kommentare samt ihrer Daten zum Kommentar-Dictionary
                comDict[comment_id] = {"Beitragsnummer": contribution_id, "Kommentartext": content}

        return comDict
    
    except FileNotFoundError:

        # Fehlermeldung, sofern die Datei mit Kommentaren nicht gefunden werden konnte
        print(f"Die Kommentar-Datei '{com_file}' konnte nicht gefunden werden.")

        return None

def joinConsComs(contributions: dict, comments: dict) -> dict:
    """
    Fügt Beiträge und zugehörige Kommentare über die Beitrags-ID in einem Dictionary zusammen.
    Beiträge ohne zugehörigen Kommentar werden aussortiert.

    Parameters:
        contributions: Dictionary mit Beiträgen der Form {BID: Beitragsinhalt}
        comments: Dictionary mit Kommentaren der Form {KID: {BID: Kommentarinhalt}}

    Returns:
        joinedData: Dictionary, das Beitrag-IDs Beitragsinhalte sowie alle zugehörigen Kommentare zuodrnet. Es hat die Form
            {Beitrags-ID: {Beitragstext: {Kommentar-ID: Kommentartext}}}.

    """
    joinedData = {}

    # Iteration über die Beiträge
    for contribution_id, contribution_content in contributions.items():
        #Erstellen eines Eintrags für jeden Beitrag
        joinedEntry = {'Beitrag': contribution_content, 'Kommentare': {}}
        # Iteration über die Kommentare
        for comment_id, comment_data in comments.items():
            # Zuordnung der Kommentare zu ihrem jeweiligen Beitrag
            if comment_data['Beitragsnummer'] == str(contribution_id):
                joinedEntry['Kommentare'][comment_id] = comment_data['Kommentartext']
        # Aufnahme der Einträge, sofern Kommentare zum Beitrag existieren
        if joinedEntry['Kommentare']:
            joinedData[contribution_id] = joinedEntry

    return joinedData

def loadApiKey(file_path: str) -> str:
    """
    Lädt den API-Schlüssel für den Zugriff auf das LLM über HuggingFace aus einer Textdatei.

    Parameters:
        file_path: Dateipfad zur Datei, die den API-Schlüssel enthält.

    Returns:
        api_key: Der geladene API-Schlüssel als Zeichenfolge.
    """

    try:

        # Einlesen der Textdatei mit dem API-Key
        with open(file_path, "r") as file:
            api_key = file.read().strip()

        return api_key
    
    except FileNotFoundError:

        # Fehlermeldung, sofern die Datei mit dem API-Key nicht gefunden werden konnte
        print(f"Die API-Key-Datei '{file_path}' konnte nicht gefunden werden.")

        return None

def loadLLM(model_id: str):
    """
    Lädt ein LLM und den zugehörigen Tokenizer von HuggingFace und initialisiert es.

    Parameters:
        model_id: Die Modell-ID aus dem HuggingFace-Hub.

    Returns:
        pipe: Pipeline für die spätere Verarbeitungung der Prompts hinsichtlich Tokenisierung und Textgenerierung.
        
    """
    # Quantisierung des Modells zur Reduktion der benötigten Ressourcen
    bnb_config = transformers.BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=bfloat16
    )

    # Laden von Tokenizer und Modell unter der Quantisierung
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
    llm = transformers.AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        quantization_config=bnb_config,
        device_map='auto',
    )
    llm.eval()

    # Initialisieren der Pipeline
    pipe = transformers.pipeline(
        model=llm, tokenizer=tokenizer,
        task='text-generation',
        temperature=0.7,
        max_new_tokens=500,
        repetition_penalty=1.1
    )

    return pipe

def generatePromptsForTargetIdentification(cons: dict) -> dict:
    """
    Generiert Prompts für die Ermittlung von Kernaussagen innerhalb von Beiträgen, welche die späteren Ziele (Targets) der Stance Detection bilden.

    Parameters:
        contribs: Dictionary mit Beiträgen {BID: Inhalt}

    Returns:
        prompt_dict: Dictionary der Form {BID: Prompt} bestehend aus Beitrags-IDs mit zugehörigen Prompts
        zur Ermittlung der Kernaussagen.
    """

    prompt_dict = {}

    # Statischer System-Prompt für die Target Identification mit allgemeinen Informationen für das LLM
    ti_system = """
    <s>[INST] <<SYS>>
    Du bist ein hilfreicher, respektvoller und ehrlicher Assistent eines Stadtplaners.
    Deine Aufgabe ist es, aus Beiträgen die enthaltenen Hauptaussagen herauszuarbeiten.
    Fasse die Hauptaussagen auf Deutsch und stichpunktartig zusammen.
    Stelle sicher, dass jede Hauptaussage entweder ein Problem, ein Lob oder ein Vorschlag ist.
    Es sind ausschließlich diese Kategorien zulässig. Andere Kategorien sollen nicht ausgegeben werden.
    Die Hauptaussagen sollen als nummerierte Liste ausgegeben werden.

    Definitionen:
    - Problem: Eine negative Beobachtung, ein Hindernis oder eine Gefahr, das bzw. die angesprochen wird.
    - Lob: Positive Rückmeldung oder Anerkennung für etwas, das gut funktioniert.
    - Vorschlag: Vorschläge oder Ideen zur Verbesserung einer Situation.
    <</SYS>>
    """

    # Statischer Beispiel-Prompt für die Target Identification mit einer Beispielaufgabe für das LLM
    ti_example = """
    Ermittle die Hauptaussagen des folgenden Beitrages.
    Beitrag:
    Der Erdkampsweg ist zwischen Ratsmühlendamm und Wacholderstraße das gewerbliche Zentrum von Fuhlsbüttel. In den letzten Jahren wurde viel erreicht durch neue Aufpflasterungen, Bänke und zaghafte Schritte zur Verkehrsberuhigung.
    Bei Neubauten wie zuletzt dem ReweCity wird leider immer noch Parkraum geschaffen, der die Fußwege kreuzt. Ein verkehrsberuhigtes, fußgänger- und radfahrerfreundliches Fuhlsbüttel könnte die Lebensqualität noch einmal erhöhen.
    [/INST]
    Hauptaussagen:
    1. Aufwertung des Erdkampsweg (Lob)
    2. Parkraum bei Neubauten kreuzt Fußwege (Problem)
    3. Verkehrsberuhigung (Vorschlag)
    """

    # Iteration über die Beiträge des übergebenen Dictionarys und Erstellung eines Prompts zur Target Identification für jeden Beitrag daraus
    for con_id, con_txt in cons.items():
        ti_main = f"""
        [INST]
        Ermittle die Hauptaussagen des folgenden Beitrages.
        Beitrag:
        {con_txt}
        [/INST]
        Hauptaussagen:
        """
        # Kombination von System-, Beispiel- und beitragsspezifischem Prompt und Hinzufügen zum Prompt-Dictionary für Targetbestimmung
        prompt = ti_system + ti_example + ti_main
        prompt_dict[str(con_id)] = [prompt]

    return prompt_dict


def extractTargetsInContributions(prompt_dict: dict, pipe) -> dict:
    """
    Extrahiert die Hauptaussagen (Targets) aus den Beiträgen durch die Übergabe der Prompts an die Pipeline des LLM.

    Parameters:
        prompt_dict: Dictionary bestehend aus Beitrags-IDs mit zugehörigen Prompts zur Ermittlung der Kernaussagen.
        pipe: Pipeline zur Verarbeitung der Prompts durch das LLM

    Returns:
        aspect_results: Dictionary der Form {BID: [Hauptaussagen]} bestehend aus Beitrags-IDs und den in den Beiträgen 
            identifizierten Hauptaussagen als Liste.
    """

    aspect_results = {}

    # Iteration über die Prompts zur Extraktion der Hauptaussagen im Prompt-Dictionary
    for con_id, prompts in prompt_dict.items():
        aspect_results[con_id] = []
        # Generieren aller Hauptaussagen (Targets) eines jeweiligen Beitrags
        for prompt in prompts:
            answer = pipe(prompt)
            generated_text = answer[0]["generated_text"]
            # Extraktion der Hauptaussagen (Targets) aus der modellgenerierten Antwort
            generated_answer = generated_text.strip().split('Hauptaussagen:')[-1].strip()
            statements = [statement.strip() for statement in generated_answer.split('\n') if statement.strip()]
            aspect_results[con_id].extend(statements)

    return aspect_results

def generatePromptsForStanceDetection(aspects: dict, com_dict: dict) -> list:
    """
    Generiert Prompts für die Stance Detection. Für jede der Kernaussagen (Target), die aus dem betrachteten Beitrag ermittelt wurde,
    wird ein Prompt generiert, der die Haltung des Kommentars hinsichtlich der identifizierten Hauptaussage erfragen soll.

    Parameters:
        aspects: Dictionary, das Beitrags-IDs und zugehörige, identifizierte Kernaussagen (Targets der Stance Detection) enthält.
        comments_dict: Dictionary, das Beitrag-IDs die Beitragsinhalte sowie alle zugehörigen Kommentare zuordnet.

    Returns:
        prompts: Eine Liste bestehend aus Tupeln, wobei jedes Tupel die Beitrags-ID, die Hauptaussage, die Kommentar-ID, den Kommentartext und den entsprechenden Prompt enthält.
        
    """

    prompts = []

    # Statischer System-Prompt für die Stance Detection mit allgemeinen Informationen für das LLM
    sd_system = """
        <s>[INST] <<SYS>>
        Du bist ein hilfreicher, respektvoller und ehrlicher Assistent eines Stadtplaners.
        Deine Aufgabe ist es zu bewerten, wie ein Kommentar sich zu einer Hauptaussage positioniert.
        Gib diese Polarität auf Deutsch wieder.
        Stelle sicher, dass jede Polarität entweder Zustimmung, Widerspruch oder Neutralität ist.
        Es sind ausschließlich diese Polaritäten zulässig. Andere Polaritäten sollen nicht ausgegeben werden.
        Gib ausschließlich die Polarität aus. Andere Ergebnisse sollen nicht ausgegeben werden.

        Definitionen:
        - Zustimmung: Der Kommentar stimmt der Hauptaussage zu oder zeigt eine positive Reaktion auf den Inhalt der Hauptaussage. Das Problem, das Lob oder der Vorschlag wird befürwortet.
        - Widerspruch: Der Kommentar widerspricht der Hauptaussage oder zeigt eine negative Reaktion auf den Inhalt der Hauptaussage. Das Problem, das Lob oder der Vorschlag wird abgelehnt.
        - Neutralität: Der Kommentar thematisiert die Hauptaussage nicht und stellt keinen Bezug zu ihr her.
        <</SYS>>
        """
    # Statischer Beispiel-Prompt für die Stance Detection
    sd_example = """
        Klassifiziere, ob der folgende Kommentar der folgenden Hauptaussage zustimmt, widerspricht oder neutral gegenüber ist.
        Hauptaussage:
        - Gehwege in schlechtem Zustand (Problem)
        Kommentar:
        - Einige Gehwege sind in schlechtem Zustand mit Stolperfallen durch Unebenheiten und teilweise abschüssig und sollten dringend saniert werden.
        [/INST]
        Polarität:
        Zustimmung
        [INST]
        Klassifiziere, ob der folgende Kommentar der folgenden Hauptaussage zustimmt, widerspricht oder neutral gegenüber ist.
        Hauptaussage:
        - Lärmbelästigung durch Feuerwehreinsätze (Problem)
        Kommentar:
        - Die Feuerwehr fährt mit Sicherheit nicht mega schnell - man sollte froh sein, dass diese schnell zum Einsatzort fahren kann und die „Lärmbelästigung „ ist auch nicht wirklich gegeben da oft - gerade in der Nacht - auf das Martinshorn verzichtet wird !
        [/INST]
        Polarität:
        Widerspruch
        [INST]
        Klassifiziere, ob der folgende Kommentar der folgenden Hauptaussage zustimmt, widerspricht oder neutral gegenüber ist.
        Hauptaussage:
        - Fehlende Parkmöglichkeit für Lastenräder und mit Verankerung (Problem)
        Kommentar:
        - Mich stören schon E-Roller. Bitte keine klobigen Drahtesel überall im Stadtbild.
        [/INST]
        Polarität:
        Neutralität
        """
    
    aspects_int_keys = {int(key): value for key, value in aspects.items()}
    
    # Iteration über die identifizierten Hauptaussagen (Targets)
    for aspect_id, aspect_list in aspects_int_keys.items():
        # Überprüfung, ob der zu einer Hauptaussage (Target) gehöriger Beitrag überhaupt Kommentare besitzt
        if aspect_id in com_dict:
            # Abrufen der zugehörigen Kommentare
            comments_to_contrib = com_dict[aspect_id]['Kommentare']
            # Iteration über alle Hauptaussagen (Targets) eines Beitrags
            for aspect_txt in aspect_list:
                # Iteration über alle zum Beitrag der Hauptaussage gehörenden Kommentare
                for com_id, com_txt in comments_to_contrib.items():
                    # Generieren des Stance Detection-Prompts für jedes Hauptaussage-Kommentar-Paar
                    sd_main = f"""
                          [INST]
                          Klassifiziere, ob der folgende Kommentar der folgenden Hauptaussage zustimmt, widerspricht oder neutral gegenüber ist.
                          Hauptaussage:
                          - {aspect_txt}
                          Kommentar:
                          - {com_txt}
                          [/INST]
                          Polarität:
                          """
                    # Erstellen eines Tupels aus Beitrags-ID, Hauptaussage, Kommentar-ID, Kommentartext und Prompt und
                    # Hinzufügen zur Liste mit den Prompts für die Stance Detection
                    prompt_data = (aspect_id, aspect_txt, com_id, com_txt, sd_system + sd_example + sd_main)
                    prompts.append(prompt_data)

    return prompts

def getTargets(aspect_ids: list, aspects: dict) -> list:
    """
    Ruft Hauptaussagen anhand der Beitrags-ID ab.

    Parameters:
        aspect_ids: Liste von Beitrags-IDs.
        aspects: Dictionary, das Beitrags-IDs zugehörige Kernaussagen zuordnet.

    Returns:
        aspect_list: Liste von Hauptaussagen mit zugehöriger Beitrags-ID.
    """

    aspect_list = []

    for aspect_id in aspect_ids:
        aspect_list.extend(aspects.get(str(aspect_id), []))

    return aspect_list

def saveTargets(contributions: dict, aspects: dict, output_file: str = 'Targets.csv') -> None:
    """
    Speichert die Beitragsdaten zusammen mit den zugehörigen identifizierten Hauptaussagen (Targets) in einer CSV-Datei.

    Parameters:
        contributions: Dictionary, das Beitrag-IDs mit zugehörigen Beitragsinhalten enthält.
        aspects: Dictionary, das Beitrags-IDs enthält und diesen eine Liste der extrahierten Hauptaussagen zuordnet.
        output_file: Dateipfad zur CSV-Datei, in der die Daten gespeichert werden sollen. Der Default-Parameter ist 'Targets.csv'.

    """

    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        # Definieren der Spaltennamen für die CSV-Datei
        columns = ['Beitrags-ID', 'Beitragstext', 'Hauptaussagen']
        writer = csv.DictWriter(csvfile, fieldnames=columns, delimiter=';')
        writer.writeheader()
        
        # Iteration über die Beiträge und zugehörigen Hauptaussagen
        for aspect_id, con_text in contributions.items():
            # Ermitteln der identifizierten Hauptaussagen für den aktuellen Beitrag
            aspects_of_con = getTargets([aspect_id], aspects)
            aspects_combined = "\n".join(aspects_of_con)
            # Einfügen der Daten in die Datei
            writer.writerow({
                'Beitrags-ID': aspect_id, 
                'Beitragstext': con_text, 
                'Hauptaussagen': aspects_combined
                })
    
import csv

def saveStance(stanceDetPrompts: list, pipe, contributions: dict, output_file: str = 'Stance.csv') -> None:
    """
    Speichert die Ergebnisse der Stance Detection zusammen mit den Beitrags- und Kommentardaten in einer CSV-Datei.

    Parameters:
        stanceDetPrompts: Liste von Tupeln. Ein Tupel enthält die Beitrags-ID, die extrahierte Hauptaussage, die Kommentar-ID, den Kommentartext und den entsprechenden Prompt zur Stance Detection.
        pipe: Pipeline zur Verarbeitung des Prompts für die Stance Detection.
        contributions: Dictionary, das Beitrag-IDs mit zugehörigen Beitragsinhalten enthält.
        output_file: Dateipfad zur CSV-Datei, in der die Daten gespeichert werden sollen. Der Default-Parameter ist 'Stance.csv'.

    """

    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        # Definieren der Spaltennamen für die CSV-Datei
        columns = ['Beitrags-ID', 'Beitragstext', 'Hauptaussage', 'Kommentar-ID', 'Kommentartext', 'Haltung', 'Begründung']
        writer = csv.DictWriter(csvfile, fieldnames=columns, delimiter=';')
        writer.writeheader()
        
        # Iteration über die erstellten Prompts zur Stance Detection
        for aspect_id, aspect, com_id, comment, prompt in stanceDetPrompts:
            # Stance Detection für den jeweiligen Prompt
            result = pipe(prompt)
            generated_text = result[0]["generated_text"].strip()
            
            # Postprocessing des generierten Textes für das erwünschte Antwortformat
            last_stance_index = generated_text.rfind('Polarität:')
            if last_stance_index != -1:
                stance_text = generated_text[last_stance_index + len('Polarität:'):].strip()
                stance_lines = stance_text.split('\n', 1)
                stance = stance_lines[0].strip()
                reasoning = stance_lines[1].strip() if len(stance_lines) > 1 else ""
            else:
                stance = 'Unbekannt'
                reasoning = generated_text
            
            # # Einfügen der Daten in die Datei
            writer.writerow({
                'Beitrags-ID': aspect_id, 
                'Beitragstext': contributions[aspect_id], 
                'Hauptaussage': aspect, 
                'Kommentar-ID': com_id, 
                'Kommentartext': comment, 
                'Haltung': stance,
                'Begründung': reasoning
            })

def main(contributions_file: str, comments_file: str, api_key_file: str, model_id: str = 'mistralai/Mistral-8x7B-Instruct-v0.1') -> None:
    """
    Hauptfunktion zur Verarbeitung von Beitrags- und Kommentardaten mittels Stance Detection

    Parameters:
        contributions_file: Pfad zur Beitragsdatei.
        comments_file: Pfad zur Kommentardatei.
        api_key_file: Pfad zur Datei mit dem API-Schlüssel der HuggiongFace-API.
        model_id: Modell-ID des verwendeten Sprachmodells. Der Default-Parameter und damit das standardmäßig verwendete Modell ist 'mistralai/Mistral-8x7B-Instruct-v0.1'.

    """
    # Import der Beiträge und Kommentare
    contributions = importCons(contributions_file)
    comments = importComs(comments_file)
    entries = joinConsComs(contributions, comments)

    # Laden des API-Schlüssels
    api_key = loadApiKey(api_key_file)
    if api_key:
        login(api_key)
    else:
        print('API-Schlüssel konnte nicht geladen werden.')
        return

    # Laden des LLMs
    pipe = loadLLM(model_id)

    # Extrahieren der Hauptaussagen (Targets) aus den Beiträgen
    contributionPrompts = generatePromptsForTargetIdentification(contributions)
    targets = extractTargetsInContributions(contributionPrompts, pipe)

    # Speichern der Hauptaussagen (Targets)
    saveTargets(contributions, targets)

    # Generieren von Prompts für die Stance Detection
    stance_det_prompts = generatePromptsForStanceDetection(targets, entries)

    # Erheben und Speichern der Daten der Stance Detection
    saveStance(stance_det_prompts, pipe, contributions)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Führt eine Stance Detection für Kommentare zu Beiträgen durch.')
    parser.add_argument("contributions_file", type=str, help="Pfad zur Datei mit den Beiträgen")
    parser.add_argument("comments_file", type=str, help="Pfad zur Datei mit den Kommentaren")
    parser.add_argument("api_key_file", type=str, help="Pfad zur Datei mit dem HF-API-Schlüssel")
    parser.add_argument("model_id", type=str, default='mistralai/Mixtral-8x7B-Instruct-v0.1', help="Modell-ID des verwendeten Sprachmodells")
    args = parser.parse_args()
    main(args.contributions_file, args.comments_file, args.api_key_file, args.model_id)